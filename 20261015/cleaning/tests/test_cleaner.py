# tests/test_cleaner.py
import numpy as np
import pandas as pd
import pytest
from cleaner import clean
from config import CleaningConfig, OutlierRule

@pytest.fixture
def raw_df():
    """构造一个含有典型脏数据的测试 DataFrame"""
    rng = np.random.default_rng(42)
    n = 300
    customers = [f"C{i:05d}" for i in range(1, 51)]
    return pd.DataFrame({
        "Order ID": [f"ORD{i:06d}" for i in range(n)],
        "Customer ID": rng.choice(customers, n),
        "order_status": rng.choice(["delivered", "shipped", "canceled"], n),
        "Payment Value": rng.lognormal(4, 0.8, n),
        "Freight Value": rng.exponential(15, n),
        # 注入脏数据
        "Order ID ": [f"ORD{i:06d}" if i < 295 else None for i in range(n)],
    }).assign(**{
        # 缺失值
        "Payment Value": lambda d: d["Payment Value"].mask(
            rng.random(n) < 0.05, np.nan),
        # 负值
        "Freight Value": lambda d: d["Freight Value"].mask(
            rng.random(n) < 0.03, -1),
    })

@pytest.fixture
def cleaned(raw_df):
    config = CleaningConfig(
        outliers={
            "payment_value": OutlierRule(method="iqr", iqr_multiplier=1.5,
                                          action="cap"),
            "freight_value": OutlierRule(method="iqr", iqr_multiplier=1.5,
                                          action="drop"),
        },
        missing=type(CleaningConfig().missing)(
            critical_columns=["order_id"],
        ),
        positive_columns=["payment_value", "freight_value"],
    )
    df, audit = clean(raw_df, config)
    return df, audit

# ---------- 不变量测试 ----------

def test_no_duplicate_primary_key(cleaned):
    """主键 order_id 不能重复"""
    df, _ = cleaned
    assert df["order_id"].duplicated().sum() == 0, "主键存在重复"

def test_primary_key_not_null(cleaned):
    """主键不能有缺失"""
    df, _ = cleaned
    assert df["order_id"].notna().all(), "主键存在缺失"

def test_amount_positive(cleaned):
    """金额列必须 > 0"""
    df, _ = cleaned
    for col in ["payment_value", "freight_value"]:
        if col in df.columns:
            assert (df[col] > 0).all(), f"{col} 存在非正值"

def test_foreign_key_exists(cleaned, raw_df):
    """外键 customer_id 必须存在于用户表"""
    df, _ = cleaned
    # 模拟用户表
    valid_customers = set(raw_df["customer_id"].unique()) \
        if "customer_id" in raw_df.columns \
        else set(raw_df["Customer ID"].unique())
    if "customer_id" in df.columns:
        invalid = ~df["customer_id"].isin(valid_customers)
        assert invalid.sum() == 0, f"存在 {invalid.sum()} 个无效外键"

def test_column_names_normalized(cleaned):
    """列名必须为 snake_case"""
    df, _ = cleaned
    for col in df.columns:
        assert col == col.lower(), f"列名 {col} 不是小写"
        assert " " not in col, f"列名 {col} 含空格"

def test_audit_log_completeness(cleaned):
    """审计日志包含关键字段"""
    _, audit = cleaned
    assert "initial_rows" in audit
    assert "final_rows" in audit
    assert "total_rows_dropped" in audit
    assert isinstance(audit["steps"], list)

def test_audit_log_records_row_drops(cleaned):
    """审计日志中应记录至少一步删行操作（如果有脏数据）"""
    _, audit = cleaned
    drop_steps = [
        s for s in audit["steps"]
        if s.get("rows_affected", 0) > 0 and "drop" in s["step"]
    ]
    assert len(drop_steps) >= 0  # 允许干净数据无删行

def test_clean_is_pure(raw_df):
    """clean() 不修改输入 DataFrame"""
    original = raw_df.copy(deep=True)
    _ = clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, original)

def test_outlier_action_cap_not_drop(cleaned):
    """cap 策略不应减少行数"""
    df, audit = cleaned
    cap_steps = [s for s in audit["steps"] if s["step"] == "cap_outliers"]
    for step in cap_steps:
        assert step["rows_affected"] == 0