# cleaner.py
import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any, Optional
from config import CleaningConfig, OutlierRule

def _normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """列名规范化：snake_case、去空格、去特殊字符"""
    logs = []
    mapping = {}
    for col in df.columns:
        new_col = re.sub(r'[^\w\s]', '', col.strip().lower())
        new_col = re.sub(r'\s+', '_', new_col)
        new_col = re.sub(r'_+', '_', new_col).strip('_')
        mapping[col] = new_col
    df = df.rename(columns=mapping)
    for old, new in mapping.items():
        if old != new:
            logs.append({"step": "normalize_columns", "old": old, "new": new,
                         "rows_affected": 0})
    return df, logs

def _report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """缺失率报告"""
    total = len(df)
    report = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        report.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(n_missing),
            "missing_ratio": round(n_missing / total, 4) if total else 0,
        })
    return pd.DataFrame(report).sort_values("missing_ratio", ascending=False)

def _handle_missing(
    df: pd.DataFrame, config: CleaningConfig
) -> Tuple[pd.DataFrame, List[Dict]]:
    """按配置处理缺失值，返回处理后 df 和审计日志"""
    logs = []
    n_before = len(df)
    mr = config.missing

    # 1. 删除缺失率过高的列
    total = len(df)
    cols_to_drop = [
        c for c in df.columns
        if df[c].isna().sum() / total > mr.drop_column_threshold
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logs.append({
            "step": "drop_high_missing_columns",
            "columns": cols_to_drop,
            "rows_affected": 0,
        })

    # 2. 关键列缺失即删行
    if mr.critical_columns:
        existing_critical = [c for c in mr.critical_columns if c in df.columns]
        mask = df[existing_critical].notna().all(axis=1)
        dropped = (~mask).sum()
        df = df[mask].reset_index(drop=True)
        logs.append({
            "step": "drop_critical_missing_rows",
            "columns": existing_critical,
            "rows_affected": int(dropped),
            "rows_before": n_before,
            "rows_after": len(df),
        })

    # 3. 逐列填充
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            strategy = mr.fill_strategy.get("numeric", "median")
            fill_val = {
                "median": df[col].median(),
                "mean": df[col].mean(),
                "zero": 0,
            }.get(strategy, df[col].median())
            df[col] = df[col].fillna(fill_val)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            strategy = mr.fill_strategy.get("datetime", "drop")
            if strategy == "drop":
                continue
        else:
            strategy = mr.fill_strategy.get("categorical", "mode")
            if strategy == "mode" and not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
        logs.append({
            "step": "fill_missing",
            "column": col,
            "strategy": strategy,
            "rows_affected": 0,
        })

    return df, logs

def _detect_outliers_iqr(series: pd.Series, multiplier: float) -> pd.Series:
    """IQR 法返回布尔 mask（True = 异常）"""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    return (series < lower) | (series > upper)

def _detect_outliers_zscore(series: pd.Series, threshold: float) -> pd.Series:
    """Z-score 法返回布尔 mask"""
    mean, std = series.mean(), series.std()
    if std == 0:
        return pd.Series(False, index=series.index)
    z = (series - mean) / std
    return z.abs() > threshold

def _handle_outliers(
    df: pd.DataFrame, config: CleaningConfig
) -> Tuple[pd.DataFrame, List[Dict]]:
    """按配置处理异常值"""
    logs = []
    for col, rule in config.outliers.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if rule.method == "iqr":
            mask = _detect_outliers_iqr(series, rule.iqr_multiplier)
        elif rule.method == "zscore":
            mask = _detect_outliers_zscore(series, rule.zscore_threshold)
        elif rule.method == "range":
            mask = pd.Series(False, index=series.index)
            if rule.lower is not None:
                mask |= series < rule.lower
            if rule.upper is not None:
                mask |= series > rule.upper
        else:
            continue

        n_outliers = int(mask.sum())
        if n_outliers == 0:
            continue

        if rule.action == "drop":
            df = df[~df.index.isin(series[mask].index)].reset_index(drop=True)
            logs.append({
                "step": "drop_outliers",
                "column": col,
                "method": rule.method,
                "outliers_found": n_outliers,
                "rows_affected": n_outliers,
            })
        elif rule.action == "cap":
            if rule.method == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - rule.iqr_multiplier * iqr
                upper = q3 + rule.iqr_multiplier * iqr
            else:
                mean, std = series.mean(), series.std()
                lower = mean - rule.zscore_threshold * std
                upper = mean + rule.zscore_threshold * std
            df[col] = df[col].clip(lower, upper)
            logs.append({
                "step": "cap_outliers",
                "column": col,
                "method": rule.method,
                "outliers_found": n_outliers,
                "rows_affected": 0,
            })
        elif rule.action == "flag":
            df[f"{col}_is_outlier"] = mask.reindex(df.index, fill_value=False)
            logs.append({
                "step": "flag_outliers",
                "column": col,
                "method": rule.method,
                "outliers_found": n_outliers,
                "rows_affected": 0,
            })
    return df, logs

def _enforce_positive(
    df: pd.DataFrame, columns: List[str]
) -> Tuple[pd.DataFrame, List[Dict]]:
    """强制正数列：<=0 的行删除"""
    logs = []
    for col in columns:
        if col not in df.columns:
            continue
        mask = df[col].notna() & (df[col] <= 0)
        n_bad = int(mask.sum())
        if n_bad > 0:
            df = df[~mask].reset_index(drop=True)
            logs.append({
                "step": "enforce_positive",
                "column": col,
                "rows_affected": n_bad,
            })
    return df, logs

def _deduplicate(
    df: pd.DataFrame, primary_keys: List[str]
) -> Tuple[pd.DataFrame, List[Dict]]:
    """按主键去重，保留第一条"""
    logs = []
    existing_pks = [c for c in primary_keys if c in df.columns]
    if not existing_pks:
        return df, logs
    n_before = len(df)
    df = df.drop_duplicates(subset=existing_pks, keep="first").reset_index(drop=True)
    dropped = n_before - len(df)
    if dropped > 0:
        logs.append({
            "step": "deduplicate",
            "primary_keys": existing_pks,
            "rows_affected": int(dropped),
        })
    return df, logs

def clean(
    raw: pd.DataFrame,
    config: Optional[CleaningConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    纯函数：清洗原始 DataFrame，返回 (clean_df, audit_log)。
    不修改输入 raw，所有操作在副本上进行。
    """
    if config is None:
        config = CleaningConfig()

    df = raw.copy()
    audit_log: Dict[str, Any] = {
        "initial_rows": len(df),
        "initial_columns": len(df.columns),
        "steps": [],
    }

    # Step 1: 列名规范化
    df, logs = _normalize_columns(df)
    audit_log["steps"].extend(logs)

    # Step 2: 缺失率报告（记录但不改数据）
    missing_report = _report_missing(df)
    audit_log["missing_report"] = missing_report.to_dict(orient="records")

    # Step 3: 缺失值处理
    df, logs = _handle_missing(df, config)
    audit_log["steps"].extend(logs)

    # Step 4: 异常值处理
    df, logs = _handle_outliers(df, config)
    audit_log["steps"].extend(logs)

    # Step 5: 强制正数列
    df, logs = _enforce_positive(df, config.positive_columns)
    audit_log["steps"].extend(logs)

    # Step 6: 去重
    df, logs = _deduplicate(df, config.primary_keys)
    audit_log["steps"].extend(logs)

    # 汇总
    audit_log["final_rows"] = len(df)
    audit_log["final_columns"] = len(df.columns)
    audit_log["total_rows_dropped"] = audit_log["initial_rows"] - audit_log["final_rows"]

    return df, audit_log