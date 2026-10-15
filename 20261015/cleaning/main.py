from datasets import load_dataset
import pandas as pd
from cleaner import clean
from config import CleaningConfig, OutlierRule

# 1. 加载数据集
ds = load_dataset("miminmoons/olist-ecommerce-for-delivery-and-review-prediction")
raw = ds["train"].to_pandas()

# 2. 查看列名和主键情况
print(raw.columns.tolist())
print("order_id 唯一数:", raw["order_id"].nunique(), "总行数:", len(raw))

# 3. 根据实际数据调整配置
config = CleaningConfig(
    primary_keys=["order_id", "product_id"],   # 如果一行是一个订单商品，主键可能是复合的
    positive_columns=["price", "freight_value"],
    outliers={
        "price": OutlierRule(method="iqr", iqr_multiplier=1.5, action="cap"),
        "freight_value": OutlierRule(method="iqr", iqr_multiplier=1.5, action="cap"),
    },
)

# 4. 调用清洗
clean_df, audit = clean(raw, config)

# 5. 查看结果
print(f"清洗前 {audit['initial_rows']} 行，清洗后 {audit['final_rows']} 行")
print("删除行数:", audit["total_rows_dropped"])
for step in audit["steps"]:
    if step.get("rows_affected", 0) > 0:
        print(step)