import pandas as pd
from datasets import load_dataset

ds = load_dataset("miminmoons/olist-ecommerce-for-delivery-and-review-prediction")
raw = ds["train"].to_pandas()

# 1. 查看列名，确认有哪些字段
print(raw.columns.tolist())

# 2. 找出 (order_id, product_id) 重复的行
dup_mask = raw.duplicated(subset=["order_id", "product_id"], keep=False)
dups = raw[dup_mask].sort_values(["order_id", "product_id"])
print(f"重复行数: {len(dups)}")
print(dups.head(20))

# 3. 检查重复行是否完全一致
full_dup = raw.duplicated(keep=False)
print(f"完全重复的行数: {full_dup.sum()}")

# 4. 检查重复行里 seller_id / price / order_item_id 是否不同
if "seller_id" in raw.columns:
    print("重复行中不同 seller_id 的数量:",
          dups.groupby(["order_id", "product_id"])["seller_id"].nunique().gt(1).sum())
# 检查重复行是否在所有 18 列上都完全相同
dup_mask = raw.duplicated(subset=["order_id", "product_id"], keep=False)
dup_rows = raw[dup_mask]

# 按 (order_id, product_id) 分组，检查每组内是否所有列都一致
def all_same(g):
    return g.drop_duplicates().shape[0] == 1

groups = dup_rows.groupby(["order_id", "product_id"])
fully_identical = groups.apply(all_same).all()
print(f"所有重复组是否完全一致: {fully_identical}")

# 如果为 True，按 (order_id, product_id) 去重就是安全的
# 顺便看下去重后每个订单的商品数分布
after_dedup = raw.drop_duplicates(subset=["order_id", "product_id"])
print("去重后每订单商品数分布:")
print(after_dedup.groupby("order_id").size().value_counts().sort_index())    