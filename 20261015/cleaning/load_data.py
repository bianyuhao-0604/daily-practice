# load_data.py
import pandas as pd
import os

DATA_DIR = "data/olist"  # 下载后存放 CSV 的目录

def load_orders() -> pd.DataFrame:
    """
    加载 Olist orders 表。
    列名示例：order_id, customer_id, order_status,
             order_purchase_timestamp, order_delivered_customer_date,
             order_estimated_delivery_date
    """
    path = os.path.join(DATA_DIR, "olist_orders_dataset.csv")
    return pd.read_csv(path)

def load_customers() -> pd.DataFrame:
    """加载 customers 表，用于外键存在性校验"""
    path = os.path.join(DATA_DIR, "olist_customers_dataset.csv")
    return pd.read_csv(path)
