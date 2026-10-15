# config.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal

@dataclass
class OutlierRule:
    """单个列的异常值处理规则"""
    method: Literal["iqr", "zscore", "range"] = "iqr"
    # IQR 参数
    iqr_multiplier: float = 1.5
    # Z-score 参数
    zscore_threshold: float = 3.0
    # 硬边界（method="range" 时生效）
    lower: Optional[float] = None
    upper: Optional[float] = None
    # 处理策略：drop / cap / flag
    action: Literal["drop", "cap", "flag"] = "cap"

@dataclass
class MissingRule:
    """缺失值处理规则"""
    # 缺失率超过此阈值直接删列（默认 0.7 = 70%）
    drop_column_threshold: float = 0.7
    # 每列的具体填充策略
    fill_strategy: Dict[str, str] = field(default_factory=lambda: {
        "numeric": "median",     # median / mean / zero
        "categorical": "mode",   # mode / constant
        "datetime": "drop",      # 日期缺失通常保留原值
    })
    # 关键列：缺失即删行
    critical_columns: List[str] = field(default_factory=list)

@dataclass
class CleaningConfig:
    """全局清洗配置"""
    outliers: Dict[str, OutlierRule] = field(default_factory=lambda: {
        "price": OutlierRule(method="iqr", iqr_multiplier=1.5, action="drop"),
        "freight_value": OutlierRule(method="iqr", iqr_multiplier=3.0, action="cap"),
        "payment_value": OutlierRule(method="zscore", zscore_threshold=3.0, action="cap"),
    })
    missing: MissingRule = field(default_factory=MissingRule)
    # 主键与外键约束
    primary_keys: List[str] = field(default_factory=lambda: ["order_id"])
    # 外键: {子表列: 父表列}
    foreign_keys: Dict[str, str] = field(default_factory=lambda: {
        "customer_id": "customer_id",
    })
    # 必须为正的列
    positive_columns: List[str] = field(default_factory=lambda: [
        "price", "freight_value", "payment_value"
    ])