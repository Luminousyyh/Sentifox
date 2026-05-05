"""
舆情分析系统全局配置
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "sentiment.db")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "output")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ========== 情感分析配置 ==========
SENTIMENT_MODEL = os.path.join(BASE_DIR, "models", "sentiment")
SENTIMENT_LABELS = {0: "negative", 1: "positive"}
BATCH_SIZE = 32

# ========== 话题聚类配置 ==========
TOPIC_CLUSTER_NUM = 5
TOPIC_MIN_DF = 2
TOPIC_MAX_DF = 0.95
STOPWORDS_PATH = os.path.join(BASE_DIR, "utils", "stopwords.txt")

# ========== 告警配置 ==========
ALERT_CONFIG = {
    "negative_ratio_threshold": 0.6,      # 负面情感占比超过此值触发告警
    "volume_spike_multiplier": 3.0,       # 提及量相对前一时间窗口突增倍数
    "volume_spike_window_minutes": 30,    # 突增检测时间窗口（分钟）
    "check_interval_seconds": 300,        # 告警检查间隔
}

# ========== 传播图谱配置 ==========
GRAPH_CONFIG = {
    "default_layout": "force_atlas2",     # 网络图布局算法
    "node_size_factor": 10,               # 节点大小系数
    "edge_width_factor": 1,               # 边宽度系数
    "sir_beta": 0.3,                      # SIR 感染率
    "sir_gamma": 0.1,                     # SIR 恢复率
    "sir_steps": 50,                      # SIR 模拟步数
}

# ========== 数据源配置 ==========
PLATFORMS = ["微博", "知乎", "小红书", "抖音", "新闻", "论坛"]

# ========== 监控关键词 ==========
DEFAULT_KEYWORDS = ["品牌", "产品", "服务"]

# ========== 调度配置 ==========
SCHEDULER_INTERVAL_MINUTES = 5

# ========== LLM API 配置 (火山引擎) ==========
# 请从环境变量读取或手动填写你的 API Key
LLM_CONFIG = {
    "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
    "api_key": os.environ.get("VOLCES_API_KEY", ""),
    "chat_model": "doubao-seed-2.0-pro",      # 对话模型（用户可更换）
    "embedding_model": "doubao-embedding-vision",
    "timeout": 30,
    "max_retries": 3,
}

# ========== Cookie 配置（用户自行填写后启用真实爬虫） ==========
COOKIES = {
    "weibo": os.environ.get("WEIBO_COOKIE", ""),
    "zhihu": os.environ.get("ZHIHU_COOKIE", ""),
    "xiaohongshu": os.environ.get("XIAOHONGSHU_COOKIE", ""),
}

# ========== RAG 配置 ==========
RAG_CONFIG = {
    "top_k": 5,
    "chunk_size": 512,
    "collection_name": "sentiment_posts",
    "vector_db_path": os.path.join(DATA_DIR, "chroma_db"),
}


@dataclass
class CrawlerConfig:
    """爬虫配置"""
    platforms: List[str] = field(default_factory=lambda: PLATFORMS)
    keywords: List[str] = field(default_factory=lambda: DEFAULT_KEYWORDS)
    max_posts_per_platform: int = 200
    mock_data_ratio: float = 1.0           # 1.0 = 全部使用模拟数据


@dataclass
class AppConfig:
    """应用运行时配置"""
    page_title: str = "舆情分析系统"
    page_icon: str = "📊"
    layout: str = "wide"
    db_path: str = DB_PATH
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
    alert: Dict = field(default_factory=lambda: ALERT_CONFIG)
    graph: Dict = field(default_factory=lambda: GRAPH_CONFIG)


# 全局单例
CONFIG = AppConfig()
