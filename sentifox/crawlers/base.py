"""
爬虫基类与统一数据模型
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod


@dataclass
class Post:
    """统一的帖子数据模型"""
    post_id: str
    platform: str
    content: str
    author: str = ""
    author_id: str = ""
    publish_time: Optional[datetime] = None
    url: str = ""
    sentiment_label: Optional[str] = None
    sentiment_score: Optional[float] = None
    topic_id: Optional[int] = None
    keywords: List[str] = field(default_factory=list)
    likes: int = 0
    comments: int = 0
    reposts: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "post_id": self.post_id,
            "platform": self.platform,
            "author": self.author,
            "author_id": self.author_id,
            "content": self.content,
            "publish_time": self.publish_time,
            "url": self.url,
            "sentiment_label": self.sentiment_label,
            "sentiment_score": self.sentiment_score,
            "topic_id": self.topic_id,
            "keywords": self.keywords,
            "likes": self.likes,
            "comments": self.comments,
            "reposts": self.reposts,
        }


@dataclass
class GraphEdge:
    """传播关系边模型"""
    source_id: str          # 源节点 ID（如转发者/评论者）
    target_id: str          # 目标节点 ID（如原帖作者/被转发帖）
    relation_type: str      # 关系类型：repost, comment, mention, similar
    timestamp: Optional[datetime] = None
    weight: float = 1.0
    platform: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "timestamp": self.timestamp,
            "weight": self.weight,
            "platform": self.platform,
        }


class BaseCrawler(ABC):
    """爬虫抽象基类"""

    def __init__(self, platform: str, keywords: List[str]):
        self.platform = platform
        self.keywords = keywords

    @abstractmethod
    def crawl(self, max_posts: int = 100) -> List[Post]:
        """执行采集，返回帖子列表"""
        pass

    @abstractmethod
    def extract_edges(self, posts: List[Post]) -> List[GraphEdge]:
        """从帖子中提取传播关系边"""
        pass

    def normalize_post(self, raw: Dict[str, Any]) -> Post:
        """将原始数据标准化为 Post 对象（子类可覆盖）"""
        return Post(
            post_id=str(raw.get("id", "")),
            platform=self.platform,
            content=raw.get("content", ""),
            author=raw.get("author", ""),
            author_id=str(raw.get("author_id", "")),
            publish_time=raw.get("publish_time"),
            url=raw.get("url", ""),
            likes=raw.get("likes", 0),
            comments=raw.get("comments", 0),
            reposts=raw.get("reposts", 0),
        )
