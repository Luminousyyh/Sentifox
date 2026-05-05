"""
模拟数据生成器
用于快速验证系统功能，无需真实爬取
"""
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Optional
from crawlers.base import Post, GraphEdge, BaseCrawler


# 模拟内容模板
POSITIVE_TEMPLATES = [
    "{brand}的产品真的太棒了，用起来非常顺手，强烈推荐给大家！",
    "刚刚体验了{brand}的新功能，感觉{product}做得越来越好了，点赞！",
    "{brand}的客服态度很好，问题很快就解决了，很满意这次的服务。",
    "用了{brand}的{product}一段时间了，体验非常不错，性价比很高。",
    "{brand}这次的活动力度很大，果断下单了，期待收货！",
]

NEGATIVE_TEMPLATES = [
    "{brand}的{product}质量真的太差了，用了两天就坏了，非常失望。",
    "对{brand}的服务非常不满，投诉了几次都没有回应，建议大家谨慎选择。",
    "{brand}这次的操作真的让人寒心，完全没有把用户当回事。",
    "买了{brand}的{product}，体验极差，退款流程也很麻烦。",
    "{brand}的宣传和实际完全不符，感觉被欺骗了，不会再买了。",
]

NEUTRAL_TEMPLATES = [
    "{brand}发布了新款{product}，大家怎么看？",
    "有人用过{brand}的{product}吗？求真实评价。",
    "{brand}最近好像有一些争议，具体情况不太清楚。",
    "看到很多人在讨论{brand}，来凑个热闹。",
    "{brand}的{product}和竞品相比，有什么优势吗？",
]

AUTHORS = [
    "科技达人小明", "数码控_阿杰", "生活观察者", "北漂青年", "产品经理老王",
    "设计狮Lisa", "代码农工", "美食探店员", "旅行日记", "财经评论员",
    "校园小编", "职场新人", "宝妈育儿经", "健身狂人", "电影爱好者",
    "游戏主播KK", "读书君", "汽车发烧友", "房产观察", "健康小贴士",
]

PLATFORMS = ["微博", "知乎", "小红书", "抖音", "新闻", "论坛"]

BRANDS = ["某品牌", "某公司", "某产品"]
PRODUCTS = ["手机", "笔记本", "耳机", "App", "服务", "会员"]


def _random_time(days_back: int = 7) -> datetime:
    """生成随机时间"""
    now = datetime.now()
    delta = timedelta(
        days=random.randint(0, days_back),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )
    return now - delta


def generate_mock_posts(
    keywords: Optional[List[str]] = None,
    num_posts: int = 200,
    sentiment_ratio: tuple = (0.3, 0.4, 0.3),  # (负面, 中性, 正面)
    days_back: int = 7,
) -> List[Post]:
    """
    生成模拟帖子数据
    :param keywords: 监控关键词，用于替换模板中的占位符
    :param num_posts: 生成数量
    :param sentiment_ratio: 情感比例 (negative, neutral, positive)
    :param days_back: 时间范围（天）
    """
    if keywords is None:
        keywords = ["品牌"]

    brand = random.choice(keywords) if keywords else "某品牌"
    product = random.choice(PRODUCTS)

    posts = []
    labels = ["negative"] * int(num_posts * sentiment_ratio[0]) + \
             ["neutral"] * int(num_posts * sentiment_ratio[1]) + \
             ["positive"] * int(num_posts * sentiment_ratio[2])

    # 补齐数量
    while len(labels) < num_posts:
        labels.append(random.choice(["negative", "neutral", "positive"]))
    random.shuffle(labels)

    for i, label in enumerate(labels[:num_posts]):
        if label == "positive":
            template = random.choice(POSITIVE_TEMPLATES)
        elif label == "negative":
            template = random.choice(NEGATIVE_TEMPLATES)
        else:
            template = random.choice(NEUTRAL_TEMPLATES)

        content = template.format(brand=brand, product=product)
        # 添加一些随机变化避免完全重复
        if random.random() < 0.3:
            content += random.choice(["", " #话题讨论", " [图片]", " [视频]", " @好友"])

        post_id = f"mock_{uuid.uuid4().hex[:12]}"
        platform = random.choice(PLATFORMS)

        post = Post(
            post_id=post_id,
            platform=platform,
            content=content,
            author=random.choice(AUTHORS) + f"_{random.randint(1, 999)}",
            author_id=f"uid_{random.randint(10000, 99999)}",
            publish_time=_random_time(days_back),
            url=f"https://example.com/{platform}/{post_id}",
            sentiment_label=label,
            likes=random.randint(0, 5000),
            comments=random.randint(0, 1000),
            reposts=random.randint(0, 2000),
        )
        posts.append(post)

    return posts


def generate_mock_edges(posts: List[Post], edge_ratio: float = 0.15) -> List[GraphEdge]:
    """
    从帖子中模拟生成传播关系边
    :param posts: 帖子列表
    :param edge_ratio: 产生边的帖子比例
    """
    edges = []
    if len(posts) < 2:
        return edges

    num_edges = max(1, int(len(posts) * edge_ratio))

    for _ in range(num_edges):
        source_post = random.choice(posts)
        target_post = random.choice(posts)
        # Support both Post objects and dicts
        s_id = source_post.post_id if hasattr(source_post, "post_id") else source_post.get("post_id")
        t_id = target_post.post_id if hasattr(target_post, "post_id") else target_post.get("post_id")
        if s_id == t_id:
            continue

        relation_types = ["repost", "comment", "mention"]
        s_author = source_post.author_id if hasattr(source_post, "author_id") else source_post.get("author_id")
        t_author = target_post.author_id if hasattr(target_post, "author_id") else target_post.get("author_id")
        s_time = source_post.publish_time if hasattr(source_post, "publish_time") else source_post.get("publish_time")
        s_plat = source_post.platform if hasattr(source_post, "platform") else source_post.get("platform")
        edge = GraphEdge(
            source_id=s_author,
            target_id=t_author,
            relation_type=random.choice(relation_types),
            timestamp=s_time,
            weight=random.uniform(0.5, 3.0),
            platform=s_plat,
        )
        edges.append(edge)

    return edges


class MockCrawler(BaseCrawler):
    """模拟爬虫实现"""

    def __init__(self, keywords: Optional[List[str]] = None):
        super().__init__(platform="模拟数据", keywords=keywords or ["品牌"])

    def crawl(self, max_posts: int = 200) -> List[Post]:
        return generate_mock_posts(
            keywords=self.keywords,
            num_posts=max_posts,
        )

    def extract_edges(self, posts: List[Post]) -> List[GraphEdge]:
        return generate_mock_edges(posts)
