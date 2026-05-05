"""
爬虫管理器
统一管理各平台爬虫，支持并发采集
"""
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from crawlers.base import Post
from crawlers.mock_generator import MockCrawler
from crawlers.weibo import WeiboCrawler
from crawlers.zhihu import ZhihuCrawler
from crawlers.xiaohongshu import XiaohongshuCrawler
from crawlers.news import NewsCrawler
from config import CONFIG, COOKIES


class CrawlerManager:
    """爬虫管理器"""

    PLATFORM_MAP = {
        "微博": WeiboCrawler,
        "知乎": ZhihuCrawler,
        "小红书": XiaohongshuCrawler,
        "新闻": NewsCrawler,
        "抖音": None,     # 预留
        "论坛": None,     # 预留
    }

    def __init__(self, keywords: Optional[List[str]] = None):
        self.keywords = keywords or CONFIG.crawler.keywords

    def crawl_all(
        self,
        platforms: Optional[List[str]] = None,
        max_posts_per_platform: int = 50,
        use_mock_fallback: bool = True,
    ) -> Dict[str, List[Post]]:
        """
        并发采集多个平台
        :return: {platform: [Post, ...]}
        """
        platforms = platforms or CONFIG.crawler.platforms
        results = {}

        def crawl_platform(platform: str) -> tuple:
            crawler_cls = self.PLATFORM_MAP.get(platform)
            if crawler_cls is None:
                # 未实现的平台使用模拟数据
                if use_mock_fallback:
                    crawler = MockCrawler(keywords=self.keywords)
                    posts = crawler.crawl(max_posts=max_posts_per_platform)
                    for p in posts:
                        p.platform = platform
                    return platform, posts
                return platform, []

            try:
                crawler = crawler_cls(keywords=self.keywords)
                posts = crawler.crawl(max_posts=max_posts_per_platform)
                return platform, posts
            except Exception as e:
                print(f"[{platform}] 采集失败: {e}")
                if use_mock_fallback:
                    crawler = MockCrawler(keywords=self.keywords)
                    posts = crawler.crawl(max_posts=max_posts_per_platform)
                    for p in posts:
                        p.platform = platform
                    return platform, posts
                return platform, []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(crawl_platform, p): p for p in platforms}
            for future in as_completed(futures):
                platform, posts = future.result()
                results[platform] = posts

        return results

    def crawl_and_process(
        self,
        platforms: Optional[List[str]] = None,
        max_posts_per_platform: int = 50,
        use_mock_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        采集 + 情感分析 + 话题聚类 + 入库 + 向量库同步（完整流水线）
        :return: 所有帖子的字典列表
        """
        from analysis.sentiment import get_analyzer
        from analysis.topic_clustering import cluster_posts
        from utils.database import insert_posts, update_post_topic, insert_edges
        from rag.document_processor import sync_posts_to_vector_store

        # 1. 采集
        results = self.crawl_all(platforms, max_posts_per_platform, use_mock_fallback)
        all_posts = []
        all_edges = []

        for platform, posts in results.items():
            all_posts.extend(posts)
            # 提取边
            crawler_cls = self.PLATFORM_MAP.get(platform)
            if crawler_cls:
                try:
                    crawler = crawler_cls(keywords=self.keywords)
                    edges = crawler.extract_edges(posts)
                    all_edges.extend(edges)
                except Exception:
                    pass

        if not all_posts:
            return []

        # 2. 情感分析
        analyzer = get_analyzer()
        post_dicts = [p.to_dict() for p in all_posts]
        analyzed = analyzer.analyze_posts(post_dicts)

        # 3. 话题聚类
        clustered, topics = cluster_posts(analyzed, n_clusters=5)

        # 4. 入库
        insert_posts(clustered)
        for p in clustered:
            if p.get("topic_id") is not None:
                update_post_topic(p["post_id"], p["topic_id"])

        if all_edges:
            insert_edges([e.to_dict() for e in all_edges])

        # 5. 同步向量库
        try:
            sync_posts_to_vector_store(clustered)
        except Exception as e:
            print(f"向量库同步失败: {e}")

        return clustered

    def get_platform_status(self) -> Dict[str, Dict[str, Any]]:
        """获取各平台采集状态（Cookie 是否配置）"""
        status = {}
        for platform in CONFIG.crawler.platforms:
            crawler_cls = self.PLATFORM_MAP.get(platform)
            if crawler_cls is None:
                status[platform] = {"available": False, "reason": "暂未实现真实采集，将使用模拟数据"}
            elif platform in ["微博", "知乎", "小红书"]:
                key = platform.lower() if platform != "小红书" else "xiaohongshu"
                cookie = COOKIES.get(key, "")
                if not cookie:
                    status[platform] = {"available": False, "reason": f"未配置 {platform} Cookie"}
                else:
                    status[platform] = {"available": True, "reason": "已配置 Cookie"}
            else:
                status[platform] = {"available": True, "reason": "无需登录"}
        return status
