"""
知乎数据采集器
基于知乎搜索 API
"""
import json
from datetime import datetime
from typing import List, Optional
import httpx

from crawlers.base import BaseCrawler, Post, GraphEdge
from config import COOKIES


class ZhihuCrawler(BaseCrawler):
    """知乎爬虫"""

    SEARCH_API = "https://www.zhihu.com/api/v4/search_v3"

    def __init__(self, keywords: List[str], cookie: Optional[str] = None):
        super().__init__(platform="知乎", keywords=keywords)
        self.cookie = cookie or COOKIES.get("zhihu", "")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "x-requested-with": "fetch",
            "Referer": "https://www.zhihu.com/search?type=content",
        }
        if self.cookie:
            self.headers["Cookie"] = f"z_c0={self.cookie}"

    def crawl(self, max_posts: int = 100) -> List[Post]:
        """采集知乎搜索结果"""
        if not self.cookie:
            print("[知乎] 未配置 Cookie，跳过采集。请在 config.py 的 COOKIES['zhihu'] 中设置 z_c0 Cookie")
            return []

        posts = []
        keyword = self.keywords[0] if self.keywords else ""
        offset = 0
        limit = 20

        with httpx.Client(headers=self.headers, follow_redirects=True, timeout=15) as client:
            while len(posts) < max_posts and offset < 100:
                try:
                    params = {
                        "t": "general",
                        "q": keyword,
                        "offset": offset,
                        "limit": limit,
                    }

                    resp = client.get(self.SEARCH_API, params=params)
                    if resp.status_code != 200:
                        print(f"[知乎] 请求失败: {resp.status_code}")
                        break

                    data = resp.json()
                    items = data.get("data", [])

                    if not items:
                        break

                    for item in items:
                        if len(posts) >= max_posts:
                            break
                        post = self._parse_item(item)
                        if post:
                            posts.append(post)

                    offset += limit
                    if not data.get("paging", {}).get("is_end", True):
                        continue
                    break

                except Exception as e:
                    print(f"[知乎] 采集异常: {e}")
                    break

        print(f"[知乎] 采集完成: {len(posts)} 条")
        return posts

    def _parse_item(self, item: dict) -> Optional[Post]:
        """解析知乎搜索结果项"""
        obj = item.get("object", {})
        if not obj:
            return None

        content_type = obj.get("type", "")
        if content_type == "answer":
            content = obj.get("excerpt", "") or obj.get("content", "")
            author = obj.get("author", {})
            question = obj.get("question", {})
            qid = question.get("id", "")
            aid = obj.get("id", "")
            url = f"https://www.zhihu.com/question/{qid}/answer/{aid}"
        elif content_type == "article":
            content = obj.get("excerpt", "") or obj.get("content", "")
            author = obj.get("author", {})
            aid = obj.get("id", "")
            url = f"https://zhuanlan.zhihu.com/p/{aid}"
        else:
            return None

        content = content.strip()
        if not content or len(content) < 10:
            return None

        created_time = obj.get("created_time", 0)
        publish_time = datetime.fromtimestamp(created_time) if created_time else None

        return Post(
            post_id=f"zhihu_{aid}",
            platform="知乎",
            content=content[:500],  # 知乎内容较长，截断
            author=author.get("name", ""),
            author_id=str(author.get("id", "")),
            publish_time=publish_time,
            url=url,
            likes=obj.get("voteup_count", 0) or 0,
            comments=obj.get("comment_count", 0) or 0,
            reposts=0,
        )

    def extract_edges(self, posts: List[Post]) -> List[GraphEdge]:
        """知乎边提取（简化）"""
        return []
