"""
新闻采集器
基于百度新闻搜索
"""
import re
from datetime import datetime
from typing import List, Optional
import httpx
from bs4 import BeautifulSoup

from crawlers.base import BaseCrawler, Post, GraphEdge


class NewsCrawler(BaseCrawler):
    """新闻爬虫"""

    SEARCH_URL = "https://www.baidu.com/s"

    def __init__(self, keywords: List[str]):
        super().__init__(platform="新闻", keywords=keywords)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

    def crawl(self, max_posts: int = 100) -> List[Post]:
        """采集百度新闻搜索结果"""
        posts = []
        keyword = self.keywords[0] if self.keywords else ""
        page = 0

        with httpx.Client(headers=self.headers, follow_redirects=True, timeout=15) as client:
            while len(posts) < max_posts and page < 3:
                try:
                    params = {
                        "rtt": "1",       # 按时间排序
                        "bsst": "1",
                        "cl": "2",
                        "tn": "news",
                        "word": keyword,
                        "pn": page * 10,
                    }

                    resp = client.get(self.SEARCH_URL, params=params)
                    if resp.status_code != 200:
                        print(f"[新闻] 请求失败: {resp.status_code}")
                        break

                    soup = BeautifulSoup(resp.text, "html.parser")
                    results = soup.find_all("div", class_="result")

                    if not results:
                        # 尝试其他选择器
                        results = soup.find_all("div", class_=re.compile("result"))

                    for result in results:
                        if len(posts) >= max_posts:
                            break
                        post = self._parse_result(result)
                        if post:
                            posts.append(post)

                    page += 1

                except Exception as e:
                    print(f"[新闻] 采集异常: {e}")
                    break

        print(f"[新闻] 采集完成: {len(posts)} 条")
        return posts

    def _parse_result(self, result) -> Optional[Post]:
        """解析百度新闻结果"""
        try:
            # 标题和链接
            title_tag = result.find("h3")
            if not title_tag:
                return None

            a_tag = title_tag.find("a")
            if not a_tag:
                return None

            title = a_tag.get_text(strip=True)
            url = a_tag.get("href", "")

            # 摘要
            summary_tag = result.find("div", class_="content-right_8Zs40")
            if not summary_tag:
                summary_tag = result.find("span", class_="content-right_8Zs40")
            if not summary_tag:
                # 尝试通用摘要提取
                summary_tag = result.find("div", class_=re.compile("summary|content"))

            summary = summary_tag.get_text(strip=True) if summary_tag else ""
            content = f"{title}。{summary}".strip()

            if not content or len(content) < 10:
                return None

            # 来源和时间
            source_info = result.find("div", class_="news-source_2KcpX")
            source = ""
            pub_time_str = ""
            if source_info:
                spans = source_info.find_all("span")
                if spans:
                    source = spans[0].get_text(strip=True)
                if len(spans) > 1:
                    pub_time_str = spans[1].get_text(strip=True)

            publish_time = None
            if pub_time_str:
                # 百度新闻时间格式多样，简单处理
                try:
                    if "年" in pub_time_str:
                        publish_time = datetime.strptime(pub_time_str, "%Y年%m月%d日 %H:%M")
                    elif "小时前" in pub_time_str or "分钟前" in pub_time_str:
                        publish_time = datetime.now()
                except Exception:
                    pass

            return Post(
                post_id=f"news_{hash(url) & 0xFFFFFFFF}",
                platform="新闻",
                content=content,
                author=source,
                author_id=source,
                publish_time=publish_time,
                url=url,
                likes=0,
                comments=0,
                reposts=0,
            )

        except Exception:
            return None

    def extract_edges(self, posts: List[Post]) -> List[GraphEdge]:
        """新闻边提取（简化）"""
        return []
