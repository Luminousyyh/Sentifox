"""
小红书数据采集器
基于小红书 Web 搜索 API
注意：小红书反爬严格（jsvmp 签名），仅配置 Cookie 可能仍被拦截，
      此时会自动降级为模拟数据。
"""
import re
from datetime import datetime
from typing import List, Optional
import httpx

from crawlers.base import BaseCrawler, Post, GraphEdge
from config import COOKIES


class XiaohongshuCrawler(BaseCrawler):
    """小红书爬虫"""

    SEARCH_API = "https://edith.xiaohongshu.com/api/sns/web/v1/search/notes"

    def __init__(self, keywords: List[str], cookie: Optional[str] = None):
        super().__init__(platform="小红书", keywords=keywords)
        self.cookie = cookie or COOKIES.get("xiaohongshu", "")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Origin": "https://www.xiaohongshu.com",
            "Referer": "https://www.xiaohongshu.com/",
        }
        if self.cookie:
            self.headers["Cookie"] = self.cookie

    def crawl(self, max_posts: int = 100) -> List[Post]:
        """采集小红书搜索结果"""
        if not self.cookie:
            print("[小红书] 未配置 Cookie，跳过采集。请在 config.py 的 COOKIES['xiaohongshu'] 中设置 web_session Cookie")
            return []

        posts = []
        keyword = self.keywords[0] if self.keywords else ""
        page = 1

        with httpx.Client(headers=self.headers, follow_redirects=True, timeout=15) as client:
            while len(posts) < max_posts and page <= 5:
                try:
                    payload = {
                        "keyword": keyword,
                        "page": page,
                        "page_size": 20,
                        "search_id": "",
                        "sort": "general",
                        "note_type": -1,
                    }

                    resp = client.post(self.SEARCH_API, json=payload)

                    # 小红书大概率返回 406/403（签名验证失败）
                    if resp.status_code in (403, 406, 401):
                        print(f"[小红书] 请求被拦截 (HTTP {resp.status_code})，签名验证失败。小红书反爬严格，建议：")
                        print("  1. 使用浏览器开发者工具 Network 面板抓取完整请求头（含 x-s, x-t 签名）")
                        print("  2. 或暂时使用模拟数据（系统将自动降级）")
                        break

                    if resp.status_code != 200:
                        print(f"[小红书] 请求失败: {resp.status_code}")
                        break

                    data = resp.json()
                    if data.get("code") != 0:
                        print(f"[小红书] API 返回错误: {data.get('msg', 'unknown')}")
                        break

                    items = data.get("data", {}).get("items", [])
                    if not items:
                        break

                    for item in items:
                        if len(posts) >= max_posts:
                            break
                        note = item.get("note_card") or item.get("notes", [{}])[0]
                        if not note:
                            continue
                        post = self._parse_note(note)
                        if post:
                            posts.append(post)

                    page += 1

                except Exception as e:
                    print(f"[小红书] 采集异常: {e}")
                    break

        print(f"[小红书] 采集完成: {len(posts)} 条")
        return posts

    def _parse_note(self, note: dict) -> Optional[Post]:
        """解析小红书笔记"""
        note_id = str(note.get("id", ""))
        title = note.get("title", "")
        desc = note.get("desc", "")
        text = f"{title}\n{desc}".strip()
        if not text:
            return None

        user = note.get("user", {})
        time_str = note.get("time", "")

        # 尝试解析时间戳（毫秒）
        publish_time = None
        if time_str and str(time_str).isdigit():
            try:
                ts = int(time_str)
                if ts > 1_000_000_000_000:  # 毫秒
                    ts = ts // 1000
                publish_time = datetime.fromtimestamp(ts)
            except Exception:
                pass

        # 互动数据
        interact_info = note.get("interact_info", {})

        return Post(
            post_id=f"xhs_{note_id}",
            platform="小红书",
            content=text,
            author=user.get("nickname", ""),
            author_id=str(user.get("user_id", "")),
            publish_time=publish_time,
            url=f"https://www.xiaohongshu.com/explore/{note_id}",
            likes=interact_info.get("liked_count", 0) or 0,
            comments=interact_info.get("comment_count", 0) or 0,
            reposts=interact_info.get("collected_count", 0) or 0,
            extra={"image_list": note.get("image_list", [])},
        )

    def extract_edges(self, posts: List[Post]) -> List[GraphEdge]:
        """从小红书笔记中提取关系（话题标签提及）"""
        edges = []
        for post in posts:
            # 提取话题标签 #话题#
            tags = re.findall(r"#([^#]+)#", post.content)
            for tag in tags:
                edges.append(GraphEdge(
                    source_id=post.author_id,
                    target_id=f"tag_{tag}",
                    relation_type="tag",
                    timestamp=post.publish_time,
                    platform="小红书",
                ))
            # 提取 @用户
            mentions = re.findall(r"@([\w\u4e00-\u9fa5]+)", post.content)
            for mention in mentions:
                edges.append(GraphEdge(
                    source_id=post.author_id,
                    target_id=mention,
                    relation_type="mention",
                    timestamp=post.publish_time,
                    platform="小红书",
                ))
        return edges
