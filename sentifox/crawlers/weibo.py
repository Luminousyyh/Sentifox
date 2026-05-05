"""
微博数据采集器（PC 端 HTML 解析模式）
基于 s.weibo.com 搜索页面
"""
import re
from datetime import datetime
from typing import List, Optional
from urllib.parse import quote
import httpx
from bs4 import BeautifulSoup

from crawlers.base import BaseCrawler, Post, GraphEdge
from config import COOKIES


class WeiboCrawler(BaseCrawler):
    """微博爬虫（PC 端 HTML 解析）"""

    SEARCH_URL = "https://s.weibo.com/weibo"

    def __init__(self, keywords: List[str], cookie: Optional[str] = None):
        super().__init__(platform="微博", keywords=keywords)
        self.cookie = cookie or COOKIES.get("weibo", "")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://s.weibo.com/",
        }
        if self.cookie:
            self.headers["Cookie"] = self.cookie

    def crawl(self, max_posts: int = 100) -> List[Post]:
        """采集微博搜索结果"""
        if not self.cookie:
            print("[微博] 未配置 Cookie，跳过采集。请在 config.py 的 COOKIES['weibo'] 中设置完整 Cookie")
            return []

        posts = []
        keyword = self.keywords[0] if self.keywords else ""
        page = 1

        with httpx.Client(headers=self.headers, follow_redirects=True, timeout=20) as client:
            while len(posts) < max_posts and page <= 5:
                try:
                    url = f"{self.SEARCH_URL}?q={quote(keyword)}&page={page}"
                    resp = client.get(url)

                    if resp.status_code != 200:
                        print(f"[微博] 请求失败: {resp.status_code}")
                        break

                    # 检查是否被重定向到登录页
                    if "passport" in resp.text or "login" in resp.text.lower()[:2000]:
                        print("[微博] Cookie 失效或被拦截，自动降级为模拟数据")
                        break

                    new_posts = self._parse_html(resp.text)
                    if not new_posts:
                        break

                    posts.extend(new_posts)
                    page += 1

                except Exception as e:
                    print(f"[微博] 采集异常: {e}")
                    break

        print(f"[微博] 采集完成: {len(posts)} 条")
        return posts[:max_posts]

    def _parse_html(self, html: str) -> List[Post]:
        """解析微博搜索 HTML"""
        soup = BeautifulSoup(html, "html.parser")
        cards = soup.find_all("div", class_="card-wrap")
        posts = []

        for card in cards:
            inner = card.find("div", class_="card")
            if not inner:
                continue

            # 跳过用户卡片
            classes = inner.get("class", [])
            if "card-user-b" in classes:
                continue

            # 必须有内容文本才算微博帖子
            txt_p = inner.find("p", class_="txt")
            if not txt_p:
                continue

            post = self._parse_post_card(card, inner, txt_p)
            if post:
                posts.append(post)

        return posts

    def _parse_post_card(self, card_wrap, card, txt_p) -> Optional[Post]:
        """解析单条微博帖子"""
        # 内容文本
        text = txt_p.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        if not text:
            return None

        # 作者
        name_a = card.find("a", class_="name")
        author = name_a.get_text(strip=True) if name_a else ""

        # 作者主页 URL（用于提取 uid）
        author_id = ""
        if name_a and name_a.get("href"):
            href = name_a["href"]
            m = re.search(r"u/(\d+)", href)
            if m:
                author_id = m.group(1)

        # 微博 mid（用于构建 URL）
        mid = card_wrap.get("mid", "")
        if not mid:
            # 尝试从 card 或子元素的 action-data 中提取
            for elem in [card, card_wrap]:
                action_data = elem.get("action-data", "")
                m = re.search(r"mid=(\d+)", action_data)
                if m:
                    mid = m.group(1)
                    break
                # 查找子元素
                for child in elem.find_all(attrs={"action-data": True}):
                    m = re.search(r"mid=(\d+)", child.get("action-data", ""))
                    if m:
                        mid = m.group(1)
                        break
                if mid:
                    break

        # 如果还没拿到 uid，尝试从 action-data 中提取
        if not author_id:
            for elem in [card, card_wrap]:
                for child in elem.find_all(attrs={"action-data": True}):
                    ad = child.get("action-data", "")
                    m = re.search(r"uid=(\d+)", ad)
                    if m:
                        author_id = m.group(1)
                        break
                if author_id:
                    break

        # 互动数据
        likes = comments = reposts = 0
        actions = card.find("div", class_="card-act")
        if actions:
            for li in actions.find_all("li"):
                # 优先从 <a> 标签的文本中提取
                a_tag = li.find("a")
                if a_tag:
                    li_text = a_tag.get_text(strip=True)
                else:
                    li_text = li.get_text(strip=True)
                num_match = re.search(r"(\d+)", li_text)
                num = int(num_match.group(1)) if num_match else 0
                if "转发" in li_text:
                    reposts = num
                elif "评论" in li_text:
                    comments = num
                elif "赞" in li_text or "like" in li_text.lower():
                    likes = num

        # 发布时间（尝试从 info 区域提取）
        publish_time = None
        info_div = card.find("div", class_="info")
        if info_div:
            # 找 info 里的第一个 <a> 标签（通常是时间链接）
            time_a = info_div.find("a")
            if time_a:
                time_text = time_a.get_text(strip=True)
                publish_time = self._parse_weibo_time(time_text)

        return Post(
            post_id=f"weibo_{mid}" if mid else f"weibo_{hash(text) & 0xFFFFFFFF}",
            platform="微博",
            content=text,
            author=author,
            author_id=author_id,
            publish_time=publish_time,
            url=f"https://weibo.com/{author_id}/{mid}" if author_id and mid else "",
            likes=likes,
            comments=comments,
            reposts=reposts,
        )

    def _parse_weibo_time(self, time_text: str) -> Optional[datetime]:
        """解析微博时间字符串"""
        if not time_text:
            return None
        # 常见格式: "今天 12:30", "5月5日 10:20", "2024-05-05 10:20"
        now = datetime.now()
        try:
            if "今天" in time_text:
                t = time_text.replace("今天", "").strip()
                return datetime.strptime(f"{now.year}-{now.month:02d}-{now.day:02d} {t}", "%Y-%m-%d %H:%M")
            elif "分钟前" in time_text:
                mins = int(re.search(r"(\d+)", time_text).group(1))
                return now.replace(second=0, microsecond=0) - __import__("datetime").timedelta(minutes=mins)
            elif "月" in time_text and "日" in time_text:
                m = re.search(r"(\d+)月(\d+)日\s+(\d+):(\d+)", time_text)
                if m:
                    month, day, hour, minute = map(int, m.groups())
                    year = now.year
                    # 如果月份大于当前月，可能是去年
                    if month > now.month:
                        year -= 1
                    return datetime(year, month, day, hour, minute)
            else:
                return datetime.strptime(time_text, "%Y-%m-%d %H:%M")
        except Exception:
            pass
        return None

    def extract_edges(self, posts: List[Post]) -> List[GraphEdge]:
        """从微博帖子中提取提及关系"""
        edges = []
        for post in posts:
            mentions = re.findall(r"@([\w\u4e00-\u9fa5]+)", post.content)
            for mention in mentions:
                edges.append(GraphEdge(
                    source_id=post.author_id,
                    target_id=mention,
                    relation_type="mention",
                    timestamp=post.publish_time,
                    platform="微博",
                ))
        return edges
