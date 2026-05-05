"""
SQLite 数据库操作封装
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from contextlib import contextmanager
from config import DB_PATH


def init_db():
    """初始化数据库表结构"""
    with get_connection() as conn:
        cursor = conn.cursor()

        # 主数据表：舆情帖子
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id TEXT UNIQUE NOT NULL,
                platform TEXT NOT NULL,
                author TEXT,
                author_id TEXT,
                content TEXT NOT NULL,
                publish_time TIMESTAMP,
                url TEXT,
                sentiment_label TEXT,
                sentiment_score REAL,
                topic_id INTEGER,
                keywords TEXT,
                likes INTEGER DEFAULT 0,
                comments INTEGER DEFAULT 0,
                reposts INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 图边表：传播关系
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                timestamp TIMESTAMP,
                weight REAL DEFAULT 1.0,
                platform TEXT,
                UNIQUE(source_id, target_id, relation_type)
            )
        """)

        # 告警记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT,
                triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                is_resolved INTEGER DEFAULT 0
            )
        """)

        # 索引优化
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_platform ON posts(platform)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_time ON posts(publish_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_sentiment ON posts(sentiment_label)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_id)")

        conn.commit()


@contextmanager
def get_connection():
    """获取数据库连接上下文管理器"""
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def insert_posts(posts: List[Dict[str, Any]]) -> int:
    """批量插入帖子数据，自动去重"""
    if not posts:
        return 0

    with get_connection() as conn:
        cursor = conn.cursor()
        inserted = 0
        for post in posts:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO posts 
                    (post_id, platform, author, author_id, content, publish_time, url,
                     sentiment_label, sentiment_score, topic_id, keywords, likes, comments, reposts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    post.get("post_id"),
                    post.get("platform"),
                    post.get("author"),
                    post.get("author_id"),
                    post.get("content"),
                    post.get("publish_time"),
                    post.get("url"),
                    post.get("sentiment_label"),
                    post.get("sentiment_score"),
                    post.get("topic_id"),
                    json.dumps(post.get("keywords", []), ensure_ascii=False) if isinstance(post.get("keywords"), list) else post.get("keywords"),
                    post.get("likes", 0),
                    post.get("comments", 0),
                    post.get("reposts", 0),
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            except Exception as e:
                print(f"Insert error: {e}")
        conn.commit()
        return inserted


def update_post_sentiment(post_id: str, label: str, score: float):
    """更新帖子情感分析结果"""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE posts SET sentiment_label = ?, sentiment_score = ? WHERE post_id = ?
        """, (label, score, post_id))
        conn.commit()


def update_post_topic(post_id: str, topic_id: int):
    """更新帖子话题归属"""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE posts SET topic_id = ? WHERE post_id = ?", (topic_id, post_id))
        conn.commit()


def get_posts(
    platforms: Optional[List[str]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    sentiment: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """多条件查询帖子"""
    with get_connection() as conn:
        query = "SELECT * FROM posts WHERE 1=1"
        params = []

        if platforms:
            placeholders = ",".join(["?"] * len(platforms))
            query += f" AND platform IN ({placeholders})"
            params.extend(platforms)
        if start_time:
            query += " AND publish_time >= ?"
            params.append(start_time)
        if end_time:
            query += " AND publish_time <= ?"
            params.append(end_time)
        if sentiment:
            query += " AND sentiment_label = ?"
            params.append(sentiment)

        query += " ORDER BY publish_time DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_post_count(
    platforms: Optional[List[str]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    sentiment: Optional[str] = None
) -> int:
    """统计帖子数量"""
    with get_connection() as conn:
        query = "SELECT COUNT(*) as cnt FROM posts WHERE 1=1"
        params = []

        if platforms:
            placeholders = ",".join(["?"] * len(platforms))
            query += f" AND platform IN ({placeholders})"
            params.extend(platforms)
        if start_time:
            query += " AND publish_time >= ?"
            params.append(start_time)
        if end_time:
            query += " AND publish_time <= ?"
            params.append(end_time)
        if sentiment:
            query += " AND sentiment_label = ?"
            params.append(sentiment)

        cursor = conn.execute(query, params)
        return cursor.fetchone()["cnt"]


def get_sentiment_distribution(
    platforms: Optional[List[str]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> Dict[str, int]:
    """获取情感分布统计"""
    with get_connection() as conn:
        query = """
            SELECT sentiment_label, COUNT(*) as cnt 
            FROM posts 
            WHERE sentiment_label IS NOT NULL
        """
        params = []

        if platforms:
            placeholders = ",".join(["?"] * len(platforms))
            query += f" AND platform IN ({placeholders})"
            params.extend(platforms)
        if start_time:
            query += " AND publish_time >= ?"
            params.append(start_time)
        if end_time:
            query += " AND publish_time <= ?"
            params.append(end_time)

        query += " GROUP BY sentiment_label"
        cursor = conn.execute(query, params)
        return {row["sentiment_label"]: row["cnt"] for row in cursor.fetchall()}


def get_platform_distribution(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> Dict[str, int]:
    """获取平台分布统计"""
    with get_connection() as conn:
        query = "SELECT platform, COUNT(*) as cnt FROM posts WHERE 1=1"
        params = []

        if start_time:
            query += " AND publish_time >= ?"
            params.append(start_time)
        if end_time:
            query += " AND publish_time <= ?"
            params.append(end_time)

        query += " GROUP BY platform"
        cursor = conn.execute(query, params)
        return {row["platform"]: row["cnt"] for row in cursor.fetchall()}


def get_trend_by_time(
    interval: str = "hour",
    platforms: Optional[List[str]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """按时间窗口统计趋势"""
    with get_connection() as conn:
        if interval == "hour":
            time_fmt = "%Y-%m-%d %H:00:00"
        elif interval == "day":
            time_fmt = "%Y-%m-%d"
        else:
            time_fmt = "%Y-%m-%d %H:00:00"

        query = f"""
            SELECT 
                strftime('{time_fmt}', publish_time) as time_bucket,
                sentiment_label,
                COUNT(*) as cnt
            FROM posts
            WHERE sentiment_label IS NOT NULL
        """
        params = []

        if platforms:
            placeholders = ",".join(["?"] * len(platforms))
            query += f" AND platform IN ({placeholders})"
            params.extend(platforms)
        if start_time:
            query += " AND publish_time >= ?"
            params.append(start_time)
        if end_time:
            query += " AND publish_time <= ?"
            params.append(end_time)

        query += f" GROUP BY time_bucket, sentiment_label ORDER BY time_bucket"
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def insert_edges(edges: List[Dict[str, Any]]) -> int:
    """批量插入图边数据"""
    if not edges:
        return 0

    with get_connection() as conn:
        cursor = conn.cursor()
        inserted = 0
        for edge in edges:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO graph_edges 
                    (source_id, target_id, relation_type, timestamp, weight, platform)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    edge.get("source_id"),
                    edge.get("target_id"),
                    edge.get("relation_type"),
                    edge.get("timestamp"),
                    edge.get("weight", 1.0),
                    edge.get("platform"),
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            except Exception as e:
                print(f"Edge insert error: {e}")
        conn.commit()
        return inserted


def get_edges(
    relation_type: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """查询图边"""
    with get_connection() as conn:
        query = "SELECT * FROM graph_edges WHERE 1=1"
        params = []

        if relation_type:
            query += " AND relation_type = ?"
            params.append(relation_type)
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def insert_alert(alert_type: str, severity: str, message: str, details: Optional[str] = None):
    """插入告警记录"""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alerts (alert_type, severity, message, details)
            VALUES (?, ?, ?, ?)
        """, (alert_type, severity, message, details))
        conn.commit()
        return cursor.lastrowid


def get_alerts(limit: int = 100, unresolved_only: bool = False) -> List[Dict[str, Any]]:
    """获取告警记录"""
    with get_connection() as conn:
        query = "SELECT * FROM alerts WHERE 1=1"
        params = []
        if unresolved_only:
            query += " AND is_resolved = 0"
        query += " ORDER BY triggered_at DESC LIMIT ?"
        params.append(limit)
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_recent_posts_count(minutes: int = 30) -> int:
    """获取最近 N 分钟的帖子数"""
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT COUNT(*) as cnt FROM posts 
            WHERE publish_time >= datetime('now', '-{} minutes')
        """.format(minutes))
        return cursor.fetchone()["cnt"]
