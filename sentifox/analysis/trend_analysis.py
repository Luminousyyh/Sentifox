"""
趋势分析模块
在 database.get_trend_by_time 基础上进行高级计算
"""
import math
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from utils.database import get_trend_by_time, get_posts


def calculate_sentiment_index(trend_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    计算情感指数（正面占比 - 负面占比，归一化到 -100~100）
    """
    if not trend_data:
        return []

    # 按时间聚合
    buckets = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})
    for row in trend_data:
        bucket = row.get("time_bucket")
        label = row.get("sentiment_label")
        cnt = row.get("cnt", 0)
        if bucket and label:
            buckets[bucket][label] += cnt

    results = []
    for bucket, counts in sorted(buckets.items()):
        total = counts["positive"] + counts["negative"] + counts["neutral"]
        if total == 0:
            index = 0
        else:
            index = (counts["positive"] - counts["negative"]) / total * 100
        results.append({
            "time_bucket": bucket,
            "sentiment_index": round(index, 2),
            "positive": counts["positive"],
            "negative": counts["negative"],
            "neutral": counts["neutral"],
            "total": total,
        })

    return results


def detect_anomaly(
    trend_data: List[Dict[str, Any]],
    method: str = "zscore",
    threshold: float = 2.0
) -> List[Dict[str, Any]]:
    """
    检测情感趋势异常点
    :param method: "zscore" 或 "iqr"
    :return: 异常点列表
    """
    if len(trend_data) < 3:
        return []

    indices = [row["sentiment_index"] for row in trend_data]

    if method == "zscore":
        mean = sum(indices) / len(indices)
        std = math.sqrt(sum((x - mean) ** 2 for x in indices) / len(indices))
        if std == 0:
            return []
        anomalies = []
        for row in trend_data:
            z = abs(row["sentiment_index"] - mean) / std
            if z > threshold:
                row["zscore"] = round(z, 2)
                anomalies.append(row)
        return anomalies

    elif method == "iqr":
        sorted_indices = sorted(indices)
        q1 = sorted_indices[len(sorted_indices) // 4]
        q3 = sorted_indices[3 * len(sorted_indices) // 4]
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        anomalies = []
        for row in trend_data:
            val = row["sentiment_index"]
            if val < lower or val > upper:
                row["iqr_boundary"] = [lower, upper]
                anomalies.append(row)
        return anomalies

    return []


def get_hot_topics_evolution(
    platforms: Optional[List[str]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    获取最热门的话题（按帖子数排序）
    """
    posts = get_posts(
        platforms=platforms,
        start_time=start_time,
        end_time=end_time,
        limit=10000,
    )

    topic_counts = defaultdict(int)
    for p in posts:
        tid = p.get("topic_id")
        if tid is not None:
            topic_counts[tid] += 1

    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
    return [{"topic_id": tid, "post_count": cnt} for tid, cnt in sorted_topics[:top_n]]


def calculate_engagement_rate(posts: List[Dict]) -> Dict[str, float]:
    """
    计算整体互动率（点赞+评论+转发 / 帖子数）
    """
    if not posts:
        return {"avg_likes": 0, "avg_comments": 0, "avg_reposts": 0, "total_engagement": 0}

    total_likes = sum(p.get("likes", 0) for p in posts)
    total_comments = sum(p.get("comments", 0) for p in posts)
    total_reposts = sum(p.get("reposts", 0) for p in posts)
    n = len(posts)

    return {
        "avg_likes": round(total_likes / n, 2),
        "avg_comments": round(total_comments / n, 2),
        "avg_reposts": round(total_reposts / n, 2),
        "total_engagement": total_likes + total_comments + total_reposts,
    }
