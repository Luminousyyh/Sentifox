"""
告警引擎模块
基于规则检测舆情异常并触发告警
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from utils.database import (
    get_posts, get_post_count, get_sentiment_distribution,
    get_recent_posts_count, insert_alert, get_alerts
)
from config import ALERT_CONFIG


class AlertEngine:
    """告警引擎"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or ALERT_CONFIG
        self.rules = [
            self._check_negative_spike,
            self._check_volume_spike,
            self._check_sentiment_drop,
        ]

    def check(self, platforms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        执行所有告警规则检查
        :return: 触发的告警列表
        """
        triggered = []
        for rule in self.rules:
            alerts = rule(platforms)
            triggered.extend(alerts)
        return triggered

    def _check_negative_spike(self, platforms: Optional[List[str]] = None) -> List[Dict]:
        """检查负面情感占比是否超过阈值"""
        dist = get_sentiment_distribution(platforms=platforms)
        total = sum(dist.values())
        if total == 0:
            return []

        neg_count = dist.get("negative", 0)
        neg_ratio = neg_count / total
        threshold = self.config.get("negative_ratio_threshold", 0.6)

        if neg_ratio > threshold:
            return [{
                "alert_type": "negative_spike",
                "severity": "high" if neg_ratio > 0.8 else "medium",
                "message": f"负面情感占比异常: {neg_ratio*100:.1f}% (阈值: {threshold*100:.0f}%)",
                "details": f"负面: {neg_count}, 总计: {total}",
            }]
        return []

    def _check_volume_spike(self, platforms: Optional[List[str]] = None) -> List[Dict]:
        """检查短时间内提及量是否突增"""
        window = self.config.get("volume_spike_window_minutes", 30)
        multiplier = self.config.get("volume_spike_multiplier", 3.0)

        recent = get_recent_posts_count(minutes=window)
        # 对比前一个窗口
        prev = get_posts(
            platforms=platforms,
            start_time=datetime.now() - timedelta(minutes=window * 2),
            end_time=datetime.now() - timedelta(minutes=window),
            limit=100000,
        )
        prev_count = len(prev)

        if prev_count > 0 and recent > prev_count * multiplier:
            return [{
                "alert_type": "volume_spike",
                "severity": "high",
                "message": f"提及量突增: 最近{window}分钟 {recent} 条 (前{window}分钟 {prev_count} 条, 倍数: {recent/prev_count:.1f}x)",
                "details": f"触发倍数阈值: {multiplier}x",
            }]
        return []

    def _check_sentiment_drop(self, platforms: Optional[List[str]] = None) -> List[Dict]:
        """检查情感指数是否突然下降"""
        from utils.database import get_trend_by_time
        from analysis.trend_analysis import calculate_sentiment_index

        trend = get_trend_by_time(
            interval="hour",
            platforms=platforms,
            start_time=datetime.now() - timedelta(hours=6),
            end_time=datetime.now(),
        )
        if len(trend) < 3:
            return []

        indices = calculate_sentiment_index(trend)
        if len(indices) < 3:
            return []

        # 比较最近两个时间窗口
        recent = indices[-1]["sentiment_index"]
        previous = indices[-2]["sentiment_index"]
        drop = previous - recent

        if drop > 30:  # 情感指数下降超过 30
            return [{
                "alert_type": "sentiment_drop",
                "severity": "medium",
                "message": f"情感指数骤降: 从 {previous:.1f} 降至 {recent:.1f} (降幅: {drop:.1f})",
                "details": f"时间窗口: {indices[-2]['time_bucket']} -> {indices[-1]['time_bucket']}",
            }]
        return []

    def run_and_save(self, platforms: Optional[List[str]] = None) -> List[int]:
        """
        运行告警检查并将结果保存到数据库
        :return: 插入的告警 ID 列表
        """
        alerts = self.check(platforms)
        alert_ids = []
        for alert in alerts:
            aid = insert_alert(
                alert_type=alert["alert_type"],
                severity=alert["severity"],
                message=alert["message"],
                details=alert.get("details"),
            )
            alert_ids.append(aid)
        return alert_ids
