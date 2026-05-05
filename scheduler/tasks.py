"""
定时任务调度模块
使用 APScheduler 实现后台采集-分析-告警流水线
"""
import time
import threading
from typing import Optional, List
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import SCHEDULER_INTERVAL_MINUTES, CONFIG
from crawlers.mock_generator import MockCrawler, generate_mock_edges
from analysis.sentiment import get_analyzer
from analysis.topic_clustering import cluster_posts
from analysis.alert_engine import AlertEngine
from utils.database import (
    init_db, insert_posts, insert_edges, update_post_topic
)


class SentimentScheduler:
    """舆情监控调度器"""

    def __init__(self, interval_minutes: int = SCHEDULER_INTERVAL_MINUTES):
        self.scheduler = BackgroundScheduler()
        self.interval = interval_minutes
        self.is_running = False
        self._job = None

    def _pipeline(self):
        """采集-分析-告警流水线"""
        print(f"[{datetime.now()}] 开始执行定时任务...")
        try:
            # 1. 采集数据
            crawler = MockCrawler(keywords=CONFIG.crawler.keywords)
            posts = crawler.crawl(max_posts=50)  # 每次增量采集

            if not posts:
                print("无新数据")
                return

            # 2. 情感分析
            analyzer = get_analyzer()
            post_dicts = [p.to_dict() for p in posts]
            analyzed = analyzer.analyze_posts(post_dicts)

            # 3. 话题聚类
            clustered, topics = cluster_posts(analyzed, n_clusters=5)

            # 4. 入库
            inserted = insert_posts(clustered)
            for p in clustered:
                if p.get("topic_id") is not None:
                    update_post_topic(p["post_id"], p["topic_id"])

            edges = generate_mock_edges(posts)
            insert_edges([e.to_dict() for e in edges])

            print(f"插入 {inserted} 条新数据, {len(topics)} 个话题")

            # 5. 告警检查
            engine = AlertEngine()
            alert_ids = engine.run_and_save()
            if alert_ids:
                print(f"触发 {len(alert_ids)} 条告警: {alert_ids}")

            # 6. 可选：轻量级 Sentifox 传播趋势检测（10步快速仿真）
            if getattr(CONFIG, 'enable_scheduler_simulation', False):
                try:
                    from graph_analysis.sentifox_simulator import SentifoxSimulator
                    from graph_analysis.interventions import SeedInfectionIntervention
                    from graph_analysis.graph_builder import build_graph_from_posts
                    from graph_analysis.influencer_detection import detect_influencers
                    from utils.database import get_posts, get_edges

                    posts = get_posts(limit=500)
                    edges = get_edges()
                    if posts and len(posts) >= 10:
                        simulator = SentifoxSimulator.from_data(posts, edges, config={
                            "total_steps": 10,
                            "use_cognitive": True,
                        })
                        G = build_graph_from_posts(posts, edges)
                        influencers = detect_influencers(G, metric="pagerank", top_n=2)
                        seed_ids = [inf["author_id"] for inf in influencers if inf.get("author_id")]
                        if seed_ids:
                            simulator.add_intervention(SeedInfectionIntervention(seed_ids, step=0))
                            result = simulator.run(steps=10)
                            if result.timeline:
                                final = result.timeline[-1]
                                peak = max(s.infected_count for s in result.timeline)
                                print(f"[仿真快照] 感染={final.infected_count}/{len(simulator.agents)} 峰值={peak}")
                except Exception as e:
                    print(f"[仿真快照] 跳过: {e}")

        except Exception as e:
            print(f"定时任务执行失败: {e}")

    def start(self):
        """启动调度器"""
        if self.is_running:
            return

        init_db()
        self._job = self.scheduler.add_job(
            self._pipeline,
            trigger=IntervalTrigger(minutes=self.interval),
            id="sentiment_pipeline",
            replace_existing=True,
        )
        self.scheduler.start()
        self.is_running = True
        print(f"调度器已启动，间隔: {self.interval} 分钟")

    def stop(self):
        """停止调度器"""
        if not self.is_running:
            return
        self.scheduler.shutdown(wait=False)
        self.is_running = False
        print("调度器已停止")

    def run_once(self):
        """立即执行一次"""
        self._pipeline()


# 全局单例
_scheduler_instance: Optional[SentimentScheduler] = None


def get_scheduler() -> SentimentScheduler:
    """获取调度器单例"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = SentimentScheduler()
    return _scheduler_instance


def start_scheduler():
    """便捷函数：启动调度器"""
    scheduler = get_scheduler()
    scheduler.start()
    return scheduler


def stop_scheduler():
    """便捷函数：停止调度器"""
    scheduler = get_scheduler()
    scheduler.stop()
