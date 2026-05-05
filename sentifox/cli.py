#!/usr/bin/env python3
"""
舆情分析系统 CLI
命令行入口，替代/补充 Streamlit GUI，便于 AI 自动化测试
"""
import argparse
import os
import sys
import json
import re
import webbrowser
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from cli.ascii_fox import play_startup_animation, print_welcome_with_fox
from cli.theme import THEME, get_status_style, sentiment_color, platform_color
from cli.ui import (
    StepProgress, LiveLog, kpi_row, kpi_card, sparkline, labeled_sparkline,
    bar_chart, sentiment_bar, status_badge, section_header, section_divider,
    result_panel, styled_table, topic_grid, topic_card,
    platform_status_card, alert_card, chat_bubble_user, chat_bubble_assistant,
    test_result_table, CommandHistory,
)

# Fix Windows encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
if os.name == "nt":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass

# Rich for pretty output (enabled everywhere with UTF-8)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

from config import CONFIG, PLATFORMS, COOKIES, ALERT_CONFIG
from utils.database import (
    init_db, get_posts, get_post_count, get_sentiment_distribution,
    get_platform_distribution, get_trend_by_time, get_edges, get_alerts,
    insert_alert, get_recent_posts_count
)
# 业务模块采用延迟导入，避免 CLI 启动时加载全部依赖
# (see each cmd_* function for local imports)


def print_table(headers: List[str], rows: List[List[Any]], title: str = ""):
    """统一表格输出（兼容旧代码，内部使用 styled_table）"""
    if HAS_RICH:
        table = styled_table(headers, rows, title=title)
        console.print(table)
    else:
        if title:
            print(f"\n{'='*60}")
            print(f"  {title}")
            print(f"{'='*60}")
        col_widths = [max(len(str(h)), max((len(str(r[i])) if i < len(r) else 0) for r in rows + [[]])) for i, h in enumerate(headers)]
        header_line = " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print(" | ".join(str(c).ljust(col_widths[i]) if i < len(col_widths) else str(c) for i, c in enumerate(row)))
        print()


def print_panel(content: str, title: str = ""):
    """统一面板输出（兼容旧代码）"""
    if HAS_RICH:
        console.print(result_panel(content, title=title, status="info"))
    else:
        if title:
            print(f"\n{'─'*60}")
            print(f"  {title}")
            print(f"{'─'*60}")
        print(content)
        print()


def parse_time_range(time_range: str) -> tuple:
    """解析时间范围字符串为 (start_time, end_time)"""
    now = datetime.now()
    end_time = now
    if time_range == "24h":
        start_time = now - timedelta(hours=24)
    elif time_range == "3d":
        start_time = now - timedelta(days=3)
    elif time_range == "7d":
        start_time = now - timedelta(days=7)
    elif time_range == "30d":
        start_time = now - timedelta(days=30)
    else:
        start_time = datetime.min
    return start_time, end_time


def get_time_bucket(start_time: datetime, end_time: datetime) -> str:
    """根据时间范围决定聚合粒度"""
    delta = end_time - start_time
    if delta <= timedelta(hours=24):
        return "hour"
    return "day"


# ═══════════════════════════════════════════════════════════════
# 子命令实现
# ═══════════════════════════════════════════════════════════════

def cmd_init(args):
    """初始化数据库"""
    if HAS_RICH:
        console.print(f"[{THEME.brand}]🦊[/] 初始化 SQLite 数据库...")
    else:
        print("[init] 初始化 SQLite 数据库...")
    init_db()
    if HAS_RICH:
        console.print(result_panel("数据库已初始化\n路径: data/sentiment.db", title="[green]✓ 初始化完成[/]", status="ok"))
    else:
        print("[init] [OK] 数据库已初始化")


def cmd_mock(args):
    """生成模拟数据并完整处理"""
    from crawlers.mock_generator import MockCrawler, generate_mock_edges
    from crawlers.manager import CrawlerManager
    from analysis.sentiment import get_analyzer
    from analysis.topic_clustering import cluster_posts
    from rag.document_processor import sync_posts_to_vector_store

    keywords = [k.strip() for k in args.keywords.split(",")] if args.keywords else CONFIG.crawler.keywords
    count = args.count

    print(f"[mock] 生成 {count} 条模拟数据，关键词: {keywords}")
    crawler = MockCrawler(keywords=keywords)
    posts = crawler.crawl(max_posts=count)
    print(f"[mock] 生成完成: {len(posts)} 条")

    # 情感分析
    print("[mock] 情感分析中...")
    analyzer = get_analyzer()
    post_dicts = [p.to_dict() for p in posts]
    analyzed = analyzer.analyze_posts(post_dicts)
    print(f"[mock] 情感分析完成")

    # 话题聚类
    print("[mock] 话题聚类中...")
    clustered, topics = cluster_posts(analyzed, n_clusters=args.clusters)
    print(f"[mock] 话题聚类完成: {len(topics)} 个话题")

    # 入库
    from utils.database import insert_posts, update_post_topic, insert_edges
    insert_posts(clustered)
    for p in clustered:
        if p.get("topic_id") is not None:
            update_post_topic(p["post_id"], p["topic_id"])

    # 传播边
    edges = generate_mock_edges(clustered)
    if edges:
        insert_edges([e.to_dict() for e in edges])
        print(f"[mock] 插入 {len(edges)} 条传播边")

    # 向量库同步
    if not args.no_vectors:
        print("[mock] 同步到向量库...")
        try:
            sync_posts_to_vector_store(clustered)
            print("[mock] 向量库同步完成")
        except Exception as e:
            print(f"[mock] [WARN] 向量库同步失败: {e}")

    print(f"[mock] [OK] 全部完成，共 {len(clustered)} 条数据已入库")


def cmd_crawl(args):
    """真实采集"""
    from crawlers.manager import CrawlerManager

    platforms = args.platforms or PLATFORMS
    keywords = [k.strip() for k in args.keywords.split(",")] if args.keywords else CONFIG.crawler.keywords
    count = args.count

    print(f"[crawl] 开始采集: 平台={platforms}, 关键词={keywords}, 每平台{count}条")
    manager = CrawlerManager(keywords=keywords)
    results = manager.crawl_and_process(
        platforms=platforms,
        max_posts_per_platform=count,
        use_mock_fallback=True,
    )
    print(f"[crawl] [OK] 采集完成，共 {len(results)} 条数据已入库")


def cmd_status(args):
    """查看平台状态（不依赖 CrawlerManager，避免加载 bs4 等重型依赖）"""
    if HAS_RICH:
        cards = []
        for platform in CONFIG.crawler.platforms:
            if platform in ("抖音", "论坛"):
                cards.append(platform_status_card(platform, "[OFF]", "暂未实现真实采集"))
            elif platform in ("微博", "知乎", "小红书"):
                key = platform.lower() if platform != "小红书" else "xiaohongshu"
                cookie = COOKIES.get(key, "")
                if cookie:
                    cards.append(platform_status_card(platform, "[ON]", f"已配置 Cookie ({len(cookie)} chars)"))
                else:
                    cards.append(platform_status_card(platform, "[OFF]", f"未配置 Cookie"))
            else:
                cards.append(platform_status_card(platform, "[ON]", "无需登录"))
        
        console.print(section_header("平台采集状态", "📡"))
        console.print(kpi_row(cards))
        
        # Cookie 表格
        cookie_rows = []
        for key, val in COOKIES.items():
            masked = val[:20] + "..." if len(val) > 20 else val
            status_icon = "✓" if val else "✗"
            status_color = "green" if val else "red"
            cookie_rows.append([f"[{status_color}]{status_icon}[/{status_color}]", key, masked if val else "(未配置)"])
        console.print("")
        console.print(styled_table(["状态", "平台", "Cookie"], cookie_rows, title="Cookie 配置"))
    else:
        # 纯文本回退
        rows = []
        for platform in CONFIG.crawler.platforms:
            if platform in ("抖音", "论坛"):
                rows.append(["[OFF]", platform, "暂未实现真实采集，将使用模拟数据"])
            elif platform in ("微博", "知乎", "小红书"):
                key = platform.lower() if platform != "小红书" else "xiaohongshu"
                cookie = COOKIES.get(key, "")
                if cookie:
                    rows.append(["[ON]", platform, "已配置 Cookie"])
                else:
                    rows.append(["[OFF]", platform, f"未配置 {platform} Cookie"])
            else:
                rows.append(["[ON]", platform, "无需登录"])
        print_table(["状态", "平台", "说明"], rows, title="平台采集状态")
        cookie_rows = []
        for key, val in COOKIES.items():
            masked = val[:20] + "..." if len(val) > 20 else val
            cookie_rows.append([key, masked if val else "(未配置)"])
        print_table(["平台", "Cookie"], cookie_rows, title="Cookie 配置")


def cmd_overview(args):
    """概览统计（升级版：KPI卡片 + Sparkline + 情感分布条图）"""
    from analysis.topic_clustering import extract_keywords
    from analysis.trend_analysis import detect_anomaly
    from analysis.insight_generator import generate_insight

    start_time, end_time = parse_time_range(args.time_range)
    platforms = args.platforms or PLATFORMS
    selected = set(platforms)

    # KPI 数据
    total = get_post_count(start_time=start_time, end_time=end_time)
    pos = get_post_count(start_time=start_time, end_time=end_time, sentiment="positive")
    neg = get_post_count(start_time=start_time, end_time=end_time, sentiment="negative")
    neu = get_post_count(start_time=start_time, end_time=end_time, sentiment="neutral")
    recent = get_recent_posts_count(minutes=30)
    neg_ratio = (neg / total * 100) if total > 0 else 0.0

    if HAS_RICH:
        console.print(section_header("📊 数据概览", "📊"))
        
        # KPI 卡片行
        cards = [
            kpi_card("总帖子数", str(total), color=THEME.text_primary),
            kpi_card("正面", str(pos), color=THEME.success),
            kpi_card("负面", str(neg), color=THEME.error),
            kpi_card("中性", str(neu), color=THEME.muted),
            kpi_card("30min新增", str(recent), color=THEME.brand),
        ]
        console.print(kpi_row(cards))
        console.print("")
        
        # 负面占比条
        neg_bar = sentiment_bar(pos, neg, neu, width=50)
        console.print(Panel(neg_bar, title="[bold]情感分布[/]", border_style=THEME.border, box=box.ROUNDED))
        console.print(f"负面占比: [bold]{'%.1f' % neg_ratio}%[/] (阈值: {ALERT_CONFIG['negative_ratio_threshold']*100:.0f}%)")
        console.print("")
        
        # 平台分布条形图
        plat_dist = get_platform_distribution(start_time=start_time, end_time=end_time)
        plat_items = [(plat, count) for plat, count in plat_dist.items() if plat in selected]
        if plat_items:
            console.print(bar_chart(plat_items, title="平台分布", color=THEME.info))
            console.print("")
    else:
        # 纯文本回退
        kpi_text = (
            f"总帖子数: {total}\n"
            f"正面: {pos} | 负面: {neg} | 中性: {neu}\n"
            f"负面占比: {neg_ratio:.1f}% (阈值: {ALERT_CONFIG['negative_ratio_threshold']*100:.0f}%)\n"
            f"30分钟新增: {recent}"
        )
        print_panel(kpi_text, title="[DATA] 关键指标")
        
        sent_dist = get_sentiment_distribution(start_time=start_time, end_time=end_time)
        rows = [[label, count, f"{count/total*100:.1f}%"] for label, count in sent_dist.items()]
        print_table(["情感", "数量", "占比"], rows, title="情感分布")
        
        plat_dist = get_platform_distribution(start_time=start_time, end_time=end_time)
        rows = [[plat, count] for plat, count in plat_dist.items() if plat in selected]
        print_table(["平台", "数量"], rows, title="平台分布")

    # 趋势（两种模式都需要的核心逻辑）
    bucket = get_time_bucket(start_time, end_time)
    trend_raw = get_trend_by_time(start_time=start_time, end_time=end_time, interval=bucket)
    trend = {}
    for item in trend_raw:
        ts = item["time_bucket"]
        if ts not in trend:
            trend[ts] = {"time_bucket": ts, "positive": 0, "negative": 0, "neutral": 0, "total": 0}
        count = item.get("cnt", 0)
        sentiment = item.get("sentiment_label", "neutral")
        trend[ts][sentiment] = trend[ts].get(sentiment, 0) + count
        trend[ts]["total"] += count
    trend_list = sorted(trend.values(), key=lambda x: x["time_bucket"])
    for item in trend_list:
        total_t = item["total"]
        pos_t = item.get("positive", 0)
        neg_t = item.get("negative", 0)
        item["sentiment_index"] = (pos_t - neg_t) / total_t * 100 if total_t > 0 else 0
    
    if trend_list:
        sentiment_indices = [item.get("sentiment_index", 0) for item in trend_list]
        if HAS_RICH:
            console.print(section_header(f"情感趋势 ({bucket} 粒度)", "📈"))
            console.print(labeled_sparkline("情感指数", sentiment_indices, color=THEME.brand))
            console.print("")
        
        print(f"\n[UP] 情感趋势 ({bucket} 粒度):")
        for item in trend_list[-10:]:
            ts = item["time_bucket"]
            total_t = item["total"]
            pos_t = item.get("positive", 0)
            neg_t = item.get("negative", 0)
            idx = (pos_t - neg_t) / total_t * 100 if total_t > 0 else 0
            print(f"  {ts}: 总计{total_t} 正{pos_t} 负{neg_t} 指数{idx:.1f}")

    # 异常检测
    anomalies = detect_anomaly(trend_list, method="zscore", threshold=2.0)
    if anomalies:
        if HAS_RICH:
            console.print(f"\n[bold red]🚨 检测到 {len(anomalies)} 个异常点[/]")
        else:
            print(f"\n[ALERT] 检测到 {len(anomalies)} 个异常点:")
        for a in anomalies[:5]:
            print(f"  {a['time_bucket']}: Z-score={a['zscore']:.2f}, 指数={a['sentiment_index']:.1f}")

    # 关键词
    posts = get_posts(start_time=start_time, end_time=end_time, limit=1000)
    texts = [p["content"] for p in posts]
    if texts:
        keywords = extract_keywords(texts, top_n=20)
        kw_str = ", ".join([k for k, _ in keywords])
        if HAS_RICH:
            console.print(f"\n[bold {THEME.brand}]🔥 热门关键词:[/] {kw_str}")
        else:
            print(f"\n[KEY] 热门关键词: {kw_str}")

    # AI 洞察
    if args.insight:
        if HAS_RICH:
            console.print(f"\n[{THEME.brand}]🧠 正在生成 AI 洞察...[/]")
        else:
            print("\n[AI] 正在生成 AI 洞察...")
        try:
            insight = generate_insight(posts[:500], 
                get_sentiment_distribution(start_time, end_time),
                get_platform_distribution(start_time, end_time))
            if HAS_RICH:
                console.print(result_panel(insight, title="[bold green]💡 AI 舆情洞察[/]", status="ok"))
            else:
                print_panel(insight, title="[TIP] AI 舆情洞察")
        except Exception as e:
            print(f"[WARN] AI 洞察生成失败: {e}")


def cmd_topics(args):
    """话题分析（升级版：话题卡片网格）"""
    from analysis.topic_clustering import cluster_posts, extract_keywords

    start_time, end_time = parse_time_range(args.time_range)
    posts = get_posts(start_time=start_time, end_time=end_time, limit=2000)
    texts = [p["content"] for p in posts]

    if not texts:
        if HAS_RICH:
            console.print("[yellow]⚠ 无数据，请先运行 mock 或 crawl[/]")
        else:
            print("[topics] [WARN] 无数据，请先运行 mock 或 crawl")
        return

    n_clusters = min(args.clusters, len(texts) // 5)
    if n_clusters < 2:
        n_clusters = 2

    if HAS_RICH:
        console.print(f"[{THEME.brand}]🔥[/] 对 {len(texts)} 条帖子进行话题聚类 (K={n_clusters})...")
    else:
        print(f"[topics] 对 {len(texts)} 条帖子进行话题聚类 (K={n_clusters})...")
    
    clustered, topics = cluster_posts(posts, n_clusters=n_clusters)

    if HAS_RICH:
        cards = []
        for tid, topic in topics.items():
            t_posts = [p for p in clustered if p.get("topic_id") == tid]
            keywords = topic.get("top_keywords", [])[:5]
            # 统计该话题的情感分布
            sent_dist = {"positive": 0, "negative": 0, "neutral": 0}
            for p in t_posts:
                lbl = p.get("sentiment_label", "neutral")
                sent_dist[lbl] = sent_dist.get(lbl, 0) + 1
            cards.append(topic_card(str(tid), len(t_posts), keywords, sent_dist))
        
        console.print(section_header(f"话题分析 ({len(topics)} 个话题)", "🔥"))
        console.print(topic_grid(cards))
    else:
        rows = []
        for tid, topic in topics.items():
            t_posts = [p for p in clustered if p.get("topic_id") == tid]
            keywords = ", ".join(topic.get("top_keywords", [])[:3])
            rows.append([f"话题{tid}", len(t_posts), keywords])
        print_table(["话题", "帖子数", "关键词"], rows, title="[FIRE] 话题分析")


def cmd_graph(args):
    """传播图谱分析（升级版：网络统计卡片 + KOL条形图 + 负面放大器表格）"""
    from graph_analysis.graph_builder import build_graph_from_posts, get_graph_stats
    from graph_analysis.influencer_detection import detect_influencers, detect_negative_amplifiers
    from graph_analysis.propagation_simulator import simulate_sir, simulate_agent_based

    start_time, end_time = parse_time_range(args.time_range)
    posts = get_posts(start_time=start_time, end_time=end_time, limit=2000)
    edges = get_edges(start_time=start_time, end_time=end_time)

    if not posts:
        if HAS_RICH:
            console.print("[yellow]⚠ 无数据[/]")
        else:
            print("[graph] [WARN] 无数据")
        return

    G = build_graph_from_posts(posts, edges)
    stats = get_graph_stats(G)

    if HAS_RICH:
        # 网络统计卡片
        console.print(section_header("🕸️ 传播图谱分析", "🕸️"))
        stat_cards = [
            kpi_card("节点数", str(stats['nodes']), color=THEME.info),
            kpi_card("边数", str(stats['edges']), color=THEME.info),
            kpi_card("密度", f"{stats['density']:.4f}", color=THEME.muted),
            kpi_card("连通分量", str(stats.get('components', 0)), color=THEME.muted),
        ]
        console.print(kpi_row(stat_cards))
        console.print("")
    else:
        stat_text = (
            f"节点数: {stats['nodes']}\n"
            f"边数: {stats['edges']}\n"
            f"密度: {stats['density']:.4f}\n"
            f"连通分量: {stats.get('components', 0)}"
        )
        print_panel(stat_text, title="[WEB] 网络统计")

    # KOL
    influencers = detect_influencers(G, metric="pagerank", top_n=10)
    if influencers:
        if HAS_RICH:
            # PageRank 条形图
            pr_items = [(inf["author"][:12], inf["pagerank"]) for inf in influencers[:10]]
            console.print(bar_chart(pr_items, title="⭐ 关键传播节点 (PageRank)", color=THEME.brand))
            console.print("")
        else:
            rows = []
            for inf in influencers:
                rows.append([
                    inf["author"], inf["platform"], inf["post_count"],
                    f"{inf['pagerank']:.4f}", f"{inf['betweenness']:.4f}",
                ])
            print_table(["作者", "平台", "帖子数", "PageRank", "Betweenness"], rows, title="[STAR] 关键传播节点 (KOL)")

    # 负面情绪放大器
    amplifiers = detect_negative_amplifiers(G, top_n=10)
    if amplifiers:
        if HAS_RICH:
            amp_items = [(amp["author"][:12], amp["composite_score"]) for amp in amplifiers[:10]]
            console.print(bar_chart(amp_items, title="😠 负面情绪放大器", color=THEME.error))
            console.print("")
        else:
            rows = []
            for amp in amplifiers:
                rows.append([amp["author"], amp["platform"], f"{amp['composite_score']:.2f}", amp["post_count"]])
            print_table(["作者", "平台", "复合得分", "帖子数"], rows, title="[OFF] 负面情绪放大器")

    # 传播模拟
    if args.simulate:
        seed_nodes = [inf["author_id"] for inf in influencers[:3] if inf.get("author_id")]
        if not seed_nodes and list(G.nodes()):
            seed_nodes = list(G.nodes())[:3]

        if args.model == "sir":
            if HAS_RICH:
                console.print(f"\n[bold {THEME.brand}]🦠 SIR 模拟[/] (β={args.beta}, γ={args.gamma}, steps={args.steps})")
            else:
                print(f"\n[VIRUS] SIR 模拟 (β={args.beta}, γ={args.gamma}, steps={args.steps})")
            result = simulate_sir(G, seed_nodes, beta=args.beta, gamma=args.gamma, steps=args.steps)
            final = result.timeline[-1] if result.timeline else {"susceptible": len(result.susceptible), "infected": len(result.infected), "recovered": len(result.recovered)}
            print(f"  最终: 易感={final['susceptible']:.0f}, 感染={final['infected']:.0f}, 恢复={final['recovered']:.0f}")
        else:
            if HAS_RICH:
                console.print(f"\n[bold {THEME.brand}]🤖 Agent-based 模拟[/] (steps={args.steps})")
            else:
                print(f"\n[AI] Agent-based 模拟 (steps={args.steps})")
            result = simulate_agent_based(G, seed_nodes, steps=args.steps)
            print(f"  最终传播节点数: {len(result.get('infected_nodes', []))}")

    # 保存网络图
    output_path = args.output
    if args.open and not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"sentifox_graph_{ts}.html"

    if output_path:
        if HAS_RICH:
            console.print(f"\n[dim]💾 保存网络图到 {output_path}...[/]")
        else:
            print(f"\n[SAVE] 保存网络图到 {output_path}...")
        try:
            from graph_analysis.graph_viz import create_pyvis_network, add_nodes_to_network, add_edges_to_network
            net = create_pyvis_network(G, directed=True)
            add_nodes_to_network(net, G, size_attr="engagement", color_attr="sentiment")
            add_edges_to_network(net, G)
            net.save_graph(output_path)
            abs_path = os.path.abspath(output_path)
            if HAS_RICH:
                console.print(f"  [green]✓ 已保存: {abs_path}[/]")
            else:
                print(f"  [OK] 已保存: {abs_path}")
            if args.open:
                if HAS_RICH:
                    console.print(f"[dim]🌐 正在打开浏览器...[/]")
                else:
                    print(f"[OPEN] 正在打开浏览器...")
                webbrowser.open(f"file://{abs_path}")
        except Exception as e:
            if HAS_RICH:
                console.print(f"  [red]✗ 保存失败: {e}[/]")
            else:
                print(f"  [FAIL] 保存失败: {e}")


def cmd_rag(args):
    """RAG 智能问答（升级版：对话气泡样式）"""
    from rag.rag_engine import ask as rag_ask

    if args.query:
        # 单次问答
        if HAS_RICH:
            console.print(chat_bubble_user(args.query))
        else:
            print(f"[rag] [BRAIN] 问题: {args.query}")
        
        result = rag_ask(
            args.query,
            platform_filter=args.platform or None,
            sentiment_filter=args.sentiment or None,
        )
        
        if HAS_RICH:
            console.print(chat_bubble_assistant(result["answer"], result.get("sources")))
        else:
            print_panel(result["answer"], title="[CHAT] 回答")
            if result.get("sources"):
                print("[BOOK] 参考来源:")
                for s in result["sources"][:5]:
                    print(f"  [{s['index']}] {s['platform']} | {s['author']} | 相关度{s['relevance']}")
    else:
        # 交互式
        if HAS_RICH:
            console.print(result_panel("输入问题开始对话，输入 quit/exit 退出\n示例: 最近负面舆情主要集中在哪些方面？",
                         title="[bold blue]💬 RAG 智能问答[/]", status="info"))
        else:
            print("[rag] [CHAT] RAG 智能问答 (输入 'quit' 退出)")
            print("示例: 最近负面舆情主要集中在哪些方面？")
        
        while True:
            try:
                if HAS_RICH:
                    console.print("\n[bold blue]👤 你[/]", end=" ")
                q = input().strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            
            result = rag_ask(
                q,
                platform_filter=args.platform or None,
                sentiment_filter=args.sentiment or None,
            )
            
            if HAS_RICH:
                console.print(chat_bubble_assistant(result["answer"], result.get("sources")))
            else:
                print_panel(result["answer"], title="[CHAT] 回答")
                if result.get("sources"):
                    print("[BOOK] 参考来源:")
                    for s in result["sources"][:5]:
                        print(f"  [{s['index']}] {s['platform']} | {s['author']} | 相关度{s['relevance']}")


def cmd_insight(args):
    """生成 AI 洞察"""
    from analysis.insight_generator import generate_insight

    start_time, end_time = parse_time_range(args.time_range)
    posts = get_posts(start_time=start_time, end_time=end_time, limit=500)
    sent_dist = get_sentiment_distribution(start_time, end_time)
    plat_dist = get_platform_distribution(start_time, end_time)

    print("[insight] [AI] 正在生成 AI 舆情洞察...")
    insight = generate_insight(posts, sent_dist, plat_dist)
    print_panel(insight, title="[TIP] AI 舆情洞察")


def cmd_alert(args):
    """查看告警（升级版：严重度彩色标签）"""
    alerts = get_alerts(limit=args.limit)
    if not alerts:
        if HAS_RICH:
            console.print(result_panel("当前无告警", title="[green]✓ 告警状态[/]", status="ok"))
        else:
            print("[alert] [OK] 当前无告警")
        return

    if HAS_RICH:
        console.print(section_header(f"🚨 最近 {len(alerts)} 条告警", "🚨"))
        for a in alerts[:args.limit]:
            console.print(alert_card(
                a.get("alert_type", ""),
                a.get("severity", "low"),
                a.get("message", "")[:80],
                a.get("created_at", "")
            ))
    else:
        rows = []
        for a in alerts:
            severity = a.get("severity", "low")
            icon = "[OFF]" if severity == "high" else "[WARN2]" if severity == "medium" else "[ON]"
            rows.append([
                icon, a.get("alert_type", ""), severity,
                a.get("message", "")[:50], a.get("created_at", ""),
            ])
        print_table(["级别", "类型", "严重度", "消息", "时间"], rows, title=f"[ALERT] 最近 {len(alerts)} 条告警")


def cmd_report(args):
    """生成报告（升级版：进度动画）"""
    from reports.generator import generate_report

    start_time, end_time = parse_time_range(args.time_range)
    platforms = args.platforms or PLATFORMS

    if HAS_RICH:
        console.print(f"[{THEME.brand}]📝[/] 正在生成报告: [bold]{args.title}[/]")
        with console.status("[bold green]正在生成 Word 报告...[/]", spinner="dots"):
            try:
                path = generate_report(
                    title=args.title,
                    platforms=platforms,
                    start_time=start_time,
                    end_time=end_time,
                    include_graph=True,
                )
                import os
                size = os.path.getsize(path) / 1024
                console.print(result_panel(
                    f"路径: {path}\n大小: {size:.1f} KB",
                    title=f"[green]✓ 报告已生成[/]",
                    status="ok"
                ))
            except Exception as e:
                console.print(f"[red]✗ 生成失败: {e}[/]")
    else:
        print(f"[report] 正在生成报告: {args.title}")
        try:
            path = generate_report(
                title=args.title,
                platforms=platforms,
                start_time=start_time,
                end_time=end_time,
                include_graph=True,
            )
            print(f"[report] [OK] 报告已生成: {path}")
        except Exception as e:
            print(f"[report] [FAIL] 生成失败: {e}")


def cmd_sync_vectors(args):
    """同步到向量库"""
    from rag.document_processor import sync_posts_to_vector_store

    print("[sync-vectors] 从数据库读取帖子并同步到 ChromaDB...")
    posts = get_posts(limit=5000)
    if not posts:
        print("[sync-vectors] [WARN] 数据库无数据")
        return
    try:
        sync_posts_to_vector_store(posts)
        print(f"[sync-vectors] [OK] 已同步 {len(posts)} 条帖子到向量库")
    except Exception as e:
        print(f"[sync-vectors] [FAIL] 同步失败: {e}")


def cmd_pipeline(args):
    """完整流水线: 采集 -> 分析 -> 聚类 -> 向量同步 -> 传播图谱 -> RAG洞察 -> 报告（Live 进度条版）"""
    from crawlers.manager import CrawlerManager
    from analysis.sentiment import get_analyzer
    from analysis.topic_clustering import cluster_posts
    from graph_analysis.graph_builder import build_graph_from_posts, get_graph_stats
    from graph_analysis.influencer_detection import detect_influencers
    from rag.rag_engine import ask as rag_ask
    from rag.document_processor import sync_posts_to_vector_store
    from reports.generator import generate_report
    from graph_analysis.sentifox_simulator import SentifoxSimulator
    from graph_analysis.interventions import SeedInfectionIntervention
    from graph_analysis.simulation_report import generate_simulation_report
    from graph_analysis.graph_builder import build_graph_from_posts
    from graph_analysis.influencer_detection import detect_influencers

    keywords = [k.strip() for k in args.keywords.split(",")] if args.keywords else CONFIG.crawler.keywords
    platforms = args.platforms or PLATFORMS
    count = args.count
    output_dir = args.output_dir or os.getcwd()
    simulate_steps = args.simulate_steps if not args.no_simulate else 0
    os.makedirs(output_dir, exist_ok=True)

    pipeline_steps = [
        "数据采集", "情感分析", "话题聚类", "向量同步",
        "传播图谱", "RAG 洞察", "生成报告",
        "Sentifox 仿真", "仿真报告"
    ]

    if HAS_RICH:
        console.print(f"[bold {THEME.brand}]🦊 启动完整流水线[/]")
        console.print(f"  关键词: {keywords}")
        console.print(f"  平台: {platforms}")
        console.print(f"  每平台采集: {count}")
        console.print(f"  输出目录: {output_dir}\n")
        
        step_progress = StepProgress(pipeline_steps, title="Sentifox Pipeline")
        live_log = LiveLog(title="执行日志", max_lines=8)
        
        # 组合显示
        from rich.layout import Layout
        layout = Layout()
        layout.split_column(
            Layout(step_progress, name="progress", size=len(pipeline_steps) + 5),
            Layout(live_log, name="log"),
        )
        
        live_ctx = Live(layout, console=console, refresh_per_second=8, transient=False)
        live_ctx.start()
    else:
        print(f"[pipeline] [GO] 启动完整流水线")
        print(f"  关键词: {keywords}")
        print(f"  平台: {platforms}")
        print(f"  每平台采集: {count}")
        print(f"  输出目录: {output_dir}")
        live_ctx = None
        step_progress = None
        live_log = None

    def _log(msg: str, level: str = "info"):
        if live_log:
            live_log.append(msg, level)
        else:
            prefix = {"info": "  ", "warn": "  [WARN] ", "error": "  [ERR] ", "ok": "  [OK] "}.get(level, "  ")
            print(f"{prefix}{msg}")

    def _start_step(idx: int):
        if step_progress:
            step_progress.start_step(idx)
            live_ctx.update(layout)

    def _finish_step(idx: int, detail: str = "", status: str = "done"):
        if step_progress:
            step_progress.finish_step(idx, detail=detail, status=status)
            live_ctx.update(layout)

    graph_html_path = None

    try:
        # ── Step 1: 数据采集 ──────────────────
        _start_step(0)
        _log("启动多平台采集...")
        manager = CrawlerManager(keywords=keywords)
        results = manager.crawl_and_process(
            platforms=platforms,
            max_posts_per_platform=count,
            use_mock_fallback=True,
        )
        _log(f"采集完成: {len(results)} 条数据")
        _finish_step(0, f"{len(results)} 条")

        # ── Step 2: 情感分析 ──────────────────
        _start_step(1)
        try:
            posts = get_posts(limit=5000)
            if posts:
                _log(f"加载 {len(posts)} 条帖子进行情感分析...")
                analyzer = get_analyzer()
                analyzed = analyzer.analyze_posts(posts)
                sent_dist = {"positive": 0, "negative": 0, "neutral": 0}
                for p in analyzed:
                    lbl = p.get("sentiment_label", "neutral")
                    sent_dist[lbl] = sent_dist.get(lbl, 0) + 1
                detail = f"正{sent_dist.get('positive',0)} 负{sent_dist.get('negative',0)} 中{sent_dist.get('neutral',0)}"
                _log(f"情感分析完成: {detail}")
                _finish_step(1, detail)
            else:
                _log("无数据，跳过情感分析", "warn")
                _finish_step(1, "跳过", "warn")
        except Exception as e:
            _log(f"情感分析失败: {e}", "error")
            _finish_step(1, f"失败: {e}", "error")

        # ── Step 3: 话题聚类 ──────────────────
        _start_step(2)
        try:
            posts = get_posts(limit=2000)
            if posts and len(posts) >= 5:
                n_clusters = min(5, len(posts) // 10)
                if n_clusters < 2:
                    n_clusters = 2
                _log(f"对 {len(posts)} 条帖子聚类 (K={n_clusters})...")
                clustered, topics = cluster_posts(posts, n_clusters=n_clusters)
                _log(f"聚类完成: {len(topics)} 个话题")
                if args.verbose:
                    for tid, topic in topics.items():
                        t_posts = [p for p in clustered if p.get("topic_id") == tid]
                        kw = ", ".join(topic.get("top_keywords", [])[:3])
                        _log(f"  话题{tid}: {len(t_posts)}条 关键词:{kw}")
                _finish_step(2, f"{len(topics)} 个话题")
            else:
                _log("数据不足，跳过话题聚类", "warn")
                _finish_step(2, "跳过", "warn")
        except Exception as e:
            _log(f"话题聚类失败: {e}", "error")
            _finish_step(2, f"失败: {e}", "error")

        # ── Step 4: 向量同步 ──────────────────
        _start_step(3)
        try:
            posts = get_posts(limit=5000)
            if posts:
                _log(f"同步 {len(posts)} 条帖子到 ChromaDB...")
                sync_posts_to_vector_store(posts)
                _log(f"向量库同步完成")
                _finish_step(3, f"{len(posts)} 条")
            else:
                _log("无数据，跳过向量同步", "warn")
                _finish_step(3, "跳过", "warn")
        except Exception as e:
            _log(f"向量同步失败: {e}", "error")
            _finish_step(3, f"失败: {e}", "error")

        # ── Step 5: 传播图谱 ──────────────────
        _start_step(4)
        try:
            posts = get_posts(limit=2000)
            edges = get_edges()
            if posts:
                _log("构建传播网络...")
                G = build_graph_from_posts(posts, edges)
                stats = get_graph_stats(G)
                _log(f"图谱构建完成: {stats['nodes']}节点 {stats['edges']}边")

                influencers = detect_influencers(G, metric="pagerank", top_n=5)
                if influencers and args.verbose:
                    for inf in influencers[:5]:
                        _log(f"  KOL: {inf['author']} ({inf['platform']}) PR={inf['pagerank']:.4f}")

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                graph_html_path = os.path.join(output_dir, f"sentifox_graph_{ts}.html")
                from graph_analysis.graph_viz import create_pyvis_network, add_nodes_to_network, add_edges_to_network
                net = create_pyvis_network(G, directed=True)
                add_nodes_to_network(net, G, size_attr="engagement", color_attr="sentiment")
                add_edges_to_network(net, G)
                net.save_graph(graph_html_path)
                _log(f"图谱 HTML 已保存")
                _finish_step(4, f"{stats['nodes']}节点 {stats['edges']}边")

                if args.open:
                    abs_path = os.path.abspath(graph_html_path)
                    webbrowser.open(f"file://{abs_path}")
            else:
                _log("无数据，跳过传播图谱", "warn")
                _finish_step(4, "跳过", "warn")
        except Exception as e:
            _log(f"传播图谱失败: {e}", "error")
            _finish_step(4, f"失败: {e}", "error")

        # ── Step 6: RAG 洞察 ──────────────────
        _start_step(5)
        try:
            _log("生成 RAG 智能洞察...")
            rag_prompt = (
                "基于当前舆情数据，请总结："
                "1. 整体情感态势如何？2. 负面舆情的主要焦点是什么？"
                "3. 有哪些值得关注的传播节点？4. 给出3条应对建议。"
            )
            rag_result = rag_ask(rag_prompt, top_k=10)
            _log("RAG 洞察生成完成")
            if args.verbose:
                if HAS_RICH:
                    live_ctx.stop()
                    console.print(result_panel(rag_result["answer"], title="[bold green]💡 RAG 洞察[/]", status="ok"))
                    live_ctx.start()
            _finish_step(5, "完成")
        except Exception as e:
            _log(f"RAG 洞察失败: {e}", "error")
            _finish_step(5, f"失败: {e}", "error")

        # ── Step 7: 生成报告 ──────────────────
        _start_step(6)
        try:
            _log("生成 Word 分析报告...")
            report_path = generate_report(
                title=f"舆情报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                platforms=platforms,
                include_graph=True,
            )
            if output_dir and os.path.dirname(report_path) != os.path.abspath(output_dir):
                import shutil
                new_path = os.path.join(output_dir, os.path.basename(report_path))
                shutil.move(report_path, new_path)
                report_path = new_path
            _log(f"报告已生成: {report_path}")
            _finish_step(6, os.path.basename(report_path))
        except Exception as e:
            _log(f"报告生成失败: {e}", "error")
            _finish_step(6, f"失败: {e}", "error")

        # ── Step 8: Sentifox 仿真 ─────────────
        if simulate_steps > 0:
            _start_step(7)
            try:
                posts = get_posts(limit=args.max_posts)
                edges = get_edges()
                if posts:
                    _log(f"构建 Sentifox 仿真环境 (认知层+通信层)...")
                    simulator = SentifoxSimulator.from_data(posts, edges, config={
                        "total_steps": simulate_steps,
                        "use_cognitive": True,
                        "use_communication": True,
                    })
                    # 自动选择 KOL 作为种子
                    G = build_graph_from_posts(posts, edges)
                    influencers = detect_influencers(G, metric="pagerank", top_n=3)
                    seed_ids = [inf["author_id"] for inf in influencers if inf.get("author_id")]
                    if not seed_ids and posts:
                        seed_ids = [posts[0].get("author_id", "") or posts[0].get("author", "")]
                    simulator.add_intervention(SeedInfectionIntervention(seed_ids, step=0))
                    _log(f"Agent 数: {len(simulator.agents)}, 种子: {seed_ids[:3]}")
                    
                    result = simulator.run(steps=simulate_steps)
                    final = result.timeline[-1] if result.timeline else None
                    if final:
                        _log(f"仿真完成: 感染={final.infected_count} 恢复={final.recovered_count} 峰值={max(s.infected_count for s in result.timeline)}")
                        _finish_step(7, f"感染{final.infected_count}")
                    else:
                        _finish_step(7, "完成")
                else:
                    _log("无数据，跳过仿真", "warn")
                    _finish_step(7, "跳过", "warn")
            except Exception as e:
                _log(f"Sentifox 仿真失败: {e}", "error")
                _finish_step(7, f"失败: {e}", "error")
        else:
            _log("跳过 Sentifox 仿真 (--no-simulate)")
            _finish_step(7, "跳过")

        # ── Step 9: 仿真报告 ──────────────────
        if simulate_steps > 0 and args.simulate_report:
            _start_step(8)
            try:
                _log("生成 Sentifox 仿真 HTML 报告...")
                # 收集认知层数据
                cognitive_data = {}
                if simulator.use_cognitive:
                    emotion_samples = []
                    for aid, cog in list(simulator.cognitive_agents.items())[:10]:
                        agent = simulator.agents.get(aid)
                        emotion_samples.append({
                            "agent_id": aid,
                            "agent_name": agent.name if agent else aid[:8],
                            "emotion": cog.emotion.dominant,
                            "intensity": round(cog.emotion.intensity, 2),
                            "arousal": round(cog.emotion.arousal, 2),
                            "valence": round(cog.emotion.valence, 2),
                        })
                    bdi_samples = []
                    for aid, cog in simulator.cognitive_agents.items():
                        if cog.reasoning_log:
                            for log in cog.reasoning_log[-3:]:
                                agent = simulator.agents.get(aid)
                                bdi_samples.append({
                                    "agent_id": aid,
                                    "agent_name": agent.name if agent else aid[:8],
                                    "step": log.get("step"),
                                    "emotion": log.get("emotion", {}).get("dominant", "?"),
                                    "desire": log.get("desire", "?"),
                                    "intention": log.get("intention", "?"),
                                    "reasoning": log.get("reasoning", []),
                                })
                    bdi_samples = bdi_samples[-20:]
                    bridge_nodes = []
                    if result.timeline:
                        final_snapshot = result.timeline[-1]
                        for node_id, score in getattr(final_snapshot, "bridge_nodes", []):
                            agent = simulator.agents.get(node_id)
                            bridge_nodes.append({"id": node_id, "name": agent.name if agent else node_id[:8], "score": score})
                    cognitive_data = {
                        "emotion_samples": emotion_samples,
                        "bdi_samples": bdi_samples,
                        "bridge_nodes": bridge_nodes,
                    }
                sim_report_path = generate_simulation_report(
                    result,
                    temporal_graph_data=simulator.temporal_graph.to_dict(),
                    cognitive_data=cognitive_data,
                    simulator=simulator,
                )
                if output_dir and os.path.dirname(sim_report_path) != os.path.abspath(output_dir):
                    import shutil
                    new_path = os.path.join(output_dir, os.path.basename(sim_report_path))
                    shutil.move(sim_report_path, new_path)
                    sim_report_path = new_path
                _log(f"仿真报告: {sim_report_path}")
                _finish_step(8, os.path.basename(sim_report_path))
            except Exception as e:
                _log(f"仿真报告生成失败: {e}", "error")
                _finish_step(8, f"失败: {e}", "error")
        else:
            _finish_step(8, "跳过")

        # ── 告警检查 ──────────────────────────
        try:
            posts_all = get_posts()
            if posts_all:
                neg_count = sum(1 for p in posts_all if p.get("sentiment_label") == "negative")
                total = len(posts_all)
                neg_ratio = neg_count / total if total > 0 else 0
                if neg_ratio > ALERT_CONFIG["negative_ratio_threshold"]:
                    _log(f"负面占比 {neg_ratio*100:.1f}% 超过阈值!", "warn")
                else:
                    _log(f"负面占比 {neg_ratio*100:.1f}% 正常")
        except Exception as e:
            _log(f"告警检查失败: {e}", "warn")

    finally:
        if live_ctx:
            live_ctx.stop()

    if HAS_RICH:
        console.print(f"\n[bold green]✓ 全流程完成[/]")
        if 'report_path' in dir() and report_path:
            console.print(f"  Word 报告: [link=file://{os.path.abspath(report_path)}]{report_path}[/link]")
        if 'sim_report_path' in dir() and sim_report_path and simulate_steps > 0 and args.simulate_report:
            console.print(f"  仿真报告: [link=file://{os.path.abspath(sim_report_path)}]{sim_report_path}[/link]")
    else:
        print("\n[pipeline] [OK] 全流程完成")


def cmd_test(args):
    """运行功能测试（升级版：测试结果表格）"""
    from crawlers.mock_generator import MockCrawler
    from analysis.sentiment import get_analyzer
    from analysis.topic_clustering import cluster_posts
    from rag.rag_engine import ask as rag_ask

    if HAS_RICH:
        console.print(section_header("🔧 功能测试", "🔧"))
    else:
        print("[test] [TEST] 开始功能测试\n")
    
    results = []

    tests = [
        ("数据库", lambda: init_db() or True),
        ("情感模型", lambda: get_analyzer().predict("测试文本") or True),
        ("Mock爬虫", lambda: len(MockCrawler(keywords=["测试"]).crawl(max_posts=5)) > 0),
        ("数据库查询", lambda: get_post_count() >= 0),
        ("话题聚类", lambda: cluster_posts([{"content": "测试1"}, {"content": "测试2"}, {"content": "测试3"}]) or True),
        ("RAG引擎", lambda: rag_ask("测试") or True),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            results.append((name, True, None))
            if not HAS_RICH:
                print(f"  [OK] {name}")
        except Exception as e:
            results.append((name, False, str(e)))
            if not HAS_RICH:
                print(f"  [FAIL] {name}: {e}")

    if HAS_RICH:
        console.print(test_result_table(results))
    
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    
    if not HAS_RICH:
        print(f"\n[test] 结果: {passed}/{total} 通过")
    
    if passed < total:
        if not HAS_RICH:
            print(f"[test] [FAIL] {total - passed} 个失败")
        sys.exit(1)
    else:
        if HAS_RICH:
            console.print("\n[bold green]✓ 全部通过[/]")
        else:
            print("[test] [OK] 全部通过")




def cmd_simulate(args):
    """Sentifox 多智能体传播仿真"""
    from graph_analysis.sentifox_simulator import SentifoxSimulator
    from graph_analysis.temporal_graph import TemporalGraph
    from graph_analysis.persona_agent import create_agents_from_graph
    from graph_analysis.interventions import (
        parse_intervention, SeedInfectionIntervention, compare_scenarios
    )
    from graph_analysis.simulation_report import generate_simulation_report
    from cli.simulation_display import SimulationLiveDisplay
    from graph_analysis.graph_builder import build_graph_from_posts
    from graph_analysis.influencer_detection import detect_influencers

    steps = args.steps
    realtime = args.realtime
    compare = args.compare_baseline
    
    # 构建数据
    if args.from_data:
        if HAS_RICH:
            console.print(f"[{THEME.brand}]🦊[/] 从当前数据构建时态图谱...")
        else:
            print("[simulate] 从当前数据构建时态图谱...")
        
        posts = get_posts(limit=args.max_posts)
        edges = get_edges()
        
        if not posts:
            if HAS_RICH:
                console.print("[yellow]⚠ 数据库无数据，请先运行 mock 或 crawl[/]")
            else:
                print("[simulate] [WARN] 数据库无数据")
            return
        
        # 创建仿真器
        simulator = SentifoxSimulator.from_data(posts, edges, config={
            "total_steps": steps,
            "speed": 1.0,
        })
        
        # 设置种子
        if args.seed:
            seed_ids = args.seed
        else:
            # 从 KOL 中选种子
            G = build_graph_from_posts(posts, edges)
            influencers = detect_influencers(G, metric="pagerank", top_n=3)
            seed_ids = [inf["author_id"] for inf in influencers if inf.get("author_id")]
            if not seed_ids and posts:
                seed_ids = [posts[0].get("author_id", "") or posts[0].get("author", "")]
        
        simulator.add_intervention(SeedInfectionIntervention(seed_ids, step=0))
    else:
        if HAS_RICH:
            console.print("[yellow]⚠ 请使用 --from-data 从现有数据启动仿真[/]")
        else:
            print("[simulate] [WARN] 请使用 --from-data")
        return
    
    # 添加干预
    for intervention_spec in args.intervention:
        try:
            intervention = parse_intervention(intervention_spec)
            simulator.add_intervention(intervention)
            if HAS_RICH:
                console.print(f"  [dim]已添加干预: {intervention.name} @ step {intervention.step}[/]")
        except Exception as e:
            if HAS_RICH:
                console.print(f"  [red]✗ 干预解析失败: {intervention_spec} - {e}[/]")
            else:
                print(f"[WARN] 干预解析失败: {intervention_spec} - {e}")
    
    # 采访模式
    if args.interview:
        agent_id = args.interview
        question = args.query or "你为什么参与这次传播？"
        answer = simulator.interview_agent(agent_id, question)
        if HAS_RICH:
            console.print(result_panel(answer, title=f"[bold]{agent_id} 的回答[/]", status="info"))
        else:
            print(f"\n[INTERVIEW] {agent_id}:")
            print(f"  Q: {question}")
            print(f"  A: {answer}")
        return
    
    # 运行仿真
    if HAS_RICH:
        console.print(f"\n[bold {THEME.brand}]🚀 启动 Sentifox 传播仿真[/]")
        console.print(f"  步数: {steps} (每步=1小时)")
        console.print(f"  Agent 数: {len(simulator.agents)}")
        console.print(f"  种子: {seed_ids}")
        console.print(f"  干预: {len(simulator.interventions)} 项\n")
    else:
        print(f"\n[simulate] 启动仿真: {steps} 步, {len(simulator.agents)} 个 Agent")
    
    # 实时显示
    display = SimulationLiveDisplay(simulator) if realtime else None
    
    def realtime_callback(step, snapshot, actions):
        if display and HAS_RICH:
            layout = display.render(step, snapshot, actions)
            # Live 对象在 simulator.run 中管理
    
    try:
        if realtime and HAS_RICH:
            from rich.live import Live
            with Live(console=console, refresh_per_second=8, transient=False) as live:
                def wrapped_callback(step, snapshot, actions):
                    if display:
                        layout = display.render(step, snapshot, actions)
                        live.update(layout)
                
                result = simulator.run(steps=steps, realtime_callback=wrapped_callback)
        else:
            result = simulator.run(steps=steps)
    except KeyboardInterrupt:
        if HAS_RICH:
            console.print("\n[yellow]⚠ 仿真被用户中断[/]")
        else:
            print("\n[simulate] 仿真被中断")
        return
    
    # 输出结果摘要
    if result.timeline:
        final = result.timeline[-1]
        if HAS_RICH:
            console.print(f"\n[bold {THEME.brand}]📊 仿真结果摘要[/]")
            stat_cards = [
                kpi_card("最终感染", str(final.infected_count), color=THEME.error),
                kpi_card("最终易感", str(final.susceptible_count), color="grey50"),
                kpi_card("最终恢复", str(final.recovered_count), color=THEME.success),
                kpi_card("峰值感染", str(max(s.infected_count for s in result.timeline)), color=THEME.brand),
            ]
            console.print(kpi_row(stat_cards))
        else:
            print(f"\n[simulate] 仿真完成")
            print(f"  最终感染: {final.infected_count}")
            print(f"  最终易感: {final.susceptible_count}")
            print(f"  最终恢复: {final.recovered_count}")
    
    # 生成报告
    if args.report:
        if HAS_RICH:
            console.print(f"\n[{THEME.brand}]📝[/] 正在生成 HTML 报告...")
        else:
            print("\n[simulate] 生成报告...")
        
        try:
            # 收集认知层数据用于报告
            cognitive_data = {}
            if simulator.use_cognitive:
                # 情绪状态采样（Top 10）
                emotion_samples = []
                for aid, cog in list(simulator.cognitive_agents.items())[:10]:
                    agent = simulator.agents.get(aid)
                    emotion_samples.append({
                        "agent_id": aid,
                        "agent_name": agent.name if agent else aid[:8],
                        "emotion": cog.emotion.dominant,
                        "intensity": round(cog.emotion.intensity, 2),
                        "arousal": round(cog.emotion.arousal, 2),
                        "valence": round(cog.emotion.valence, 2),
                    })
                # BDI 决策链采样（最近5条）
                bdi_samples = []
                for aid, cog in simulator.cognitive_agents.items():
                    if cog.reasoning_log:
                        for log in cog.reasoning_log[-3:]:
                            agent = simulator.agents.get(aid)
                            bdi_samples.append({
                                "agent_id": aid,
                                "agent_name": agent.name if agent else aid[:8],
                                "step": log.get("step"),
                                "emotion": log.get("emotion", {}).get("dominant", "?"),
                                "desire": log.get("desire", "?"),
                                "intention": log.get("intention", "?"),
                                "reasoning": log.get("reasoning", []),
                            })
                bdi_samples = bdi_samples[-20:]  # 只保留最近20条
                # 桥接节点
                bridge_nodes = []
                if result.timeline:
                    final_snapshot = result.timeline[-1]
                    for node_id, score in getattr(final_snapshot, "bridge_nodes", []):
                        agent = simulator.agents.get(node_id)
                        bridge_nodes.append({
                            "id": node_id,
                            "name": agent.name if agent else node_id[:8],
                            "score": score,
                        })
                cognitive_data = {
                    "emotion_samples": emotion_samples,
                    "bdi_samples": bdi_samples,
                    "bridge_nodes": bridge_nodes,
                }
            
            report_path = generate_simulation_report(
                result,
                temporal_graph_data=simulator.temporal_graph.to_dict(),
                cognitive_data=cognitive_data,
                simulator=simulator,
            )
            if HAS_RICH:
                console.print(f"  [green]✓ 报告已生成: {report_path}[/]")
            else:
                print(f"  [OK] 报告: {report_path}")
            
            if args.open:
                webbrowser.open(f"file://{report_path}")
        except Exception as e:
            if HAS_RICH:
                console.print(f"  [red]✗ 报告生成失败: {e}[/]")
            else:
                print(f"  [FAIL] 报告生成失败: {e}")


# ═══════════════════════════════════════════════════════════════
# argparse 主入口 + 交互式 REPL
# ═══════════════════════════════════════════════════════════════

class CLIExit(Exception):
    """CLI 内部退出信号，不终止整个进程"""
    pass


class NoExitArgumentParser(argparse.ArgumentParser):
    """不调用 sys.exit() 的 ArgumentParser，用于 REPL 模式"""
    def error(self, message):
        if HAS_RICH:
            console.print(f"[bold red][FAIL][/] 参数错误: {message}")
            console.print("[dim]输入 [bold]help[/] 查看用法[/]")
        else:
            print(f"[FAIL] 参数错误: {message}")
            print("输入 'help' 查看用法")
        raise CLIExit(2)

    def exit(self, status=0, message=None):
        if message:
            if HAS_RICH:
                console.print(message)
            else:
                print(message)
        raise CLIExit(status)


def build_parser():
    parser = NoExitArgumentParser(
        prog="sentifox",
        description="舆情分析系统 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""",
示例:
  sentifox init
  sentifox mock --count 200 --keywords "品牌,产品"
  sentifox crawl --platforms 微博 知乎 --count 30
  sentifox status
  sentifox overview --time-range 7d --insight
  sentifox topics --clusters 5
  sentifox graph --simulate --model sir --output graph.html --open
  sentifox rag --query "最近负面舆情主要集中在哪些方面？"
  sentifox insight
  sentifox alert
  sentifox report --title "周报"
  sentifox pipeline --keywords "品牌" --count 50 --open --simulate-steps 48
  sentifox pipeline --keywords "品牌" --count 50 --no-simulate
  sentifox simulate --from-data --steps 48 --realtime --report
  sentifox test
        """,

    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # init
    subparsers.add_parser("init", help="初始化数据库")

    # mock
    p_mock = subparsers.add_parser("mock", help="生成模拟数据并处理")
    p_mock.add_argument("--count", type=int, default=200, help="帖子数量")
    p_mock.add_argument("--keywords", type=str, default="", help="关键词，逗号分隔")
    p_mock.add_argument("--clusters", type=int, default=5, help="话题聚类数")
    p_mock.add_argument("--no-vectors", action="store_true", help="跳过向量库同步")

    # crawl
    p_crawl = subparsers.add_parser("crawl", help="真实采集")
    p_crawl.add_argument("--platforms", nargs="+", help="平台列表")
    p_crawl.add_argument("--keywords", type=str, default="", help="关键词")
    p_crawl.add_argument("--count", type=int, default=50, help="每平台数量")

    # status
    subparsers.add_parser("status", help="查看平台状态")

    # overview
    p_overview = subparsers.add_parser("overview", help="概览统计")
    p_overview.add_argument("--time-range", type=str, default="7d", choices=["24h", "3d", "7d", "30d", "all"], help="时间范围")
    p_overview.add_argument("--platforms", nargs="+", help="平台筛选")
    p_overview.add_argument("--insight", action="store_true", help="生成AI洞察")

    # topics
    p_topics = subparsers.add_parser("topics", help="话题分析")
    p_topics.add_argument("--time-range", type=str, default="all", choices=["24h", "3d", "7d", "30d", "all"])
    p_topics.add_argument("--clusters", type=int, default=5, help="聚类数")

    # graph
    p_graph = subparsers.add_parser("graph", help="传播图谱")
    p_graph.add_argument("--time-range", type=str, default="all", choices=["24h", "3d", "7d", "30d", "all"])
    p_graph.add_argument("--simulate", action="store_true", help="运行传播模拟")
    p_graph.add_argument("--model", type=str, default="sir", choices=["sir", "agent"], help="模拟模型")
    p_graph.add_argument("--beta", type=float, default=0.3, help="SIR感染率")
    p_graph.add_argument("--gamma", type=float, default=0.1, help="SIR恢复率")
    p_graph.add_argument("--steps", type=int, default=50, help="模拟步数")
    p_graph.add_argument("--layout", type=str, default="force_atlas2", help="网络布局")
    p_graph.add_argument("--output", type=str, default="", help="输出HTML文件路径")
    p_graph.add_argument("--open", action="store_true", help="生成后自动打开浏览器")

    # rag
    p_rag = subparsers.add_parser("rag", help="RAG智能问答")
    p_rag.add_argument("--query", type=str, default="", help="问题（为空则进入交互模式）")
    p_rag.add_argument("--platform", type=str, default="", help="过滤平台")
    p_rag.add_argument("--sentiment", type=str, default="", choices=["", "positive", "negative", "neutral"], help="过滤情感")

    # insight
    p_insight = subparsers.add_parser("insight", help="AI洞察")
    p_insight.add_argument("--time-range", type=str, default="all", choices=["24h", "3d", "7d", "30d", "all"])

    # alert
    p_alert = subparsers.add_parser("alert", help="告警列表")
    p_alert.add_argument("--limit", type=int, default=20, help="显示条数")

    # report
    p_report = subparsers.add_parser("report", help="生成报告")
    p_report.add_argument("--title", type=str, default="舆情分析报告", help="报告标题")
    p_report.add_argument("--time-range", type=str, default="all", choices=["24h", "3d", "7d", "30d", "all"])
    p_report.add_argument("--platforms", nargs="+", help="平台筛选")

    # sync-vectors
    subparsers.add_parser("sync-vectors", help="同步到向量库")

    # pipeline
    p_pipeline = subparsers.add_parser("pipeline", help="完整流水线")
    p_pipeline.add_argument("--keywords", type=str, default="", help="关键词")
    p_pipeline.add_argument("--platforms", nargs="+", help="平台列表")
    p_pipeline.add_argument("--count", type=int, default=50, help="每平台数量")
    p_pipeline.add_argument("--verbose", action="store_true", help="详细输出")
    p_pipeline.add_argument("--output-dir", type=str, default="", help="输出目录（报告和图谱HTML）")
    p_pipeline.add_argument("--open", action="store_true", help="自动打开传播图谱")
    p_pipeline.add_argument("--simulate-steps", type=int, default=48, help="Sentifox 仿真步数（默认48，0=不仿真）")
    p_pipeline.add_argument("--no-simulate", action="store_true", help="跳过 Sentifox 仿真")
    p_pipeline.add_argument("--simulate-report", action="store_true", default=True, help="生成仿真 HTML 报告")
    p_pipeline.add_argument("--max-posts", type=int, default=2000, help="仿真时最大加载帖子数")

    # test
    subparsers.add_parser("test", help="功能测试")
    
    # simulate (Sentifox 传播仿真)
    p_simulate = subparsers.add_parser("simulate", help="Sentifox 多智能体传播仿真")
    p_simulate.add_argument("--from-data", action="store_true", help="从当前数据库数据构建仿真")
    p_simulate.add_argument("--steps", type=int, default=48, help="仿真步数（每步=1小时）")
    p_simulate.add_argument("--seed", nargs="+", default=[], help="种子节点ID列表")
    p_simulate.add_argument("--max-posts", type=int, default=2000, help="最大加载帖子数")
    p_simulate.add_argument("--intervention", nargs="+", default=[], help="干预措施（如 delete:user@step10）")
    p_simulate.add_argument("--compare-baseline", action="store_true", help="与无干预基准对比")
    p_simulate.add_argument("--realtime", action="store_true", help="CLI 实时展示仿真过程")
    p_simulate.add_argument("--report", action="store_true", help="生成 HTML 报告")
    p_simulate.add_argument("--open", action="store_true", help="仿真结束后自动打开报告")
    p_simulate.add_argument("--interview", type=str, default="", help="采访指定 Agent")
    p_simulate.add_argument("--query", type=str, default="", help="采访问题")

    commands = {
        "init": cmd_init,
        "mock": cmd_mock,
        "crawl": cmd_crawl,
        "status": cmd_status,
        "overview": cmd_overview,
        "topics": cmd_topics,
        "graph": cmd_graph,
        "rag": cmd_rag,
        "insight": cmd_insight,
        "alert": cmd_alert,
        "report": cmd_report,
        "sync-vectors": cmd_sync_vectors,
        "pipeline": cmd_pipeline,
        "test": cmd_test,
        "simulate": cmd_simulate,
    }

    return parser, commands, subparsers


def print_welcome():
    print_welcome_with_fox()


def print_interactive_help(subparsers):
    if HAS_RICH:
        console.print("")
        table = Table(title="[bold]可用命令[/]", show_header=True, header_style="bold magenta", box=box.SIMPLE)
        table.add_column("命令", style="bold green")
        table.add_column("说明")
        table.add_column("常用示例")
        rows = [
            ("init", "初始化数据库", "init"),
            ("mock", "生成模拟数据", "mock --count 100"),
            ("crawl", "真实采集", "crawl --platforms 微博 知乎"),
            ("status", "查看平台状态", "status"),
            ("overview", "概览统计", "overview --time-range 7d"),
            ("topics", "话题分析", "topics --clusters 5"),
            ("graph", "传播图谱", "graph --simulate --output g.html --open"),
            ("rag", "智能问答", 'rag --query "负面舆情?"'),
            ("insight", "AI洞察", "insight"),
            ("alert", "告警列表", "alert"),
            ("report", "生成报告", "report --title 周报"),
            ("sync-vectors", "同步向量库", "sync-vectors"),
            ("pipeline", "完整流水线", "pipeline --keywords 品牌 --open"),
            ("simulate", "Sentifox传播仿真", "simulate --from-data --steps 48 --realtime"),
            ("test", "功能测试", "test"),
        ]
        for cmd, desc, example in rows:
            table.add_row(cmd, desc, example)
        console.print(table)
        console.print("\n[dim]提示: 直接输入命令即可执行，支持 Tab 类自动补全（输入部分命令按 Tab）[/]")
        console.print("[dim]      输入 [bold]clear[/] 清屏，[bold]exit[/] 退出[/]\n")
    else:
        print("\n可用命令:")
        print("  init          初始化数据库")
        print("  mock          生成模拟数据")
        print("  crawl         真实采集")
        print("  status        查看平台状态")
        print("  overview      概览统计")
        print("  topics        话题分析")
        print("  graph         传播图谱")
        print("  rag           智能问答")
        print("  insight       AI洞察")
        print("  alert         告警列表")
        print("  report        生成报告")
        print("  sync-vectors  同步向量库")
        print("  pipeline      完整流水线")
        print("  simulate      Sentifox传播仿真")
        print("  test          功能测试")
        print("\n提示: 输入 help 查看帮助，exit 退出\n")


def interactive_mode(parser, commands, subparsers):
    play_startup_animation()
    print_welcome()
    
    # 初始化命令历史
    history = CommandHistory()
    
    while True:
        try:
            if HAS_RICH:
                console.print(f"[bold {THEME.brand}]{THEME.icon_fox} sentifox>[/] ", end="")
            else:
                print("sentifox> ", end="")
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            history.save()
            if HAS_RICH:
                console.print("\n[dim]再见！[/]")
            else:
                print("\n再见！")
            break

        # 强力清除 BOM / 零宽字符 / 不可见控制字符
        user_input = user_input.strip()
        for ch in ('\ufeff', '\xef\xbb\xbf', '\u200b', '\u200c', '\u200d', '\x00'):
            user_input = user_input.replace(ch, '')
        user_input = re.sub(r'^[\s\x00-\x1f\x7f\ufeff]+|[\s\x00-\x1f\x7f\ufeff]+$', '', user_input)
        if not user_input:
            continue

        cmd_lower = user_input.lower()
        if cmd_lower in ("exit", "quit", "q"):
            history.save()
            if HAS_RICH:
                console.print("[bold green]再见！[/]")
            else:
                print("再见！")
            break

        if cmd_lower in ("help", "?", "h"):
            print_interactive_help(subparsers)
            continue

        if cmd_lower == "clear":
            if HAS_RICH:
                console.clear()
                print_welcome_with_fox()  # 清屏后保留 banner
            else:
                os.system("cls" if os.name == "nt" else "clear")
            continue
        
        if cmd_lower == "history":
            recent = history.get_recent(20)
            if HAS_RICH:
                console.print(section_header("📜 最近命令", "📜"))
                for i, cmd in enumerate(recent, 1):
                    console.print(f"  [dim]{i:2}.[/] {cmd}")
            else:
                print("\n最近命令:")
                for i, cmd in enumerate(recent, 1):
                    print(f"  {i:2}. {cmd}")
            continue

        # 记录到历史
        history.add(user_input)

        # 解析并执行命令
        try:
            args = parser.parse_args(user_input.split())
            if args.command and args.command in commands:
                commands[args.command](args)
                if HAS_RICH:
                    console.print(section_divider())
            elif not args.command:
                pass  # 空命令，静默忽略
        except CLIExit:
            continue
        except Exception as e:
            if HAS_RICH:
                console.print(f"[bold red][FAIL][/] 执行错误: {e}")
            else:
                print(f"[FAIL] 执行错误: {e}")


def main():
    parser, commands, subparsers = build_parser()

    if len(sys.argv) == 1:
        # 无参数 -> 交互式 REPL
        interactive_mode(parser, commands, subparsers)
    else:
        # 有参数 -> 直接执行单条命令（兼容脚本调用）
        try:
            args = parser.parse_args()
        except CLIExit as e:
            sys.exit(e.args[0] if e.args else 0)
        if not args.command:
            parser.print_help()
            sys.exit(0)
        if args.command in commands:
            commands[args.command](args)
        else:
            print(f"[FAIL] 未知命令: {args.command}")
            sys.exit(1)


if __name__ == "__main__":
    main()
