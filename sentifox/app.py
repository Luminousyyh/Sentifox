"""
舆情分析系统 - Streamlit 主入口
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from config import CONFIG, PLATFORMS
from utils.database import (
    init_db, insert_posts, get_posts, get_post_count,
    get_sentiment_distribution, get_platform_distribution, get_trend_by_time,
    get_alerts, get_recent_posts_count, insert_edges, update_post_topic, get_edges,
    insert_alert
)
from crawlers.mock_generator import MockCrawler, generate_mock_edges
from crawlers.manager import CrawlerManager
from analysis.sentiment import get_analyzer
from analysis.topic_clustering import cluster_posts, extract_keywords, TopicClusterer
from analysis.trend_analysis import calculate_sentiment_index, detect_anomaly, get_hot_topics_evolution
from analysis.insight_generator import generate_insight
from visualization.charts import (
    create_pie_chart, create_bar_chart, create_line_chart,
    create_stacked_area_chart, generate_wordcloud, create_topic_scatter
)
from graph_analysis.graph_builder import build_graph_from_posts, get_graph_stats
from graph_analysis.influencer_detection import detect_influencers, detect_negative_amplifiers
from graph_analysis.sentifox_simulator import SentifoxSimulator
from graph_analysis.graph_viz import render_network, render_dynamic_graph
from rag.rag_engine import ask
from rag.document_processor import sync_posts_to_vector_store

st.set_page_config(
    page_title=CONFIG.page_title,
    page_icon=CONFIG.page_icon,
    layout=CONFIG.layout,
)

# ========== 初始化 ==========
init_db()

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ========== 侧边栏 ==========
with st.sidebar:
    st.title("📊 舆情分析系统")
    st.markdown("---")

    keywords_input = st.text_input(
        "监控关键词",
        value=", ".join(CONFIG.crawler.keywords),
        help="多个关键词用逗号分隔"
    )
    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

    time_range = st.selectbox(
        "时间范围",
        options=["最近24小时", "最近3天", "最近7天", "最近30天", "全部"],
        index=2,
    )

    selected_platforms = st.multiselect(
        "选择平台",
        options=PLATFORMS,
        default=PLATFORMS,
    )

    st.markdown("---")
    st.subheader("数据管理")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 生成模拟数据", use_container_width=True):
            with st.spinner("正在生成并分析模拟数据..."):
                crawler = MockCrawler(keywords=keywords)
                posts = crawler.crawl(max_posts=CONFIG.crawler.max_posts_per_platform)

                analyzer = get_analyzer()
                post_dicts = [p.to_dict() for p in posts]
                analyzed = analyzer.analyze_posts(post_dicts)

                clustered, topics = cluster_posts(analyzed, n_clusters=5)

                inserted_posts = insert_posts(clustered)
                for p in clustered:
                    if p.get("topic_id") is not None:
                        update_post_topic(p["post_id"], p["topic_id"])

                edges = generate_mock_edges(posts)
                inserted_edges = insert_edges([e.to_dict() for e in edges])

                # 同步向量库
                try:
                    sync_posts_to_vector_store(clustered)
                except Exception as e:
                    st.warning(f"向量库同步跳过: {e}")

                st.session_state.data_loaded = True
                st.success(f"已生成 {inserted_posts} 条帖子, {inserted_edges} 条关系, {len(topics)} 个话题")
                st.rerun()

    with col2:
        if st.button("🗑️ 清空数据", use_container_width=True):
            import os
            from config import DB_PATH
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
            init_db()
            st.session_state.data_loaded = False
            st.success("数据已清空")
            st.rerun()

    # 真实采集面板
    st.markdown("---")
    st.subheader("🔌 真实采集")
    with st.expander("配置与运行"):
        manager = CrawlerManager(keywords=keywords)
        status = manager.get_platform_status()

        for plat, info in status.items():
            icon = "🟢" if info["available"] else "🔴"
            st.caption(f"{icon} {plat}: {info['reason']}")

        crawl_platforms = st.multiselect(
            "选择采集平台",
            options=[p for p in PLATFORMS if status.get(p, {}).get("available", False) or True],
            default=[p for p in PLATFORMS if status.get(p, {}).get("available", False)],
            key="real_crawl_platforms",
        )

        if st.button("🚀 开始真实采集", use_container_width=True):
            with st.spinner("正在采集真实数据..."):
                try:
                    results = manager.crawl_and_process(
                        platforms=crawl_platforms,
                        max_posts_per_platform=30,
                        use_mock_fallback=True,
                    )
                    st.success(f"采集完成，共 {len(results)} 条数据已入库")
                    st.session_state.data_loaded = True
                    st.rerun()
                except Exception as e:
                    st.error(f"采集失败: {e}")

    # RAG 同步
    st.markdown("---")
    st.subheader("🧠 向量库")
    if st.button("🔄 同步到向量库", use_container_width=True):
        with st.spinner("正在同步..."):
            try:
                from utils.database import get_posts
                posts = get_posts(limit=5000)
                count = sync_posts_to_vector_store(posts)
                st.success(f"已同步 {count} 条到向量库")
            except Exception as e:
                st.error(f"同步失败: {e}")

    st.markdown("---")
    st.caption("v2.0.0 | 舆情分析系统 + LLM + RAG")


# ========== 主内容区 ==========
st.title("📈 舆情监控仪表盘")


def get_time_filter():
    now = datetime.now()
    if time_range == "最近24小时":
        return now - timedelta(hours=24), now
    elif time_range == "最近3天":
        return now - timedelta(days=3), now
    elif time_range == "最近7天":
        return now - timedelta(days=7), now
    elif time_range == "最近30天":
        return now - timedelta(days=30), now
    else:
        return None, None


start_time, end_time = get_time_filter()

# KPI 卡片
total_posts = get_post_count(platforms=selected_platforms, start_time=start_time, end_time=end_time)
sentiment_dist = get_sentiment_distribution(platforms=selected_platforms, start_time=start_time, end_time=end_time)
platform_dist = get_platform_distribution(start_time=start_time, end_time=end_time)

negative_count = sentiment_dist.get("negative", 0)
positive_count = sentiment_dist.get("positive", 0)
neutral_count = sentiment_dist.get("neutral", 0)
neg_rate = negative_count / total_posts * 100 if total_posts > 0 else 0
recent_count = get_recent_posts_count(minutes=30)

if neg_rate > CONFIG.alert["negative_ratio_threshold"] * 100:
    st.error(f"⚠️ 负面情感占比过高: {neg_rate:.1f}%")
    try:
        insert_alert(
            alert_type="negative_spike",
            severity="high",
            message=f"负面情感占比达到 {neg_rate:.1f}%",
            details=f"平台: {', '.join(selected_platforms)}"
        )
    except Exception:
        pass

kpi_cols = st.columns(5)
kpi_data = [
    ("📄 总帖子数", f"{total_posts:,}", None),
    ("😊 正面", f"{positive_count:,}", None),
    ("😐 中性", f"{neutral_count:,}", None),
    ("😠 负面", f"{negative_count:,}", f"{neg_rate:.1f}%"),
    ("🕐 30min新增", f"{recent_count:,}", None),
]

for col, (title, value, delta) in zip(kpi_cols, kpi_data):
    with col:
        if title == "😠 负面":
            st.metric(title, value, delta=delta, delta_color="inverse")
        else:
            st.metric(title, value)

st.markdown("---")

if total_posts == 0:
    st.info("👈 请在侧边栏点击「生成模拟数据」或「开始真实采集」开始分析")
    st.stop()

# ==================== Tab 布局 ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 概览", "🔥 话题分析", "🕸️ 传播图谱", "💬 智能问答", "🚨 告警与报告"])

# ========== Tab 1: 概览 ==========
with tab1:
    # LLM 洞察卡片
    st.subheader("💡 AI 舆情洞察")
    if st.button("✨ 生成洞察", key="gen_insight"):
        with st.spinner("AI 分析中..."):
            try:
                all_posts = get_posts(platforms=selected_platforms, start_time=start_time, end_time=end_time, limit=500)
                graph_posts = get_posts(platforms=selected_platforms, start_time=start_time, end_time=end_time, limit=1000)
                graph_edges = get_edges(start_time=start_time, end_time=end_time)
                G = build_graph_from_posts(graph_posts, graph_edges, directed=True)
                influencers = detect_influencers(G, top_n=5)
                insight = generate_insight(all_posts, sentiment_dist, platform_dist, influencers)
                st.session_state.insight_text = insight
            except Exception as e:
                st.error(f"洞察生成失败: {e}")

    if "insight_text" in st.session_state:
        st.info(st.session_state.insight_text)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("情感分布")
        fig = create_pie_chart(sentiment_dist, title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("平台来源分布")
        filtered_platform = {k: v for k, v in platform_dist.items() if k in selected_platforms}
        fig = create_bar_chart(filtered_platform, title="", orientation="v")
        st.plotly_chart(fig, use_container_width=True)

    # 情感趋势
    st.subheader("情感趋势")
    interval = "hour" if time_range in ["最近24小时", "最近3天"] else "day"
    trend_data = get_trend_by_time(
        interval=interval,
        platforms=selected_platforms,
        start_time=start_time,
        end_time=end_time,
    )
    if trend_data:
        fig_area = create_stacked_area_chart(trend_data, title="")
        st.plotly_chart(fig_area, use_container_width=True)

        st.subheader("情感指数趋势")
        sentiment_index = calculate_sentiment_index(trend_data)
        if sentiment_index:
            fig_idx = create_line_chart(
                sentiment_index,
                x_key="time_bucket",
                y_keys=["sentiment_index"],
                title=""
            )
            fig_idx.update_traces(line=dict(color="#3498db", width=3))
            fig_idx.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_idx, use_container_width=True)

            anomalies = detect_anomaly(sentiment_index, method="zscore", threshold=2.0)
            if anomalies:
                with st.expander(f"⚠️ 检测到 {len(anomalies)} 个异常点"):
                    for a in anomalies:
                        st.write(f"- **{a['time_bucket']}**: 情感指数 {a['sentiment_index']}, Z-Score: {a.get('zscore', 'N/A')}")
    else:
        st.info("暂无趋势数据")

    # 词云
    st.subheader("关键词云")
    posts = get_posts(
        platforms=selected_platforms,
        start_time=start_time,
        end_time=end_time,
        limit=1000,
    )
    if posts:
        texts = [p.get("content", "") for p in posts]
        keywords_list = extract_keywords(texts, top_n=50)
        if keywords_list:
            word_freq = {w: s for w, s in keywords_list}
            img_b64 = generate_wordcloud(word_freq, width=900, height=350)
            st.markdown(f"<img src='{img_b64}' style='width:100%;max-height:350px;object-fit:contain;'/>", unsafe_allow_html=True)
        else:
            st.info("暂无足够数据生成词云")
    else:
        st.info("暂无数据")

    # 最新帖子
    st.subheader("📋 最新帖子")
    posts = get_posts(
        platforms=selected_platforms,
        start_time=start_time,
        end_time=end_time,
        limit=30,
    )
    if posts:
        df = pd.DataFrame(posts)
        display_cols = ["platform", "author", "content", "sentiment_label", "sentiment_score", "topic_id", "publish_time", "likes", "comments", "reposts"]
        df = df[[c for c in display_cols if c in df.columns]]

        def sentiment_badge(label):
            badges = {"positive": "🟢 正面", "negative": "🔴 负面", "neutral": "⚪ 中性"}
            return badges.get(label, label)

        if "sentiment_label" in df.columns:
            df["sentiment_label"] = df["sentiment_label"].apply(sentiment_badge)

        st.dataframe(df, use_container_width=True, hide_index=True)

# ========== Tab 2: 话题分析 ==========
with tab2:
    st.subheader("🔥 热门话题")

    all_posts = get_posts(
        platforms=selected_platforms,
        start_time=start_time,
        end_time=end_time,
        limit=2000,
    )

    if len(all_posts) < 10:
        st.info("数据量不足，请先生成更多模拟数据")
    else:
        texts = [p.get("content", "") for p in all_posts]
        clusterer = TopicClusterer(n_clusters=min(5, len(texts) // 5))
        topics = clusterer.fit(texts)

        fig = create_topic_scatter(topics, title="")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("话题详情")
        topic_cols = st.columns(len(topics))
        for col, (tid, info) in zip(topic_cols, topics.items()):
            with col:
                st.metric(f"话题 {tid}", f"{info['size']} 帖")
                st.caption(f"关键词: {', '.join(info['keywords'][:3])}")

        st.subheader("话题情感分布")
        topic_sentiment = {}
        labels = clusterer.predict(texts)
        for post, label in zip(all_posts, labels):
            key = f"话题 {label}"
            if key not in topic_sentiment:
                topic_sentiment[key] = {"positive": 0, "negative": 0, "neutral": 0}
            sl = post.get("sentiment_label", "neutral")
            topic_sentiment[key][sl] = topic_sentiment[key].get(sl, 0) + 1

        topic_sent_df = []
        for topic, dist in topic_sentiment.items():
            for sent, cnt in dist.items():
                topic_sent_df.append({"topic": topic, "sentiment": sent, "count": cnt})

        if topic_sent_df:
            df_ts = pd.DataFrame(topic_sent_df)
            fig = px.bar(df_ts, x="topic", y="count", color="sentiment",
                         color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
                         barmode="group")
            fig.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

# ========== Tab 3: 传播图谱 ==========
with tab3:
    st.subheader("🕸️ 舆情传播图谱")

    graph_posts = get_posts(
        platforms=selected_platforms,
        start_time=start_time,
        end_time=end_time,
        limit=2000,
    )
    graph_edges = get_edges(start_time=start_time, end_time=end_time)

    if len(graph_posts) < 5:
        st.info("数据量不足，无法构建传播图谱")
    else:
        G = build_graph_from_posts(graph_posts, graph_edges, directed=True)
        stats = get_graph_stats(G)

        stat_cols = st.columns(4)
        stat_data = [
            ("节点数", stats["nodes"]),
            ("边数", stats["edges"]),
            ("网络密度", stats["density"]),
            ("连通分量", stats["components"]),
        ]
        for col, (label, val) in zip(stat_cols, stat_data):
            with col:
                st.metric(label, val)

        st.subheader("交互式传播网络")
        color_mode = st.radio("节点着色", ["情感极性", "平台类型"], horizontal=True, key="graph_color")
        color_attr = "sentiment" if color_mode == "情感极性" else "platform"
        render_network(G, height="600px", color_attr=color_attr, key="main_network")

        st.markdown("---")
        st.subheader("⭐ 关键传播节点 (KOL)")
        influencers = detect_influencers(G, top_n=10, metric="pagerank")
        if influencers:
            df_inf = pd.DataFrame(influencers)
            display_cols = ["author", "platform", "post_count", "engagement", "pagerank", "betweenness"]
            df_inf = df_inf[[c for c in display_cols if c in df_inf.columns]]
            st.dataframe(df_inf, use_container_width=True, hide_index=True)
        else:
            st.info("未识别到关键节点")

        st.subheader("🔴 负面情绪放大器")
        amplifiers = detect_negative_amplifiers(G, top_n=10)
        if amplifiers:
            df_amp = pd.DataFrame(amplifiers)
            display_cols = ["author", "platform", "post_count", "engagement", "composite_score"]
            df_amp = df_amp[[c for c in display_cols if c in df_amp.columns]]
            st.dataframe(df_amp, use_container_width=True, hide_index=True)
        else:
            st.info("未识别到负面情绪放大器")

        st.markdown("---")
        st.subheader("🔬 Sentifox 多智能体传播仿真")

        sim_col1, sim_col2 = st.columns([1, 3])
        with sim_col1:
            sim_steps = st.slider("仿真步数", 8, 72, 24, key="sentifox_steps")
            use_cognitive = st.toggle("启用认知层 (BDI+情绪)", value=True, key="sentifox_cognitive")
            use_communication = st.toggle("启用通信层 (讨论+回音室)", value=True, key="sentifox_comm")
            run_sim = st.button("▶ 运行仿真", use_container_width=True)

            if "sentifox_result" in st.session_state:
                if st.button("📥 下载仿真报告", use_container_width=True):
                    from graph_analysis.simulation_report import generate_simulation_report
                    try:
                        result = st.session_state.sentifox_result
                        simulator = st.session_state.sentifox_simulator
                        cognitive_data = {
                            "emotional_states": {},
                            "bdi_chains": [],
                            "bridge_nodes": [],
                        }
                        if hasattr(simulator, "cognitive_agents"):
                            for aid, cog in list(simulator.cognitive_agents.items())[:20]:
                                cognitive_data["emotional_states"][aid] = cog.emotion.to_dict()
                        report_path = generate_simulation_report(
                            result, result.timeline, cognitive_data=cognitive_data,
                            output_path=f"sentifox_simulation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        )
                        with open(report_path, "rb") as f:
                            st.download_button(
                                label="📄 点击下载 HTML 报告",
                                data=f,
                                file_name=os.path.basename(report_path),
                                mime="text/html",
                                use_container_width=True,
                            )
                    except Exception as e:
                        st.error(f"报告生成失败: {e}")

        with sim_col2:
            if run_sim:
                with st.spinner("构建 Sentifox 仿真环境并运行..."):
                    try:
                        simulator = SentifoxSimulator.from_data(
                            graph_posts, graph_edges,
                            config={
                                "total_steps": sim_steps,
                                "use_cognitive": use_cognitive,
                                "use_communication": use_communication,
                            }
                        )
                        result = simulator.run(steps=sim_steps)
                        st.session_state.sentifox_result = result
                        st.session_state.sentifox_simulator = simulator
                        st.success(f"仿真完成: {len(result.timeline)} 步, {len(result.events)} 个事件")
                    except Exception as e:
                        st.error(f"仿真失败: {e}")
                        st.session_state.sentifox_result = None
                        st.session_state.sentifox_simulator = None

            if "sentifox_result" in st.session_state and st.session_state.sentifox_result:
                result = st.session_state.sentifox_result
                timeline = result.timeline

                # SIR 曲线
                sim_df = pd.DataFrame([
                    {
                        "step": s.step,
                        "susceptible": s.susceptible_count,
                        "infected": s.infected_count,
                        "recovered": s.recovered_count,
                    }
                    for s in timeline
                ])
                if not sim_df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=sim_df["step"], y=sim_df["susceptible"], mode="lines", name="未感染 S", line=dict(color="#95a5a6")))
                    fig.add_trace(go.Scatter(x=sim_df["step"], y=sim_df["infected"], mode="lines", name="传播中 I", line=dict(color="#e74c3c")))
                    fig.add_trace(go.Scatter(x=sim_df["step"], y=sim_df["recovered"], mode="lines", name="已衰减 R", line=dict(color="#2ecc71")))
                    fig.update_layout(
                        title="SIR 传播曲线",
                        xaxis_title="时间步 (小时)",
                        yaxis_title="节点数",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(t=50, b=10, l=10, r=10),
                        height=320,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Zep 风格动态认知图谱
                st.subheader("🎬 动态认知图谱 (Zep 风格)")
                simulator = st.session_state.sentifox_simulator
                render_dynamic_graph(result, simulator, height=650, key="sentifox_dynamic_graph")
            else:
                st.info("点击「运行仿真」启动 Sentifox 多智能体传播仿真")

# ========== Tab 4: 智能问答 (RAG) ==========
with tab4:
    st.subheader("💬 智能问答")
    st.caption("基于舆情数据的 RAG 问答，支持自然语言查询")

    # 示例问题
    examples = [
        "最近负面舆情主要集中在哪些方面？",
        "哪些 KOL 在传播正面内容？",
        "关于产品质量的讨论趋势如何？",
        "各平台的情感分布有什么差异？",
    ]

    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        with col:
            if st.button(ex, use_container_width=True, key=f"ex_{hash(ex) & 0xFFFFFF}"):
                st.session_state.rag_query = ex

    query = st.text_input(
        "输入你的问题",
        value=st.session_state.get("rag_query", ""),
        placeholder="例如：最近用户对产品有哪些负面反馈？",
        key="rag_input",
    )

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        rag_platform = st.selectbox("平台过滤", ["全部"] + PLATFORMS, key="rag_platform")
    with filter_col2:
        rag_sentiment = st.selectbox("情感过滤", ["全部", "positive", "negative", "neutral"], key="rag_sentiment")

    if st.button("🔍 提问", use_container_width=True, key="rag_ask"):
        if not query.strip():
            st.warning("请输入问题")
        else:
            with st.spinner("检索中..."):
                try:
                    platform_filter = rag_platform if rag_platform != "全部" else None
                    sentiment_filter = rag_sentiment if rag_sentiment != "全部" else None

                    result = ask(
                        query,
                        top_k=5,
                        platform_filter=platform_filter,
                        sentiment_filter=sentiment_filter,
                    )

                    # 添加到历史
                    st.session_state.chat_history.append({
                        "query": query,
                        "answer": result["answer"],
                        "sources": result["sources"],
                    })

                except Exception as e:
                    st.error(f"问答失败: {e}")

    # 显示对话历史
    if st.session_state.chat_history:
        st.markdown("---")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):
            with st.chat_message("user"):
                st.write(chat["query"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])
                if chat.get("sources"):
                    with st.expander("📎 参考来源"):
                        for src in chat["sources"]:
                            st.caption(f"[{src['index']}] {src['platform']} | {src['author']} | 相关度: {src['relevance']}")

    if st.button("🗑️ 清空对话"):
        st.session_state.chat_history = []
        st.rerun()

# ========== Tab 5: 告警与报告 ==========
with tab5:
    st.subheader("🚨 最近告警")
    alerts = get_alerts(limit=20, unresolved_only=False)
    if alerts:
        alert_df = pd.DataFrame(alerts)
        display_cols = ["alert_type", "severity", "message", "triggered_at", "is_resolved"]
        alert_df = alert_df[[c for c in display_cols if c in alert_df.columns]]

        def severity_badge(s):
            colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            return f"{colors.get(s, '⚪')} {s}"

        if "severity" in alert_df.columns:
            alert_df["severity"] = alert_df["severity"].apply(severity_badge)
        st.dataframe(alert_df, use_container_width=True, hide_index=True)
    else:
        st.info("暂无告警")

    st.markdown("---")
    st.subheader("📄 报告导出")

    report_col1, report_col2 = st.columns([1, 2])
    with report_col1:
        if st.button("📥 生成报告", use_container_width=True):
            with st.spinner("正在生成报告..."):
                from reports.generator import generate_report, get_report_list
                try:
                    filepath = generate_report(
                        title=f"舆情分析报告 - {datetime.now().strftime('%Y-%m-%d')}",
                        platforms=selected_platforms,
                        start_time=start_time,
                        end_time=end_time,
                        include_graph=True,
                    )
                    st.success(f"报告已生成: {os.path.basename(filepath)}")
                    st.session_state.report_ready = filepath
                except Exception as e:
                    st.error(f"生成失败: {e}")

    with report_col2:
        from reports.generator import get_report_list
        reports = get_report_list()
        if reports:
            st.write("**已生成的报告:**")
            for r in reports[:5]:
                with open(r["path"], "rb") as f:
                    st.download_button(
                        label=f"📄 {r['filename']} ({r['created']})",
                        data=f,
                        file_name=r["filename"],
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                        key=f"dl_{r['filename']}",
                    )
        else:
            st.info("暂无报告")
