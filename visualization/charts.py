"""
图表组件封装
使用 Plotly 和 WordCloud
"""
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64


def create_pie_chart(
    data: Dict[str, int],
    title: str = "",
    color_map: Optional[Dict[str, str]] = None
) -> go.Figure:
    """饼图/环形图"""
    df = pd.DataFrame([{"name": k, "value": v} for k, v in data.items() if v > 0])
    if df.empty:
        return go.Figure()

    default_colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
    colors = color_map or default_colors

    fig = px.pie(df, values="value", names="name", color="name",
                 color_discrete_map=colors, hole=0.4, title=title)
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(showlegend=False, margin=dict(t=30, b=10, l=10, r=10))
    return fig


def create_bar_chart(
    data: Dict[str, int],
    title: str = "",
    orientation: str = "v"
) -> go.Figure:
    """柱状图"""
    df = pd.DataFrame([{"name": k, "value": v} for k, v in data.items()])
    if df.empty:
        return go.Figure()

    if orientation == "h":
        fig = px.bar(df, y="name", x="value", color="name", text="value", orientation="h", title=title)
    else:
        fig = px.bar(df, x="name", y="value", color="name", text="value", title=title)

    fig.update_layout(showlegend=False, margin=dict(t=30, b=10, l=10, r=10))
    return fig


def create_line_chart(
    data: List[Dict[str, Any]],
    x_key: str = "time_bucket",
    y_keys: List[str] = None,
    title: str = ""
) -> go.Figure:
    """多系列折线图"""
    if not data:
        return go.Figure()

    df = pd.DataFrame(data)
    if x_key not in df.columns:
        return go.Figure()

    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    y_keys = y_keys or [c for c in df.columns if c != x_key]
    for i, col in enumerate(y_keys):
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df[x_key],
            y=df[col],
            mode="lines+markers",
            name=col,
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="时间",
        yaxis_title="数值",
        margin=dict(t=50, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_stacked_area_chart(
    trend_data: List[Dict[str, Any]],
    title: str = "情感趋势堆叠图"
) -> go.Figure:
    """堆叠面积图"""
    if not trend_data:
        return go.Figure()

    # 数据透视
    df = pd.DataFrame(trend_data)
    pivot = df.pivot(index="time_bucket", columns="sentiment_label", values="cnt").fillna(0)

    fig = go.Figure()
    colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}

    for col in pivot.columns:
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot[col],
            mode="lines",
            stackgroup="one",
            name=col,
            line=dict(color=colors.get(col, "#3498db")),
            fillcolor=colors.get(col, "#3498db"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="时间",
        yaxis_title="帖子数",
        margin=dict(t=50, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def generate_wordcloud(
    word_freq: Dict[str, float],
    width: int = 800,
    height: int = 400,
    background_color: str = "white"
) -> str:
    """
    生成词云图片并返回 base64 编码
    """
    if not word_freq:
        return ""

    wc = WordCloud(
        font_path=None,  # 使用默认字体
        width=width,
        height=height,
        background_color=background_color,
        max_words=100,
        relative_scaling=0.5,
        colormap="viridis",
    ).generate_from_frequencies(word_freq)

    img = wc.to_image()
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def create_heatmap(
    data: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    value_key: str,
    title: str = ""
) -> go.Figure:
    """热力图"""
    if not data:
        return go.Figure()

    df = pd.DataFrame(data)
    pivot = df.pivot(index=y_key, columns=x_key, values=value_key).fillna(0)

    fig = px.imshow(
        pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        color_continuous_scale="RdYlGn",
        title=title,
    )
    fig.update_layout(margin=dict(t=50, b=10, l=10, r=10))
    return fig


def create_topic_scatter(
    topics: Dict[int, Dict],
    title: str = "话题分布"
) -> go.Figure:
    """话题气泡图（大小=帖子数）"""
    if not topics:
        return go.Figure()

    df_data = []
    for tid, info in topics.items():
        keywords = info.get("keywords", [])
        size = info.get("size", 0)
        df_data.append({
            "topic_id": f"话题 {tid}",
            "keywords": " ".join(keywords[:3]),
            "size": size,
            "x": np.random.random(),  # 简单随机布局，可用 t-SNE 替代
            "y": np.random.random(),
        })

    df = pd.DataFrame(df_data)
    fig = px.scatter(
        df,
        x="x",
        y="y",
        size="size",
        color="topic_id",
        hover_data=["keywords", "size"],
        text="topic_id",
        title=title,
        size_max=60,
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        showlegend=False,
        xaxis_visible=False,
        yaxis_visible=False,
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig
