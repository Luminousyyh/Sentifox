"""
舆情传播图构建模块
从帖子数据和边数据构建 NetworkX 图
"""
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import networkx as nx

from crawlers.base import Post, GraphEdge


def build_graph_from_posts(
    posts: List[Dict[str, Any]],
    edges: Optional[List[Dict[str, Any]]] = None,
    directed: bool = True
) -> nx.DiGraph:
    """
    从帖子列表和边列表构建传播图
    :param posts: 帖子字典列表
    :param edges: 边字典列表（可选，如未提供则只构建孤立节点）
    :param directed: 是否构建有向图
    :return: NetworkX DiGraph
    """
    G = nx.DiGraph() if directed else nx.Graph()

    # 添加节点（以作者为节点）
    author_posts = {}
    for post in posts:
        author_id = post.get("author_id", "")
        author = post.get("author", "")
        if not author_id:
            continue

        if author_id not in author_posts:
            author_posts[author_id] = {
                "author": author,
                "posts": [],
                "platforms": set(),
                "sentiments": [],
                "total_likes": 0,
                "total_comments": 0,
                "total_reposts": 0,
            }

        author_posts[author_id]["posts"].append(post)
        author_posts[author_id]["platforms"].add(post.get("platform", ""))
        author_posts[author_id]["sentiments"].append(post.get("sentiment_label", "neutral"))
        author_posts[author_id]["total_likes"] += post.get("likes", 0)
        author_posts[author_id]["total_comments"] += post.get("comments", 0)
        author_posts[author_id]["total_reposts"] += post.get("reposts", 0)

    # 统计每个作者的主情感
    for author_id, data in author_posts.items():
        sentiments = data["sentiments"]
        pos = sentiments.count("positive")
        neg = sentiments.count("negative")
        neu = sentiments.count("neutral")

        if neg > pos and neg > neu:
            dominant_sentiment = "negative"
        elif pos > neg and pos > neu:
            dominant_sentiment = "positive"
        else:
            dominant_sentiment = "neutral"

        G.add_node(
            author_id,
            label=data["author"],
            platform=",".join(data["platforms"]),
            post_count=len(data["posts"]),
            sentiment=dominant_sentiment,
            likes=data["total_likes"],
            comments=data["total_comments"],
            reposts=data["total_reposts"],
            engagement=data["total_likes"] + data["total_comments"] + data["total_reposts"],
        )

    # 添加边
    if edges:
        for edge in edges:
            src = edge.get("source_id", "")
            tgt = edge.get("target_id", "")
            if src and tgt and src in G.nodes and tgt in G.nodes:
                existing = G.get_edge_data(src, tgt)
                if existing:
                    # 合并多条边，累加权重
                    G[src][tgt]["weight"] += edge.get("weight", 1.0)
                    G[src][tgt]["count"] = G[src][tgt].get("count", 1) + 1
                else:
                    G.add_edge(
                        src, tgt,
                        relation_type=edge.get("relation_type", "unknown"),
                        weight=edge.get("weight", 1.0),
                        timestamp=edge.get("timestamp"),
                        platform=edge.get("platform", ""),
                        count=1,
                    )

    return G


def build_post_graph(
    posts: List[Dict[str, Any]],
    edges: Optional[List[Dict[str, Any]]] = None
) -> nx.DiGraph:
    """
    以帖子为节点构建图（而非以用户为节点）
    适合分析具体内容的传播路径
    """
    G = nx.DiGraph()

    for post in posts:
        pid = post.get("post_id", "")
        if not pid:
            continue
        G.add_node(
            pid,
            author=post.get("author", ""),
            author_id=post.get("author_id", ""),
            content=post.get("content", "")[:100],
            platform=post.get("platform", ""),
            sentiment=post.get("sentiment_label", "neutral"),
            sentiment_score=post.get("sentiment_score", 0.5),
            likes=post.get("likes", 0),
            comments=post.get("comments", 0),
            reposts=post.get("reposts", 0),
            publish_time=post.get("publish_time"),
        )

    if edges:
        for edge in edges:
            # 帖子图中，source/target 也视为作者关系，但需要映射到帖子
            # 简化：使用作者ID匹配帖子
            src_author = edge.get("source_id", "")
            tgt_author = edge.get("target_id", "")

            src_posts = [p.get("post_id") for p in posts if p.get("author_id") == src_author]
            tgt_posts = [p.get("post_id") for p in posts if p.get("author_id") == tgt_author]

            for sp in src_posts[:1]:  # 每个作者取一篇代表
                for tp in tgt_posts[:1]:
                    if sp and tp and sp != tp and G.has_node(sp) and G.has_node(tp):
                        G.add_edge(
                            sp, tp,
                            relation_type=edge.get("relation_type", "unknown"),
                            weight=edge.get("weight", 1.0),
                        )

    return G


def get_graph_stats(G: nx.DiGraph) -> Dict[str, Any]:
    """获取图的基本统计信息"""
    if G.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "density": 0, "components": 0}

    # 将 DiGraph 转为无向图计算连通分量
    UG = G.to_undirected()
    components = list(nx.connected_components(UG))

    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 4),
        "components": len(components),
        "largest_component_size": max(len(c) for c in components) if components else 0,
        "avg_degree": round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2) if G.number_of_nodes() > 0 else 0,
    }

    try:
        stats["avg_clustering"] = round(nx.average_clustering(UG), 4)
    except Exception:
        stats["avg_clustering"] = 0

    return stats


def get_temporal_subgraph(
    G: nx.DiGraph,
    start_time: datetime,
    end_time: datetime
) -> nx.DiGraph:
    """
    按时间范围提取子图
    """
    sub = nx.DiGraph()
    for node, data in G.nodes(data=True):
        pt = data.get("publish_time")
        if pt and start_time <= pt <= end_time:
            sub.add_node(node, **data)

    for u, v, data in G.edges(data=True):
        if u in sub.nodes and v in sub.nodes:
            ts = data.get("timestamp")
            if ts and start_time <= ts <= end_time:
                sub.add_edge(u, v, **data)

    return sub
