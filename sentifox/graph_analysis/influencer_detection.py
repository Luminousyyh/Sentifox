"""
关键传播节点识别模块
基于图中心性算法识别 KOL / 负面情绪放大器
"""
from typing import List, Dict, Any, Optional
import networkx as nx


def compute_centrality(
    G: nx.DiGraph,
    metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    计算多种中心性指标
    :param metrics: ["degree", "in_degree", "out_degree", "betweenness", "closeness", "pagerank", "eigenvector"]
    :return: {node_id: {"pagerank": 0.05, ...}, ...}
    """
    if metrics is None:
        metrics = ["degree", "in_degree", "out_degree", "betweenness", "pagerank"]

    results = {}
    for node in G.nodes():
        results[node] = {}

    if "degree" in metrics:
        deg = dict(G.degree())
        for n, v in deg.items():
            results[n]["degree"] = v

    if "in_degree" in metrics:
        in_deg = dict(G.in_degree())
        for n, v in in_deg.items():
            results[n]["in_degree"] = v

    if "out_degree" in metrics:
        out_deg = dict(G.out_degree())
        for n, v in out_deg.items():
            results[n]["out_degree"] = v

    if "betweenness" in metrics and G.number_of_nodes() > 1:
        try:
            btw = nx.betweenness_centrality(G, weight="weight")
            for n, v in btw.items():
                results[n]["betweenness"] = round(v, 6)
        except Exception:
            pass

    if "closeness" in metrics and G.number_of_nodes() > 1:
        try:
            # 在有向图上计算 closeness 需要强连通，转无向图
            UG = G.to_undirected()
            cls = nx.closeness_centrality(UG)
            for n, v in cls.items():
                results[n]["closeness"] = round(v, 6)
        except Exception:
            pass

    if "pagerank" in metrics:
        try:
            pr = nx.pagerank(G, weight="weight")
            for n, v in pr.items():
                results[n]["pagerank"] = round(v, 6)
        except Exception:
            pass

    if "eigenvector" in metrics and G.number_of_nodes() > 1:
        try:
            UG = G.to_undirected()
            ev = nx.eigenvector_centrality(UG, max_iter=1000, weight="weight")
            for n, v in ev.items():
                results[n]["eigenvector"] = round(v, 6)
        except Exception:
            pass

    return results


def detect_influencers(
    G: nx.DiGraph,
    top_n: int = 10,
    metric: str = "pagerank",
    min_posts: int = 1
) -> List[Dict[str, Any]]:
    """
    识别关键传播节点（KOL）
    :param metric: 排序依据的中心性指标
    :return: 排序后的节点列表，含详细属性
    """
    centrality = compute_centrality(G, metrics=[metric, "degree", "betweenness"])

    candidates = []
    for node_id, scores in centrality.items():
        data = G.nodes[node_id]
        if data.get("post_count", 0) < min_posts:
            continue

        candidates.append({
            "node_id": node_id,
            "author": data.get("label", ""),
            "platform": data.get("platform", ""),
            "post_count": data.get("post_count", 0),
            "engagement": data.get("engagement", 0),
            "sentiment": data.get("sentiment", "neutral"),
            "pagerank": scores.get("pagerank", 0),
            "betweenness": scores.get("betweenness", 0),
            "degree": scores.get("degree", 0),
            "score": scores.get(metric, 0),
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]


def detect_negative_amplifiers(
    G: nx.DiGraph,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    识别负面情绪放大器
    综合考虑：负面情感 + 高中心性 + 高互动量
    """
    centrality = compute_centrality(G, metrics=["pagerank", "betweenness", "out_degree"])

    amplifiers = []
    for node_id, scores in centrality.items():
        data = G.nodes[node_id]
        if data.get("sentiment") != "negative":
            continue

        # 综合得分：PageRank * 出度 * 互动量
        composite_score = (
            scores.get("pagerank", 0) *
            (1 + scores.get("out_degree", 0)) *
            (1 + data.get("engagement", 0) / 100)
        )

        amplifiers.append({
            "node_id": node_id,
            "author": data.get("label", ""),
            "platform": data.get("platform", ""),
            "post_count": data.get("post_count", 0),
            "engagement": data.get("engagement", 0),
            "pagerank": scores.get("pagerank", 0),
            "out_degree": scores.get("out_degree", 0),
            "composite_score": round(composite_score, 4),
        })

    amplifiers.sort(key=lambda x: x["composite_score"], reverse=True)
    return amplifiers[:top_n]


def detect_communities(G: nx.DiGraph) -> Dict[int, List[str]]:
    """
    社区发现（Louvain 算法）
    :return: {community_id: [node_ids], ...}
    """
    try:
        import community as community_louvain
        UG = G.to_undirected()
        partition = community_louvain.best_partition(UG, weight="weight")

        communities = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(node)
        return communities
    except ImportError:
        # fallback: 简单的连通分量
        UG = G.to_undirected()
        communities = {}
        for i, comp in enumerate(nx.connected_components(UG)):
            communities[i] = list(comp)
        return communities


def get_bridge_nodes(G: nx.DiGraph, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    识别桥接节点（连接不同社区的关键节点）
    """
    communities = detect_communities(G)
    if len(communities) <= 1:
        return []

    # 构建社区归属映射
    node_community = {}
    for cid, nodes in communities.items():
        for n in nodes:
            node_community[n] = cid

    bridges = []
    centrality = compute_centrality(G, metrics=["betweenness"])

    for node_id in G.nodes():
        # 统计该节点连接的异社区邻居
        cross_community_edges = 0
        my_comm = node_community.get(node_id, -1)
        for neighbor in G.successors(node_id):
            if node_community.get(neighbor, -1) != my_comm:
                cross_community_edges += 1

        if cross_community_edges > 0:
            data = G.nodes[node_id]
            bridges.append({
                "node_id": node_id,
                "author": data.get("label", ""),
                "cross_edges": cross_community_edges,
                "betweenness": centrality.get(node_id, {}).get("betweenness", 0),
                "platform": data.get("platform", ""),
            })

    bridges.sort(key=lambda x: (x["cross_edges"], x["betweenness"]), reverse=True)
    return bridges[:top_n]
