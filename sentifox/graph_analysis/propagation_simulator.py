"""
舆情传播模拟模块
支持 SIR 模型、Agent-based 模拟、LLM API 驱动接口
"""
import random
import copy
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import networkx as nx
import numpy as np


# ========== SIR 传播模型 ==========

@dataclass
class SIRState:
    """SIR 模型状态"""
    susceptible: set      # 未接触
    infected: set         # 已传播（感染）
    recovered: set        # 已衰减（恢复/遗忘）
    timeline: List[Dict]  # 时间线记录


def simulate_sir(
    G: nx.DiGraph,
    seed_nodes: List[str],
    beta: float = 0.3,       # 感染率
    gamma: float = 0.1,      # 恢复率
    steps: int = 50,
    influence_attr: str = "engagement",  # 节点影响力属性
) -> SIRState:
    """
    在图上运行 SIR 传播模拟
    :param G: 传播网络（有向图，边方向表示信息流向）
    :param seed_nodes: 初始种子节点
    :param beta: 感染概率（每条边每次尝试）
    :param gamma: 恢复概率
    :param steps: 模拟步数
    :return: SIRState 包含完整时间线
    """
    all_nodes = set(G.nodes())
    state = SIRState(
        susceptible=all_nodes - set(seed_nodes),
        infected=set(seed_nodes),
        recovered=set(),
        timeline=[],
    )

    for step in range(steps):
        new_infected = set()
        new_recovered = set()

        # 恢复过程
        for node in list(state.infected):
            if random.random() < gamma:
                new_recovered.add(node)

        # 感染过程
        for node in list(state.infected):
            # 获取出边邻居
            neighbors = list(G.successors(node))
            for neighbor in neighbors:
                if neighbor in state.susceptible:
                    # 根据节点影响力调整感染率
                    node_weight = G.nodes[node].get(influence_attr, 1)
                    edge_weight = G[node][neighbor].get("weight", 1.0)
                    effective_beta = min(1.0, beta * (1 + np.log1p(node_weight) / 10) * edge_weight)

                    if random.random() < effective_beta:
                        new_infected.add(neighbor)

        # 更新状态
        state.infected -= new_recovered
        state.infected |= new_infected
        state.susceptible -= new_infected
        state.recovered |= new_recovered

        state.timeline.append({
            "step": step,
            "susceptible": len(state.susceptible),
            "infected": len(state.infected),
            "recovered": len(state.recovered),
            "new_infected": len(new_infected),
            "new_recovered": len(new_recovered),
        })

        # 提前终止
        if len(state.infected) == 0:
            break

    return state


# ========== Agent-based 传播模拟 ==========

@dataclass
class AgentNode:
    """传播 Agent"""
    node_id: str
    influence: float = 1.0          # 影响力基数
    receptivity: float = 0.5        # 信息接受度
    platform_boost: float = 1.0     # 平台放大系数
    sentiment_bias: float = 0.0     # 情感偏向 (-1=负面偏好, 0=中性, 1=正面偏好)
    fatigue: float = 0.0            # 传播疲劳度
    state: str = "S"                # S/I/R


def simulate_agent_based(
    G: nx.DiGraph,
    seed_nodes: List[str],
    steps: int = 50,
    base_prob: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    基于 Agent 的传播模拟
    每个节点作为独立 Agent，根据自身属性和邻居属性决策是否传播
    """
    agents = {}
    for node_id in G.nodes():
        data = G.nodes[node_id]
        platform = data.get("platform", "")
        boost = {"微博": 1.5, "抖音": 1.8, "小红书": 1.3, "知乎": 1.0, "新闻": 0.8, "论坛": 0.9}.get(platform, 1.0)

        sentiment = data.get("sentiment", "neutral")
        bias = {"positive": 0.3, "negative": 0.5, "neutral": 0.0}.get(sentiment, 0.0)

        agents[node_id] = AgentNode(
            node_id=node_id,
            influence=data.get("engagement", 10) / 100 + 0.5,
            receptivity=random.uniform(0.3, 0.8),
            platform_boost=boost,
            sentiment_bias=bias,
            state="I" if node_id in seed_nodes else "S",
        )

    timeline = []
    for step in range(steps):
        new_infections = []
        new_recoveries = []

        for node_id, agent in agents.items():
            if agent.state == "I":
                # 疲劳累积
                agent.fatigue += 0.05
                if random.random() < agent.fatigue:
                    agent.state = "R"
                    new_recoveries.append(node_id)
                    continue

                # 尝试感染邻居
                for neighbor in G.successors(node_id):
                    if agents[neighbor].state != "S":
                        continue

                    neighbor_agent = agents[neighbor]
                    # 计算传播概率
                    prob = base_prob * agent.influence * agent.platform_boost
                    prob *= neighbor_agent.receptivity
                    prob *= (1 + agent.sentiment_bias)  # 情感内容更易传播
                    prob *= G[node_id][neighbor].get("weight", 1.0)
                    prob = min(1.0, prob)

                    if random.random() < prob:
                        neighbor_agent.state = "I"
                        new_infections.append(neighbor)

        timeline.append({
            "step": step,
            "susceptible": sum(1 for a in agents.values() if a.state == "S"),
            "infected": sum(1 for a in agents.values() if a.state == "I"),
            "recovered": sum(1 for a in agents.values() if a.state == "R"),
            "new_infected": len(new_infections),
        })

        if not any(a.state == "I" for a in agents.values()):
            break

    return timeline


# ========== LLM API 驱动接口 ==========

class LLMDrivenSimulator:
    """
    LLM API 驱动的传播模拟接口
    将子图上下文传给外部 LLM，由模型预测下一步传播
    """

    def __init__(self, api_callback: Optional[Callable] = None):
        """
        :param api_callback: 外部 API 回调函数
            签名: callback(graph_context: Dict) -> Dict
            返回应包含: {"new_edges": [...], "activated_nodes": [...], "reasoning": "..."}
        """
        self.api_callback = api_callback

    def build_context(self, G: nx.DiGraph, active_nodes: List[str]) -> Dict[str, Any]:
        """
        构建图上下文，供 LLM 消费
        """
        nodes_info = []
        for node in active_nodes:
            if node not in G.nodes:
                continue
            data = G.nodes[node]
            nodes_info.append({
                "id": node,
                "label": data.get("label", ""),
                "platform": data.get("platform", ""),
                "sentiment": data.get("sentiment", ""),
                "engagement": data.get("engagement", 0),
                "post_count": data.get("post_count", 0),
            })

        edges_info = []
        for u, v, data in G.edges(data=True):
            if u in active_nodes or v in active_nodes:
                edges_info.append({
                    "source": u,
                    "target": v,
                    "relation": data.get("relation_type", ""),
                    "weight": data.get("weight", 1.0),
                })

        return {
            "active_nodes": nodes_info,
            "relevant_edges": edges_info,
            "graph_stats": {
                "total_nodes": G.number_of_nodes(),
                "total_edges": G.number_of_edges(),
                "active_count": len(active_nodes),
            },
        }

    def simulate_step(
        self,
        G: nx.DiGraph,
        active_nodes: List[str]
    ) -> Dict[str, Any]:
        """
        单步 LLM 驱动传播预测
        """
        if self.api_callback is None:
            # Fallback: 使用简单的启发式规则
            return self._fallback_step(G, active_nodes)

        context = self.build_context(G, active_nodes)
        try:
            result = self.api_callback(context)
            return result
        except Exception as e:
            print(f"LLM API call failed: {e}, using fallback")
            return self._fallback_step(G, active_nodes)

    def _fallback_step(self, G: nx.DiGraph, active_nodes: List[str]) -> Dict[str, Any]:
        """LLM 不可用时使用简单启发式规则"""
        new_edges = []
        activated = []

        for node in active_nodes:
            for neighbor in G.successors(node):
                if neighbor in active_nodes:
                    continue
                weight = G[node][neighbor].get("weight", 1.0)
                node_engagement = G.nodes[node].get("engagement", 10)
                prob = min(1.0, 0.3 * weight * (1 + np.log1p(node_engagement) / 10))
                if random.random() < prob:
                    new_edges.append({"source": node, "target": neighbor})
                    activated.append(neighbor)

        return {
            "new_edges": new_edges,
            "activated_nodes": list(set(activated)),
            "reasoning": "Fallback heuristic based on edge weight and node engagement",
        }

    def simulate_full(
        self,
        G: nx.DiGraph,
        seed_nodes: List[str],
        steps: int = 10
    ) -> List[Dict[str, Any]]:
        """
        多步 LLM 驱动传播模拟
        """
        active = set(seed_nodes)
        timeline = []

        for step in range(steps):
            result = self.simulate_step(G, list(active))
            new_activated = set(result.get("activated_nodes", []))

            timeline.append({
                "step": step,
                "active_count": len(active),
                "new_activated": len(new_activated),
                "total_active": len(active | new_activated),
                "reasoning": result.get("reasoning", ""),
            })

            active |= new_activated

            if not new_activated:
                break

        return timeline


# ========== 便捷函数 ==========

def compare_simulations(
    G: nx.DiGraph,
    seed_nodes: List[str],
    steps: int = 50,
    sir_beta: float = 0.3,
    sir_gamma: float = 0.1,
) -> Dict[str, List[Dict]]:
    """
    对比 SIR 和 Agent-based 两种模拟结果
    """
    sir_state = simulate_sir(G, seed_nodes, beta=sir_beta, gamma=sir_gamma, steps=steps)
    agent_timeline = simulate_agent_based(G, seed_nodes, steps=steps)

    return {
        "sir": sir_state.timeline,
        "agent": agent_timeline,
    }
