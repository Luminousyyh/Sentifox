#!/usr/bin/env python3
"""
Agent Communication — 多智能体通信与群体动力学

核心功能：
1. 直接消息：Agent A → Agent B 的私信/评论
2. 群体讨论：围绕热点事件形成讨论线程
3. 信息级联：Agent 观察邻居行为后更新 Belief
4. 回音室检测：同质 Agent 聚集识别
5. 立场极化：群体互动中的立场强化/软化

参考：
- Information Cascades (Bikhchandani et al., 1992)
- Echo Chambers in Social Media (Barberá et al., 2015)
"""
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import networkx as nx

from graph_analysis.cognitive_agent import CognitiveAgent
from graph_analysis.persona_agent import AgentAction
from graph_analysis.temporal_graph import TemporalGraph, TemporalRelation


@dataclass
class Message:
    """Agent 间消息"""
    sender_id: str
    receiver_id: str
    content_type: str       # comment / direct_message / reply / mention
    content: str
    step: int
    sentiment: float = 0.0  # -1 ~ +1
    stance: str = "neutral" # support / oppose / neutral
    influence: float = 1.0  # 消息影响力


@dataclass
class DiscussionThread:
    """讨论线程"""
    thread_id: str
    topic: str              # 讨论主题/事件ID
    creator_id: str
    created_step: int
    messages: List[Message] = field(default_factory=list)
    participant_stances: Dict[str, str] = field(default_factory=dict)  # agent_id -> stance
    sentiment_trajectory: List[float] = field(default_factory=list)    # 情感变化轨迹

    @property
    def participant_count(self) -> int:
        return len(set(m.sender_id for m in self.messages))

    @property
    def dominant_stance(self) -> str:
        counts = defaultdict(int)
        for stance in self.participant_stances.values():
            counts[stance] += 1
        if not counts:
            return "neutral"
        return max(counts.keys(), key=lambda s: counts[s])

    @property
    def polarization_score(self) -> float:
        """讨论内部极化程度：0=一致, 1=完全分裂"""
        counts = defaultdict(int)
        for stance in self.participant_stances.values():
            counts[stance] += 1
        total = sum(counts.values())
        if total < 2:
            return 0.0
        # Gini-like 不平衡度
        max_count = max(counts.values())
        return 1.0 - (max_count / total)

    def add_message(self, msg: Message):
        self.messages.append(msg)
        self.participant_stances[msg.sender_id] = msg.stance
        self.sentiment_trajectory.append(msg.sentiment)


class AgentCommunication:
    """
    Agent 通信管理器
    管理消息传递、讨论线程、信息级联
    """

    def __init__(self):
        self.message_log: List[Message] = []
        self.threads: Dict[str, DiscussionThread] = {}
        self.agent_inboxes: Dict[str, List[Message]] = defaultdict(list)
        # 观察历史：agent_id -> [(observed_agent, action, step)]
        self.observation_history: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)

    # ── 直接通信 ──────────────────────────────

    def send_message(self, sender: CognitiveAgent, receiver_id: str,
                     content_type: str, content: str, step: int,
                     sentiment: float = 0.0) -> Message:
        """发送直接消息"""
        msg = Message(
            sender_id=sender.entity_id,
            receiver_id=receiver_id,
            content_type=content_type,
            content=content,
            step=step,
            sentiment=sentiment,
            stance=sender.stance,
            influence=sender.persona.influence_base * 0.1,
        )
        self.message_log.append(msg)
        self.agent_inboxes[receiver_id].append(msg)
        return msg

    def process_inbox(self, agent: CognitiveAgent, step: int,
                       temporal_graph: TemporalGraph) -> List[AgentAction]:
        """
        Agent 处理收件箱，可能产生行动
        Returns: 产生的 AgentAction 列表
        """
        actions = []
        inbox = self.agent_inboxes.get(agent.entity_id, [])
        if not inbox:
            return actions

        # 只处理最近的消息
        recent_msgs = [m for m in inbox if m.step >= step - 2]
        self.agent_inboxes[agent.entity_id] = []  # 清空已处理

        for msg in recent_msgs:
            # 情感影响
            if msg.sentiment < -0.3:
                agent.emotion.update("exposed_to_negative", event_intensity=msg.sentiment, step=step)

            # 立场影响：如果来自信任的来源
            trust = agent.memory.semantic.get_trust(msg.sender_id)
            if trust > 0.6 and msg.stance != agent.stance and msg.stance != "neutral":
                # 高信任 + 不同立场 → 可能软化立场
                if agent.emotion.dominant != "anger":
                    # 记录到记忆
                    agent.memory.perceive(
                        MemoryItem(
                            content=f"Received opposing view from trusted {msg.sender_id}",
                            memory_type="event",
                            step=step,
                            importance=0.6,
                            emotion_tag="anxiety",
                            metadata={"sender": msg.sender_id, "stance": msg.stance},
                        ),
                        step,
                    )

            # 回复概率
            reply_prob = 0.1 * trust * (1.0 - agent.persona.fatigue)
            if agent.emotion.dominant == "anger":
                reply_prob *= 2.0  # 愤怒时更爱回复

            if random.random() < reply_prob:
                actions.append(AgentAction(
                    action_type="comment",
                    target_id=msg.sender_id,
                    sentiment=agent.persona.sentiment_tendency,
                ))

        return actions

    # ── 群体讨论 ──────────────────────────────

    def create_discussion(self, creator_id: str, topic: str,
                          step: int) -> DiscussionThread:
        """创建讨论线程"""
        thread_id = f"thread_{step}_{creator_id}_{random.randint(1000, 9999)}"
        thread = DiscussionThread(
            thread_id=thread_id,
            topic=topic,
            creator_id=creator_id,
            created_step=step,
        )
        self.threads[thread_id] = thread
        return thread

    def participate_discussion(self, thread_id: str, agent: CognitiveAgent,
                                step: int, temporal_graph: TemporalGraph) -> Optional[Message]:
        """
        Agent 参与讨论，发表观点
        Returns: 发表的消息（如有）
        """
        thread = self.threads.get(thread_id)
        if not thread:
            return None

        # 评估是否参与
        thread_sentiment = 0.0
        if thread.sentiment_trajectory:
            thread_sentiment = sum(thread.sentiment_trajectory[-5:]) / \
                min(5, len(thread.sentiment_trajectory))

        # 立场匹配度
        stance_match = 1.0 if thread.dominant_stance == agent.stance else 0.3

        # 参与概率
        participate_prob = 0.3 * stance_match * (1.0 - agent.persona.fatigue)
        if agent.emotion.dominant in ("anger", "excitement"):
            participate_prob *= 1.5

        if random.random() > participate_prob:
            return None

        # 生成消息内容（简化版描述）
        if agent.stance == "support":
            content = f"Support for {thread.topic}"
            sentiment = 0.5
        elif agent.stance == "oppose":
            content = f"Opposition to {thread.topic}"
            sentiment = -0.5
        else:
            content = f"Comment on {thread.topic}"
            sentiment = 0.0

        # 群体极化效应：如果讨论已经偏向一方，Agent 更可能强化而非软化
        if thread.dominant_stance == agent.stance and agent.stance != "neutral":
            # 同质强化
            sentiment *= 1.3
            agent.persona.sentiment_tendency = max(-1.0, min(1.0,
                agent.persona.sentiment_tendency + 0.05 * (1 if agent.stance == "support" else -1)))
        elif thread.dominant_stance != agent.stance and thread.dominant_stance != "neutral":
            # 异质对抗：愤怒时对抗，冷静时可能软化
            if agent.emotion.dominant == "anger":
                sentiment *= -1.2
            elif agent.emotion.dominant == "calm":
                # 冷静时可能中立化
                sentiment *= 0.5

        msg = Message(
            sender_id=agent.entity_id,
            receiver_id=thread_id,
            content_type="reply",
            content=content,
            step=step,
            sentiment=sentiment,
            stance=agent.stance,
            influence=agent.persona.influence_base * 0.1,
        )
        thread.add_message(msg)
        self.message_log.append(msg)

        # 消耗注意力
        agent.persona.attention -= 0.05
        agent.persona.fatigue += 0.02

        return msg

    # ── 信息级联 ──────────────────────────────

    def observe_and_update(self, observer: CognitiveAgent,
                           observed_actions: List[Tuple[str, AgentAction]],
                           step: int, temporal_graph: TemporalGraph):
        """
        信息级联：Agent 观察邻居的行为后更新自身 Belief
        参考：Bikhchandani et al. Information Cascades
        """
        if not observed_actions:
            return

        # 统计观察到的行为
        action_counts = defaultdict(int)
        stance_influence = defaultdict(float)

        for actor_id, action in observed_actions:
            if actor_id == observer.entity_id:
                continue
            action_counts[action.action_type] += 1

            # 从信任权重加权
            trust = observer.memory.semantic.get_trust(actor_id)
            if action.action_type == "repost" and action.target_id:
                stance_influence[action.target_id] += trust

            # 记录观察历史
            self.observation_history[observer.entity_id].append(
                (actor_id, action.action_type, step)
            )

        # 信息级联：如果多数邻居都在转发某内容，Observer 也更可能跟随
        total_observed = sum(action_counts.values())
        if total_observed >= 3:
            repost_ratio = action_counts.get("repost", 0) / total_observed
            if repost_ratio > 0.6:
                # 级联阈值突破：增加 Observer 的转发欲望
                observer.emotion.arousal = min(1.0, observer.emotion.arousal + 0.1)
                observer.memory.perceive(
                    MemoryItem(
                        content=f"Observed cascade: {action_counts.get('repost', 0)} reposts among neighbors",
                        memory_type="event",
                        step=step,
                        importance=0.5,
                        emotion_tag="excitement",
                        metadata={"cascade_size": action_counts.get("repost", 0)},
                    ),
                    step,
                )

        # 立场级联：如果多数信任邻居转变立场
        if stance_influence:
            # 简化：不在这里做复杂立场更新，留给 cognitive_agent 的 belief 更新
            pass

    # ── 回音室检测 ──────────────────────────────

    def detect_echo_chambers(self, agents: Dict[str, CognitiveAgent],
                             temporal_graph: TemporalGraph,
                             step: int,
                             min_size: int = 3) -> List[Dict[str, Any]]:
        """
        检测回音室社区：高内聚 + 同质立场的 Agent 群体
        
        Returns:
            [{
                "members": [agent_id, ...],
                "dominant_stance": str,
                "internal_density": float,
                "external_sparsity": float,
                "echo_score": float,
            }, ...]
        """
        snapshot = temporal_graph.get_snapshot_at(step)
        chambers = []

        # 按立场分组
        stance_groups = defaultdict(list)
        for aid, agent in agents.items():
            stance_groups[agent.stance].append(aid)

        for stance, members in stance_groups.items():
            if len(members) < min_size or stance == "neutral":
                continue

            # 计算组内连接密度
            internal_edges = 0
            possible_internal = len(members) * (len(members) - 1)
            for i, a in enumerate(members):
                for b in members[i+1:]:
                    if snapshot.has_edge(a, b) or snapshot.has_edge(b, a):
                        internal_edges += 1

            internal_density = internal_edges / max(1, possible_internal)

            # 计算组外连接稀疏度
            external_edges = 0
            for a in members:
                for b in set(snapshot.nodes()) - set(members):
                    if snapshot.has_edge(a, b) or snapshot.has_edge(b, a):
                        external_edges += 1
            possible_external = len(members) * (len(snapshot.nodes()) - len(members))
            external_density = external_edges / max(1, possible_external)

            # 回音室分数：高内聚 + 低外联
            echo_score = internal_density * (1.0 - external_density)
            if echo_score > 0.3:
                chambers.append({
                    "members": members,
                    "dominant_stance": stance,
                    "internal_density": round(internal_density, 3),
                    "external_sparsity": round(1.0 - external_density, 3),
                    "echo_score": round(echo_score, 3),
                })

        # 按回音室分数排序
        chambers.sort(key=lambda x: x["echo_score"], reverse=True)
        return chambers

    def get_discussion_stats(self) -> Dict[str, Any]:
        """获取讨论统计"""
        if not self.threads:
            return {"thread_count": 0}

        total_msgs = sum(len(t.messages) for t in self.threads.values())
        avg_participants = sum(t.participant_count for t in self.threads.values()) / len(self.threads)
        avg_polarization = sum(t.polarization_score for t in self.threads.values()) / len(self.threads)

        return {
            "thread_count": len(self.threads),
            "total_messages": total_msgs,
            "avg_participants": round(avg_participants, 2),
            "avg_polarization": round(avg_polarization, 3),
            "most_active_thread": max(
                self.threads.values(),
                key=lambda t: len(t.messages)
            ).thread_id if self.threads else None,
        }


# 导入 MemoryItem 用于通信模块
from graph_analysis.memory_system import MemoryItem
