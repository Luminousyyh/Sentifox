#!/usr/bin/env python3
"""
CognitiveAgent — BDI 认知架构 + 情感状态机

将 Agent 从「概率掷骰子」升级为「信念-欲望-意图」推理驱动的自主实体。

核心组件：
- Belief: 对世界的信念集合（从时态图谱查询构建）
- Desire: 目标栈（传播真相 / 维护立场 / 获得关注 / 保持中立）
- Intention: 当前执行计划（阅读 → 评估 → 行动）
- EmotionalState: 情绪状态机（愤怒/焦虑/兴奋/平静/疲惫）
- MemorySystem: 三层记忆（工作/情景/语义）

决策流程：
感知(perceive) → 记忆检索(retrieve) → 信念更新(update_beliefs)
→ 目标选择(select_desire) → 计划生成(form_plan) → 行动执行(execute)
"""
import random
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from graph_analysis.persona_agent import PersonaAgent, AgentAction
from graph_analysis.memory_system import MemorySystem, MemoryItem
from graph_analysis.temporal_graph import TemporalGraph, TemporalRelation
from graph_analysis.platform_dynamics import PlatformRule, get_platform_rule


# ── 情绪定义 ──────────────────────────────

EMOTION_TYPES = ["anger", "anxiety", "excitement", "calm", "fatigue"]

EMOTION_MODULATORS = {
    "anger":       {"post_boost": 1.6, "repost_boost": 1.5, "rational_filter": 0.4, "stance_rigidity": 1.4},
    "anxiety":     {"post_boost": 1.2, "repost_boost": 1.8, "rational_filter": 0.6, "stance_rigidity": 0.8},
    "excitement":  {"post_boost": 1.5, "repost_boost": 1.6, "rational_filter": 0.7, "stance_rigidity": 1.0},
    "calm":        {"post_boost": 0.8, "repost_boost": 0.7, "rational_filter": 1.3, "stance_rigidity": 0.9},
    "fatigue":     {"post_boost": 0.3, "repost_boost": 0.3, "rational_filter": 1.0, "stance_rigidity": 0.5},
}


@dataclass
class Belief:
    """Agent 对世界的信念"""
    hot_events: List[Tuple[str, float]] = field(default_factory=list)
    neighbor_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    community_stance: str = "neutral"
    community_sentiment: float = 0.0
    personal_trust: Dict[str, float] = field(default_factory=dict)
    recent_influences: List[TemporalRelation] = field(default_factory=list)
    self_state: Dict[str, Any] = field(default_factory=dict)
    # 对当前热点事件的评估
    event_evaluations: Dict[str, float] = field(default_factory=dict)


@dataclass
class Desire:
    """Agent 的目标"""
    goal_type: str          # spread_truth / defend_stance / gain_attention / stay_neutral / seek_safety
    priority: float = 0.5   # 0~1
    target_topic: str = ""  # 相关话题/事件
    urgency: float = 0.5    # 紧急程度


@dataclass
class Intention:
    """Agent 的当前计划"""
    plan_steps: List[str] = field(default_factory=list)  # ["observe", "evaluate", "act"]
    current_step_idx: int = 0
    target_action: Optional[str] = None
    target_id: Optional[str] = None
    reasoning_chain: List[str] = field(default_factory=list)

    @property
    def current_phase(self) -> str:
        if self.current_step_idx < len(self.plan_steps):
            return self.plan_steps[self.current_step_idx]
        return "done"

    def advance(self):
        self.current_step_idx += 1

    def add_reasoning(self, reason: str):
        self.reasoning_chain.append(reason)


class EmotionalState:
    """
    情感状态机
    - 基础情绪（主导情绪 + 强度）
    - 情绪唤起度（arousal）
    - 效价（valence: 正/负）
    - 情绪传染接口
    """

    def __init__(self, initial_emotion: str = "calm", initial_intensity: float = 0.2):
        self.dominant = initial_emotion
        self.intensity = initial_intensity
        self.arousal = 0.2
        self.valence = 0.0  # -1(负) ~ +1(正)
        # 情绪历史（用于追踪情绪波动）
        self.history: List[Tuple[int, str, float]] = []

    def update(self, event_type: str, event_intensity: float = 0.0, step: int = 0):
        """
        根据事件更新情绪
        event_type: infection / reposted / opposed / supported / ignored / exposed_to_negative
        """
        self.history.append((step, self.dominant, self.intensity))

        if event_type == "infection":
            # 被感染 → 兴奋或焦虑（取决于内容情感）
            if event_intensity > 0:
                self.dominant = "excitement"
                self.valence = min(1.0, self.valence + 0.3)
            else:
                self.dominant = "anxiety"
                self.valence = max(-1.0, self.valence - 0.3)
            self.intensity = min(1.0, self.intensity + 0.4)
            self.arousal = min(1.0, self.arousal + 0.3)

        elif event_type == "opposed":
            self.dominant = "anger"
            self.valence = max(-1.0, self.valence - 0.4)
            self.intensity = min(1.0, self.intensity + 0.3)
            self.arousal = min(1.0, self.arousal + 0.2)

        elif event_type == "supported":
            if self.dominant in ("anger", "anxiety"):
                self.dominant = "calm"
            self.valence = min(1.0, self.valence + 0.2)
            self.intensity = max(0.1, self.intensity - 0.1)
            self.arousal = max(0.0, self.arousal - 0.1)

        elif event_type == "exposed_to_negative":
            # 暴露于负面内容
            if self.dominant == "calm":
                self.dominant = "anxiety"
            self.valence = max(-1.0, self.valence - 0.15)
            self.intensity = min(1.0, self.intensity + 0.1)
            self.arousal = min(1.0, self.arousal + 0.15)

        elif event_type == "fatigue_accumulation":
            self.dominant = "fatigue"
            self.arousal = max(0.0, self.arousal - 0.2)
            self.intensity = max(0.1, self.intensity - 0.15)

        elif event_type == "natural_decay":
            # 自然衰减向平静
            if self.dominant != "calm":
                self.intensity *= 0.9
                if self.intensity < 0.2:
                    self.dominant = "calm"
                    self.intensity = 0.2
            self.arousal = max(0.0, self.arousal - 0.05)

    def get_modulators(self) -> Dict[str, float]:
        """获取当前情绪对行为的调制系数"""
        base = EMOTION_MODULATORS.get(self.dominant, EMOTION_MODULATORS["calm"])
        # 强度缩放：高强度时效果放大
        scale = 0.5 + 0.5 * self.intensity
        return {k: 1.0 + (v - 1.0) * scale for k, v in base.items()}

    def is_receptive_to_influence(self) -> bool:
        """高唤起 + 非疲劳状态下更容易被影响"""
        return self.arousal > 0.4 and self.dominant != "fatigue"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dominant": self.dominant,
            "intensity": round(self.intensity, 2),
            "arousal": round(self.arousal, 2),
            "valence": round(self.valence, 2),
        }


class CognitiveAgent:
    """
    BDI 认知 Agent

    将人格化 Agent (PersonaAgent) 包装为具有认知推理能力的实体。
    保留 PersonaAgent 的所有属性，决策逻辑由 BDI 引擎驱动。
    """

    def __init__(self, persona: PersonaAgent):
        self.persona = persona
        self.memory = MemorySystem(wm_capacity=9, em_max_size=200)
        self.emotion = EmotionalState(
            initial_emotion="calm",
            initial_intensity=0.2,
        )
        self.belief = Belief()
        self.desire: Optional[Desire] = None
        self.intention = Intention()

        # BDI 推理日志（用于可解释性）
        self.reasoning_log: List[Dict[str, Any]] = []

    # ── 属性代理 ──────────────────────────────

    @property
    def entity_id(self) -> str:
        return self.persona.entity_id

    @property
    def name(self) -> str:
        return self.persona.name

    @property
    def stance(self) -> str:
        return self.persona.stance

    @stance.setter
    def stance(self, value: str):
        self.persona.stance = value

    @property
    def state(self) -> str:
        return self.persona.state

    @property
    def platform(self) -> str:
        return self.persona.platform

    @property
    def mbti(self) -> str:
        return self.persona.mbti

    # ── BDI 核心循环 ──────────────────────────────

    def perceive_and_decide(self, temporal_graph: TemporalGraph,
                            step: int, platform_rule: PlatformRule) -> Optional[AgentAction]:
        """
        完整的感知-决策循环
        返回 AgentAction 或 None
        """
        # 1. 感知环境
        self._perceive(temporal_graph, step)

        # 2. 记忆检索与存储
        self._update_memory(step)

        # 3. 信念更新
        self._update_beliefs(temporal_graph, step)

        # 4. 情绪自然衰减（每步）
        self.emotion.update("natural_decay", step=step)

        # 5. 目标选择
        self._select_desire()

        # 6. 计划生成
        self._form_plan()

        # 7. 行动执行
        action = self._execute_intention(temporal_graph, step, platform_rule)

        # 8. 记录推理日志
        self.reasoning_log.append({
            "step": step,
            "emotion": self.emotion.to_dict(),
            "desire": self.desire.goal_type if self.desire else "none",
            "intention": self.intention.target_action,
            "reasoning": list(self.intention.reasoning_chain),
        })
        self.intention.reasoning_chain.clear()

        return action

    def _perceive(self, temporal_graph: TemporalGraph, step: int):
        """感知：从时态图谱获取信息，存入工作记忆"""
        # 热点事件
        hot_events = temporal_graph.get_hot_events(step, top_n=5)
        for event_id, score in hot_events:
            self.memory.perceive(MemoryItem(
                content=f"Hot event: {event_id} (score={score:.1f})",
                memory_type="event",
                step=step,
                importance=min(1.0, score / 10.0),
                emotion_tag=self.emotion.dominant,
                emotion_intensity=self.emotion.intensity,
                metadata={"event_id": event_id, "score": score},
            ), step)

        # 最近的关系
        recent_rels = temporal_graph.query_relations(
            self.entity_id, at_step=step, since_step=max(0, step - 5)
        )
        for rel in recent_rels[:5]:  # 最多5条
            self.memory.perceive(MemoryItem(
                content=f"Relation: {rel.relation_type} with {rel.target_id}",
                memory_type="relation",
                step=step,
                importance=min(1.0, rel.weight / 5.0),
                emotion_tag=self.emotion.dominant,
                metadata={"relation_type": rel.relation_type, "target": rel.target_id},
            ), step)

    def _update_memory(self, step: int):
        """检索相关记忆，辅助信念形成"""
        # 定期固化（每20步）
        if step > 0 and step % 20 == 0:
            self.memory.consolidate_to_semantic(step)
            self.memory.forget_old_memories(step, threshold=0.03)

    def _update_beliefs(self, temporal_graph: TemporalGraph, step: int):
        """基于感知和记忆更新信念"""
        # 社区情绪
        snapshot = temporal_graph.get_snapshot_at(step)
        neighbor_ids = list(snapshot.successors(self.entity_id)) + \
                       list(snapshot.predecessors(self.entity_id))

        neighbor_states = {}
        for nid in neighbor_ids[:10]:  # 最多10个邻居
            # 从语义记忆中获取信任度
            trust = self.memory.semantic.get_trust(nid)
            # 从图谱获取最近关系
            rels = temporal_graph.query_relations(self.entity_id, target_id=nid, at_step=step)
            total_weight = sum(r.weight for r in rels)
            neighbor_states[nid] = {
                "trust": trust,
                "relation_weight": total_weight,
                "has_influence": any(r.relation_type == "influence" for r in rels),
            }

        # 社区主导立场（从记忆中推断）
        recent_wm = self.memory.working.retrieve(current_step=step, top_k=9)
        stance_counts = defaultdict(int)
        for m in recent_wm:
            if m.memory_type == "relation":
                # 简化：根据关系类型推断立场
                if m.metadata.get("relation_type") in ("support", "influence"):
                    stance_counts["support"] += 1
                elif m.metadata.get("relation_type") == "oppose":
                    stance_counts["oppose"] += 1

        dominant_stance = max(stance_counts.keys(), key=lambda s: stance_counts[s]) \
            if stance_counts else "neutral"

        # 更新信念
        self.belief = Belief(
            hot_events=temporal_graph.get_hot_events(step, top_n=3),
            neighbor_states=neighbor_states,
            community_stance=dominant_stance,
            community_sentiment=sum(self.memory.semantic.stance_anchors.values()) / \
                max(1, len(self.memory.semantic.stance_anchors)),
            personal_trust=dict(self.memory.semantic.trust_network),
            recent_influences=temporal_graph.query_relations(
                self.entity_id, relation_type="influence", at_step=step
            )[:5],
            self_state={
                "attention": self.persona.attention,
                "fatigue": self.persona.fatigue,
                "stance": self.persona.stance,
                "influence": self.persona.influence_base,
            },
        )

    def _select_desire(self):
        """基于信念、情绪和人格选择目标"""
        modulators = self.emotion.get_modulators()
        traits = self.persona.get_mbti_traits()

        # 评估各目标的吸引力
        goals = []

        # 1. 维护立场
        stance_defend_score = 0.4
        if self.persona.stance != "neutral":
            stance_defend_score += 0.3
        stance_defend_score *= traits.get("stance_stability", 1.0)
        stance_defend_score *= modulators["stance_rigidity"]
        # 如果情绪愤怒，更想捍卫立场
        if self.emotion.dominant == "anger":
            stance_defend_score += 0.3
        goals.append(("defend_stance", stance_defend_score, self.persona.stance))

        # 2. 传播信息（获得关注）
        spread_score = 0.3
        spread_score *= traits.get("post_boost", 1.0) * 0.5 + traits.get("repost_boost", 1.0) * 0.5
        spread_score *= modulators["repost_boost"]
        # 兴奋时更爱传播
        if self.emotion.dominant == "excitement":
            spread_score += 0.2
        # 疲劳时不想传播
        if self.emotion.dominant == "fatigue":
            spread_score -= 0.4
        goals.append(("gain_attention", spread_score, ""))

        # 3. 保持中立/安全
        safety_score = 0.2
        if self.persona.stance == "neutral":
            safety_score += 0.3
        safety_score *= modulators["rational_filter"]
        # 焦虑时更想保持安全
        if self.emotion.dominant == "anxiety":
            safety_score += 0.2
        goals.append(("seek_safety", safety_score, ""))

        # 4. 传播"真相"（理性 Agent 倾向）
        truth_score = 0.2
        truth_score *= traits.get("rational_filter", 1.0)
        if self.emotion.dominant == "calm":
            truth_score += 0.1
        goals.append(("spread_truth", truth_score, ""))

        # 选择最高优先级目标
        goals.sort(key=lambda x: x[1], reverse=True)
        chosen = goals[0]

        self.desire = Desire(
            goal_type=chosen[0],
            priority=chosen[1],
            target_topic=chosen[2],
            urgency=self.emotion.arousal,
        )

    def _form_plan(self):
        """基于目标生成计划"""
        if not self.desire:
            self.intention = Intention()
            return

        goal = self.desire.goal_type
        plan = []

        if goal == "defend_stance":
            plan = ["observe", "evaluate_opposition", "counter_post"]
            self.intention.target_action = "post"
        elif goal == "gain_attention":
            if self.belief.recent_influences:
                plan = ["observe", "select_content", "repost"]
                self.intention.target_action = "repost"
            else:
                plan = ["observe", "create_content", "post"]
                self.intention.target_action = "post"
        elif goal == "seek_safety":
            plan = ["observe", "assess_risk", "ignore_or_clarify"]
            self.intention.target_action = "ignore"
        elif goal == "spread_truth":
            plan = ["observe", "verify", "share"]
            self.intention.target_action = "post"
        else:
            plan = ["observe"]
            self.intention.target_action = "ignore"

        self.intention = Intention(
            plan_steps=plan,
            target_action=self.intention.target_action,
        )
        self.intention.add_reasoning(f"Goal: {goal} (priority={self.desire.priority:.2f})")
        self.intention.add_reasoning(f"Emotion: {self.emotion.dominant} (intensity={self.emotion.intensity:.2f})")

    def _execute_intention(self, temporal_graph: TemporalGraph,
                           step: int, platform_rule: PlatformRule) -> Optional[AgentAction]:
        """将计划转化为具体行动"""
        if not self.desire or self.desire.priority < 0.2:
            self.intention.add_reasoning("Priority too low, ignore")
            return None

        if self.persona.attention <= 0:
            self.intention.add_reasoning("No attention left")
            return None

        action_type = self.intention.target_action

        # 根据目标类型执行不同行动
        if action_type == "repost":
            return self._do_repost(temporal_graph, step, platform_rule)
        elif action_type == "post":
            return self._do_post(platform_rule)
        elif action_type == "stance_shift":
            return self._do_stance_shift()
        elif action_type == "comment":
            return self._do_comment(temporal_graph, step)

        self.intention.add_reasoning("No action taken")
        return None

    def _do_repost(self, temporal_graph: TemporalGraph,
                   step: int, platform_rule: PlatformRule) -> Optional[AgentAction]:
        """执行转发"""
        # 选择最具影响力的可信邻居
        candidates = []
        for nid, info in self.belief.neighbor_states.items():
            trust = info.get("trust", 0.3)
            weight = info.get("relation_weight", 0)
            # 从语义记忆获取印象
            impression = self.memory.semantic.get_impression(nid)
            score = trust * 0.4 + weight * 0.3 + max(0, impression) * 0.3
            candidates.append((nid, score))

        if not candidates:
            self.intention.add_reasoning("No trusted neighbor to repost")
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        target_id = candidates[0][0]
        self.intention.target_id = target_id

        # 更新记忆
        self.memory.semantic.update_trust(target_id, 0.02)
        self.persona.attention -= 0.1
        self.persona.fatigue += 0.03
        self.persona.amplification_count += 1

        self.intention.add_reasoning(f"Repost from {target_id} (trust_score={candidates[0][1]:.2f})")

        return AgentAction(
            action_type="repost",
            target_id=target_id,
            sentiment=self.persona.sentiment_tendency,
        )

    def _do_post(self, platform_rule: PlatformRule) -> AgentAction:
        """执行发布"""
        self.persona.attention -= 0.15
        self.persona.fatigue += 0.05

        # 内容情感由人格 + 情绪共同决定
        sentiment = self.persona.sentiment_tendency
        if self.emotion.dominant == "anger":
            sentiment = max(-1.0, sentiment - 0.2)
        elif self.emotion.dominant == "excitement":
            sentiment = min(1.0, sentiment + 0.1)

        self.intention.add_reasoning(f"Post with sentiment={sentiment:.2f}")

        return AgentAction(
            action_type="post",
            sentiment=sentiment,
        )

    def _do_stance_shift(self) -> Optional[AgentAction]:
        """执行立场转变"""
        new_stance = self.belief.community_stance
        if new_stance == self.persona.stance or new_stance == "neutral":
            return None

        # 记录旧立场到语义记忆
        self.memory.semantic.update_stance("community", 1.0 if new_stance == "support" else -1.0)

        self.intention.add_reasoning(f"Stance shift: {self.persona.stance} -> {new_stance}")

        return AgentAction(
            action_type="stance_shift",
            new_stance=new_stance,
        )

    def _do_comment(self, temporal_graph: TemporalGraph, step: int) -> Optional[AgentAction]:
        """执行评论"""
        recent = temporal_graph.query_relations(self.entity_id, at_step=step)
        if not recent:
            return None
        target = random.choice(recent)
        target_id = target.target_id if target.source_id == self.entity_id else target.source_id

        self.persona.attention -= 0.08
        self.persona.fatigue += 0.02

        return AgentAction(
            action_type="comment",
            target_id=target_id,
            sentiment=self.persona.sentiment_tendency,
        )

    # ── 情绪传染接口 ──────────────────────────────

    def receive_emotional_influence(self, source_emotion: str, source_intensity: float,
                                    source_valence: float, trust_weight: float):
        """
        接收来自邻居的情绪影响
        trust_weight: 对来源 Agent 的信任权重
        """
        # 情绪传染公式：自己的状态 = 原状态 + 来源情绪 × 信任权重 × 接受度
        receptivity = self.persona.receptivity * (1.0 + self.emotion.arousal)
        influence = source_intensity * trust_weight * receptivity * 0.3

        if influence > 0.15:
            # 被显著影响
            if source_emotion in ("anger", "anxiety") and self.emotion.dominant not in ("anger", "anxiety"):
                self.emotion.update("exposed_to_negative", event_intensity=source_valence)
            elif source_emotion == "excitement":
                self.emotion.arousal = min(1.0, self.emotion.arousal + influence * 0.5)

    def record_infection(self, step: int, source_id: str, source_sentiment: float = 0.0):
        """记录被感染，触发情绪和记忆更新"""
        self.persona.record_infection(step, source_id)
        self.emotion.update("infection", event_intensity=source_sentiment, step=step)

        # 记录到记忆
        self.memory.perceive(MemoryItem(
            content=f"Infected by {source_id} at step {step}",
            memory_type="event",
            step=step,
            importance=0.8,
            emotion_tag=self.emotion.dominant,
            emotion_intensity=self.emotion.intensity,
            metadata={"source": source_id, "sentiment": source_sentiment},
        ), step)

        # 更新对感染源的信任（负面感染降低信任）
        if source_sentiment < -0.3:
            self.memory.semantic.update_trust(source_id, -0.1)
        else:
            self.memory.semantic.update_trust(source_id, 0.05)

    def get_decision_explanation(self) -> Dict[str, Any]:
        """获取最近一次决策的可解释性报告"""
        if not self.reasoning_log:
            return {}
        latest = self.reasoning_log[-1]
        memory_ctx = self.memory.get_context_for_decision(
            current_step=latest["step"],
            current_emotion=self.emotion.dominant,
            current_arousal=self.emotion.arousal,
        )
        return {
            "agent_id": self.entity_id,
            "name": self.name,
            "mbti": self.mbti,
            "step": latest["step"],
            "belief_summary": {
                "community_stance": self.belief.community_stance,
                "neighbor_count": len(self.belief.neighbor_states),
                "hot_events": [e[0] for e in self.belief.hot_events[:3]],
            },
            "emotion": self.emotion.to_dict(),
            "desire": self.desire.goal_type if self.desire else "none",
            "intention": self.intention.target_action,
            "reasoning_chain": latest["reasoning"],
            "memory_context": memory_ctx,
        }

    def interview(self, question: str) -> str:
        """
        基于认知状态生成采访回答
        返回自然语言解释，包含 BDI 链
        """
        parts = []

        if "为什么" in question or "怎么" in question:
            exp = self.get_decision_explanation()
            if exp:
                parts.append(f"作为{self.name}（{self.mbti}型{self.persona.occupation}），")
                parts.append(f"我当前的情绪状态是{self.emotion.dominant}（强度{self.emotion.intensity:.1f}），")
                if self.desire:
                    parts.append(f"我的主要目标是「{self.desire.goal_type}」")
                if self.intention.reasoning_chain:
                    parts.append(f"，因为{'；'.join(self.intention.reasoning_chain[:2])}。")

        if "立场" in question or "态度" in question:
            parts.append(f"我目前的立场是{self.stance}，")
            if self.memory.semantic.stance_anchors:
                avg_stance = sum(self.memory.semantic.stance_anchors.values()) / \
                    len(self.memory.semantic.stance_anchors)
                parts.append(f"对社区话题的整体倾向是{'支持' if avg_stance > 0 else '反对' if avg_stance < 0 else '中立'}。")

        if "记忆" in question or "记得" in question:
            recent = self.memory.working.get_all()
            if recent:
                parts.append(f"我最近记得的事情：{recent[0].content}。")
            else:
                parts.append("最近没有什么特别的印象。")

        if not parts:
            parts.append(f"我是{self.name}，{self.persona.occupation}，")
            parts.append(f"当前情绪{self.emotion.dominant}，立场{self.stance}。")

        return " ".join(parts)
