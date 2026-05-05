#!/usr/bin/env python3
"""
Agent 三层记忆系统（MemGPT 风格简化版）

工作记忆(WM): 当前步感知，容量 7±2，FIFO+优先级淘汰
情景记忆(EM): 个人经历事件序列，支持情感/时间/内容检索
语义记忆(SM): 长期知识（信任网络、立场锚点、品牌印象）

记忆检索受情绪唤起度和时间衰减共同调制。
"""
import random
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class MemoryItem:
    """记忆单元"""
    content: str                      # 记忆内容描述
    memory_type: str                  # event / relation / emotion / fact
    step: int                         # 创建时间步
    importance: float = 0.5           # 重要性 0~1
    emotion_tag: str = "neutral"      # 情感标签
    emotion_intensity: float = 0.0    # 情感强度 0~1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def recency_score(self, current_step: int, decay_lambda: float = 0.1) -> float:
        """时间衰减分数（指数衰减）"""
        delta = max(0, current_step - self.step)
        return math.exp(-decay_lambda * delta)

    def retrieval_strength(self, current_step: int, current_emotion: str,
                           current_arousal: float) -> float:
        """
        综合检索强度 = 重要性 × 时间衰减 × 情感匹配加成
        """
        recency = self.recency_score(current_step)
        # 情感匹配：相同情感标签时加成
        emotion_match = 1.0
        if self.emotion_tag == current_emotion and current_emotion != "neutral":
            emotion_match = 1.0 + 0.5 * current_arousal
        # 高情感强度记忆更容易被唤起
        intensity_boost = 1.0 + 0.3 * self.emotion_intensity
        return self.importance * recency * emotion_match * intensity_boost


class WorkingMemory:
    """
    工作记忆：容量有限（默认9），当前意识内容
    """

    def __init__(self, capacity: int = 9):
        self.capacity = capacity
        self.items: List[MemoryItem] = []

    def store(self, item: MemoryItem) -> Optional[MemoryItem]:
        """
        存入工作记忆。若溢出，淘汰重要性最低项，并返回被淘汰项（用于固化到情景记忆）
        """
        self.items.append(item)
        # 按（重要性 + 时间）排序，保持高优先级在前
        self.items.sort(
            key=lambda m: (m.importance, m.step),
            reverse=True,
        )
        if len(self.items) > self.capacity:
            evicted = self.items.pop()  # 最低优先级的被淘汰
            return evicted
        return None

    def retrieve(self, query: str = "", current_step: int = 0,
                 current_emotion: str = "neutral", current_arousal: float = 0.0,
                 top_k: int = 3) -> List[MemoryItem]:
        """检索最相关的工作记忆"""
        scored = [
            (m, m.retrieval_strength(current_step, current_emotion, current_arousal))
            for m in self.items
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_k]]

    def clear(self):
        """清空（如睡眠/断线时）"""
        self.items.clear()

    def get_all(self) -> List[MemoryItem]:
        return list(self.items)


class EpisodicMemory:
    """
    情景记忆：个人经历的事件序列
    支持基于情感标签、时间窗口、内容关键词的检索
    """

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.episodes: List[MemoryItem] = []

    def store(self, item: MemoryItem):
        """存储事件，超限时淘汰最旧的低重要性事件"""
        self.episodes.append(item)
        if len(self.episodes) > self.max_size:
            # 淘汰：旧且不重要
            self.episodes.sort(
                key=lambda m: (m.step, m.importance),
            )
            self.episodes = self.episodes[-self.max_size:]

    def retrieve(self, current_step: int = 0,
                 current_emotion: str = "neutral", current_arousal: float = 0.0,
                 emotion_filter: Optional[str] = None,
                 since_step: Optional[int] = None,
                 top_k: int = 5) -> List[MemoryItem]:
        """
        检索情景记忆
        """
        candidates = self.episodes
        if since_step is not None:
            candidates = [m for m in candidates if m.step >= since_step]
        if emotion_filter:
            candidates = [m for m in candidates if m.emotion_tag == emotion_filter]

        scored = [
            (m, m.retrieval_strength(current_step, current_emotion, current_arousal))
            for m in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_k]]

    def retrieve_similar_experiences(self, reference: MemoryItem,
                                     current_step: int = 0,
                                     top_k: int = 3) -> List[MemoryItem]:
        """检索与参考事件相似的经历（同类型 + 同情感 + 时间近）"""
        candidates = [
            m for m in self.episodes
            if m.memory_type == reference.memory_type and m.step != reference.step
        ]
        scored = []
        for m in candidates:
            score = m.retrieval_strength(
                current_step, reference.emotion_tag, reference.emotion_intensity
            )
            # 额外加成：同情感标签
            if m.emotion_tag == reference.emotion_tag:
                score *= 1.3
            scored.append((m, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_k]]

    def consolidate(self, current_step: int) -> List[Dict[str, Any]]:
        """
        记忆固化分析：找出重复出现的模式，返回可提升为语义记忆的知识条目
        """
        patterns = defaultdict(list)
        for m in self.episodes:
            if m.step < current_step - 20:  # 只分析较旧的记忆
                key = (m.memory_type, m.emotion_tag)
                patterns[key].append(m)

        consolidated = []
        for (mtype, emotion), items in patterns.items():
            if len(items) >= 3:  # 重复3次以上形成语义知识
                avg_importance = sum(m.importance for m in items) / len(items)
                consolidated.append({
                    "type": mtype,
                    "emotion": emotion,
                    "count": len(items),
                    "avg_importance": avg_importance,
                    "summary": f"Repeated {mtype} experiences with {emotion} emotion",
                })
        return consolidated


class SemanticMemory:
    """
    语义记忆：长期知识存储
    键值对 + 置信度，支持版本更新
    """

    def __init__(self):
        self.facts: Dict[str, Dict[str, Any]] = {}
        # 信任网络: {target_id: trust_score}
        self.trust_network: Dict[str, float] = {}
        # 立场锚点: {topic: stance_value}
        self.stance_anchors: Dict[str, float] = {}
        # 品牌/实体印象: {entity_name: impression_score}
        self.impressions: Dict[str, float] = {}

    def store_fact(self, key: str, value: Any, confidence: float = 1.0):
        """存储事实，若已存在则按置信度加权更新"""
        if key in self.facts:
            old = self.facts[key]
            # 贝叶斯式更新
            old_conf = old.get("confidence", 0.5)
            old_val = old.get("value")
            # 简单加权平均（数值型）
            if isinstance(old_val, (int, float)) and isinstance(value, (int, float)):
                total_conf = old_conf + confidence
                new_val = (old_val * old_conf + value * confidence) / total_conf
                self.facts[key] = {
                    "value": new_val,
                    "confidence": min(1.0, total_conf * 0.8),
                    "updated_at": old.get("updated_at", 0),
                }
            else:
                # 非数值型，高置信度覆盖
                if confidence >= old_conf:
                    self.facts[key] = {"value": value, "confidence": confidence}
        else:
            self.facts[key] = {"value": value, "confidence": confidence}

    def retrieve_fact(self, key: str) -> Optional[Any]:
        """检索事实值"""
        entry = self.facts.get(key)
        if entry and entry.get("confidence", 0) > 0.3:
            return entry["value"]
        return None

    def update_trust(self, target_id: str, delta: float):
        """更新对某 Agent 的信任度"""
        current = self.trust_network.get(target_id, 0.5)
        self.trust_network[target_id] = max(0.0, min(1.0, current + delta))

    def get_trust(self, target_id: str) -> float:
        return self.trust_network.get(target_id, 0.3)

    def update_stance(self, topic: str, value: float):
        """更新对某话题的立场锚点"""
        current = self.stance_anchors.get(topic, 0.0)
        # 立场有惯性，小幅更新
        self.stance_anchors[topic] = current * 0.8 + value * 0.2

    def get_stance(self, topic: str) -> float:
        return self.stance_anchors.get(topic, 0.0)

    def update_impression(self, entity: str, delta: float):
        """更新对某实体/品牌的印象"""
        current = self.impressions.get(entity, 0.0)
        self.impressions[entity] = max(-1.0, min(1.0, current + delta))

    def get_impression(self, entity: str) -> float:
        return self.impressions.get(entity, 0.0)


class MemorySystem:
    """
    统一记忆管理系统
    协调 WM / EM / SM 三层，提供自动固化和检索接口
    """

    def __init__(self, wm_capacity: int = 9, em_max_size: int = 200):
        self.working = WorkingMemory(capacity=wm_capacity)
        self.episodic = EpisodicMemory(max_size=em_max_size)
        self.semantic = SemanticMemory()

    def perceive(self, item: MemoryItem, current_step: int):
        """
        感知输入：存入工作记忆，溢出项自动固化到情景记忆
        """
        evicted = self.working.store(item)
        if evicted and evicted.importance > 0.4:
            # 重要记忆固化到情景记忆
            self.episodic.store(evicted)

    def retrieve(self, query: str = "", current_step: int = 0,
                 current_emotion: str = "neutral", current_arousal: float = 0.0,
                 top_k: int = 5, layer: str = "all") -> List[MemoryItem]:
        """
        跨层检索记忆
        layer: "wm" | "em" | "all"
        """
        results = []
        if layer in ("wm", "all"):
            results.extend(self.working.retrieve(
                query, current_step, current_emotion, current_arousal, top_k
            ))
        if layer in ("em", "all"):
            results.extend(self.episodic.retrieve(
                current_step, current_emotion, current_arousal, top_k=top_k
            ))
        # 去重并按检索强度排序
        seen = set()
        unique = []
        for m in results:
            uid = (m.content, m.step)
            if uid not in seen:
                seen.add(uid)
                unique.append(m)
        unique.sort(
            key=lambda m: m.retrieval_strength(current_step, current_emotion, current_arousal),
            reverse=True,
        )
        return unique[:top_k]

    def retrieve_similar(self, reference: MemoryItem, current_step: int = 0,
                         top_k: int = 3) -> List[MemoryItem]:
        """检索相似经历"""
        return self.episodic.retrieve_similar_experiences(reference, current_step, top_k)

    def consolidate_to_semantic(self, current_step: int):
        """
        情景记忆 → 语义记忆的批量固化
        返回新生成的语义知识条目
        """
        patterns = self.episodic.consolidate(current_step)
        new_facts = []
        for p in patterns:
            key = f"pattern_{p['type']}_{p['emotion']}"
            self.semantic.store_fact(key, p, confidence=min(1.0, p["count"] * 0.2))
            new_facts.append(p)
        return new_facts

    def forget_old_memories(self, current_step: int, threshold: float = 0.05):
        """清理极低检索强度的旧记忆"""
        self.episodic.episodes = [
            m for m in self.episodic.episodes
            if m.recency_score(current_step) > threshold or m.importance > 0.7
        ]

    def get_context_for_decision(self, current_step: int,
                                  current_emotion: str, current_arousal: float) -> Dict[str, Any]:
        """
        为决策提供记忆上下文：工作记忆 + 最相关的情景记忆 + 关键语义知识
        """
        wm_items = self.working.retrieve(
            current_step=current_step,
            current_emotion=current_emotion,
            current_arousal=current_arousal,
            top_k=5,
        )
        em_items = self.episodic.retrieve(
            current_step=current_step,
            current_emotion=current_emotion,
            current_arousal=current_arousal,
            since_step=max(0, current_step - 48),  # 最近48步
            top_k=3,
        )
        return {
            "working_memory": [
                {"content": m.content, "type": m.memory_type, "step": m.step}
                for m in wm_items
            ],
            "episodic_memory": [
                {"content": m.content, "emotion": m.emotion_tag, "step": m.step}
                for m in em_items
            ],
            "trust_network": dict(self.semantic.trust_network),
            "stance_anchors": dict(self.semantic.stance_anchors),
        }
