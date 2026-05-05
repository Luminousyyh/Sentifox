#!/usr/bin/env python3
"""
干预系统
模拟舆情管控措施的效果
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from graph_analysis.temporal_graph import TemporalGraph, TemporalEntity, TemporalRelation
from graph_analysis.persona_agent import PersonaAgent


@dataclass
class Intervention:
    """干预措施基类"""
    name: str
    step: int
    description: str = ""
    
    def apply(self, temporal_graph: TemporalGraph, 
              agents: Dict[str, PersonaAgent], step: int) -> Dict[str, Any]:
        """应用干预，返回效果描述"""
        raise NotImplementedError


class DeletionIntervention(Intervention):
    """删除关键节点（模拟删帖/封号）"""
    
    def __init__(self, target_id: str, step: int, cascade_prevention: float = 0.7):
        super().__init__(
            name="节点删除",
            step=step,
            description=f"删除节点 {target_id} 及其出边"
        )
        self.target_id = target_id
        self.cascade_prevention = cascade_prevention
    
    def apply(self, temporal_graph: TemporalGraph,
              agents: Dict[str, PersonaAgent], step: int) -> Dict[str, Any]:
        # 1. 删除实体
        temporal_graph.remove_entity(self.target_id)
        
        # 2. 删除 Agent
        if self.target_id in agents:
            del agents[self.target_id]
        
        # 3. 统计被切断的关系数
        return {
            "type": "deletion",
            "target": self.target_id,
            "effect": f"已删除节点 {self.target_id}",
        }


class InfluencerIntervention(Intervention):
    """KOL 正面引导"""
    
    def __init__(self, target_id: str, step: int, 
                 new_stance: str = "support", influence_boost: float = 2.0,
                 sentiment_shift: float = 0.3):
        super().__init__(
            name="KOL引导",
            step=step,
            description=f"引导 KOL {target_id} 转向 {new_stance}"
        )
        self.target_id = target_id
        self.new_stance = new_stance
        self.influence_boost = influence_boost
        self.sentiment_shift = sentiment_shift
    
    def apply(self, temporal_graph: TemporalGraph,
              agents: Dict[str, PersonaAgent], step: int) -> Dict[str, Any]:
        # 1. 修改 Agent 属性
        if self.target_id in agents:
            agent = agents[self.target_id]
            old_stance = agent.stance
            agent.stance = self.new_stance
            agent.sentiment_tendency = min(1.0, agent.sentiment_tendency + self.sentiment_shift)
            agent.influence_base *= self.influence_boost
        
        # 2. 添加引导关系
        temporal_graph.add_relation(TemporalRelation(
            source_id="official_guidance",
            target_id=self.target_id,
            relation_type="influence",
            created_at=step,
            weight=self.influence_boost,
            confidence=0.9,
        ))
        
        return {
            "type": "influencer",
            "target": self.target_id,
            "effect": f"KOL {self.target_id} 立场转向 {self.new_stance}",
        }


class PlatformIntervention(Intervention):
    """平台限流/降权"""
    
    def __init__(self, platform: str, step: int,
                 spread_speed_multiplier: float = 0.3,
                 virality_threshold: float = 0.8):
        super().__init__(
            name="平台限流",
            step=step,
            description=f"对 {platform} 进行限流"
        )
        self.platform = platform
        self.spread_speed_multiplier = spread_speed_multiplier
        self.virality_threshold = virality_threshold
    
    def apply(self, temporal_graph: TemporalGraph,
              agents: Dict[str, PersonaAgent], step: int) -> Dict[str, Any]:
        # 找到该平台的所有 Agent，降低其影响力
        affected = 0
        for agent in agents.values():
            if agent.platform == self.platform:
                agent.influence_base *= self.spread_speed_multiplier
                affected += 1
        
        return {
            "type": "platform",
            "target": self.platform,
            "effect": f"{self.platform} 平台 {affected} 个 Agent 影响力降低",
        }


class OfficialResponseIntervention(Intervention):
    """官方回应"""
    
    def __init__(self, step: int, global_sentiment_shift: float = 0.2,
                 neutralize_negative: float = 0.5):
        super().__init__(
            name="官方回应",
            step=step,
            description="发布官方声明"
        )
        self.global_sentiment_shift = global_sentiment_shift
        self.neutralize_negative = neutralize_negative
    
    def apply(self, temporal_graph: TemporalGraph,
              agents: Dict[str, PersonaAgent], step: int) -> Dict[str, Any]:
        # 1. 添加官方实体（如果不存在）
        official_id = "official_account"
        if official_id not in temporal_graph.entities:
            temporal_graph.add_entity(TemporalEntity(
                entity_id=official_id,
                entity_type="organization",
                name="官方账号",
                properties={"role": "official", "authority": 10.0},
                first_seen=step,
            ))
        
        # 2. 对所有负面倾向 Agent 施加正面影响
        affected = 0
        for agent in agents.values():
            if agent.sentiment_tendency < -0.2:
                # 添加 influence 关系
                temporal_graph.add_relation(TemporalRelation(
                    source_id=official_id,
                    target_id=agent.entity_id,
                    relation_type="influence",
                    created_at=step,
                    weight=self.global_sentiment_shift * 2,
                    confidence=0.9,
                ))
                # 轻微修正情感倾向
                agent.sentiment_tendency = min(1.0, 
                    agent.sentiment_tendency + self.global_sentiment_shift)
                affected += 1
        
        return {
            "type": "official",
            "target": "global",
            "effect": f"官方回应影响 {affected} 个负面 Agent",
        }


class SeedInfectionIntervention(Intervention):
    """初始种子感染（用于启动传播模拟）"""
    
    def __init__(self, seed_ids: List[str], step: int = 0):
        super().__init__(
            name="种子感染",
            step=step,
            description=f"设置种子节点: {seed_ids}"
        )
        self.seed_ids = seed_ids
    
    def apply(self, temporal_graph: TemporalGraph,
              agents: Dict[str, PersonaAgent], step: int) -> Dict[str, Any]:
        infected = 0
        for seed_id in self.seed_ids:
            if seed_id in agents:
                agents[seed_id].state = "I"
                agents[seed_id].infection_step = step
                agents[seed_id].emotional_arousal = 0.5
                infected += 1
        
        return {
            "type": "seed",
            "target": str(self.seed_ids),
            "effect": f"{infected} 个种子节点已感染",
        }


# ── 干预解析工具 ──────────────────────────────

def parse_intervention(spec: str) -> Intervention:
    """
    从字符串解析干预措施
    
    格式: "type:target@step[:param=value,...]"
    
    示例:
        "delete:user_001@step10"
        "kol:user_002@step20:positive"
        "platform:微博@step15"
        "official@step30"
    """
    parts = spec.split(":", 1)
    intervention_type = parts[0].lower()
    
    if intervention_type == "delete":
        # delete:target@step
        rest = parts[1] if len(parts) > 1 else ""
        target, step_str = rest.split("@")
        step = int(step_str.replace("step", ""))
        return DeletionIntervention(target.strip(), step)
    
    elif intervention_type == "kol":
        # kol:target@step[:positive]
        rest = parts[1] if len(parts) > 1 else ""
        target_step = rest.split(":", 1)
        target_step_part = target_step[0]
        target, step_str = target_step_part.split("@")
        step = int(step_str.replace("step", ""))
        stance = "positive" if len(target_step) > 1 else "support"
        return InfluencerIntervention(target.strip(), step, new_stance=stance)
    
    elif intervention_type == "platform":
        # platform:平台名@step
        rest = parts[1] if len(parts) > 1 else ""
        platform, step_str = rest.split("@")
        step = int(step_str.replace("step", ""))
        return PlatformIntervention(platform.strip(), step)
    
    elif intervention_type == "official":
        # official@step
        step_str = parts[1] if len(parts) > 1 else "0"
        step = int(step_str.replace("step", "").replace("@", ""))
        return OfficialResponseIntervention(step)
    
    else:
        raise ValueError(f"未知干预类型: {intervention_type}")


def compare_scenarios(baseline: Dict, intervention: Dict) -> Dict[str, Any]:
    """
    对比基准场景和干预场景
    
    Returns:
        对比报告字典
    """
    def get_final(timeline: List[Dict], key: str):
        return timeline[-1].get(key, 0) if timeline else 0
    
    def get_peak(timeline: List[Dict], key: str):
        return max((t.get(key, 0) for t in timeline), default=0) if timeline else 0
    
    b_infected = get_final(baseline.get("timeline", []), "infected_count")
    i_infected = get_final(intervention.get("timeline", []), "infected_count")
    
    b_peak = get_peak(baseline.get("timeline", []), "infected_count")
    i_peak = get_peak(intervention.get("timeline", []), "infected_count")
    
    b_negative = get_final(baseline.get("timeline", []), "negative_ratio")
    i_negative = get_final(intervention.get("timeline", []), "negative_ratio")
    
    return {
        "infected_total": {"baseline": b_infected, "intervention": i_infected,
                          "reduction": b_infected - i_infected,
                          "reduction_pct": (b_infected - i_infected) / b_infected * 100 if b_infected > 0 else 0},
        "peak_infected": {"baseline": b_peak, "intervention": i_peak},
        "negative_ratio": {"baseline": b_negative, "intervention": i_negative},
        "intervention_effectiveness": "有效" if i_infected < b_infected * 0.8 else "中等" if i_infected < b_infected else "有限",
    }
