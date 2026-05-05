#!/usr/bin/env python3
"""
PersonaAgent 人格化模型
时态图谱中 Person 实体的仿真化身
从图谱属性推断人格，支持自主决策
"""
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from graph_analysis.temporal_graph import TemporalGraph, TemporalEntity
from graph_analysis.platform_dynamics import PlatformRule, get_platform_rule


# ── MBTI 行为影响系数 ──────────────────────────────

MBTI_TRAITS = {
    # E/I: 外向/内向 → 影响传播主动性
    "E": {"post_boost": 1.4, "repost_boost": 1.3, "comment_boost": 1.2},
    "I": {"post_boost": 0.7, "repost_boost": 0.8, "comment_boost": 1.0},
    # N/S: 直觉/实感 → 影响对热点/熟人信息的敏感度
    "N": {"hot_event_sensitivity": 1.4, "social_sensitivity": 0.9},
    "S": {"hot_event_sensitivity": 0.8, "social_sensitivity": 1.3},
    # F/T: 情感/思考 → 影响情感内容传播倾向
    "F": {"emotional_boost": 1.5, "rational_filter": 0.7},
    "T": {"emotional_boost": 0.8, "rational_filter": 1.4},
    # J/P: 判断/感知 → 影响决策速度和立场稳定性
    "J": {"decision_speed": 1.2, "stance_stability": 1.3},
    "P": {"decision_speed": 1.4, "stance_stability": 0.7},
}

MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]

# 平台用户画像分布（用于随机采样）
PLATFORM_USER_PROFILE = {
    "微博": {"age_median": 28, "mbti_bias": ["E", "N", "F", "P"]},
    "知乎": {"age_median": 30, "mbti_bias": ["I", "N", "T", "J"]},
    "小红书": {"age_median": 25, "mbti_bias": ["E", "S", "F", "P"]},
    "抖音": {"age_median": 24, "mbti_bias": ["E", "S", "F", "P"]},
    "新闻": {"age_median": 38, "mbti_bias": ["I", "S", "T", "J"]},
    "论坛": {"age_median": 30, "mbti_bias": ["I", "N", "T", "P"]},
}

OCCUPATIONS = ["学生", "白领", "自由职业", "媒体从业者", "KOL", "公务员", 
               "工程师", "教师", "医生", "销售", "退休人员"]


@dataclass
class AgentAction:
    """Agent 动作"""
    action_type: str   # post / repost / comment / stance_shift / ignore
    target_id: str = ""
    content: str = ""
    sentiment: float = 0.0
    new_stance: str = ""


@dataclass
class PersonaAgent:
    """时态图谱中 Person 实体的仿真化身"""
    entity_id: str
    name: str
    # 人口统计学
    age: int = 30
    gender: str = "M"
    mbti: str = "INTJ"
    occupation: str = "白领"
    # 社交属性
    platform: str = "微博"
    affiliated_orgs: List[str] = field(default_factory=list)
    trusted_agents: Dict[str, float] = field(default_factory=dict)
    influenced_by: Dict[str, float] = field(default_factory=dict)
    # 传播动力学
    influence_base: float = 5.0
    receptivity: float = 0.5
    stance: str = "neutral"   # support / oppose / neutral
    sentiment_tendency: float = 0.0  # -1.0 ~ 1.0
    # 活跃模式
    activity_pattern: str = "evening_peak"
    peak_hours: List[int] = field(default_factory=list)
    # 动态状态
    attention: float = 1.0
    fatigue: float = 0.0
    emotional_arousal: float = 0.0
    memory: List[Dict] = field(default_factory=list)
    # 传播状态
    state: str = "S"           # S=susceptible, I=infected, R=recovered
    infection_step: int = -1
    amplification_count: int = 0
    
    def __post_init__(self):
        if not self.peak_hours:
            self.peak_hours = [19, 20, 21, 22]
    
    def get_mbti_traits(self) -> Dict[str, float]:
        """获取 MBTI 行为影响系数"""
        traits = {}
        for letter in self.mbti:
            if letter in MBTI_TRAITS:
                traits.update(MBTI_TRAITS[letter])
        return traits
    
    def decide(self, perception: Dict, temporal_graph: TemporalGraph,
               step: int, platform_rule: PlatformRule) -> Optional[AgentAction]:
        """
        Agent 独立决策
        
        Returns:
            AgentAction 或 None（不行动）
        """
        if self.attention <= 0:
            return None
        
        traits = self.get_mbti_traits()
        
        # 感知信息
        hot_events = perception.get("hot_events", [])
        recent_relations = perception.get("recent_relations", [])
        community_sentiment = perception.get("community_sentiment", {})
        
        # 基础概率（受 MBTI 影响）
        base_post = 0.02 * traits.get("post_boost", 1.0)
        base_repost = 0.08 * traits.get("repost_boost", 1.0)
        base_comment = 0.05 * traits.get("comment_boost", 1.0)
        
        # 热点事件敏感度
        if hot_events:
            hot_event_boost = traits.get("hot_event_sensitivity", 1.0)
            base_post *= (1 + len(hot_events) * 0.2 * hot_event_boost)
            base_repost *= (1 + len(hot_events) * 0.3 * hot_event_boost)
        
        # 社交关系影响
        if recent_relations:
            social_boost = traits.get("social_sensitivity", 1.0)
            base_repost *= (1 + len(recent_relations) * 0.05 * social_boost)
        
        # 情绪唤起影响
        if self.emotional_arousal > 0.5:
            emotional_boost = traits.get("emotional_boost", 1.0)
            base_post *= (1 + self.emotional_arousal * emotional_boost)
            base_repost *= (1 + self.emotional_arousal * emotional_boost * 0.8)
        
        # 疲劳衰减
        fatigue_factor = max(0.2, 1.0 - self.fatigue)
        base_post *= fatigue_factor
        base_repost *= fatigue_factor
        base_comment *= fatigue_factor
        
        # 立场转变概率
        stance_shift_prob = 0.005
        if community_sentiment:
            # 如果社区情绪与个人立场相反，更容易转变
            community_stance = community_sentiment.get("dominant_stance", "neutral")
            if community_stance != self.stance and community_stance != "neutral":
                stance_shift_prob *= 3.0
        stance_shift_prob *= (1.0 - traits.get("stance_stability", 1.0) * 0.3)
        
        # 决策
        r = random.random()
        cumulative = 0.0
        
        cumulative += base_post
        if r < cumulative:
            return self._action_post(hot_events, platform_rule)
        
        cumulative += base_repost
        if r < cumulative:
            return self._action_repost(recent_relations, platform_rule)
        
        cumulative += base_comment
        if r < cumulative:
            return self._action_comment(recent_relations)
        
        cumulative += stance_shift_prob
        if r < cumulative:
            return self._action_stance_shift(community_sentiment)
        
        return None  # ignore
    
    def _action_post(self, hot_events, platform_rule) -> AgentAction:
        """发布新内容"""
        self.attention -= 0.15
        self.fatigue += 0.05
        return AgentAction(
            action_type="post",
            content=f"post_{random.randint(1000, 9999)}",
            sentiment=self.sentiment_tendency,
        )
    
    def _action_repost(self, recent_relations, platform_rule) -> Optional[AgentAction]:
        """转发"""
        if not recent_relations:
            return None
        
        # 选择权重最高的关系目标
        best_rel = max(recent_relations, key=lambda r: r.weight)
        self.attention -= 0.1
        self.fatigue += 0.03
        self.amplification_count += 1
        
        return AgentAction(
            action_type="repost",
            target_id=best_rel.target_id if best_rel.source_id == self.entity_id else best_rel.source_id,
            sentiment=self.sentiment_tendency,
        )
    
    def _action_comment(self, recent_relations) -> Optional[AgentAction]:
        """评论"""
        if not recent_relations:
            return None
        
        target_rel = random.choice(recent_relations)
        self.attention -= 0.08
        self.fatigue += 0.02
        
        return AgentAction(
            action_type="comment",
            target_id=target_rel.target_id if target_rel.source_id == self.entity_id else target_rel.source_id,
            sentiment=self.sentiment_tendency,
        )
    
    def _action_stance_shift(self, community_sentiment) -> Optional[AgentAction]:
        """立场转变"""
        new_stance = community_sentiment.get("dominant_stance", self.stance)
        if new_stance == self.stance or new_stance == "neutral":
            return None
        
        return AgentAction(
            action_type="stance_shift",
            new_stance=new_stance,
        )
    
    def record_infection(self, step: int, source_id: str):
        """记录被感染"""
        self.state = "I"
        self.infection_step = step
        self.emotional_arousal = min(1.0, self.emotional_arousal + 0.3)
        self.memory.append({
            "type": "infection",
            "step": step,
            "source": source_id,
        })
    
    def recover(self):
        """恢复/遗忘"""
        self.state = "R"
        self.emotional_arousal = max(0.0, self.emotional_arousal - 0.5)


def infer_persona_from_graph(entity: TemporalEntity, temporal_graph: TemporalGraph,
                             platform: str = "微博") -> PersonaAgent:
    """
    从时态图谱推断 Agent 人格属性
    """
    agent = PersonaAgent(
        entity_id=entity.entity_id,
        name=entity.name,
        platform=platform,
    )
    
    # 从图谱关系推断属性
    relations = temporal_graph.query_relations(entity.entity_id)
    
    # influence 出度 -> 影响力
    influence_rels = [r for r in relations if r.relation_type == "influence"]
    agent.influence_base = min(10.0, 2.0 + len(influence_rels) * 0.5)
    
    # trust 关系 -> 接受度
    trust_rels = [r for r in relations if r.relation_type == "trust"]
    if trust_rels:
        avg_trust = sum(r.weight for r in trust_rels) / len(trust_rels)
        agent.receptivity = min(1.0, 0.3 + avg_trust * 0.3)
    
    # support/oppose 关系 -> 立场
    support_count = sum(1 for r in relations if r.relation_type == "support")
    oppose_count = sum(1 for r in relations if r.relation_type == "oppose")
    if support_count > oppose_count * 1.5:
        agent.stance = "support"
        agent.sentiment_tendency = 0.3
    elif oppose_count > support_count * 1.5:
        agent.stance = "oppose"
        agent.sentiment_tendency = -0.3
    
    # belong_to -> 归属组织
    belong_rels = [r for r in relations if r.relation_type == "belong_to"]
    agent.affiliated_orgs = [r.target_id for r in belong_rels]
    
    # 平台用户画像采样
    profile = PLATFORM_USER_PROFILE.get(platform, PLATFORM_USER_PROFILE["微博"])
    
    # 年龄
    age_min, age_max, age_median = profile["age_median"] - 10, profile["age_median"] + 15, profile["age_median"]
    agent.age = random.randint(age_min, age_max)
    
    # MBTI（带平台偏置）
    bias = profile.get("mbti_bias", [])
    agent.mbti = _generate_mbti(bias)
    
    # 性别
    agent.gender = random.choice(["M", "F"])
    
    # 职业（基于平台）
    if platform == "知乎":
        agent.occupation = random.choice(["工程师", "教师", "白领", "公务员"])
    elif platform == "抖音":
        agent.occupation = random.choice(["学生", "自由职业", "销售", "KOL"])
    elif platform == "小红书":
        agent.occupation = random.choice(["学生", "白领", "自由职业", "媒体从业者"])
    else:
        agent.occupation = random.choice(OCCUPATIONS)
    
    # 活跃模式（基于年龄）
    if agent.age < 25:
        agent.activity_pattern = random.choice(["night_owl", "evening_peak", "all_day"])
    elif agent.age > 45:
        agent.activity_pattern = random.choice(["early_bird", "work_hours"])
    else:
        agent.activity_pattern = random.choice(["work_hours", "evening_peak"])
    
    # peak_hours 基于活跃模式
    pattern_hours = {
        "night_owl": [22, 23, 0, 1],
        "early_bird": [6, 7, 8, 9],
        "work_hours": [9, 10, 11, 14, 15, 16],
        "evening_peak": [19, 20, 21, 22],
        "all_day": list(range(8, 24)),
    }
    agent.peak_hours = pattern_hours.get(agent.activity_pattern, [19, 20, 21, 22])
    
    return agent


def _generate_mbti(bias: List[str]) -> str:
    """生成 MBTI（带偏置）"""
    mbti = ""
    dimensions = [
        ("E", "I"),  # 外向/内向
        ("N", "S"),  # 直觉/实感
        ("F", "T"),  # 情感/思考
        ("J", "P"),  # 判断/感知
    ]
    
    for i, (a, b) in enumerate(dimensions):
        if i < len(bias):
            # 70% 概率选择偏置方向
            mbti += a if random.random() < 0.7 else b
        else:
            mbti += random.choice([a, b])
    
    return mbti


def create_agents_from_graph(temporal_graph: TemporalGraph, 
                             platform_map: Optional[Dict[str, str]] = None) -> Dict[str, PersonaAgent]:
    """
    从时态图谱的所有 Person 实体创建 Agent
    
    Returns:
        {entity_id: PersonaAgent}
    """
    agents = {}
    persons = temporal_graph.get_entities_by_type("person")
    
    for person in persons:
        platform = "微博"
        if platform_map and person.entity_id in platform_map:
            platform = platform_map[person.entity_id]
        elif person.properties.get("platform"):
            platform = person.properties["platform"]
        
        agent = infer_persona_from_graph(person, temporal_graph, platform)
        agents[person.entity_id] = agent
    
    return agents
