#!/usr/bin/env python3
"""
Sentifox 多智能体+时态图谱仿真引擎

核心设计理念：
- 每个 Agent 是自主决策实体
- 时态图谱是 Agent 共享的"世界记忆"
- Agent 感知来自图谱，动作修改图谱
- 传播是 Agent 群体互动的涌现结果
"""
import copy
import random
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import networkx as nx

from graph_analysis.temporal_graph import TemporalGraph, TemporalEntity, TemporalRelation
from graph_analysis.persona_agent import PersonaAgent, AgentAction
from graph_analysis.cognitive_agent import CognitiveAgent
from graph_analysis.temporal_engine import TemporalEngine
from graph_analysis.platform_dynamics import PlatformRule, get_platform_rule
from graph_analysis.interventions import Intervention, SeedInfectionIntervention
from graph_analysis.agent_communication import AgentCommunication, DiscussionThread


@dataclass
class SimulationSnapshot:
    """单步仿真快照"""
    step: int
    hour: int
    infected_count: int
    susceptible_count: int
    recovered_count: int
    active_agents: int
    new_actions: int
    new_relations: int
    expired_relations: int
    hot_events: List[Tuple[str, float]]
    sentiment_distribution: Dict[str, int]
    stance_distribution: Dict[str, int]
    top_influencers: List[Tuple[str, int]]
    # 新增：认知层指标
    polarization_index: float = 0.0
    echo_chamber_count: int = 0
    discussion_count: int = 0
    emotional_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    bridge_nodes: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class SimulationResult:
    """仿真结果"""
    timeline: List[SimulationSnapshot]
    agent_logs: Dict[str, List[Dict]]
    events: List[Dict]
    final_graph_stats: Dict[str, Any]
    interventions_applied: List[Dict]


class SentifoxSimulator:
    """
    Sentifox 多智能体仿真引擎
    """
    
    def __init__(self, temporal_graph: TemporalGraph,
                 agents: Dict[str, PersonaAgent],
                 platform_rules: Optional[Dict[str, PlatformRule]] = None,
                 config: Optional[Dict] = None):
        self.temporal_graph = temporal_graph
        self.agents = agents
        self.platform_rules = platform_rules or {}
        self.config = config or {}
        
        # 时间引擎
        self.temporal_engine = TemporalEngine(speed=self.config.get("speed", 1.0))
        
        # 记录
        self.timeline: List[SimulationSnapshot] = []
        self.agent_logs: Dict[str, List[Dict]] = defaultdict(list)
        self.events: List[Dict] = []
        self.interventions: List[Intervention] = []
        
        # 配置参数
        self.base_infection_prob = self.config.get("base_infection_prob", 0.15)
        self.decay_factor = self.config.get("decay_factor", 0.95)
        self.expire_threshold = self.config.get("expire_threshold", 0.1)
        self.echo_chamber_bonus = self.config.get("echo_chamber_bonus", 1.5)
        self.polarization_strength = self.config.get("polarization_strength", 0.15)
        
        # 认知层开关
        self.use_cognitive = self.config.get("use_cognitive", True)
        self.cognitive_agents: Dict[str, CognitiveAgent] = {}
        if self.use_cognitive:
            for aid, agent in self.agents.items():
                self.cognitive_agents[aid] = CognitiveAgent(agent)
        
        # 通信层
        self.communication = AgentCommunication()
        self.use_communication = self.config.get("use_communication", True)
    
    def add_intervention(self, intervention: Intervention):
        """添加干预措施"""
        self.interventions.append(intervention)
    
    def run(self, steps: int = 72, 
            realtime_callback: Optional[Callable] = None) -> SimulationResult:
        """
        运行完整仿真
        
        Args:
            steps: 仿真步数（每步=1小时）
            realtime_callback: 实时回调函数 (step, snapshot, actions) -> None
        
        Returns:
            SimulationResult
        """
        # 排序干预措施
        self.interventions.sort(key=lambda x: x.step)
        intervention_idx = 0
        
        for step in range(steps):
            self.temporal_engine.current_step = step
            current_hour = step % 24
            
            # ── Step 1: 应用干预 ──
            while intervention_idx < len(self.interventions) and \
                  self.interventions[intervention_idx].step == step:
                intervention = self.interventions[intervention_idx]
                effect = intervention.apply(self.temporal_graph, self.agents, step)
                self.events.append({
                    "step": step,
                    "type": "intervention",
                    "intervention": intervention.name,
                    "effect": effect,
                })
                intervention_idx += 1
            
            # ── Step 2: 环境感知 ──
            perceptions = self._agent_perceive(step)
            
            # ── Step 3: 独立决策 ──
            actions: List[Tuple[str, AgentAction]] = []
            for agent_id, agent in self.agents.items():
                # 检查 Agent 是否活跃
                if not self.temporal_engine.is_agent_active(agent.peak_hours):
                    continue
                
                # 已恢复的 Agent 不再行动
                if agent.state == "R":
                    continue
                
                # 获取平台规则
                platform_rule = self.platform_rules.get(agent.platform,
                    get_platform_rule(agent.platform))
                
                # 决策
                perception = perceptions.get(agent_id, {})
                if self.use_cognitive and agent_id in self.cognitive_agents:
                    cog_agent = self.cognitive_agents[agent_id]
                    action = cog_agent.perceive_and_decide(
                        self.temporal_graph, step, platform_rule
                    )
                else:
                    action = agent.decide(perception, self.temporal_graph, step, platform_rule)
                
                if action:
                    actions.append((agent_id, action))
                    self.agent_logs[agent_id].append({
                        "step": step,
                        "action": action.action_type,
                        "target": action.target_id,
                    })
            
            # ── Step 4: 动作执行 + 图谱更新 ──
            new_relations = 0
            for agent_id, action in actions:
                count = self._execute_action(agent_id, action, step)
                new_relations += count
            
            # ── Step 4.5: 通信层处理 ──
            comm_actions = []
            if self.use_communication and self.use_cognitive:
                # 信息级联：Agent 观察邻居行为
                for agent_id, cog in self.cognitive_agents.items():
                    if self.agents[agent_id].state == "R":
                        continue
                    neighbor_actions = [
                        (aid, act) for aid, act in actions
                        if aid != agent_id and aid in cog.belief.neighbor_states
                    ]
                    self.communication.observe_and_update(
                        cog, neighbor_actions, step, self.temporal_graph
                    )
                
                # 处理收件箱（直接消息产生的新行动）
                for agent_id, cog in self.cognitive_agents.items():
                    if self.agents[agent_id].state == "R":
                        continue
                    inbox_actions = self.communication.process_inbox(
                        cog, step, self.temporal_graph
                    )
                    for act in inbox_actions:
                        comm_actions.append((agent_id, act))
                        self.agent_logs[agent_id].append({
                            "step": step,
                            "action": act.action_type,
                            "target": act.target_id,
                            "source": "communication",
                        })
                
                # 执行通信产生的动作
                for agent_id, action in comm_actions:
                    count = self._execute_action(agent_id, action, step)
                    new_relations += count
            
            # ── Step 5: 传播动力学（感染扩散 + 情绪传染）──
            self._propagate_infection(step)
            
            # ── Step 6: 环境演化 ──
            expired = self.temporal_graph.evolve(step, self.decay_factor, self.expire_threshold)
            self._environment_evolve(step)
            
            # ── Step 7: 记录快照 ──
            snapshot = self._create_snapshot(step, len(actions), new_relations, expired)
            self.timeline.append(snapshot)
            
            # ── Step 8: 实时回调 ──
            if realtime_callback:
                try:
                    realtime_callback(step, snapshot, actions)
                except Exception:
                    pass
        
        return SimulationResult(
            timeline=self.timeline,
            agent_logs=dict(self.agent_logs),
            events=self.events,
            final_graph_stats=self.temporal_graph.get_stats(),
            interventions_applied=[
                {"name": i.name, "step": i.step, "description": i.description}
                for i in self.interventions
            ],
        )
    
    def _agent_perceive(self, step: int) -> Dict[str, Dict]:
        """Agent 环境感知"""
        perceptions = {}
        
        # 获取当前热点事件
        hot_events = self.temporal_graph.get_hot_events(step, top_n=5)
        
        # 获取社区情绪分布（基于 stance）
        stance_counts = defaultdict(int)
        for agent in self.agents.values():
            stance_counts[agent.stance] += 1
        total = len(self.agents)
        dominant_stance = max(stance_counts.keys(), key=lambda s: stance_counts[s]) if stance_counts else "neutral"
        
        for agent_id, agent in self.agents.items():
            # 查询最近的关系
            recent_relations = self.temporal_graph.query_relations(
                agent_id,
                at_step=step,
                since_step=max(0, step - 5)
            )
            
            # 过滤高权重关系
            significant_rels = [r for r in recent_relations if r.weight > 0.3]
            
            perceptions[agent_id] = {
                "recent_relations": significant_rels,
                "hot_events": hot_events,
                "community_sentiment": {
                    "dominant_stance": dominant_stance,
                    "stance_distribution": dict(stance_counts),
                },
                "attention_left": agent.attention,
                "current_hour": step % 24,
            }
        
        return perceptions
    
    def _execute_action(self, agent_id: str, action: AgentAction, step: int) -> int:
        """
        执行 Agent 动作，返回新增关系数
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return 0
        
        new_relations = 0
        
        if action.action_type == "repost":
            # 转发：创建 influence 关系
            target_id = action.target_id
            if target_id and target_id in self.agents:
                platform_rule = get_platform_rule(agent.platform)
                weight = agent.influence_base * platform_rule.virality_boost * 0.1
                
                self.temporal_graph.add_relation(TemporalRelation(
                    source_id=agent_id,
                    target_id=target_id,
                    relation_type="influence",
                    created_at=step,
                    weight=min(5.0, weight),
                    confidence=0.8,
                    evidence=[f"repost_step_{step}"],
                ))
                new_relations += 1
                
                # 记录事件
                self.events.append({
                    "step": step,
                    "type": "repost",
                    "source": agent_id,
                    "target": target_id,
                })
        
        elif action.action_type == "post":
            # 发布新内容：创建 Event 实体
            event_id = f"event_{step}_{agent_id}_{random.randint(1000, 9999)}"
            self.temporal_graph.add_entity(TemporalEntity(
                entity_id=event_id,
                entity_type="event",
                name=f"Post by {agent.name}",
                properties={
                    "creator": agent_id,
                    "sentiment": action.sentiment,
                    "platform": agent.platform,
                },
                first_seen=step,
            ))
            # 记录活动
            self.temporal_graph.record_event_activity(event_id, step)
            
            self.events.append({
                "step": step,
                "type": "post",
                "source": agent_id,
                "event_id": event_id,
            })
        
        elif action.action_type == "comment":
            # 评论：创建 mention 关系
            target_id = action.target_id
            if target_id and target_id in self.agents:
                self.temporal_graph.add_relation(TemporalRelation(
                    source_id=agent_id,
                    target_id=target_id,
                    relation_type="mention",
                    created_at=step,
                    weight=0.5,
                    confidence=0.7,
                ))
                new_relations += 1
        
        elif action.action_type == "stance_shift":
            # 立场转变
            old_stance = agent.stance
            agent.stance = action.new_stance
            
            # 更新图谱中的 stance 关系
            if old_stance == "support":
                # 删除旧的 support 关系
                old_rels = self.temporal_graph.query_relations(
                    agent_id, relation_type="support", at_step=step
                )
                for rel in old_rels:
                    rel.expires_at = step
            
            # 添加新的 stance 关系
            if action.new_stance in ("support", "oppose"):
                self.temporal_graph.add_relation(TemporalRelation(
                    source_id=agent_id,
                    target_id="community",
                    relation_type=action.new_stance,
                    created_at=step,
                    weight=1.0,
                    confidence=0.8,
                ))
                new_relations += 1
            
            self.events.append({
                "step": step,
                "type": "stance_shift",
                "agent": agent_id,
                "from": old_stance,
                "to": action.new_stance,
            })
        
        return new_relations
    
    def _propagate_infection(self, step: int):
        """
        基于时态图谱快照的传播动力学 + 情绪传染
        Agent 的感染状态在真实关系网络上扩散
        """
        # 获取当前快照
        snapshot = self.temporal_graph.get_snapshot_at(step)
        
        new_infections = []
        
        for agent_id, agent in self.agents.items():
            if agent.state != "I":
                continue
            
            # 获取该 Agent 的出边邻居（当前有效的关系）
            for _, target_id in snapshot.out_edges(agent_id):
                target_agent = self.agents.get(target_id)
                if not target_agent or target_agent.state != "S":
                    continue
                
                # 计算传播概率
                platform_rule = get_platform_rule(agent.platform)
                
                # 基础概率
                prob = self.base_infection_prob
                
                # 影响力加成（认知层：动态影响力）
                if self.use_cognitive and agent_id in self.cognitive_agents:
                    # 认知 Agent 的影响力受情绪和记忆调制
                    cog = self.cognitive_agents[agent_id]
                    influence = agent.influence_base * (1.0 + cog.emotion.arousal * 0.3)
                    prob *= (influence / 5.0)
                else:
                    prob *= (agent.influence_base / 5.0)
                
                # 平台传播速度
                prob *= platform_rule.spread_speed
                
                # 时间活跃度
                prob *= self.temporal_engine.get_activity_multiplier()
                
                # 情绪唤起
                if self.use_cognitive and agent_id in self.cognitive_agents:
                    arousal = self.cognitive_agents[agent_id].emotion.arousal
                    prob *= (1.0 + arousal)
                else:
                    prob *= (1.0 + agent.emotional_arousal)
                
                # 从时态图谱获取信任权重（认知层：从语义记忆获取）
                if self.use_cognitive and agent_id in self.cognitive_agents:
                    trust_weight = self.cognitive_agents[agent_id].memory.semantic.get_trust(target_id)
                else:
                    trust_rels = self.temporal_graph.query_relations(
                        agent_id, relation_type="trust", target_id=target_id, at_step=step
                    )
                    trust_weight = max((r.weight for r in trust_rels), default=0.3)
                prob *= trust_weight
                
                # 回声室效应
                if agent.stance == target_agent.stance and agent.stance != "neutral":
                    prob *= self.echo_chamber_bonus
                
                # 疲劳衰减
                prob *= max(0.2, 1.0 - agent.fatigue)
                
                # 情感传染（负面内容更易传播）
                sentiment = agent.sentiment_tendency
                if self.use_cognitive and agent_id in self.cognitive_agents:
                    sentiment = self.cognitive_agents[agent_id].emotion.valence
                if sentiment < -0.3:
                    prob *= 1.2
                
                prob = min(1.0, prob)
                
                if random.random() < prob:
                    # 认知层：使用认知 Agent 的感染记录（触发情绪和记忆）
                    if self.use_cognitive and target_id in self.cognitive_agents:
                        self.cognitive_agents[target_id].record_infection(
                            step, agent_id, source_sentiment=sentiment
                        )
                    else:
                        target_agent.record_infection(step, agent_id)
                    new_infections.append((agent_id, target_id))
                    
                    # 群体极化
                    if sentiment < -0.3:
                        target_agent.sentiment_tendency = max(-1.0,
                            target_agent.sentiment_tendency - self.polarization_strength)
        
        # 情绪传染：感染完成后，在邻居间传播情绪
        if self.use_cognitive:
            self._emotional_contagion(step, snapshot)
        
        # 记录传播事件
        for src, tgt in new_infections:
            self.events.append({
                "step": step,
                "type": "infection",
                "source": src,
                "target": tgt,
            })
    
    def _emotional_contagion(self, step: int, snapshot: nx.DiGraph):
        """
        情绪传染：感染 Agent 的情绪沿信任边传播给易感邻居
        """
        for agent_id, agent in self.agents.items():
            if agent.state != "I" or agent_id not in self.cognitive_agents:
                continue
            
            cog = self.cognitive_agents[agent_id]
            if cog.emotion.intensity < 0.3:
                continue  # 情绪不够强烈，不传染
            
            for _, target_id in snapshot.out_edges(agent_id):
                if target_id not in self.cognitive_agents:
                    continue
                target_cog = self.cognitive_agents[target_id]
                if target_cog.persona.state != "S":
                    continue
                
                # 获取信任权重
                trust_weight = cog.memory.semantic.get_trust(target_id)
                # 情绪传染
                target_cog.receive_emotional_influence(
                    source_emotion=cog.emotion.dominant,
                    source_intensity=cog.emotion.intensity,
                    source_valence=cog.emotion.valence,
                    trust_weight=trust_weight,
                )
    
    def _environment_evolve(self, step: int):
        """环境演化：Agent 状态更新"""
        for agent in self.agents.values():
            # 注意力恢复
            agent.attention = min(1.0, agent.attention + 0.05)
            
            # 疲劳自然恢复
            agent.fatigue = max(0.0, agent.fatigue - 0.02)
            
            # 情绪唤起衰减
            agent.emotional_arousal = max(0.0, agent.emotional_arousal - 0.05)
            
            # 感染 Agent 的恢复
            if agent.state == "I" and agent.infection_step >= 0:
                # 感染 5-15 步后可能恢复
                infected_duration = step - agent.infection_step
                if infected_duration > 5:
                    recovery_prob = 0.1 * (infected_duration - 5) / 10
                    if random.random() < recovery_prob:
                        agent.recover()
    
    def _create_snapshot(self, step: int, new_actions: int, 
                         new_relations: int, expired_relations: int) -> SimulationSnapshot:
        """创建统计快照（含认知层指标）"""
        infected = sum(1 for a in self.agents.values() if a.state == "I")
        susceptible = sum(1 for a in self.agents.values() if a.state == "S")
        recovered = sum(1 for a in self.agents.values() if a.state == "R")
        
        # 情感分布
        sent_dist = defaultdict(int)
        for a in self.agents.values():
            if a.sentiment_tendency > 0.2:
                sent_dist["positive"] += 1
            elif a.sentiment_tendency < -0.2:
                sent_dist["negative"] += 1
            else:
                sent_dist["neutral"] += 1
        
        # 立场分布
        stance_dist = defaultdict(int)
        for a in self.agents.values():
            stance_dist[a.stance] += 1
        
        # Top 影响力
        influencers = sorted(
            [(aid, a.amplification_count) for aid, a in self.agents.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        # ── 认知层指标 ──
        polarization_index = 0.0
        echo_chamber_count = 0
        emotional_states = {}
        bridge_nodes = []
        
        if self.use_cognitive:
            # 情绪状态采样（Top 10）
            for aid, cog in list(self.cognitive_agents.items())[:10]:
                emotional_states[aid] = cog.emotion.to_dict()
            
            # 回音室检测（每10步检测一次，减少开销）
            if step % 10 == 0:
                chambers = self.communication.detect_echo_chambers(
                    self.cognitive_agents, self.temporal_graph, step, min_size=3
                )
                echo_chamber_count = len(chambers)
            elif self.timeline:
                echo_chamber_count = self.timeline[-1].echo_chamber_count
            
            # 极化指数：从时态图谱嵌入计算
            if self.temporal_graph.use_embedding:
                support_ids = [aid for aid, a in self.agents.items() if a.stance == "support"]
                oppose_ids = [aid for aid, a in self.agents.items() if a.stance == "oppose"]
                if support_ids and oppose_ids:
                    polarization_index = self.temporal_graph.get_polarization_index(
                        support_ids, oppose_ids, step
                    )
                
                # 桥接节点
                bridge_nodes = self.temporal_graph.get_bridge_nodes(
                    support_ids[:30], oppose_ids[:30], step, top_k=3
                )
        
        # 讨论统计
        discussion_stats = self.communication.get_discussion_stats() if self.use_communication else {}
        
        return SimulationSnapshot(
            step=step,
            hour=step % 24,
            infected_count=infected,
            susceptible_count=susceptible,
            recovered_count=recovered,
            active_agents=infected + susceptible + recovered,
            new_actions=new_actions,
            new_relations=new_relations,
            expired_relations=expired_relations,
            hot_events=self.temporal_graph.get_hot_events(step, top_n=3),
            sentiment_distribution=dict(sent_dist),
            stance_distribution=dict(stance_dist),
            top_influencers=influencers,
            polarization_index=round(polarization_index, 3),
            echo_chamber_count=echo_chamber_count,
            discussion_count=discussion_stats.get("thread_count", 0),
            emotional_states=emotional_states,
            bridge_nodes=bridge_nodes,
        )
    
    def interview_agent(self, agent_id: str, question: str) -> str:
        """
        采访 Agent
        认知层下使用 BDI 推理链生成回答，否则回退到规则引擎
        """
        # 优先使用认知层
        if self.use_cognitive and agent_id in self.cognitive_agents:
            cog = self.cognitive_agents[agent_id]
            return cog.interview(question)
        
        # 回退到原有规则引擎
        agent = self.agents.get(agent_id)
        if not agent:
            return f"Agent {agent_id} 不存在"
        
        answers = []
        if "为什么" in question or "怎么" in question:
            if agent.stance == "support":
                answers.append(f"作为{agent.occupation}，我认为这件事有其积极的一面。")
            elif agent.stance == "oppose":
                answers.append(f"从我的角度看，这件事存在一些问题。我是{agent.mbti}型人格，比较关注细节。")
            else:
                answers.append("我还在观察这件事，目前保持中立态度。")
        
        if "转发" in question or "传播" in question:
            if agent.amplification_count > 5:
                answers.append(f"我确实转发了不少相关内容（{agent.amplification_count}次），因为我觉得这值得关注。")
            else:
                answers.append("我只转发了少量内容，主要还是在观望。")
        
        if "立场" in question or "态度" in question:
            answers.append(f"我目前的立场是{agent.stance}。")
        
        if not answers:
            answers.append(f"我是{agent.name}，在{agent.platform}上活跃。作为{agent.occupation}，我的MBTI是{agent.mbti}。")
        
        return " ".join(answers)
    
    @classmethod
    def from_data(cls, posts: List[Dict], edges: List[Dict],
                  config: Optional[Dict] = None) -> "SentifoxSimulator":
        """
        从 posts 和 edges 数据快速构建仿真器
        """
        from graph_analysis.temporal_graph import TemporalGraph
        from graph_analysis.persona_agent import create_agents_from_graph
        from graph_analysis.entity_extractor import EntityExtractor
        from graph_analysis.relation_extractor import RelationExtractor
        
        # 1. 构建时态图谱
        tg = TemporalGraph.from_posts_and_edges(posts, edges)
        
        # 2. 抽取实体和关系增强图谱
        extractor = EntityExtractor()
        entities = extractor.extract_from_posts(posts)
        for e in entities:
            tg.add_entity(TemporalEntity(
                entity_id=f"{e.entity_type}_{e.name}",
                entity_type=e.entity_type,
                name=e.name,
                properties={"source_post": e.source_post_id},
                first_seen=0,
            ))
        
        # 3. 创建 Agents
        platform_map = {}
        for p in posts:
            aid = p.get("author_id", "") or p.get("author", "")
            if aid:
                platform_map[aid] = p.get("platform", "微博")
        
        agents = create_agents_from_graph(tg, platform_map)
        
        return cls(tg, agents, config=config)
