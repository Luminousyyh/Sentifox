#!/usr/bin/env python3
"""
时态知识图谱引擎（Zep-like）
sentifox 内置时态图谱层，不依赖外部服务

核心能力：
1. 实体/关系存储，支持多类型实体
2. 时态关系：每条关系带 (created_at, expires_at, confidence, weight)
3. 时间快照：get_snapshot_at(step) 返回指定时刻的有效关系子图
4. 动态演化：关系权重衰减、过期关系清理
5. 热点事件检测
"""
import uuid
import random
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import networkx as nx

from graph_analysis.temporal_embedding import TemporalEmbedding


@dataclass
class TemporalEntity:
    """时态图谱中的实体"""
    entity_id: str
    entity_type: str        # person / organization / brand / product / event / location
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    first_seen: int = 0     # 首次出现的时间步
    last_seen: int = 0      # 最近出现的时间步
    confidence: float = 1.0


@dataclass
class TemporalRelation:
    """时态图谱中的关系 —— Zep 核心：关系带时间维度"""
    relation_id: str = ""  # 空字符串时 add_relation 会自动生成
    source_id: str = ""
    target_id: str = ""
    relation_type: str = "mention"  # support / oppose / mention / belong_to / trust / influence / follow
    created_at: int = 0
    expires_at: Optional[int] = None
    confidence: float = 1.0
    weight: float = 1.0
    evidence: List[str] = field(default_factory=list)
    
    def is_valid_at(self, step: int) -> bool:
        """Zep 核心能力：查询某时刻关系是否有效"""
        return self.created_at <= step and (self.expires_at is None or self.expires_at > step)
    
    def time_to_live(self, step: int) -> Optional[int]:
        """返回剩余存活步数（None=永久）"""
        if self.expires_at is None:
            return None
        return max(0, self.expires_at - step)


class TemporalGraph:
    """
    时态知识图谱 —— sentifox 内置 Zep-like 引擎
    
    设计要点：
    - 所有实体和关系存储在内存 Dict 中（规模可控，舆情数据通常 <10K 实体）
    - 快照按需生成并缓存（LRU 策略）
    - 支持从 NetworkX 图导入和导出
    """
    
    def __init__(self, use_embedding: bool = True, embedding_dim: int = 64):
        self.entities: Dict[str, TemporalEntity] = {}
        self.relations: Dict[str, TemporalRelation] = {}
        # 实体 -> 关系ID 的倒排索引
        self.entity_relations: Dict[str, Set[str]] = defaultdict(set)
        # 快照缓存
        self._snapshot_cache: Dict[int, nx.DiGraph] = {}
        self._cache_max_size = 50
        # 热点事件追踪
        self._event_activity: Dict[str, List[int]] = defaultdict(list)  # event_id -> [活跃步数列表]
        # 时序嵌入
        self.use_embedding = use_embedding
        self.embedding = TemporalEmbedding(dim=embedding_dim) if use_embedding else None
    
    # ── 实体管理 ──────────────────────────────
    
    def add_entity(self, entity: TemporalEntity) -> str:
        """添加或更新实体"""
        if not entity.entity_id:
            entity.entity_id = f"{entity.entity_type}_{uuid.uuid4().hex[:8]}"
        self.entities[entity.entity_id] = entity
        return entity.entity_id
    
    def get_entity(self, entity_id: str) -> Optional[TemporalEntity]:
        """获取实体"""
        return self.entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: str) -> List[TemporalEntity]:
        """按类型获取实体"""
        return [e for e in self.entities.values() if e.entity_type == entity_type]
    
    def remove_entity(self, entity_id: str):
        """删除实体及其所有关系"""
        if entity_id not in self.entities:
            return
        # 删除相关关系
        rel_ids = list(self.entity_relations.get(entity_id, set()))
        for rid in rel_ids:
            self.remove_relation(rid)
        del self.entities[entity_id]
        if entity_id in self.entity_relations:
            del self.entity_relations[entity_id]
    
    def update_entity_property(self, entity_id: str, key: str, value: Any):
        """更新实体属性"""
        if entity_id in self.entities:
            self.entities[entity_id].properties[key] = value
    
    # ── 关系管理 ──────────────────────────────
    
    def add_relation(self, relation: TemporalRelation) -> str:
        """添加或更新关系"""
        if not relation.relation_id:
            relation.relation_id = f"rel_{uuid.uuid4().hex[:8]}"
        
        self.relations[relation.relation_id] = relation
        self.entity_relations[relation.source_id].add(relation.relation_id)
        self.entity_relations[relation.target_id].add(relation.relation_id)
        
        # 更新实体 last_seen
        if relation.source_id in self.entities:
            self.entities[relation.source_id].last_seen = max(
                self.entities[relation.source_id].last_seen, relation.created_at
            )
        if relation.target_id in self.entities:
            self.entities[relation.target_id].last_seen = max(
                self.entities[relation.target_id].last_seen, relation.created_at
            )
        
        # 时序嵌入更新
        if self.use_embedding and self.embedding is not None:
            self.embedding.process_relation(
                relation.relation_type,
                relation.source_id,
                relation.target_id,
                relation.created_at,
                relation.weight,
            )
        
        # 清理快照缓存（关系变化后缓存失效）
        self._invalidate_cache()
        
        return relation.relation_id
    
    def get_relation(self, relation_id: str) -> Optional[TemporalRelation]:
        """获取关系"""
        return self.relations.get(relation_id)
    
    def remove_relation(self, relation_id: str):
        """删除关系"""
        if relation_id not in self.relations:
            return
        rel = self.relations[relation_id]
        self.entity_relations[rel.source_id].discard(relation_id)
        self.entity_relations[rel.target_id].discard(relation_id)
        del self.relations[relation_id]
        self._invalidate_cache()
    
    def update_relation_weight(self, relation_id: str, delta: float, step: int):
        """动态更新关系权重"""
        if relation_id in self.relations:
            self.relations[relation_id].weight = max(0.0, min(10.0, 
                self.relations[relation_id].weight + delta))
            self.relations[relation_id].last_updated = step
            self._invalidate_cache()
    
    # ── 时态查询 ──────────────────────────────
    
    def query_relations(self, entity_id: str, relation_type: Optional[str] = None,
                        at_step: Optional[int] = None, target_id: Optional[str] = None,
                        since_step: Optional[int] = None) -> List[TemporalRelation]:
        """
        查询实体的关系，支持多维度过滤
        
        Args:
            entity_id: 源实体ID
            relation_type: 关系类型过滤
            at_step: 指定时间步（只返回该时刻有效的关系）
            target_id: 目标实体过滤
            since_step: 只返回 created_at >= since_step 的关系
        """
        result = []
        for rid in self.entity_relations.get(entity_id, set()):
            rel = self.relations.get(rid)
            if not rel:
                continue
            
            # 类型过滤
            if relation_type and rel.relation_type != relation_type:
                continue
            
            # 目标过滤
            if target_id and rel.target_id != target_id and rel.source_id != target_id:
                continue
            
            # 时态过滤
            if at_step is not None and not rel.is_valid_at(at_step):
                continue
            
            # 时间窗口过滤
            if since_step is not None and rel.created_at < since_step:
                continue
            
            result.append(rel)
        
        # 按权重降序
        result.sort(key=lambda r: r.weight, reverse=True)
        return result
    
    def get_snapshot_at(self, step: int) -> nx.DiGraph:
        """
        获取指定时间步的图谱快照 —— 只包含该时刻有效的关系
        
        Returns:
            NetworkX DiGraph，节点带实体属性，边带关系属性
        """
        # 检查缓存
        if step in self._snapshot_cache:
            return self._snapshot_cache[step]
        
        G = nx.DiGraph()
        
        # 添加所有实体作为节点
        for eid, entity in self.entities.items():
            G.add_node(eid, **{
                "name": entity.name,
                "entity_type": entity.entity_type,
                **entity.properties,
            })
        
        # 添加有效关系作为边
        for rid, rel in self.relations.items():
            if rel.is_valid_at(step):
                if rel.source_id in self.entities and rel.target_id in self.entities:
                    G.add_edge(
                        rel.source_id, rel.target_id,
                        relation_type=rel.relation_type,
                        weight=rel.weight,
                        confidence=rel.confidence,
                        created_at=rel.created_at,
                        relation_id=rid,
                    )
        
        # 缓存（LRU）
        self._snapshot_cache[step] = G
        if len(self._snapshot_cache) > self._cache_max_size:
            oldest = min(self._snapshot_cache.keys())
            del self._snapshot_cache[oldest]
        
        return G
    
    def get_valid_relations_at(self, step: int) -> List[TemporalRelation]:
        """获取指定时间步所有有效关系"""
        return [rel for rel in self.relations.values() if rel.is_valid_at(step)]
    
    # ── 图谱演化 ──────────────────────────────
    
    def evolve(self, step: int, decay_factor: float = 0.95, 
               expire_threshold: float = 0.1):
        """
        每步演化：关系权重衰减、弱关系过期
        
        Args:
            step: 当前时间步
            decay_factor: 衰减系数（默认 0.95，每步衰减 5%）
            expire_threshold: 过期阈值，低于此值的关系标记为过期
        """
        expired_count = 0
        for rel in list(self.relations.values()):
            if not rel.is_valid_at(step):
                continue
            
            # 信任/影响关系衰减
            if rel.relation_type in ("trust", "influence"):
                rel.weight *= decay_factor
                if rel.weight < expire_threshold:
                    rel.expires_at = step
                    expired_count += 1
            
            # mention 关系半衰期更短
            elif rel.relation_type == "mention":
                rel.weight *= (decay_factor ** 2)
                if rel.weight < expire_threshold:
                    rel.expires_at = step
                    expired_count += 1
        
        # 清理过期缓存
        self._invalidate_cache()
        
        return expired_count
    
    def _invalidate_cache(self):
        """清除快照缓存"""
        self._snapshot_cache.clear()
    
    # ── 热点事件 ──────────────────────────────
    
    def record_event_activity(self, event_id: str, step: int):
        """记录事件在某步的活跃度"""
        self._event_activity[event_id].append(step)
        # 只保留最近 20 步的记录
        self._event_activity[event_id] = self._event_activity[event_id][-20:]
    
    def get_hot_events(self, step: int, top_n: int = 5, 
                       window: int = 5) -> List[Tuple[str, float]]:
        """
        获取当前热点事件
        
        Returns:
            [(event_id, 热度分数), ...]
        """
        scores = {}
        for event_id, activity in self._event_activity.items():
            # 计算窗口内的活跃度
            recent = [s for s in activity if step - window <= s <= step]
            # 热度 = 最近活跃度 + 时间衰减
            score = len(recent) * (1.0 + 0.1 * len(recent))
            if score > 0:
                scores[event_id] = score
        
        # 按热度排序
        sorted_events = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_events[:top_n]
    
    # ── 统计信息 ──────────────────────────────
    
    def get_stats(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        entity_types = defaultdict(int)
        for e in self.entities.values():
            entity_types[e.entity_type] += 1
        
        relation_types = defaultdict(int)
        for r in self.relations.values():
            relation_types[r.relation_type] += 1
        
        stats = {
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "entity_types": dict(entity_types),
            "relation_types": dict(relation_types),
        }
        
        # 嵌入统计
        if self.use_embedding and self.embedding is not None:
            stats["embedding_dim"] = self.embedding.dim
            stats["embedded_entities"] = len(self.embedding.memory)
        
        return stats
    
    # ── 时态嵌入接口 ──────────────────────────────
    
    def get_entity_embedding(self, entity_id: str, step: int = 0) -> Optional[List[float]]:
        """获取实体时态嵌入"""
        if not self.use_embedding or self.embedding is None:
            return None
        emb = self.embedding.get_embedding(entity_id, step)
        return emb.tolist()
    
    def predict_future_links(self, entity_id: str, step: int = 0,
                              top_k: int = 5) -> List[Tuple[str, float]]:
        """预测该实体最可能建立的新关系"""
        if not self.use_embedding or self.embedding is None:
            return []
        candidates = [eid for eid in self.entities.keys() if eid != entity_id]
        return self.embedding.predict_links(entity_id, candidates, step, top_k)
    
    def find_similar_entities(self, entity_id: str, step: int = 0,
                               top_k: int = 5) -> List[Tuple[str, float]]:
        """基于嵌入找出最相似的实体"""
        if not self.use_embedding or self.embedding is None:
            return []
        candidates = [eid for eid in self.entities.keys() if eid != entity_id]
        return self.embedding.find_similar_entities(entity_id, candidates, step, top_k)
    
    def get_bridge_nodes(self, group_a: List[str], group_b: List[str],
                         step: int = 0, top_k: int = 3) -> List[Tuple[str, float]]:
        """识别跨群体桥接节点"""
        if not self.use_embedding or self.embedding is None:
            return []
        return self.embedding.get_bridge_nodes(group_a, group_b, step, top_k)
    
    def get_polarization_index(self, group_a: List[str], group_b: List[str],
                                step: int = 0) -> float:
        """计算群体极化指数"""
        if not self.use_embedding or self.embedding is None:
            return 0.0
        return self.embedding.get_polarization_index(group_a, group_b, step)
    
    # ── 导入/导出 ──────────────────────────────
    
    @classmethod
    def from_posts_and_edges(cls, posts: List[Dict], edges: List[Dict],
                             entity_extractor=None, relation_extractor=None) -> "TemporalGraph":
        """
        从 posts 和 edges 构建时态图谱
        
        简化版本：将作者作为 Person 实体，从 edges 创建关系
        完整版本应调用 entity_extractor 和 relation_extractor
        """
        tg = cls()
        
        # 创建 Person 实体（作者）
        seen_authors = set()
        for post in posts:
            author = post.get("author", "")
            author_id = post.get("author_id", "") or author
            if author and author_id not in seen_authors:
                seen_authors.add(author_id)
                tg.add_entity(TemporalEntity(
                    entity_id=author_id,
                    entity_type="person",
                    name=author,
                    properties={
                        "platform": post.get("platform", ""),
                        "post_count": 1,
                    },
                    first_seen=0,
                ))
            elif author_id in tg.entities:
                # 更新帖子计数
                tg.entities[author_id].properties["post_count"] = \
                    tg.entities[author_id].properties.get("post_count", 0) + 1
        
        # 从 edges 创建关系
        for edge in edges:
            source = edge.get("source_id", "")
            target = edge.get("target_id", "")
            rel_type = edge.get("relation_type", "mention")
            weight = edge.get("weight", 1.0)
            
            if source and target and source in tg.entities and target in tg.entities:
                tg.add_relation(TemporalRelation(
                    source_id=source,
                    target_id=target,
                    relation_type=rel_type,
                    created_at=0,
                    weight=weight,
                    confidence=0.8,
                ))
        
        return tg
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "entities": [
                {
                    "entity_id": e.entity_id,
                    "entity_type": e.entity_type,
                    "name": e.name,
                    "properties": e.properties,
                    "first_seen": e.first_seen,
                    "last_seen": e.last_seen,
                    "confidence": e.confidence,
                }
                for e in self.entities.values()
            ],
            "relations": [
                {
                    "relation_id": r.relation_id,
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "relation_type": r.relation_type,
                    "created_at": r.created_at,
                    "expires_at": r.expires_at,
                    "confidence": r.confidence,
                    "weight": r.weight,
                    "evidence": r.evidence,
                }
                for r in self.relations.values()
            ],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalGraph":
        """从字典反序列化"""
        tg = cls()
        for e_data in data.get("entities", []):
            tg.add_entity(TemporalEntity(**e_data))
        for r_data in data.get("relations", []):
            tg.add_relation(TemporalRelation(**r_data))
        return tg
