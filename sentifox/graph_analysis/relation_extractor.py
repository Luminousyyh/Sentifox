#!/usr/bin/env python3
"""
关系抽取器
从帖子内容和图边中抽取实体间的语义关系
"""
import re
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass

from graph_analysis.temporal_graph import TemporalEntity, TemporalRelation


@dataclass
class ExtractedRelation:
    """抽取出的关系"""
    source_id: str
    target_id: str
    relation_type: str
    confidence: float = 1.0
    evidence: List[str] = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


class RelationExtractor:
    """关系抽取器"""
    
    # 立场关键词
    SUPPORT_WORDS = {"支持", "赞同", "认可", "好评", "推荐", "优秀", "满意", "不错", "好"}
    OPPOSE_WORDS = {"反对", "批评", "质疑", "差评", "失望", "糟糕", "恶心", "骗人", "假", "差"}
    
    # 归属模式
    BELONG_PATTERNS = [
        re.compile(r'(\w+)员工'), re.compile(r'(\w+)用户'),
        re.compile(r'(\w+)粉丝'), re.compile(r'(\w+)车主'),
    ]
    
    def __init__(self):
        pass
    
    def extract_from_edges(self, edges: List[Dict]) -> List[ExtractedRelation]:
        """
        从 graph_edges 表抽取基础关系
        edges: [{source_id, target_id, relation_type, weight, ...}, ...]
        """
        relations = []
        for edge in edges:
            rel_type = edge.get("relation_type", "mention")
            # 映射到标准关系类型
            mapped_type = self._map_relation_type(rel_type)
            relations.append(ExtractedRelation(
                source_id=edge.get("source_id", ""),
                target_id=edge.get("target_id", ""),
                relation_type=mapped_type,
                confidence=edge.get("weight", 1.0),
                evidence=[edge.get("post_id", "")],
            ))
        return relations
    
    def extract_sentiment_relations(self, posts: List[Dict], 
                                    entities: List[TemporalEntity]) -> List[ExtractedRelation]:
        """
        从帖子情感和实体共现中抽取 support/oppose 关系
        """
        relations = []
        entity_names = {e.name: e.entity_id for e in entities}
        
        for post in posts:
            text = post.get("content", "")
            author_id = post.get("author_id", "") or post.get("author", "")
            sentiment = post.get("sentiment_label", "neutral")
            post_id = post.get("post_id", "")
            
            # 检测帖子中提到的实体
            mentioned = []
            for name, eid in entity_names.items():
                if name in text:
                    mentioned.append((eid, name))
            
            if len(mentioned) >= 1 and author_id:
                # 作者对提及实体的立场
                for eid, name in mentioned:
                    if sentiment == "positive":
                        rel_type = "support"
                        conf = 0.7
                    elif sentiment == "negative":
                        rel_type = "oppose"
                        conf = 0.7
                    else:
                        continue
                    
                    relations.append(ExtractedRelation(
                        source_id=author_id,
                        target_id=eid,
                        relation_type=rel_type,
                        confidence=conf,
                        evidence=[post_id],
                    ))
        
        return relations
    
    def extract_belong_relations(self, posts: List[Dict],
                                 persons: List[TemporalEntity],
                                 orgs: List[TemporalEntity]) -> List[ExtractedRelation]:
        """
        从文本模式中抽取 belong_to 关系
        如 "XX公司员工" -> Person 属于 Organization
        """
        relations = []
        org_names = {o.name: o.entity_id for o in orgs}
        
        for post in posts:
            text = post.get("content", "")
            post_id = post.get("post_id", "")
            
            for pattern in self.BELONG_PATTERNS:
                for match in pattern.finditer(text):
                    org_name = match.group(1)
                    if org_name in org_names:
                        # 找到最近的 Person（简化处理：用作者）
                        author_id = post.get("author_id", "") or post.get("author", "")
                        if author_id:
                            relations.append(ExtractedRelation(
                                source_id=author_id,
                                target_id=org_names[org_name],
                                relation_type="belong_to",
                                confidence=0.6,
                                evidence=[post_id],
                            ))
        
        return relations
    
    def extract_trust_from_interactions(self, edges: List[Dict],
                                        min_weight: float = 2.0) -> List[ExtractedRelation]:
        """
        从高互动频率的边中推断 trust 关系
        """
        relations = []
        # 统计每对节点的互动次数
        interaction_counts: Dict[Tuple[str, str], int] = {}
        for edge in edges:
            key = (edge.get("source_id", ""), edge.get("target_id", ""))
            interaction_counts[key] = interaction_counts.get(key, 0) + 1
        
        # 互动次数多的推断为信任关系
        for (src, tgt), count in interaction_counts.items():
            if count >= min_weight:
                relations.append(ExtractedRelation(
                    source_id=src,
                    target_id=tgt,
                    relation_type="trust",
                    confidence=min(1.0, count / 10.0),
                ))
        
        return relations
    
    def _map_relation_type(self, raw_type: str) -> str:
        """将原始关系类型映射到标准类型"""
        mapping = {
            "repost": "influence",
            "comment": "mention",
            "mention": "mention",
            "similar": "trust",
            "tag": "mention",
            "follow": "follow",
        }
        return mapping.get(raw_type.lower(), "mention")
    
    def merge_relations(self, relations: List[ExtractedRelation]) -> List[ExtractedRelation]:
        """合并重复关系（相同 source-target-type），权重取平均"""
        grouped: Dict[Tuple[str, str, str], List[ExtractedRelation]] = {}
        for rel in relations:
            key = (rel.source_id, rel.target_id, rel.relation_type)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(rel)
        
        merged = []
        for key, rels in grouped.items():
            avg_conf = sum(r.confidence for r in rels) / len(rels)
            all_evidence = []
            for r in rels:
                all_evidence.extend(r.evidence)
            
            merged.append(ExtractedRelation(
                source_id=key[0],
                target_id=key[1],
                relation_type=key[2],
                confidence=avg_conf,
                evidence=list(set(all_evidence))[:10],  # 去重，最多10条证据
            ))
        
        return merged
