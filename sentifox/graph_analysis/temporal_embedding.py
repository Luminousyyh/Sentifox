#!/usr/bin/env python3
"""
Temporal Embedding — TGN (Temporal Graph Networks) 简化版

纯 numpy 实现，无需 PyTorch。

核心概念：
- 每个实体维护一个时序嵌入向量 z(t) 和一个记忆向量 s(t)
- 关系事件触发消息传递，更新两端实体的记忆
- 嵌入 = 记忆 + 时间衰减
- 注意力权重从嵌入计算（GAT-style）
- 支持链路预测：预测下一步最可能形成的关系

参考：Rossi et al. "Temporal Graph Networks for Deep Learning on Dynamic Graphs" (2020)
"""
import math
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


class TemporalEmbedding:
    """
    时态图嵌入引擎
    """

    def __init__(self, dim: int = 64, seed: int = 42):
        """
        Args:
            dim: 嵌入维度
            seed: 随机种子（用于初始化）
        """
        self.dim = dim
        random.seed(seed)
        np.random.seed(seed)

        # 实体记忆向量 s(t)：聚合历史交互信息
        self.memory: Dict[str, np.ndarray] = {}
        # 实体嵌入向量 z(t)：对外使用的表示
        self.embedding: Dict[str, np.ndarray] = {}
        # 最后更新时间
        self.last_update: Dict[str, int] = {}
        # 交互计数
        self.interaction_count: Dict[str, int] = defaultdict(int)

        # 消息函数参数（简化版：线性变换）
        # W_msg: 关系类型 -> 变换矩阵 (dim, dim)
        self._init_message_params()

        # 注意力参数
        self.attention_vector = np.random.randn(dim) * 0.01

        # 时间编码参数
        self.time_encoding_dim = 16
        self.time_encoding_scale = np.random.randn(self.time_encoding_dim) * 0.01

    def _init_message_params(self):
        """初始化各关系类型的消息变换矩阵"""
        relation_types = ["support", "oppose", "mention", "belong_to", "trust", "influence", "follow"]
        self.W_msg: Dict[str, np.ndarray] = {}
        for rel_type in relation_types:
            # Xavier-like 初始化
            self.W_msg[rel_type] = np.random.randn(self.dim, self.dim) / math.sqrt(self.dim)
        # 默认矩阵
        self.W_msg["default"] = np.random.randn(self.dim, self.dim) / math.sqrt(self.dim)

    def _get_W(self, relation_type: str) -> np.ndarray:
        return self.W_msg.get(relation_type, self.W_msg["default"])

    def initialize_entity(self, entity_id: str):
        """初始化实体嵌入"""
        if entity_id not in self.memory:
            init_vec = np.random.randn(self.dim) * 0.01
            self.memory[entity_id] = init_vec
            self.embedding[entity_id] = init_vec.copy()
            self.last_update[entity_id] = 0
            self.interaction_count[entity_id] = 0

    def _time_encoding(self, delta_t: int) -> np.ndarray:
        """
        时间差编码（正弦位置编码简化版）
        delta_t: 时间差（步数）
        Returns: (time_encoding_dim,) 向量
        """
        enc = np.zeros(self.time_encoding_dim)
        for i in range(self.time_encoding_dim):
            freq = 1.0 / (10000 ** (i / self.time_encoding_dim))
            if i % 2 == 0:
                enc[i] = math.sin(delta_t * freq)
            else:
                enc[i] = math.cos(delta_t * freq)
        return enc

    def _apply_time_decay(self, entity_id: str, current_step: int) -> np.ndarray:
        """
        应用时间衰减到嵌入
        长时间未更新的实体，其嵌入会向零衰减
        """
        emb = self.embedding.get(entity_id)
        if emb is None:
            return np.zeros(self.dim)
        last_t = self.last_update.get(entity_id, current_step)
        delta = max(0, current_step - last_t)
        # 指数衰减
        decay = math.exp(-0.02 * delta)
        return emb * decay

    def compute_message(self, relation_type: str,
                        source_emb: np.ndarray,
                        target_emb: np.ndarray,
                        timestamp: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算关系事件的消息
        
        Returns:
            (source_message, target_message)
        """
        W = self._get_W(relation_type)
        # 消息 = W @ [source_emb || target_emb || time_enc] 的简化版
        # 这里简化为：消息 = W @ (source_emb + target_emb) / 2
        combined = (source_emb + target_emb) / 2.0
        # 防止过大值
        combined = np.tanh(combined)
        msg = W @ combined
        # 两端收到相同的消息（可扩展为不对称）
        return msg, msg

    def update_memory(self, entity_id: str, message: np.ndarray, timestamp: int):
        """
        用消息更新实体的记忆向量
        使用 GRU 简化版更新：s_new = (1 - gate) * s_old + gate * message
        """
        self.initialize_entity(entity_id)

        s_old = self.memory[entity_id]
        # 更新门：交互越多，越倾向于接受新信息
        gate = 1.0 / (1.0 + math.exp(-self.interaction_count[entity_id] * 0.1))
        gate = min(0.8, max(0.1, gate))  # 限制在 [0.1, 0.8]

        s_new = (1 - gate) * s_old + gate * message
        # L2 归一化，防止向量爆炸
        norm = np.linalg.norm(s_new)
        if norm > 10.0:
            s_new = s_new / norm * 10.0

        self.memory[entity_id] = s_new
        self.last_update[entity_id] = timestamp
        self.interaction_count[entity_id] += 1

        # 嵌入 = 记忆 + 小幅随机扰动（模拟不确定性）
        noise = np.random.randn(self.dim) * 0.001
        self.embedding[entity_id] = s_new + noise

    def process_relation(self, relation_type: str,
                         source_id: str, target_id: str,
                         timestamp: int, weight: float = 1.0):
        """
        处理一条关系事件，更新两端实体的嵌入
        """
        self.initialize_entity(source_id)
        self.initialize_entity(target_id)

        # 获取当前嵌入（带时间衰减）
        s_emb = self._apply_time_decay(source_id, timestamp)
        t_emb = self._apply_time_decay(target_id, timestamp)

        # 计算消息
        s_msg, t_msg = self.compute_message(relation_type, s_emb, t_emb, timestamp)

        # 权重调制消息强度
        s_msg *= weight
        t_msg *= weight

        # 更新记忆
        self.update_memory(source_id, s_msg, timestamp)
        self.update_memory(target_id, t_msg, timestamp)

    def get_embedding(self, entity_id: str, current_step: int = 0) -> np.ndarray:
        """获取实体在当前时刻的嵌入（带时间衰减）"""
        self.initialize_entity(entity_id)
        return self._apply_time_decay(entity_id, current_step)

    def compute_attention(self, source_id: str, target_id: str,
                          current_step: int = 0) -> float:
        """
        GAT-style 注意力计算
        attention(i, j) = LeakyReLU(a^T [W·z_i || W·z_j])
        返回 softmax-ready 的分数（未归一化）
        """
        z_i = self.get_embedding(source_id, current_step)
        z_j = self.get_embedding(target_id, current_step)

        # 拼接 + 注意力向量点积
        concat = np.concatenate([z_i, z_j])
        # 注意力向量需要扩展到 2*dim
        if len(self.attention_vector) != len(concat):
            self.attention_vector = np.random.randn(len(concat)) * 0.01

        score = np.dot(self.attention_vector, concat)
        # LeakyReLU
        score = max(0.01 * score, score)
        return float(score)

    def compute_attention_batch(self, source_id: str,
                                 target_ids: List[str],
                                 current_step: int = 0) -> np.ndarray:
        """批量计算注意力（向量化，高效）"""
        z_i = self.get_embedding(source_id, current_step)
        scores = []
        for tid in target_ids:
            z_j = self.get_embedding(tid, current_step)
            concat = np.concatenate([z_i, z_j])
            if len(self.attention_vector) != len(concat):
                self.attention_vector = np.random.randn(len(concat)) * 0.01
            score = np.dot(self.attention_vector, concat)
            score = max(0.01 * score, score)
            scores.append(score)
        return np.array(scores)

    def predict_links(self, source_id: str,
                      candidate_ids: List[str],
                      current_step: int = 0,
                      top_k: int = 5) -> List[Tuple[str, float]]:
        """
        链路预测：预测 source_id 最可能与哪些候选实体建立关系
        
        Returns:
            [(candidate_id, score), ...] 按分数降序
        """
        if not candidate_ids:
            return []

        scores = self.compute_attention_batch(source_id, candidate_ids, current_step)
        # Softmax 归一化
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / (np.sum(exp_scores) + 1e-10)

        ranked = sorted(
            [(cid, float(p)) for cid, p in zip(candidate_ids, probs)],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def find_similar_entities(self, entity_id: str,
                               candidate_ids: List[str],
                               current_step: int = 0,
                               top_k: int = 5) -> List[Tuple[str, float]]:
        """
        基于嵌入相似度找出最相似的实体（余弦相似度）
        """
        emb_i = self.get_embedding(entity_id, current_step)
        norm_i = np.linalg.norm(emb_i)
        if norm_i < 1e-8:
            return []

        similarities = []
        for cid in candidate_ids:
            if cid == entity_id:
                continue
            emb_j = self.get_embedding(cid, current_step)
            norm_j = np.linalg.norm(emb_j)
            if norm_j < 1e-8:
                continue
            sim = np.dot(emb_i, emb_j) / (norm_i * norm_j)
            similarities.append((cid, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_bridge_nodes(self, community_a: List[str],
                         community_b: List[str],
                         current_step: int = 0,
                         top_k: int = 3) -> List[Tuple[str, float]]:
        """
        识别跨社区桥接节点：对两个社区都有高注意力的节点
        """
        bridge_scores = []
        all_nodes = list(self.memory.keys())

        for node in all_nodes:
            if node in community_a or node in community_b:
                continue
            # 计算该节点对两个社区的平均注意力
            att_a = []
            att_b = []
            for a in community_a[:20]:  # 限制计算量
                try:
                    att_a.append(self.compute_attention(node, a, current_step))
                except Exception:
                    pass
            for b in community_b[:20]:
                try:
                    att_b.append(self.compute_attention(node, b, current_step))
                except Exception:
                    pass
            if att_a and att_b:
                score = (np.mean(att_a) + np.mean(att_b)) / 2.0
                bridge_scores.append((node, float(score)))

        bridge_scores.sort(key=lambda x: x[1], reverse=True)
        return bridge_scores[:top_k]

    def get_polarization_index(self, group_a: List[str],
                                group_b: List[str],
                                current_step: int = 0) -> float:
        """
        计算群体极化指数：两组实体嵌入的分离程度
        返回值 0~1，越接近 1 表示越极化
        """
        if not group_a or not group_b:
            return 0.0

        embs_a = [self.get_embedding(eid, current_step) for eid in group_a if eid in self.embedding]
        embs_b = [self.get_embedding(eid, current_step) for eid in group_b if eid in self.embedding]

        if not embs_a or not embs_b:
            return 0.0

        centroid_a = np.mean(embs_a, axis=0)
        centroid_b = np.mean(embs_b, axis=0)

        # 组内距离
        intra_a = np.mean([np.linalg.norm(e - centroid_a) for e in embs_a])
        intra_b = np.mean([np.linalg.norm(e - centroid_b) for e in embs_b])

        # 组间距离
        inter = np.linalg.norm(centroid_a - centroid_b)

        # 极化指数 = 组间距离 / (组间距离 + 平均组内距离)
        polarization = inter / (inter + (intra_a + intra_b) / 2.0 + 1e-10)
        return float(polarization)

    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "dim": self.dim,
            "memory": {k: v.tolist() for k, v in self.memory.items()},
            "last_update": dict(self.last_update),
            "interaction_count": dict(self.interaction_count),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalEmbedding":
        """反序列化"""
        inst = cls(dim=data.get("dim", 64))
        for k, v in data.get("memory", {}).items():
            inst.memory[k] = np.array(v)
            inst.embedding[k] = np.array(v)
        inst.last_update = data.get("last_update", {})
        inst.interaction_count = defaultdict(int, data.get("interaction_count", {}))
        return inst
