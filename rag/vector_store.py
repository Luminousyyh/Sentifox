"""
RAG 向量存储模块
使用 ChromaDB 作为轻量级向量数据库
"""
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from config import RAG_CONFIG


class VectorStore:
    """向量存储管理器"""

    def __init__(self, collection_name: Optional[str] = None, persist_path: Optional[str] = None):
        self.collection_name = collection_name or RAG_CONFIG["collection_name"]
        self.persist_path = persist_path or RAG_CONFIG["vector_db_path"]
        os.makedirs(self.persist_path, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """批量添加文档到向量库"""
        if not ids or not documents:
            return

        # 去重：过滤已存在的 id
        existing = self.collection.get(ids=ids)["ids"]
        existing_set = set(existing)

        new_ids = []
        new_docs = []
        new_embeds = [] if embeddings else None
        new_metas = [] if metadatas else None

        for i, doc_id in enumerate(ids):
            if doc_id in existing_set:
                continue
            new_ids.append(doc_id)
            new_docs.append(documents[i])
            if embeddings:
                new_embeds.append(embeddings[i])
            if metadatas:
                new_metas.append(metadatas[i])

        if not new_ids:
            return

        kwargs = {
            "ids": new_ids,
            "documents": new_docs,
        }
        if new_embeds:
            kwargs["embeddings"] = new_embeds
        if new_metas:
            kwargs["metadatas"] = new_metas

        self.collection.add(**kwargs)

    def query(
        self,
        query_embeddings: Optional[List[List[float]]] = None,
        query_texts: Optional[List[str]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        相似度检索
        :param query_embeddings: 查询向量
        :param query_texts: 查询文本（如未提供 embedding，则使用 Chroma 内置 embedding）
        :param n_results: 返回结果数
        :param where: metadata 过滤条件，如 {"platform": "微博"}
        """
        kwargs = {"n_results": n_results}
        if query_embeddings:
            kwargs["query_embeddings"] = query_embeddings
        elif query_texts:
            kwargs["query_texts"] = query_texts
        else:
            raise ValueError("必须提供 query_embeddings 或 query_texts")

        if where:
            kwargs["where"] = where

        return self.collection.query(**kwargs)

    def delete_by_ids(self, ids: List[str]) -> None:
        """按 ID 删除文档"""
        self.collection.delete(ids=ids)

    def get_count(self) -> int:
        """获取文档总数"""
        return self.collection.count()

    def clear(self) -> None:
        """清空集合"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


# 全局单例
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """获取向量存储单例"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
