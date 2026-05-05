"""
RAG 引擎模块
检索 + 生成 完整流程
"""
from typing import List, Dict, Any, Optional

from rag.vector_store import get_vector_store
from utils.llm_client import get_llm_client


RAG_SYSTEM_PROMPT = """你是一位专业的舆情分析专家。请基于以下检索到的舆情数据，回答用户的问题。

要求：
1. 回答必须基于提供的检索内容，不要编造信息
2. 如果检索内容不足以回答问题，请明确说明
3. 引用具体的数据和来源支持你的观点
4. 保持客观、专业的分析语气
5. 回答请使用中文
"""


def retrieve(
    query: str,
    top_k: int = 5,
    platform_filter: Optional[str] = None,
    sentiment_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    检索相关文档
    :return: 检索结果列表，每个包含 document, metadata, distance
    """
    store = get_vector_store()
    llm_client = get_llm_client()

    # 获取查询向量
    query_embedding = llm_client.embeddings([query])

    # 构建过滤条件
    where = {}
    if platform_filter:
        where["platform"] = platform_filter
    if sentiment_filter:
        where["sentiment_label"] = sentiment_filter

    results = store.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where if where else None,
    )

    # 格式化结果
    formatted = []
    if results and results.get("documents"):
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        dists = results["distances"][0] if results.get("distances") else [0] * len(docs)

        for doc, meta, dist in zip(docs, metas, dists):
            formatted.append({
                "document": doc,
                "metadata": meta,
                "distance": dist,
            })

    return formatted


def generate_answer(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    基于检索结果生成答案
    :return: {"answer": "...", "sources": [...]}
    """
    if not retrieved_docs:
        return {
            "answer": "未检索到相关舆情数据，无法回答该问题。请尝试生成更多数据后提问。",
            "sources": [],
        }

    # 构建上下文
    context_parts = []
    sources = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[{i}] {doc['document']}")
        meta = doc.get("metadata", {})
        sources.append({
            "index": i,
            "platform": meta.get("platform", ""),
            "author": meta.get("author", ""),
            "sentiment": meta.get("sentiment_label", ""),
            "publish_time": meta.get("publish_time", ""),
            "relevance": f"{1 - doc.get('distance', 0):.2f}",
        })

    context = "\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"基于以下舆情数据回答问题：\n\n{context}\n\n问题：{query}\n\n请提供详细的分析回答，并标注引用来源（如[1], [2]）。"},
    ]

    llm_client = get_llm_client()
    answer = llm_client.chat_completion(messages, model=model, temperature=0.5)

    return {
        "answer": answer,
        "sources": sources,
    }


def ask(
    query: str,
    top_k: int = 5,
    platform_filter: Optional[str] = None,
    sentiment_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    RAG 完整流程：检索 + 生成
    """
    docs = retrieve(query, top_k=top_k, platform_filter=platform_filter, sentiment_filter=sentiment_filter)
    return generate_answer(query, docs)
