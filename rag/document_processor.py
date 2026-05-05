"""
RAG 文档处理模块
将舆情帖子转换为向量库文档，并管理同步
"""
from typing import List, Dict, Any, Optional
from datetime import datetime

from rag.vector_store import get_vector_store
from utils.llm_client import get_llm_client


def format_document(post: Dict[str, Any]) -> str:
    """
    将帖子格式化为文档字符串
    """
    platform = post.get("platform", "")
    author = post.get("author", "")
    content = post.get("content", "")
    sentiment = post.get("sentiment_label", "")
    topic_id = post.get("topic_id", "")

    doc = f"[{platform}]"
    if author:
        doc += f" {author}:"
    doc += f" {content}"
    if sentiment:
        doc += f" [情感:{sentiment}]"
    if topic_id is not None:
        doc += f" [话题:{topic_id}]"
    return doc


def sync_posts_to_vector_store(
    posts: List[Dict[str, Any]],
    batch_size: int = 100,
) -> int:
    """
    将帖子列表同步到向量库
    :return: 实际写入的文档数
    """
    if not posts:
        return 0

    store = get_vector_store()
    llm_client = get_llm_client()

    ids = []
    documents = []
    metadatas = []

    for post in posts:
        post_id = post.get("post_id", "")
        if not post_id:
            continue

        ids.append(str(post_id))
        documents.append(format_document(post))

        # Build metadata, ChromaDB only accepts str/int/float/bool/None
        meta = {
            "platform": str(post.get("platform", "")),
            "sentiment_label": str(post.get("sentiment_label", "")),
            "author": str(post.get("author", "")),
            "likes": int(post.get("likes", 0) or 0),
        }
        topic_id = post.get("topic_id")
        if topic_id is not None:
            meta["topic_id"] = int(topic_id)
        pt = post.get("publish_time")
        if isinstance(pt, datetime):
            meta["publish_time"] = pt.isoformat()
        elif pt:
            meta["publish_time"] = str(pt)
        metadatas.append(meta)

    # 批量获取 embedding (火山引擎限制 max 10)
    embeddings = llm_client.embeddings(documents, batch_size=10)

    # 写入向量库
    store.add_documents(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(ids)


def sync_all_posts_from_db(
    platforms: Optional[List[str]] = None,
    limit: int = 5000,
) -> int:
    """
    从数据库全量同步帖子到向量库
    """
    from utils.database import get_posts
    posts = get_posts(platforms=platforms, limit=limit)
    return sync_posts_to_vector_store(posts, batch_size=10)
