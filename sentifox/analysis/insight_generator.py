"""
智能洞察生成模块
利用 LLM 分析舆情数据，生成洞察摘要
"""
from typing import List, Dict, Any, Optional
from datetime import datetime

from utils.llm_client import get_llm_client


INSIGHT_SYSTEM_PROMPT = """你是一位资深舆情分析师。请基于提供的舆情数据，生成一份简洁但专业的洞察摘要。

要求：
1. 总结当前舆情的主要特征和趋势
2. 指出需要重点关注的问题（如负面集中点、突发热点）
3. 给出简短的可操作建议
4. 使用中文，语气专业客观
5. 控制在 300-500 字
"""


def generate_insight(
    posts: List[Dict[str, Any]],
    sentiment_dist: Dict[str, int],
    platform_dist: Dict[str, int],
    influencers: Optional[List[Dict]] = None,
) -> str:
    """
    基于统计数据生成舆情洞察
    """
    if not posts:
        return "暂无数据，无法生成洞察。"

    total = sum(sentiment_dist.values())
    neg_ratio = sentiment_dist.get("negative", 0) / total * 100 if total > 0 else 0
    pos_ratio = sentiment_dist.get("positive", 0) / total * 100 if total > 0 else 0

    # 构建数据摘要
    data_summary = f"""舆情数据概览：
- 总帖子数: {total}
- 正面: {sentiment_dist.get('positive', 0)} ({pos_ratio:.1f}%)
- 负面: {sentiment_dist.get('negative', 0)} ({neg_ratio:.1f}%)
- 中性: {sentiment_dist.get('neutral', 0)}
- 平台分布: {platform_dist}
"""

    # 选取代表性帖子
    sample_posts = []
    for label in ["negative", "positive", "neutral"]:
        label_posts = [p for p in posts if p.get("sentiment_label") == label][:3]
        for p in label_posts:
            sample_posts.append(f"[{label}] {p.get('platform', '')} | {p.get('content', '')[:100]}")

    sample_text = "\n".join(sample_posts)

    # KOL 信息
    influencer_text = ""
    if influencers:
        influencer_text = "关键传播节点:\n" + "\n".join(
            [f"- {inf.get('author', '')} ({inf.get('platform', '')}): PageRank {inf.get('pagerank', 0)}" for inf in influencers[:5]]
        )

    prompt = f"""{data_summary}

代表性帖子:
{sample_text}

{influencer_text}

请基于以上数据生成舆情洞察摘要。"""

    messages = [
        {"role": "system", "content": INSIGHT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    llm_client = get_llm_client()
    return llm_client.chat_completion(messages, temperature=0.5, max_tokens=800)


def generate_topic_summary(
    topic_id: int,
    topic_info: Dict[str, Any],
    posts: List[Dict[str, Any]],
) -> str:
    """
    为单个话题生成摘要
    """
    keywords = ", ".join(topic_info.get("keywords", []))
    size = topic_info.get("size", 0)

    sample = [p.get("content", "")[:80] for p in posts[:5]]
    sample_text = "\n".join([f"- {s}" for s in sample])

    prompt = f"""话题 {topic_id} 摘要：
- 帖子数: {size}
- 关键词: {keywords}
- 代表性内容:
{sample_text}

请用 2-3 句话概括这个话题的核心讨论内容。"""

    messages = [
        {"role": "system", "content": "你是一位舆情分析专家，擅长提炼话题核心。请用中文简洁概括。"},
        {"role": "user", "content": prompt},
    ]

    llm_client = get_llm_client()
    return llm_client.chat_completion(messages, temperature=0.4, max_tokens=200)
