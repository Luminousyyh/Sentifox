"""
热点话题聚类模块
使用 jieba 分词 + TF-IDF + KMeans 聚类
"""
import re
import jieba
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter

from config import TOPIC_CLUSTER_NUM, TOPIC_MIN_DF, TOPIC_MAX_DF, STOPWORDS_PATH


# 默认停用词（内置 fallback）
DEFAULT_STOPWORDS = set([
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也",
    "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那",
    "吗", "吧", "啊", "呢", "哦", "嗯", "还", "让", "但", "与", "而", "为", "被", "把", "给",
    "之", "将", "并", "从", "以", "及", "或", "等", "可以", "这个", "那个", "什么", "怎么",
    "今天", "现在", "已经", "觉得", "感觉", "时候", "因为", "所以", "如果", "但是", "然后",
    "虽然", "而且", "或者", "并且", "不过", "只是", "可能", "应该", "需要", "想要", "觉得",
    "http", "https", "www", "com", "cn", "html", "php", "jpg", "png", "gif",
])


def load_stopwords() -> set:
    """加载停用词"""
    stopwords = DEFAULT_STOPWORDS.copy()
    try:
        if STOPWORDS_PATH and os.path.exists(STOPWORDS_PATH):
            with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word:
                        stopwords.add(word)
    except Exception:
        pass
    return stopwords


def tokenize(text: str, stopwords: Optional[set] = None) -> List[str]:
    """
    中文分词并过滤停用词/单字/数字/英文（保留中文词汇）
    """
    if stopwords is None:
        stopwords = load_stopwords()

    # 清理特殊字符
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
    words = jieba.lcut(text)

    filtered = []
    for w in words:
        w = w.strip().lower()
        if not w or w in stopwords:
            continue
        if len(w) == 1 and not re.match(r"[\u4e00-\u9fa5]", w):
            continue
        if re.match(r"^\d+$", w):
            continue
        if re.match(r"^[a-zA-Z]+$", w) and len(w) < 3:
            continue
        filtered.append(w)

    return filtered


def extract_keywords(texts: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
    """
    从文本中提取关键词（基于 TF-IDF）
    :return: [(word, score), ...]
    """
    stopwords = load_stopwords()
    tokenized = [" ".join(tokenize(t, stopwords)) for t in texts]

    if not any(tokenized):
        return []

    vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=TOPIC_MIN_DF,
        max_df=TOPIC_MAX_DF,
    )
    tfidf_matrix = vectorizer.fit_transform(tokenized)
    feature_names = vectorizer.get_feature_names_out()

    # 计算平均 TF-IDF 权重
    scores = np.mean(tfidf_matrix.toarray(), axis=0)
    word_scores = list(zip(feature_names, scores))
    word_scores.sort(key=lambda x: x[1], reverse=True)

    return word_scores[:top_n]


class TopicClusterer:
    """话题聚类器"""

    def __init__(self, n_clusters: int = TOPIC_CLUSTER_NUM):
        self.n_clusters = n_clusters
        self.vectorizer = None
        self.kmeans = None
        self.feature_names = None
        self.stopwords = load_stopwords()

    def fit(self, texts: List[str]) -> Dict[int, Dict]:
        """
        对文本进行话题聚类
        :return: {cluster_id: {"keywords": [...], "top_words": [...], "size": int}}
        """
        if len(texts) < self.n_clusters:
            self.n_clusters = max(1, len(texts) // 2)

        # 分词
        tokenized = [" ".join(tokenize(t, self.stopwords)) for t in texts]
        valid_indices = [i for i, t in enumerate(tokenized) if t.strip()]

        if not valid_indices:
            return {0: {"keywords": [], "top_words": [], "size": len(texts)}}

        filtered_texts = [tokenized[i] for i in valid_indices]

        # TF-IDF 向量化
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            min_df=TOPIC_MIN_DF,
            max_df=TOPIC_MAX_DF,
        )
        tfidf_matrix = self.vectorizer.fit_transform(filtered_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # KMeans 聚类
        actual_clusters = min(self.n_clusters, len(filtered_texts))
        self.kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(tfidf_matrix)

        # 构建结果
        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(label, []).append(valid_indices[i])

        result = {}
        for label, indices in clusters.items():
            cluster_texts = [filtered_texts[labels.tolist().index(label) + j] for j in range(len(indices))]
            # 获取该簇的中心词
            center_idx = label
            center = self.kmeans.cluster_centers_[center_idx]
            top_indices = center.argsort()[-10:][::-1]
            top_words = [self.feature_names[i] for i in top_indices]

            result[label] = {
                "keywords": top_words[:5],
                "top_words": top_words,
                "size": len(indices),
                "indices": indices,
            }

        return result

    def predict(self, texts: List[str]) -> List[int]:
        """对新文本预测话题"""
        if self.vectorizer is None or self.kmeans is None:
            raise RuntimeError("请先调用 fit()")
        tokenized = [" ".join(tokenize(t, self.stopwords)) for t in texts]
        tfidf_matrix = self.vectorizer.transform(tokenized)
        return self.kmeans.predict(tfidf_matrix).tolist()


def cluster_posts(posts: List[Dict], n_clusters: int = TOPIC_CLUSTER_NUM) -> Tuple[List[Dict], Dict[int, Dict]]:
    """
    对帖子进行话题聚类，并将 topic_id 写回帖子
    :return: (更新后的帖子列表, 话题信息字典)
    """
    if not posts:
        return posts, {}

    texts = [p.get("content", "") for p in posts]
    clusterer = TopicClusterer(n_clusters=n_clusters)
    topics = clusterer.fit(texts)

    # 为每个帖子分配话题
    labels = clusterer.predict(texts)
    for post, label in zip(posts, labels):
        post["topic_id"] = int(label)

    return posts, topics


import os
