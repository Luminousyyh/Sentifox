"""
情感分析模块
使用 HuggingFace Transformers 预训练中文模型
"""
import os

# 中国大陆镜像加速（如需官方源可注释掉）
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
from typing import List, Dict, Union
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

from config import SENTIMENT_MODEL, BATCH_SIZE, SENTIMENT_LABELS


class SentimentAnalyzer:
    """情感分析器（单例懒加载）"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = SENTIMENT_MODEL, device: Union[int, str] = None):
        if self._initialized:
            return

        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self.model_name = model_name
        self.device = device
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        self._initialized = True

    def _load_pipeline(self):
        """懒加载 pipeline"""
        if self._pipeline is None:
            try:
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=self.device,
                    truncation=True,
                    max_length=512,
                )
            except Exception as e:
                print(f"Pipeline load failed: {e}, fallback to manual load")
                self._tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self._model = BertForSequenceClassification.from_pretrained(self.model_name)
                if self.device >= 0 and torch.cuda.is_available():
                    self._model = self._model.cuda()
        return self._pipeline

    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        单条文本情感预测
        :return: {"label": "positive"/"negative"/"neutral", "score": 0.95}
        """
        if not text or not text.strip():
            return {"label": "neutral", "score": 0.0}

        pipe = self._load_pipeline()
        if pipe:
            result = pipe(text[:512])[0]
            label = result["label"].lower()
            score = result["score"]
            # 统一标签
            if "positive" in label or label == "1" or label == "LABEL_1":
                label = "positive"
            elif "negative" in label or label == "0" or label == "LABEL_0":
                label = "negative"
            else:
                label = "neutral"
            return {"label": label, "score": float(score)}
        else:
            return self._predict_manual(text)

    def _predict_manual(self, text: str) -> Dict[str, Union[str, float]]:
        """手动推理（pipeline加载失败时fallback）"""
        inputs = self._tokenizer(
            text[:512],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        if self.device >= 0 and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            score = probs[0][pred].item()

        label = SENTIMENT_LABELS.get(pred, "neutral")
        return {"label": label, "score": score}

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        批量情感预测
        """
        if not texts:
            return []

        results = []
        pipe = self._load_pipeline()

        if pipe:
            # pipeline 批量推理
            batch_size = BATCH_SIZE
            for i in range(0, len(texts), batch_size):
                batch = [t[:512] if t else "" for t in texts[i:i + batch_size]]
                batch_results = pipe(batch)
                for res in batch_results:
                    label = res["label"].lower()
                    score = res["score"]
                    if "positive" in label or label == "1" or label == "LABEL_1":
                        label = "positive"
                    elif "negative" in label or label == "0" or label == "LABEL_0":
                        label = "negative"
                    else:
                        label = "neutral"
                    results.append({"label": label, "score": float(score)})
        else:
            # 手动批量推理
            for text in texts:
                results.append(self._predict_manual(text))

        return results

    def analyze_posts(self, posts: List[Dict]) -> List[Dict]:
        """
        对帖子列表进行情感分析并附加结果
        """
        texts = [p.get("content", "") for p in posts]
        sentiments = self.predict_batch(texts)

        for post, sentiment in zip(posts, sentiments):
            post["sentiment_label"] = sentiment["label"]
            post["sentiment_score"] = sentiment["score"]

        return posts


# 便捷函数
def get_analyzer() -> SentimentAnalyzer:
    """获取情感分析器实例"""
    return SentimentAnalyzer()


def quick_sentiment(text: str) -> Dict[str, Union[str, float]]:
    """快速单条分析"""
    analyzer = get_analyzer()
    return analyzer.predict(text)
