"""
火山引擎 LLM API 客户端封装
兼容 OpenAI 协议
"""
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI

from config import LLM_CONFIG


class LLMClient:
    """统一 LLM 客户端"""

    def __init__(self):
        self.client = OpenAI(
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"],
            timeout=LLM_CONFIG.get("timeout", 30),
            max_retries=LLM_CONFIG.get("max_retries", 3),
        )
        self.chat_model = LLM_CONFIG["chat_model"]
        self.embedding_model = LLM_CONFIG["embedding_model"]

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        response_format: Optional[Dict] = None,
    ) -> str:
        """
        对话补全
        :param messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
        :return: 生成的文本
        """
        try:
            kwargs = {
                "model": model or self.chat_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = self.client.chat.completions.create(**kwargs)

            if stream:
                # 流式输出由调用方处理
                return response

            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"LLM API 调用失败: {e}")
            return f"[错误] LLM API 调用失败: {e}"

    def embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 100,
    ) -> List[List[float]]:
        """
        批量获取文本 Embedding
        :param texts: 文本列表
        :param batch_size: 每批数量
        :return: 向量列表
        """
        all_embeddings = []
        model = model or self.embedding_model

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            success = False
            for attempt in range(3):
                try:
                    response = self.client.embeddings.create(
                        model=model,
                        input=batch,
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    success = True
                    break
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RateLimit" in err_msg or "TooManyRequests" in err_msg:
                        wait = 2 ** attempt
                        print(f"Embedding API 限流 (batch {i}), 等待 {wait}s 后重试...")
                        time.sleep(wait)
                    else:
                        print(f"Embedding API 调用失败 (batch {i}): {e}")
                        break
            if not success:
                # 失败时填充零向量
                dim = 2048  # doubao-embedding-vision 维度
                all_embeddings.extend([[0.0] * dim] * len(batch))
            # 简单限流
            if i + batch_size < len(texts):
                time.sleep(1.0)

        return all_embeddings

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        强制返回 JSON 格式的响应
        """
        try:
            response = self.client.chat.completions.create(
                model=model or self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            import json
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as e:
            print(f"JSON 生成失败: {e}")
            return {}


# 全局单例
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取 LLM 客户端单例"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
