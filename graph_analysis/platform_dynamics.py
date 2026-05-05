#!/usr/bin/env python3
"""
多平台传播规则定义
不同社交媒体平台有不同的传播动力学特性
"""
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class PlatformRule:
    """平台传播规则"""
    name: str
    content_form: str          # short_text / long_text / image / video
    spread_speed: float        # 传播速度系数（基准 1.0）
    discussion_depth: float    # 讨论深度（0-1）
    virality_threshold: float  # 病毒传播阈值（概率）
    virality_boost: float      # 病毒传播时的影响力放大系数
    algorithm: str             # 推荐算法类型
    max_cascade_depth: int     # 最大级联深度
    echo_chamber_strength: float  # 回声室强度（0-1）
    decay_rate: float          # 信息衰减率（每步）
    peak_hours: list           # 高峰时段
    user_age_profile: tuple    # 用户年龄分布 (min, max, median)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "content_form": self.content_form,
            "spread_speed": self.spread_speed,
            "discussion_depth": self.discussion_depth,
            "virality_threshold": self.virality_threshold,
            "algorithm": self.algorithm,
            "max_cascade_depth": self.max_cascade_depth,
            "echo_chamber_strength": self.echo_chamber_strength,
        }


# ── 平台规则定义 ──────────────────────────────

PLATFORM_RULES = {
    "微博": PlatformRule(
        name="微博",
        content_form="short_text",
        spread_speed=1.5,
        discussion_depth=0.3,
        virality_threshold=0.3,
        virality_boost=2.0,
        algorithm="realtime_hot",
        max_cascade_depth=8,
        echo_chamber_strength=0.6,
        decay_rate=0.15,
        peak_hours=[12, 13, 18, 19, 20, 21, 22],
        user_age_profile=(16, 45, 28),
    ),
    "知乎": PlatformRule(
        name="知乎",
        content_form="long_text",
        spread_speed=0.7,
        discussion_depth=0.9,
        virality_threshold=0.5,
        virality_boost=1.5,
        algorithm="professional_recommend",
        max_cascade_depth=5,
        echo_chamber_strength=0.4,
        decay_rate=0.08,
        peak_hours=[9, 10, 11, 14, 15, 16, 20, 21, 22],
        user_age_profile=(20, 50, 30),
    ),
    "小红书": PlatformRule(
        name="小红书",
        content_form="image",
        spread_speed=1.2,
        discussion_depth=0.5,
        virality_threshold=0.35,
        virality_boost=1.8,
        algorithm="interest_recommend",
        max_cascade_depth=6,
        echo_chamber_strength=0.7,
        decay_rate=0.12,
        peak_hours=[11, 12, 19, 20, 21, 22, 23],
        user_age_profile=(18, 35, 25),
    ),
    "抖音": PlatformRule(
        name="抖音",
        content_form="video",
        spread_speed=2.0,
        discussion_depth=0.2,
        virality_threshold=0.2,
        virality_boost=2.5,
        algorithm="algorithm_push",
        max_cascade_depth=10,
        echo_chamber_strength=0.5,
        decay_rate=0.2,
        peak_hours=[11, 12, 18, 19, 20, 21, 22, 23],
        user_age_profile=(16, 40, 24),
    ),
    "新闻": PlatformRule(
        name="新闻",
        content_form="long_text",
        spread_speed=0.5,
        discussion_depth=0.4,
        virality_threshold=0.6,
        virality_boost=1.2,
        algorithm="editor_recommend",
        max_cascade_depth=4,
        echo_chamber_strength=0.2,
        decay_rate=0.05,
        peak_hours=[7, 8, 12, 18, 19, 20],
        user_age_profile=(25, 60, 38),
    ),
    "论坛": PlatformRule(
        name="论坛",
        content_form="long_text",
        spread_speed=0.6,
        discussion_depth=0.8,
        virality_threshold=0.45,
        virality_boost=1.3,
        algorithm="community_recommend",
        max_cascade_depth=5,
        echo_chamber_strength=0.8,
        decay_rate=0.06,
        peak_hours=[10, 11, 12, 20, 21, 22, 23],
        user_age_profile=(20, 45, 30),
    ),
}


def get_platform_rule(platform: str) -> PlatformRule:
    """获取平台规则（默认回退到论坛）"""
    return PLATFORM_RULES.get(platform, PLATFORM_RULES["论坛"])


def get_all_platforms() -> list:
    """获取所有平台名称"""
    return list(PLATFORM_RULES.keys())
