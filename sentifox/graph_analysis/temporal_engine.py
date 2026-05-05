#!/usr/bin/env python3
"""
时间驱动引擎
管理仿真时间轴，基于中国社交媒体活跃时段调度 Agent 行动
"""
import random
from typing import Dict, List, Optional


# ── 中国社交媒体活跃时段配置 ──────────────────────────────

CHINA_SOCIAL_SCHEDULE = {
    0: 0.05,   # 深夜
    1: 0.03,
    2: 0.02,
    3: 0.02,
    4: 0.03,
    5: 0.05,
    6: 0.20,   # 早起
    7: 0.40,
    8: 0.60,
    9: 0.75,   # 工作时段
    10: 0.80,
    11: 0.85,
    12: 0.90,  # 午休
    13: 0.85,
    14: 0.80,
    15: 0.80,
    16: 0.75,
    17: 0.70,
    18: 0.80,  # 傍晚
    19: 1.00,  # 晚高峰
    20: 1.20,
    21: 1.30,
    22: 1.10,
    23: 0.50,  # 深夜
}

# 时段标签
TIME_PERIOD_LABELS = {
    "dead": [0, 1, 2, 3, 4, 5],
    "morning": [6, 7, 8],
    "work": [9, 10, 11, 12, 13, 14, 15, 16, 17],
    "evening": [18, 19, 20, 21, 22],
    "late": [23],
}


class TemporalEngine:
    """时间驱动引擎"""
    
    def __init__(self, schedule: Optional[Dict[int, float]] = None,
                 speed: float = 1.0):
        """
        Args:
            schedule: 每小时活跃度系数表
            speed: 时间加速系数（1.0=真实速度，100.0=100倍速）
        """
        self.schedule = schedule or CHINA_SOCIAL_SCHEDULE
        self.speed = speed
        self.current_step = 0
        self.step_labels = []  # 每步的标签（用于展示）
    
    @property
    def current_hour(self) -> int:
        """当前小时（0-23）"""
        return self.current_step % 24
    
    @property
    def current_day(self) -> int:
        """当前天数"""
        return self.current_step // 24
    
    def get_activity_multiplier(self, hour: Optional[int] = None) -> float:
        """获取指定小时的活跃度系数"""
        h = hour if hour is not None else self.current_hour
        return self.schedule.get(h, 0.5)
    
    def is_peak_hour(self, hour: Optional[int] = None) -> bool:
        """是否是高峰时段"""
        h = hour if hour is not None else self.current_hour
        return self.schedule.get(h, 0) >= 1.0
    
    def get_time_period(self, hour: Optional[int] = None) -> str:
        """获取时段标签"""
        h = hour if hour is not None else self.current_hour
        for period, hours in TIME_PERIOD_LABELS.items():
            if h in hours:
                return period
        return "unknown"
    
    def get_time_label(self, step: Optional[int] = None) -> str:
        """获取格式化时间标签"""
        s = step if step is not None else self.current_step
        hour = s % 24
        day = s // 24
        period_emoji = {
            "dead": "🌙", "morning": "🌅", "work": "💼",
            "evening": "🌆", "late": "🌃",
        }
        period = self.get_time_period(hour)
        emoji = period_emoji.get(period, "🕐")
        return f"Day{day+1} {hour:02d}:00 {emoji}"
    
    def is_agent_active(self, agent_peak_hours: List[int], 
                        base_prob: float = 0.3) -> bool:
        """
        判断 Agent 在当前时刻是否活跃
        
        Args:
            agent_peak_hours: Agent 的高峰时段
            base_prob: 基础活跃概率
        
        Returns:
            bool
        """
        hour = self.current_hour
        multiplier = self.get_activity_multiplier(hour)
        
        # Agent 个人高峰时段加成
        if hour in agent_peak_hours:
            multiplier *= 1.5
        
        # 计算最终概率
        activation_prob = base_prob * multiplier
        return random.random() < activation_prob
    
    def step(self) -> int:
        """推进到下一步，返回当前步数"""
        self.current_step += 1
        return self.current_step
    
    def reset(self):
        """重置时间轴"""
        self.current_step = 0
        self.step_labels = []
    
    def simulate_hours(self, hours: int) -> List[int]:
        """模拟指定小时数，返回每一步的步数列表"""
        steps = []
        for _ in range(hours):
            steps.append(self.step())
        return steps
