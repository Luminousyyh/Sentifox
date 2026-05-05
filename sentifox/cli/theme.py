"""
Sentifox 品牌主题系统
定义语义颜色、边框样式、暗/亮模式检测
"""
from typing import Dict, Optional
from dataclasses import dataclass
from rich import box


@dataclass
class SentifoxTheme:
    """Sentifox 品牌主题配置"""
    # 品牌色
    brand: str = "#FF6B35"           # 狐狸橙
    brand_secondary: str = "#E63946"  # 深红
    
    # 语义颜色
    success: str = "#2ECC71"         # 绿
    warning: str = "#F1C40F"         # 黄
    error: str = "#E74C3C"           # 红
    info: str = "#3498DB"            # 蓝
    muted: str = "#95A5A6"           # 灰
    
    # 文本颜色
    text_primary: str = "#ECF0F1"    # 主文本（暗色背景）
    text_secondary: str = "#BDC3C7"  # 次要文本
    text_highlight: str = "#FFFFFF"  # 高亮文本
    
    # 背景
    bg_panel: str = "#1E1E1E"        # 面板背景
    bg_surface: str = "#2D2D2D"      # 表面背景
    
    # 边框
    border: str = "#34495E"          # 边框
    border_focus: str = "#FF6B35"    # 聚焦边框
    
    # Box 样式
    box_style = box.ROUNDED
    box_header = box.SIMPLE_HEAD
    
    # 图标
    icon_fox: str = "🦊"
    icon_ok: str = "✓"
    icon_warn: str = "⚠"
    icon_error: str = "✗"
    icon_arrow: str = "→"
    icon_bullet: str = "•"
    icon_spinner: str = "◐"
    icon_fire: str = "🔥"
    icon_brain: str = "🧠"
    icon_chart: str = "📊"
    icon_alert: str = "🚨"
    icon_star: str = "⭐"


# 全局主题实例
THEME = SentifoxTheme()


def get_status_style(status: str) -> str:
    """根据状态返回 Rich 样式字符串"""
    mapping = {
        "ok": f"bold {THEME.success}",
        "success": f"bold {THEME.success}",
        "done": f"bold {THEME.success}",
        "warn": f"bold {THEME.warning}",
        "warning": f"bold {THEME.warning}",
        "error": f"bold {THEME.error}",
        "fail": f"bold {THEME.error}",
        "info": f"bold {THEME.info}",
        "pending": f"bold {THEME.muted}",
        "running": f"bold {THEME.brand}",
        "alert": f"bold {THEME.error}",
    }
    return mapping.get(status.lower(), f"bold {THEME.muted}")


def get_status_emoji(status: str) -> str:
    """根据状态返回 emoji 图标"""
    mapping = {
        "ok": "[green]✓[/]",
        "success": "[green]✓[/]",
        "done": "[green]✓[/]",
        "warn": "[yellow]⚠[/]",
        "warning": "[yellow]⚠[/]",
        "error": "[red]✗[/]",
        "fail": "[red]✗[/]",
        "info": "[blue]ℹ[/]",
        "pending": "[grey50]○[/]",
        "running": "[orange3]◐[/]",
        "alert": "[red]🚨[/]",
    }
    return mapping.get(status.lower(), "[grey50]•[/]")


def severity_color(severity: str) -> str:
    """告警严重度颜色"""
    mapping = {
        "high": "bold red",
        "critical": "bold red reverse",
        "medium": "bold yellow",
        "low": "bold green",
        "info": "bold blue",
    }
    return mapping.get(severity.lower(), "bold grey50")


def sentiment_color(label: str) -> str:
    """情感标签颜色"""
    mapping = {
        "positive": "green",
        "negative": "red",
        "neutral": "grey50",
    }
    return mapping.get(label.lower(), "grey50")


def sentiment_emoji(label: str) -> str:
    """情感标签 emoji"""
    mapping = {
        "positive": "😊",
        "negative": "😠",
        "neutral": "😐",
    }
    return mapping.get(label.lower(), "😐")


def platform_color(platform: str) -> str:
    """平台颜色"""
    mapping = {
        "微博": "#E6162D",
        "知乎": "#0084FF",
        "小红书": "#FF2442",
        "抖音": "#000000",
        "新闻": "#F39C12",
        "论坛": "#9B59B6",
    }
    return mapping.get(platform, "#3498DB")
