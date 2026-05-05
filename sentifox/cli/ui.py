"""
Sentifox Rich UI 组件库
提供统一的终端视觉组件：进度条、KPI卡片、Sparkline、日志面板等
"""
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.live import Live
from rich.layout import Layout
from rich.columns import Columns
from rich.box import Box, ROUNDED, SIMPLE

from cli.theme import THEME, get_status_style, get_status_emoji, severity_color, sentiment_color, platform_color


console = Console()


# ═══════════════════════════════════════════════════════════════
# 状态徽章
# ═══════════════════════════════════════════════════════════════

def status_badge(status: str, text: Optional[str] = None) -> Text:
    """生成彩色状态徽章"""
    label = text or status.upper()
    style = get_status_style(status)
    return Text(f" {label} ", style=style)


# ═══════════════════════════════════════════════════════════════
# 区块标题
# ═══════════════════════════════════════════════════════════════

def section_header(title: str, icon: str = "") -> Text:
    """带图标的区块标题"""
    prefix = f"{icon} " if icon else ""
    return Text(f"{prefix}{title}", style=f"bold {THEME.brand}")


def section_divider() -> Text:
    """细灰线分隔"""
    return Text("─" * console.width, style=THEME.muted)


# ═══════════════════════════════════════════════════════════════
# KPI 卡片行
# ═══════════════════════════════════════════════════════════════

def kpi_card(title: str, value: str, delta: Optional[str] = None, 
             color: str = THEME.text_primary) -> Panel:
    """单个 KPI 卡片"""
    content = Text(value, style=f"bold {color}", justify="center")
    if delta:
        content.append(f"\n{delta}", style=THEME.muted)
    return Panel(
        content,
        title=title,
        title_align="center",
        border_style=THEME.border,
        box=ROUNDED,
        padding=(1, 2),
    )


def kpi_row(cards: List[Panel]) -> Columns:
    """横向 KPI 卡片行"""
    return Columns(cards, equal=True, expand=True)


# ═══════════════════════════════════════════════════════════════
# Sparkline (ASCII 迷你趋势图)
# ═══════════════════════════════════════════════════════════════

SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: List[float], width: int = 40, 
              color: str = THEME.brand) -> Text:
    """生成 ASCII Sparkline 趋势图"""
    if not values or len(values) < 2:
        return Text("(无数据)", style=THEME.muted)
    
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return Text("▁" * min(width, len(values)), style=color)
    
    # 采样到指定宽度
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    
    # 归一化到 sparkline 字符
    chars = []
    for v in sampled:
        idx = int((v - min_val) / (max_val - min_val) * (len(SPARKLINE_CHARS) - 1))
        chars.append(SPARKLINE_CHARS[min(idx, len(SPARKLINE_CHARS) - 1)])
    
    return Text("".join(chars), style=color)


def labeled_sparkline(label: str, values: List[float], 
                      width: int = 40, color: str = THEME.brand) -> Panel:
    """带标签的 Sparkline 面板"""
    line = sparkline(values, width, color)
    min_v = f"{min(values):.1f}" if values else "-"
    max_v = f"{max(values):.1f}" if values else "-"
    footer = Text(f"min: {min_v}  max: {max_v}", style=THEME.muted)
    content = Group(line, footer)
    return Panel(content, title=label, border_style=color, box=ROUNDED)


# ═══════════════════════════════════════════════════════════════
# 水平条形图（用于排名展示）
# ═══════════════════════════════════════════════════════════════

def bar_chart_item(label: str, value: float, max_value: float, 
                   width: int = 30, color: str = THEME.brand) -> Text:
    """单行水平条形图"""
    if max_value <= 0:
        bar_len = 0
    else:
        bar_len = int((value / max_value) * width)
    bar = "█" * bar_len + "░" * (width - bar_len)
    return Text(f"{label:12} {bar} {value:.2f}", style=color)


def bar_chart(items: List[Tuple[str, float]], 
              color: str = THEME.brand, title: str = "") -> Panel:
    """多行水平条形图面板"""
    if not items:
        return Panel("(无数据)", title=title, border_style=THEME.muted)
    
    max_val = max(v for _, v in items) if items else 1
    lines = [bar_chart_item(label, value, max_val, color=color) 
             for label, value in items]
    content = Group(*lines)
    return Panel(content, title=title, border_style=color, box=ROUNDED)


# ═══════════════════════════════════════════════════════════════
# 情感分布水平条图
# ═══════════════════════════════════════════════════════════════

def sentiment_bar(positive: int, negative: int, neutral: int, 
                  width: int = 40) -> Text:
    """情感分布水平条图"""
    total = positive + negative + neutral
    if total == 0:
        return Text("(无数据)", style=THEME.muted)
    
    p_len = int(positive / total * width)
    n_len = int(negative / total * width)
    u_len = width - p_len - n_len
    
    bar = Text()
    bar.append("█" * p_len, style="green")
    bar.append("█" * n_len, style="red")
    bar.append("█" * u_len, style="grey50")
    bar.append(f"  正{positive} 负{negative} 中{neutral}", style=THEME.muted)
    return bar


# ═══════════════════════════════════════════════════════════════
# Pipeline 步骤进度条
# ═══════════════════════════════════════════════════════════════

@dataclass
class StepInfo:
    """单个步骤信息"""
    name: str
    status: str = "pending"  # pending / running / done / warn / error
    elapsed: float = 0.0
    detail: str = ""
    start_time: Optional[float] = None


class StepProgress:
    """多步骤流水线进度条组件"""
    
    def __init__(self, steps: List[str], title: str = "Pipeline"):
        self.steps = [StepInfo(name=s) for s in steps]
        self.title = title
        self.start_time = time.time()
        self._table = self._build_table()
    
    def _build_table(self) -> Table:
        """构建进度表格"""
        table = Table(
            title=f"[bold {THEME.brand}]{THEME.icon_fox} {self.title}[/]",
            box=ROUNDED,
            border_style=THEME.border,
            show_header=False,
            padding=(0, 1),
        )
        table.add_column("状态", width=4)
        table.add_column("步骤", width=16)
        table.add_column("耗时", width=8)
        table.add_column("详情", ratio=1)
        return table
    
    def _status_icon(self, step: StepInfo) -> str:
        """状态图标"""
        icons = {
            "pending": f"[grey50]○[/]",
            "running": f"[{THEME.brand}]◐[/]",
            "done": f"[green]✓[/]",
            "warn": f"[yellow]⚠[/]",
            "error": f"[red]✗[/]",
        }
        return icons.get(step.status, "○")
    
    def _render(self) -> Table:
        """渲染当前状态"""
        table = self._build_table()
        for step in self.steps:
            icon = self._status_icon(step)
            elapsed_str = f"{step.elapsed:.1f}s" if step.elapsed > 0 else "--"
            detail = step.detail or ""
            
            if step.status == "running":
                name_style = f"bold {THEME.brand}"
            elif step.status == "done":
                name_style = "bold green"
            elif step.status == "error":
                name_style = "bold red"
            elif step.status == "warn":
                name_style = "bold yellow"
            else:
                name_style = f"{THEME.muted}"
            
            table.add_row(
                icon,
                f"[{name_style}]{step.name}[/]",
                elapsed_str,
                detail,
            )
        
        # 底部统计行
        total_elapsed = time.time() - self.start_time
        done_count = sum(1 for s in self.steps if s.status == "done")
        table.add_row(
            "",
            f"[dim]进度: {done_count}/{len(self.steps)}[/]",
            f"[dim]{total_elapsed:.1f}s[/]",
            "",
        )
        return table
    
    def __rich__(self):
        return self._render()
    
    def start_step(self, index: int):
        """开始指定步骤"""
        if 0 <= index < len(self.steps):
            self.steps[index].status = "running"
            self.steps[index].start_time = time.time()
    
    def finish_step(self, index: int, detail: str = "", status: str = "done"):
        """完成指定步骤"""
        if 0 <= index < len(self.steps):
            step = self.steps[index]
            step.status = status
            if step.start_time:
                step.elapsed = time.time() - step.start_time
            step.detail = detail
    
    def update_step_detail(self, index: int, detail: str):
        """更新步骤详情"""
        if 0 <= index < len(self.steps):
            self.steps[index].detail = detail
    
    @property
    def is_complete(self) -> bool:
        """是否全部完成"""
        return all(s.status in ("done", "warn", "error") for s in self.steps)


# ═══════════════════════════════════════════════════════════════
# 实时日志面板
# ═══════════════════════════════════════════════════════════════

class LiveLog:
    """可追加的实时日志面板"""
    
    def __init__(self, title: str = "日志", max_lines: int = 20):
        self.title = title
        self.max_lines = max_lines
        self.lines: List[Text] = []
    
    def append(self, message: str, level: str = "info"):
        """追加日志行"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        style = get_status_style(level).replace("bold ", "")
        line = Text(f"[{timestamp}] ", style=THEME.muted)
        line.append(message, style=style)
        self.lines.append(line)
        if len(self.lines) > self.max_lines:
            self.lines = self.lines[-self.max_lines:]
    
    def __rich__(self):
        if not self.lines:
            return Panel("(暂无日志)", title=self.title, border_style=THEME.muted)
        content = Group(*self.lines)
        return Panel(content, title=self.title, border_style=THEME.border, box=ROUNDED)


# ═══════════════════════════════════════════════════════════════
# 命令结果包装
# ═══════════════════════════════════════════════════════════════

def result_panel(content: Any, title: str = "", status: str = "info") -> Panel:
    """命令结果统一包装面板"""
    border = {
        "ok": THEME.success,
        "success": THEME.success,
        "warn": THEME.warning,
        "error": THEME.error,
        "info": THEME.info,
    }.get(status.lower(), THEME.border)
    
    return Panel(
        content,
        title=title if title else None,
        border_style=border,
        box=ROUNDED,
    )


# ═══════════════════════════════════════════════════════════════
# 命令历史持久化
# ═══════════════════════════════════════════════════════════════

class CommandHistory:
    """REPL 命令历史记录管理"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.history_file = os.path.expanduser("~/.sentifox_history")
        self.commands: List[str] = []
        self._load()
    
    def _load(self):
        """加载历史记录"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.commands = [line.strip() for line in f.readlines() if line.strip()]
            except Exception:
                pass
    
    def save(self):
        """保存历史记录"""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                for cmd in self.commands[-self.max_size:]:
                    f.write(cmd + "\n")
        except Exception:
            pass
    
    def add(self, command: str):
        """添加命令到历史"""
        if command and command not in ("exit", "quit", "q", "clear"):
            # 避免重复连续命令
            if not self.commands or self.commands[-1] != command:
                self.commands.append(command)
                if len(self.commands) > self.max_size:
                    self.commands = self.commands[-self.max_size:]
    
    def get_recent(self, n: int = 20) -> List[str]:
        """获取最近 n 条命令"""
        return self.commands[-n:]


# ═══════════════════════════════════════════════════════════════
# 精美表格包装
# ═══════════════════════════════════════════════════════════════

def styled_table(headers: List[str], rows: List[List[Any]], 
                 title: str = "", header_style: str = "bold magenta") -> Table:
    """统一样式的表格"""
    table = Table(
        title=f"[bold]{title}[/]" if title else None,
        show_header=True,
        header_style=header_style,
        box=ROUNDED,
        border_style=THEME.border,
        row_styles=["", "dim"],  # 斑马纹
    )
    for h in headers:
        table.add_column(str(h), overflow="fold")
    for row in rows:
        table.add_row(*[str(c) for c in row])
    return table


# ═══════════════════════════════════════════════════════════════
# 话题卡片
# ═══════════════════════════════════════════════════════════════

def topic_card(topic_id: str, post_count: int, keywords: List[str],
               sentiment_dist: Optional[Dict[str, int]] = None) -> Panel:
    """话题卡片"""
    content = Text()
    content.append(f"帖子数: {post_count}\n", style="bold")
    
    if sentiment_dist:
        total = sum(sentiment_dist.values())
        if total > 0:
            content.append("情感: ")
            content.append(f"正{sentiment_dist.get('positive', 0)} ", style="green")
            content.append(f"负{sentiment_dist.get('negative', 0)} ", style="red")
            content.append(f"中{sentiment_dist.get('neutral', 0)}\n", style="grey50")
    
    if keywords:
        tags = " ".join(f"[#{THEME.brand}]{kw}[/#]" for kw in keywords[:5])
        content.append(f"关键词: {tags}")
    
    return Panel(
        content,
        title=f"[bold]{THEME.icon_fire} 话题 {topic_id}[/]",
        border_style=THEME.brand,
        box=ROUNDED,
        padding=(1, 2),
    )


def topic_grid(cards: List[Panel]) -> Columns:
    """话题卡片网格"""
    return Columns(cards, equal=True, expand=True)


# ═══════════════════════════════════════════════════════════════
# 平台状态卡片
# ═══════════════════════════════════════════════════════════════

def platform_status_card(platform: str, status: str, detail: str) -> Panel:
    """平台状态卡片"""
    is_on = status == "[ON]"
    color = THEME.success if is_on else THEME.error
    icon = "🟢" if is_on else "🔴"
    
    content = Text()
    content.append(f"{icon} {status}\n", style=f"bold {color}")
    content.append(detail, style=THEME.muted)
    
    plat_color = platform_color(platform)
    return Panel(
        content,
        title=f"[bold {plat_color}]{platform}[/]",
        border_style=color,
        box=ROUNDED,
        padding=(1, 2),
    )


# ═══════════════════════════════════════════════════════════════
# 告警卡片
# ═══════════════════════════════════════════════════════════════

def alert_card(alert_type: str, severity: str, message: str, 
               created_at: str) -> Panel:
    """告警卡片"""
    sev_style = severity_color(severity)
    icon = {"high": "🚨", "medium": "⚠️", "low": "ℹ️"}.get(severity.lower(), "•")
    
    content = Text()
    content.append(f"{icon} {message}\n", style=sev_style)
    content.append(f"时间: {created_at}", style=THEME.muted)
    
    return Panel(
        content,
        title=f"[bold]{alert_type}[/]",
        border_style=sev_style.split()[1] if " " in sev_style else THEME.warning,
        box=ROUNDED,
    )


# ═══════════════════════════════════════════════════════════════
# RAG 对话气泡
# ═══════════════════════════════════════════════════════════════

def chat_bubble_user(question: str) -> Panel:
    """用户问题气泡"""
    return Panel(
        Text(question, style="bold"),
        title="[bold blue]👤 你[/]",
        title_align="left",
        border_style="blue",
        box=ROUNDED,
    )


def chat_bubble_assistant(answer: str, sources: Optional[List[Dict]] = None) -> Panel:
    """助手回答气泡"""
    content = Text(answer)
    if sources:
        content.append("\n\n", style="")
        content.append("参考来源:\n", style=f"bold {THEME.muted}")
        for i, s in enumerate(sources[:5], 1):
            content.append(f"  [{i}] {s.get('platform', '')} | {s.get('author', '')} | 相关度{s.get('relevance', 0):.2f}\n", 
                          style=THEME.muted)
    
    return Panel(
        content,
        title=f"[bold {THEME.brand}]{THEME.icon_brain} Sentifox[/]",
        title_align="left",
        border_style=THEME.brand,
        box=ROUNDED,
    )


# ═══════════════════════════════════════════════════════════════
# 测试结果表格
# ═══════════════════════════════════════════════════════════════

def test_result_table(results: List[Tuple[str, bool, Optional[str]]]) -> Table:
    """功能测试结果表格"""
    table = Table(
        title="[bold]功能测试结果[/]",
        show_header=True,
        header_style="bold magenta",
        box=ROUNDED,
        border_style=THEME.border,
    )
    table.add_column("状态", width=4)
    table.add_column("测试项", width=20)
    table.add_column("详情", ratio=1)
    
    passed = 0
    for name, ok, detail in results:
        if ok:
            passed += 1
            icon = "[green]✓[/]"
            status_style = "green"
            detail_text = detail or "通过"
        else:
            icon = "[red]✗[/]"
            status_style = "red"
            detail_text = detail or "失败"
        
        table.add_row(icon, f"[{status_style}]{name}[/]", detail_text)
    
    total = len(results)
    summary = f"{passed}/{total} 通过"
    if passed == total:
        summary = f"[green]{summary}[/]"
    else:
        summary = f"[yellow]{summary}[/]"
    
    table.caption = f"总计: {summary}"
    return table
