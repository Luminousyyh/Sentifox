#!/usr/bin/env python3
"""
Sentifox 仿真实时仪表盘
Rich Live 实时展示多智能体传播仿真过程
"""
from typing import Dict, List, Any, Optional, Tuple

from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich import box

from cli.theme import THEME
from cli.ui import sparkline, bar_chart_item


console = Console()


class SimulationLiveDisplay:
    """Rich Live 实时仿真仪表盘"""
    
    def __init__(self, simulator):
        self.simulator = simulator
        self._history = {
            "infected": [],
            "susceptible": [],
            "recovered": [],
            "new_actions": [],
            "relations": [],
        }
        self._event_log: List[str] = []
        self._max_events = 8
    
    def render(self, step: int, snapshot: Any, actions: List[Tuple[str, Any]]) -> Layout:
        """渲染仪表盘"""
        layout = Layout()
        
        # 更新历史数据
        self._history["infected"].append(snapshot.infected_count)
        self._history["susceptible"].append(snapshot.susceptible_count)
        self._history["recovered"].append(snapshot.recovered_count)
        self._history["new_actions"].append(snapshot.new_actions)
        self._history["relations"].append(snapshot.new_relations)
        
        # 限制历史长度
        for key in self._history:
            if len(self._history[key]) > 60:
                self._history[key] = self._history[key][-60:]
        
        # 记录事件
        time_label = self.simulator.temporal_engine.get_time_label(step)
        if actions:
            action_types = {}
            for _, a in actions:
                action_types[a.action_type] = action_types.get(a.action_type, 0) + 1
            event_str = f"{time_label} | 动作: " + ", ".join(f"{k}({v})" for k, v in action_types.items())
            self._event_log.append(event_str)
        if snapshot.new_relations > 0:
            self._event_log.append(f"{time_label} | 新关系: {snapshot.new_relations}")
        if len(self._event_log) > self._max_events:
            self._event_log = self._event_log[-self._max_events:]
        
        # 构建布局
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=4),
        )
        
        # Header: 时间和状态
        header_text = Text()
        header_text.append(f"🦊 Sentifox 仿真  ", style=f"bold {THEME.brand}")
        header_text.append(f"{time_label}  ", style="bold white")
        header_text.append(f"Step {step+1}  ", style="dim")
        header_text.append(
            f"😷{snapshot.infected_count}  😊{snapshot.susceptible_count}  😴{snapshot.recovered_count}",
            style="bold"
        )
        layout["header"].update(Panel(header_text, border_style=THEME.border, padding=(0, 1)))
        
        # Main: 左右分栏
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )
        
        # Left: 感染曲线 + 事件日志
        left_content = Group(
            self._render_infection_curve(),
            self._render_event_log(),
        )
        layout["left"].update(left_content)
        
        # Right: 实时统计 + 认知层指标 + Top Agent
        right_content = Group(
            self._render_stats(snapshot),
            self._render_cognitive_stats(snapshot),
            self._render_top_agents(snapshot),
        )
        layout["right"].update(right_content)
        
        # Footer: 进度条
        layout["footer"].update(self._render_progress(step))
        
        return layout
    
    def _render_infection_curve(self) -> Panel:
        """感染曲线 Sparkline"""
        if len(self._history["infected"]) < 2:
            return Panel("(等待数据...)", title="传播曲线", border_style=THEME.muted, box=box.ROUNDED)
        
        content = Text()
        
        # S 线（易感）
        content.append("易感 S: ", style="dim")
        content.append(sparkline(self._history["susceptible"], width=40, color="grey50"))
        content.append("\n")
        
        # I 线（感染）
        content.append("感染 I: ", style="dim")
        content.append(sparkline(self._history["infected"], width=40, color=THEME.error))
        content.append("\n")
        
        # R 线（恢复）
        content.append("恢复 R: ", style="dim")
        content.append(sparkline(self._history["recovered"], width=40, color=THEME.success))
        
        return Panel(content, title="📈 传播曲线", border_style=THEME.border, box=box.ROUNDED, padding=(0, 1))
    
    def _render_event_log(self) -> Panel:
        """事件日志"""
        if not self._event_log:
            return Panel("(暂无事件)", title="事件日志", border_style=THEME.muted, box=box.ROUNDED)
        
        lines = [Text(e, style="dim") for e in self._event_log]
        content = Group(*lines)
        return Panel(content, title="📋 事件日志", border_style=THEME.border, box=box.ROUNDED, padding=(0, 1))
    
    def _render_stats(self, snapshot: Any) -> Panel:
        """实时统计"""
        total = snapshot.active_agents
        if total == 0:
            total = 1
        
        content = Text()
        content.append(f"总 Agent: {total}\n", style="bold")
        content.append(f"感染:     {snapshot.infected_count} ({snapshot.infected_count/total*100:.1f}%)\n", style=THEME.error)
        content.append(f"易感:     {snapshot.susceptible_count} ({snapshot.susceptible_count/total*100:.1f}%)\n", style="grey50")
        content.append(f"恢复:     {snapshot.recovered_count} ({snapshot.recovered_count/total*100:.1f}%)\n", style=THEME.success)
        content.append(f"新动作:   {snapshot.new_actions}\n", style=THEME.info)
        content.append(f"新关系:   {snapshot.new_relations}\n", style=THEME.brand)
        
        # 立场分布
        stance = snapshot.stance_distribution
        content.append(f"\n立场:\n", style="bold")
        for s, c in stance.items():
            content.append(f"  {s}: {c}\n", style="dim")
        
        return Panel(content, title="📊 实时统计", border_style=THEME.border, box=box.ROUNDED, padding=(0, 1))
    
    def _render_cognitive_stats(self, snapshot: Any) -> Panel:
        """认知层指标：极化指数、情绪状态、桥接节点"""
        content = Text()
        
        # 极化指数
        polar = getattr(snapshot, "polarization_index", 0.0)
        polar_color = THEME.error if polar > 0.6 else THEME.warn if polar > 0.3 else THEME.success
        content.append(f"极化指数: ", style="bold")
        content.append(f"{polar:.2f}\n", style=polar_color)
        
        # 回音室
        echo = getattr(snapshot, "echo_chamber_count", 0)
        content.append(f"回音室:   {echo}\n", style="dim")
        
        # 讨论线程
        disc = getattr(snapshot, "discussion_count", 0)
        content.append(f"讨论线程: {disc}\n", style="dim")
        
        # 情绪状态（Top 3）
        emotions = getattr(snapshot, "emotional_states", {})
        if emotions:
            content.append(f"\n情绪采样:\n", style="bold")
            for aid, emo in list(emotions.items())[:3]:
                name = self.simulator.agents.get(aid, type("A", (), {"name": aid[:6]}))().name
                emoji_map = {
                    "anger": "😠", "anxiety": "😰", "excitement": "🤩",
                    "calm": "😌", "fatigue": "😴",
                }
                emoji = emoji_map.get(emo.get("dominant", "calm"), "😐")
                content.append(f"  {emoji} {name[:6]} {emo.get('dominant', '?')}({emo.get('intensity', 0):.1f})\n", style="dim")
        
        # 桥接节点
        bridges = getattr(snapshot, "bridge_nodes", [])
        if bridges:
            content.append(f"\n桥接节点:\n", style="bold")
            for node, score in bridges[:2]:
                name = self.simulator.agents.get(node, type("A", (), {"name": node[:6]}))().name
                content.append(f"  🔗 {name[:6]} {score:.2f}\n", style="dim")
        
        return Panel(content, title="🧠 认知指标", border_style=THEME.accent, box=box.ROUNDED, padding=(0, 1))
    
    def _render_top_agents(self, snapshot: Any) -> Panel:
        """Top 传播 Agent"""
        if not snapshot.top_influencers:
            return Panel("(暂无数据)", title="传播排行榜", border_style=THEME.muted, box=box.ROUNDED)
        
        lines = []
        max_amp = max((amp for _, amp in snapshot.top_influencers), default=1)
        for agent_id, amp in snapshot.top_influencers[:5]:
            agent = self.simulator.agents.get(agent_id)
            name = agent.name if agent else agent_id[:8]
            lines.append(bar_chart_item(name[:8], amp, max_amp, width=15, color=THEME.brand))
        
        content = Group(*lines)
        return Panel(content, title="🏆 传播排行", border_style=THEME.border, box=box.ROUNDED, padding=(0, 1))
    
    def _render_progress(self, step: int) -> Panel:
        """进度条"""
        total_steps = getattr(self.simulator.config, 'get', lambda k, d: d)("total_steps", 72)
        progress = (step + 1) / total_steps
        bar_width = 40
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        content = Text()
        content.append(f"进度: [{bar}] {(step+1)}/{total_steps} ({progress*100:.1f}%)", style="bold")
        
        return Panel(content, border_style=THEME.brand, box=box.ROUNDED, padding=(0, 1))
