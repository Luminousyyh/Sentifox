#!/usr/bin/env python3
"""
Sentifox ASCII Art & Startup Animation
红色小狐狸启动动画与静态 Banner
"""
import time
import sys
import os

# ── Rich 检测 ──────────────────────────────
try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.spinner import Spinner
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

# ── ASCII 帧（4帧尾巴摆动动画）────────────
FOX_FRAMES = [
    # Frame 1: 尾巴 |
    r"""
          /\      /\
         /  \    /  \
        |    \  /    |
       /      \/      \
      |  ^    /\    ^  |
       \_____/  \_____/
            |  |
           /|  |\
          / |  | \
         /  |__|  \
        /   /  \   \
       /___/    \___\
""",
    # Frame 2: 尾巴 /
    r"""
          /\      /\
         /  \    /  \
        |    \  /    |
       /      \/      \
      |  ^    /\    ^  |
       \_____/  \_____/
            |  |/
           /|  /
          / | /
         /  |/
        /   /
       /___/
""",
    # Frame 3: 尾巴 -
    r"""
          /\      /\
         /  \    /  \
        |    \  /    |
       /      \/      \
      |  ^    /\    ^  |
       \_____/  \_____/
            |  |___
           /|     \
          / |      \
         /  |_______\
        /
       /______________
""",
    # Frame 4: 尾巴 \
    r"""
          /\      /\
         /  \    /  \
        |    \  /    |
       /      \/      \
      |  ^    /\    ^  |
       \_____/  \_____/
           \|  |
            \ | |
             \| |
              \|__|
               \  \
                \___\
""",
]

# ── 静态 Banner 狐狸 ────
FOX_BANNER = r"""
        /\__      __/\
       /   `\    /`   \
      |  o o \  / o o  |
     /    \   \/   /    \
    |   ^   \__/   ^   |
     \  ___/    \___  /
      |/  |      |  \|
         /|      |\
        / |      | \
       /  |______|  \
      /   /        \
     /___/          \
""".strip("\n")

# ── 状态消息 ───────────────────────────────
BOOT_MESSAGES = [
    ("Initializing Sentifox engine...", "green"),
    ("Loading sentiment analysis model...", "cyan"),
    ("Connecting to vector store...", "blue"),
    ("Syncing ChromaDB collections...", "magenta"),
    ("Warming up propagation simulator...", "yellow"),
    ("Ready.", "bold green"),
]


def _print_typing(text: str, delay: float = 0.015):
    """打字机效果输出（纯文本回退）"""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _colorize_frame(frame: str) -> Text:
    """将 ASCII 帧转为 Rich Text（红色主题）"""
    t = Text(frame, style="bold red")
    t.highlight_regex(r"o", style="bold bright_white on_red")
    t.highlight_regex(r"\^", style="bold bright_white")
    return t


def _check_health() -> list:
    """快速健康检查，返回状态列表"""
    checks = []
    
    # 数据库检查
    try:
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sentiment.db")
        if os.path.exists(db_path):
            checks.append(("database", True, "SQLite connected"))
        else:
            checks.append(("database", False, "DB not initialized"))
    except Exception as e:
        checks.append(("database", False, str(e)))
    
    # ChromaDB 检查（轻量检测目录）
    try:
        chroma_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
        if os.path.exists(chroma_path):
            checks.append(("vector_store", True, "ChromaDB ready"))
        else:
            checks.append(("vector_store", False, "Not initialized"))
    except Exception as e:
        checks.append(("vector_store", False, str(e)))
    
    return checks


def play_startup_animation(cycles: int = 2):
    """
    播放启动动画：4帧尾巴摆动 + 打字机状态消息
    总时长约 2.0 秒
    """
    if HAS_RICH:
        frame_texts = [_colorize_frame(f) for f in FOX_FRAMES]
        total_frames = cycles * len(FOX_FRAMES)
        msg_progress = 0  # 当前已完成的消息数
        
        with Live(console=console, refresh_per_second=12, transient=False) as live:
            for i in range(total_frames):
                frame = frame_texts[i % len(FOX_FRAMES)]
                
                # 构建状态消息区
                msg_lines = []
                # 已完成的消息
                for j, (msg, style) in enumerate(BOOT_MESSAGES[:msg_progress]):
                    msg_lines.append(Text(f"  ✓ {msg}", style=f"dim {style}"))
                # 当前进行中的消息
                if msg_progress < len(BOOT_MESSAGES):
                    # 根据帧进度计算打字效果
                    current_msg, current_style = BOOT_MESSAGES[msg_progress]
                    chars_to_show = min(len(current_msg), (i % len(FOX_FRAMES)) * (len(current_msg) // len(FOX_FRAMES) + 2) + 3)
                    partial = current_msg[:chars_to_show]
                    # 添加闪烁光标
                    cursor = "▌" if i % 2 == 0 else " "
                    msg_lines.append(Text(f"  ► {partial}{cursor}", style=current_style))
                
                # 组装内容
                content_parts = [frame, Text("\n")]
                for line in msg_lines:
                    content_parts.append(line)
                    content_parts.append(Text("\n"))
                
                content = Text.assemble(*content_parts)
                live.update(Panel(
                    content,
                    border_style="red",
                    box=box.ROUNDED,
                    title="[bold red]🦊 Sentifox v1.0[/]",
                    subtitle="[dim]舆情智能监测终端[/]",
                ))
                
                time.sleep(0.14)
                
                # 每完成一轮帧，推进一条消息
                if i % len(FOX_FRAMES) == len(FOX_FRAMES) - 1 and msg_progress < len(BOOT_MESSAGES):
                    msg_progress += 1
            
            # 最终帧：显示所有完成消息 + 健康检查
            health_checks = _check_health()
            final_parts = [frame_texts[0], Text("\n")]
            for msg, style in BOOT_MESSAGES:
                final_parts.append(Text(f"  ✓ {msg}\n", style=f"dim {style}"))
            
            # 健康检查摘要
            final_parts.append(Text("\n"))
            ok_count = sum(1 for _, ok, _ in health_checks if ok)
            health_color = "green" if ok_count == len(health_checks) else "yellow"
            final_parts.append(Text(f"  系统状态: {ok_count}/{len(health_checks)} 就绪\n", style=f"bold {health_color}"))
            for name, ok, detail in health_checks:
                icon = "✓" if ok else "⚠"
                color = "green" if ok else "yellow"
                final_parts.append(Text(f"    [{color}]{icon}[/{color}] {name}: {detail}\n", style="dim"))
            
            final_content = Text.assemble(*final_parts)
            live.update(Panel(
                final_content,
                border_style="red",
                box=box.ROUNDED,
                title="[bold red]🦊 Sentifox v1.0[/]",
                subtitle="[dim]舆情智能监测终端[/]",
            ))
            time.sleep(0.5)
    
    else:
        # 纯文本回退模式
        print(FOX_FRAMES[0])
        for msg, _ in BOOT_MESSAGES:
            _print_typing(f"  >> {msg}", delay=0.02)
            time.sleep(0.15)
        print()


def get_static_banner() -> str:
    """获取静态狐狸 Banner"""
    return FOX_BANNER


def print_welcome_with_fox():
    """打印带狐狸的欢迎界面"""
    if HAS_RICH:
        banner = Text(FOX_BANNER, style="bold red")
        banner.highlight_regex(r"o", style="bold bright_white on_red")
        banner.highlight_regex(r"\^", style="bold bright_white")
        
        content = Text.assemble(
            banner, Text("\n\n"),
            Text("🦊 Sentifox ", style="bold red"),
            Text("v1.0  ", style="dim"),
            Text("舆情智能监测终端\n", style="bold white"),
            Text("├─ ", style="dim"), Text("14", style="bold yellow"), Text(" 个子命令\n", style="dim"),
            Text("├─ ", style="dim"), Text("7", style="bold yellow"), Text(" 步完整流水线\n", style="dim"),
            Text("└─ ", style="dim"), Text("Sentifox", style="bold yellow"), Text(" 多智能体传播模拟\n", style="dim"),
            Text("\n输入 ", style="dim"),
            Text("help", style="bold green"),
            Text(" 查看命令  |  ", style="dim"),
            Text("exit", style="bold green"),
            Text(" 退出", style="dim"),
        )
        console.print("")
        console.print(Panel.fit(
            content,
            title="[bold red]Welcome to Sentifox[/]",
            border_style="red",
            box=box.ROUNDED,
            padding=(0, 2),
        ))
        console.print("")
    else:
        print("")
        print("=" * 56)
        print(FOX_BANNER)
        print("  Sentifox v1.0 - 舆情智能监测终端")
        print("  输入 help 查看命令 | exit 退出")
        print("=" * 56)
        print("")
