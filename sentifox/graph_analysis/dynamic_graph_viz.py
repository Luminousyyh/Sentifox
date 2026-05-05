"""
Zep 风格动态认知图谱可视化引擎
基于 vis.js + 自定义 CSS/JS，支持时序播放、认知状态多维编码、传播动画
"""
import json
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict


# ═══════════════════════════════════════════════════════════════
# 颜色与样式配置
# ═══════════════════════════════════════════════════════════════

EMOTION_COLORS = {
    "anger": "#e74c3c",
    "anxiety": "#e67e22",
    "excitement": "#f1c40f",
    "calm": "#2ecc71",
    "fatigue": "#95a5a6",
    "unknown": "#bdc3c7",
}

INFECTION_COLORS = {
    "S": "#7f8c8d",   # 易感 - 灰
    "I": "#e74c3c",   # 感染 - 红
    "R": "#3498db",   # 恢复 - 蓝
}

PLATFORM_COLORS = {
    "微博": "#e6162d",
    "知乎": "#0084ff",
    "小红书": "#ff2442",
    "抖音": "#00f2ea",
    "新闻": "#f39c12",
    "论坛": "#9b59b6",
    "unknown": "#bdc3c7",
}

BDI_GOAL_ICONS = {
    "spread_truth": "📢",
    "defend_stance": "🛡️",
    "gain_attention": "⭐",
    "seek_safety": "🛡️",
    "unknown": "❓",
}

DARK_THEME_CSS = """
:root {
  --bg-primary: #1a1a2e;
  --bg-secondary: #16213e;
  --bg-panel: rgba(22, 33, 62, 0.95);
  --accent: #0f3460;
  --accent-light: #e94560;
  --text-primary: #eee;
  --text-secondary: #aaa;
  --border: #0f3460;
  --success: #2ecc71;
  --warning: #f1c40f;
  --danger: #e74c3c;
  --info: #3498db;
}
"""


@dataclass
class GraphNodeState:
    """节点在某一步的状态"""
    emotion: str = "unknown"
    emotion_intensity: float = 0.0
    infection: str = "S"
    bdi_goal: str = ""
    bdi_intention: str = ""
    is_active: bool = True
    size_multiplier: float = 1.0


@dataclass
class GraphKPI:
    """某一步的 KPI 指标"""
    step: int = 0
    time_label: str = ""
    s_count: int = 0
    i_count: int = 0
    r_count: int = 0
    polarization_index: float = 0.0
    active_threads: int = 0
    dominant_emotion: str = ""


# ═══════════════════════════════════════════════════════════════
# 数据转换
# ═══════════════════════════════════════════════════════════════

def extract_simulation_graph_data(
    simulator_result,
    simulator,
    top_n_nodes: int = 200,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    从 Sentifox 仿真结果提取动态图谱所需数据
    
    Returns:
        (nodes_static, edges_static, timeline_updates, kpi_series)
    """
    agents = simulator.agents
    temporal_graph = simulator.temporal_graph
    
    # ── 静态节点属性 ──
    nodes_static = []
    agent_list = list(agents.values())
    # 优先选择高影响力节点，限制数量以保证性能
    agent_list.sort(key=lambda a: getattr(a, "pagerank", 0) or 0, reverse=True)
    selected_agents = agent_list[:top_n_nodes]
    selected_ids = {a.agent_id for a in selected_agents}
    
    for agent in selected_agents:
        nodes_static.append({
            "id": agent.agent_id,
            "label": getattr(agent, "name", agent.agent_id)[:12],
            "platform": getattr(agent, "platform", "unknown"),
            "mbti": getattr(agent, "mbti", ""),
            "occupation": getattr(agent, "occupation", ""),
            "base_size": min(30, max(10, 8 + (getattr(agent, "pagerank", 0.01) * 200))),
            "is_kol": getattr(agent, "is_kol", False),
        })
    
    # ── 静态边属性 ──
    edges_static = []
    seen_edges = set()
    for agent in selected_agents:
        aid = agent.agent_id
        # 从 temporal_graph 获取关系
        neighbors = temporal_graph.get_neighbors(aid) if hasattr(temporal_graph, "get_neighbors") else []
        for nb_id in neighbors:
            if nb_id not in selected_ids:
                continue
            eid = tuple(sorted([aid, nb_id]))
            if eid in seen_edges:
                continue
            seen_edges.add(eid)
            trust = 0.5
            if hasattr(temporal_graph, "get_edge_weight"):
                trust = temporal_graph.get_edge_weight(aid, nb_id) or 0.5
            edges_static.append({
                "from": aid,
                "to": nb_id,
                "trust": trust,
            })
    
    # 如果没有从 temporal_graph 获取到边，用 agents 的邻居关系
    if not edges_static:
        for agent in selected_agents:
            aid = agent.agent_id
            nbrs = getattr(agent, "neighbor_ids", []) or []
            for nb_id in nbrs:
                if nb_id not in selected_ids:
                    continue
                eid = tuple(sorted([aid, nb_id]))
                if eid in seen_edges:
                    continue
                seen_edges.add(eid)
                edges_static.append({
                    "from": aid,
                    "to": nb_id,
                    "trust": getattr(agent, "trust_weights", {}).get(nb_id, 0.5),
                })
    
    # ── 时序状态更新 ──
    timeline_updates = []
    snapshots = simulator_result.timeline if simulator_result else []
    
    for step_idx, snapshot in enumerate(snapshots):
        step_update = {"step": step_idx, "nodes": {}}
        
        infected_set = set(getattr(snapshot, "infected_nodes", []) or [])
        recovered_set = set(getattr(snapshot, "recovered_nodes", []) or [])
        
        for agent in selected_agents:
            aid = agent.agent_id
            state = {"infection": "S"}
            if aid in infected_set:
                state["infection"] = "I"
            elif aid in recovered_set:
                state["infection"] = "R"
            
            # 认知层数据
            cog = simulator.cognitive_agents.get(aid) if hasattr(simulator, "cognitive_agents") else None
            if cog:
                state["emotion"] = getattr(cog.emotion, "dominant", "unknown")
                state["emotion_intensity"] = round(getattr(cog.emotion, "intensity", 0), 2)
                state["bdi_goal"] = getattr(cog, "current_goal", "") or ""
                state["bdi_intention"] = getattr(cog, "current_intention", "") or ""
            else:
                state["emotion"] = "unknown"
                state["emotion_intensity"] = 0.0
            
            step_update["nodes"][aid] = state
        
        timeline_updates.append(step_update)
    
    # ── KPI 序列 ──
    kpi_series = []
    for step_idx, snapshot in enumerate(snapshots):
        s_count = getattr(snapshot, "susceptible_count", 0)
        i_count = getattr(snapshot, "infected_count", 0)
        r_count = getattr(snapshot, "recovered_count", 0)
        
        # 主导情绪
        emotion_counts = {}
        if step_idx < len(timeline_updates):
            for state in timeline_updates[step_idx]["nodes"].values():
                emo = state.get("emotion", "unknown")
                emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
        dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "unknown"
        
        kpi_series.append({
            "step": step_idx,
            "time_label": f"Step {step_idx}",
            "s_count": s_count,
            "i_count": i_count,
            "r_count": r_count,
            "polarization_index": round(getattr(snapshot, "polarization_index", 0), 3),
            "active_threads": getattr(snapshot, "active_threads", 0),
            "dominant_emotion": dominant,
        })
    
    return nodes_static, edges_static, timeline_updates, kpi_series


# ═══════════════════════════════════════════════════════════════
# HTML 生成
# ═══════════════════════════════════════════════════════════════

def create_dynamic_graph_html(
    nodes_static: List[Dict],
    edges_static: List[Dict],
    timeline_updates: List[Dict],
    kpi_series: List[Dict],
    height: str = "100vh",
    title: str = "Sentifox 动态认知图谱",
) -> str:
    """生成完整的 Zep 风格动态认知图谱 HTML"""
    
    # 序列化数据为 JSON
    nodes_json = json.dumps(nodes_static, ensure_ascii=False)
    edges_json = json.dumps(edges_static, ensure_ascii=False)
    timeline_json = json.dumps(timeline_updates, ensure_ascii=False)
    kpi_json = json.dumps(kpi_series, ensure_ascii=False)
    
    emotion_colors_json = json.dumps(EMOTION_COLORS, ensure_ascii=False)
    infection_colors_json = json.dumps(INFECTION_COLORS, ensure_ascii=False)
    platform_colors_json = json.dumps(PLATFORM_COLORS, ensure_ascii=False)
    bdi_icons_json = json.dumps(BDI_GOAL_ICONS, ensure_ascii=False)
    
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script type="text/javascript" src="https://unpkg.com/vis-network@9.1.2/standalone/umd/vis-network.min.js"></script>
<style>
{DARK_THEME_CSS}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: var(--bg-primary); color: var(--text-primary); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; overflow: hidden; }}

#container {{ display: flex; height: {height}; width: 100%; }}
#main-area {{ flex: 1; position: relative; overflow: hidden; }}
#network {{ width: 100%; height: 100%; }}

/* 顶部 KPI 条 */
#top-bar {{
  position: absolute; top: 0; left: 0; right: 0; height: 56px;
  background: var(--bg-panel); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 20px; gap: 20px;
  z-index: 100; backdrop-filter: blur(8px);
}}
#top-bar .kpi-item {{ display: flex; flex-direction: column; align-items: center; min-width: 60px; }}
#top-bar .kpi-value {{ font-size: 18px; font-weight: 700; }}
#top-bar .kpi-label {{ font-size: 10px; color: var(--text-secondary); text-transform: uppercase; }}
#top-bar .kpi-sep {{ width: 1px; height: 32px; background: var(--border); }}

/* SIR 进度条 */
#sir-bar {{ flex: 1; max-width: 300px; height: 24px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden; display: flex; position: relative; }}
#sir-bar .sir-seg {{ height: 100%; transition: width 0.3s ease; }}
#sir-bar .sir-label {{ position: absolute; top: 0; left: 0; right: 0; bottom: 0; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.8); }}

/* 极化仪表盘 */
#polar-gauge {{ width: 48px; height: 48px; position: relative; }}
#polar-gauge svg {{ transform: rotate(-90deg); }}
#polar-gauge .gauge-value {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); font-size: 13px; font-weight: 700; }}

/* 控制面板 */
#controls {{
  position: absolute; bottom: 20px; left: 20px;
  background: var(--bg-panel); border: 1px solid var(--border);
  border-radius: 12px; padding: 14px 18px;
  display: flex; align-items: center; gap: 14px;
  z-index: 100; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}}
#controls button {{
  background: var(--accent); border: none; color: var(--text-primary);
  padding: 8px 14px; border-radius: 6px; cursor: pointer; font-size: 13px;
  transition: all 0.2s; display: flex; align-items: center; gap: 4px;
}}
#controls button:hover {{ background: var(--accent-light); }}
#controls button:disabled {{ opacity: 0.4; cursor: not-allowed; }}
#controls .ctrl-btn {{ width: 36px; height: 36px; padding: 0; justify-content: center; font-size: 16px; }}

#time-slider {{
  -webkit-appearance: none; width: 200px; height: 4px;
  background: rgba(255,255,255,0.1); border-radius: 2px; outline: none;
}}
#time-slider::-webkit-slider-thumb {{
  -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%;
  background: var(--accent-light); cursor: pointer; border: 2px solid #fff;
}}
#time-label {{ font-size: 12px; color: var(--text-secondary); min-width: 70px; text-align: center; }}

/* 图例 */
#legend {{
  position: absolute; top: 68px; left: 12px;
  background: var(--bg-panel); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 14px; z-index: 90;
  font-size: 11px; max-width: 140px;
}}
#legend .legend-title {{ font-weight: 600; margin-bottom: 6px; color: var(--text-secondary); font-size: 10px; text-transform: uppercase; }}
#legend .legend-row {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; }}
#legend .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}

/* 侧边栏 */
#sidebar {{
  width: 340px; background: var(--bg-secondary); border-left: 1px solid var(--border);
  overflow-y: auto; padding: 20px; display: none; z-index: 110;
}}
#sidebar.active {{ display: block; }}
#sidebar .sb-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }}
#sidebar .sb-title {{ font-size: 16px; font-weight: 700; }}
#sidebar .sb-close {{ background: none; border: none; color: var(--text-secondary); cursor: pointer; font-size: 18px; }}
#sidebar .sb-close:hover {{ color: var(--text-primary); }}

.sb-section {{ margin-bottom: 18px; }}
.sb-section-title {{ font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }}
.sb-card {{ background: rgba(255,255,255,0.03); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }}
.sb-row {{ display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; border-bottom: 1px solid rgba(255,255,255,0.03); }}
.sb-row:last-child {{ border-bottom: none; }}
.sb-row .sb-key {{ color: var(--text-secondary); }}
.sb-row .sb-val {{ font-weight: 600; }}

/* 情绪条 */
.emotion-bar {{ display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 6px; }}
.emotion-bar-seg {{ height: 100%; transition: width 0.3s; }}

/* BDI 链 */
.bdi-chain {{ display: flex; align-items: center; gap: 6px; margin-top: 8px; flex-wrap: wrap; }}
.bdi-item {{ background: var(--accent); padding: 4px 10px; border-radius: 4px; font-size: 12px; }}
.bdi-arrow {{ color: var(--text-secondary); font-size: 12px; }}

/* 筛选器 */
#filters {{
  position: absolute; top: 68px; right: 12px;
  background: var(--bg-panel); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 14px; z-index: 90;
  font-size: 11px;
}}
#filters label {{ display: flex; align-items: center; gap: 6px; margin: 4px 0; cursor: pointer; }}
#filters input[type="checkbox"] {{ accent-color: var(--accent-light); }}

/* 节点 tooltip 样式覆盖 */
div.vis-tooltip {{
  background: var(--bg-panel) !important; color: var(--text-primary) !important;
  border: 1px solid var(--border) !important; border-radius: 6px !important;
  font-family: inherit !important; padding: 8px 12px !important;
}}
</style>
</head>
<body>
<div id="container">
  <div id="main-area">
    <!-- 顶部 KPI 条 -->
    <div id="top-bar">
      <div class="kpi-item">
        <div class="kpi-value" id="kpi-step" style="color:var(--accent-light)">0</div>
        <div class="kpi-label">Step</div>
      </div>
      <div class="kpi-sep"></div>
      <div id="sir-bar">
        <div class="sir-seg" id="sir-s" style="background:#7f8c8d;width:100%"></div>
        <div class="sir-seg" id="sir-i" style="background:#e74c3c;width:0%"></div>
        <div class="sir-seg" id="sir-r" style="background:#3498db;width:0%"></div>
        <div class="sir-label" id="sir-label">S:0 I:0 R:0</div>
      </div>
      <div class="kpi-sep"></div>
      <div class="kpi-item">
        <div class="kpi-value" id="kpi-polar" style="color:var(--info)">0.00</div>
        <div class="kpi-label">极化指数</div>
      </div>
      <div class="kpi-sep"></div>
      <div class="kpi-item">
        <div class="kpi-value" id="kpi-threads" style="color:var(--warning)">0</div>
        <div class="kpi-label">讨论</div>
      </div>
      <div class="kpi-sep"></div>
      <div class="kpi-item">
        <div class="kpi-value" id="kpi-emotion" style="font-size:20px">😐</div>
        <div class="kpi-label" id="kpi-emotion-name">-</div>
      </div>
    </div>

    <!-- 图例 -->
    <div id="legend">
      <div class="legend-title">情绪状态</div>
      <div class="legend-row"><div class="legend-dot" style="background:#e74c3c"></div>愤怒</div>
      <div class="legend-row"><div class="legend-dot" style="background:#e67e22"></div>焦虑</div>
      <div class="legend-row"><div class="legend-dot" style="background:#f1c40f"></div>兴奋</div>
      <div class="legend-row"><div class="legend-dot" style="background:#2ecc71"></div>平静</div>
      <div class="legend-row"><div class="legend-dot" style="background:#95a5a6"></div>疲劳</div>
      <div class="legend-title" style="margin-top:8px">感染状态</div>
      <div class="legend-row"><div class="legend-dot" style="background:#7f8c8d"></div>易感 S</div>
      <div class="legend-row"><div class="legend-dot" style="background:#e74c3c"></div>感染 I</div>
      <div class="legend-row"><div class="legend-dot" style="background:#3498db"></div>恢复 R</div>
    </div>

    <!-- 筛选器 -->
    <div id="filters">
      <div class="legend-title">显示过滤</div>
      <label><input type="checkbox" id="filter-S" checked onchange="applyFilters()"> 易感</label>
      <label><input type="checkbox" id="filter-I" checked onchange="applyFilters()"> 感染</label>
      <label><input type="checkbox" id="filter-R" checked onchange="applyFilters()"> 恢复</label>
      <label><input type="checkbox" id="filter-bridge" onchange="applyFilters()"> 仅桥接节点</label>
    </div>

    <!-- 网络图容器 -->
    <div id="network"></div>

    <!-- 控制面板 -->
    <div id="controls">
      <button class="ctrl-btn" onclick="resetAnimation()" title="重置">⟲</button>
      <button class="ctrl-btn" onclick="stepPrev()" title="上一步">◀</button>
      <button class="ctrl-btn" id="play-btn" onclick="togglePlay()" title="播放/暂停">▶</button>
      <button class="ctrl-btn" onclick="stepNext()" title="下一步">▶</button>
      <input type="range" id="time-slider" min="0" max="{len(timeline_updates)-1 if timeline_updates else 0}" value="0" oninput="seekTo(this.value)">
      <div id="time-label">Step 0</div>
      <select id="speed-select" onchange="setSpeed(this.value)" style="background:var(--accent);color:var(--text-primary);border:none;border-radius:4px;padding:4px 8px;font-size:12px;">
        <option value="0.5">0.5x</option>
        <option value="1" selected>1x</option>
        <option value="2">2x</option>
        <option value="5">5x</option>
      </select>
    </div>
  </div>

  <!-- 节点详情侧边栏 -->
  <div id="sidebar">
    <div class="sb-header">
      <div class="sb-title" id="sb-title">节点详情</div>
      <button class="sb-close" onclick="closeSidebar()">✕</button>
    </div>
    <div id="sb-content"></div>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════════════════════
// 数据注入
// ═══════════════════════════════════════════════════════════════
const NODES_STATIC = {nodes_json};
const EDGES_STATIC = {edges_json};
const TIMELINE = {timeline_json};
const KPI_SERIES = {kpi_json};
const EMOTION_COLORS = {emotion_colors_json};
const INFECTION_COLORS = {infection_colors_json};
const PLATFORM_COLORS = {platform_colors_json};
const BDI_ICONS = {bdi_icons_json};

// ═══════════════════════════════════════════════════════════════
// 状态
// ═══════════════════════════════════════════════════════════════
let currentStep = 0;
let isPlaying = false;
let playInterval = null;
let playSpeed = 1;
let selectedNodeId = null;
let network = null;
let nodesDS = null;
let edgesDS = null;

// ═══════════════════════════════════════════════════════════════
// 初始化 vis.js
// ═══════════════════════════════════════════════════════════════
function initNetwork() {{
  const nodes = NODES_STATIC.map(n => ({{
    id: n.id,
    label: n.label,
    value: n.base_size,
    color: {{
      background: INFECTION_COLORS['S'],
      border: 'rgba(255,255,255,0.2)',
      highlight: {{ background: '#e94560', border: '#fff' }}
    }},
    font: {{ color: '#ddd', size: 12 }},
    shape: n.is_kol ? 'star' : 'dot',
    borderWidth: 1,
    shadow: {{ enabled: true, color: 'rgba(0,0,0,0.5)', size: 10 }},
    title: makeTooltip(n),
    // 自定义属性
    _platform: n.platform,
    _is_kol: n.is_kol,
    _is_bridge: false,
  }}));

  const edges = EDGES_STATIC.map(e => ({{
    from: e.from,
    to: e.to,
    width: Math.max(0.5, e.trust * 2),
    color: {{ color: 'rgba(255,255,255,0.08)', highlight: 'rgba(233,69,96,0.6)' }},
    smooth: {{ type: 'continuous' }},
    arrows: {{ to: {{ enabled: true, scaleFactor: 0.3 }} }},
    _trust: e.trust,
  }}));

  nodesDS = new vis.DataSet(nodes);
  edgesDS = new vis.DataSet(edges);

  const container = document.getElementById('network');
  const data = {{ nodes: nodesDS, edges: edgesDS }};
  const options = {{
    nodes: {{
      shape: 'dot',
      scaling: {{ min: 8, max: 35 }},
      font: {{ size: 12, face: 'system-ui', color: '#ddd' }},
    }},
    edges: {{
      width: 1,
      smooth: {{ type: 'continuous' }},
    }},
    physics: {{
      forceAtlas2Based: {{
        gravitationalConstant: -60,
        centralGravity: 0.005,
        springLength: 120,
        springConstant: 0.05,
      }},
      maxVelocity: 40,
      solver: 'forceAtlas2Based',
      timestep: 0.3,
      stabilization: {{ iterations: 200 }},
    }},
    interaction: {{
      hover: true,
      tooltipDelay: 100,
      hideEdgesOnDrag: true,
    }},
  }};

  network = new vis.Network(container, data, options);

  network.on("click", function(params) {{
    if (params.nodes.length > 0) {{
      selectNode(params.nodes[0]);
    }} else {{
      closeSidebar();
    }}
  }});

  // 初始渲染第 0 步
  renderStep(0);
}}

function makeTooltip(n) {{
  const platColor = PLATFORM_COLORS[n.platform] || '#bdc3c7';
  return `<div style="line-height:1.6">
    <strong style="font-size:14px;color:#fff">${{n.label}}</strong><br/>
    <span style="color:${{platColor}}">●</span> ${{n.platform || 'unknown'}}<br/>
    ${{n.mbti ? 'MBTI: '+n.mbti+'<br/>' : ''}}
    ${{n.occupation ? '职业: '+n.occupation+'<br/>' : ''}}
    ${{n.is_kol ? '<span style="color:#f1c40f">⭐ KOL</span>' : ''}}
  </div>`;
}}

// ═══════════════════════════════════════════════════════════════
// 渲染指定步骤
// ═══════════════════════════════════════════════════════════════
function renderStep(step) {{
  if (step < 0) step = 0;
  if (step >= TIMELINE.length) step = TIMELINE.length - 1;
  currentStep = step;

  const tdata = TIMELINE[step] || {{ nodes: {{}} }};
  const kpi = KPI_SERIES[step] || {{}};
  const updates = [];

  // 更新所有节点状态
  nodesDS.forEach(node => {{
    const state = tdata.nodes[node.id] || {{}};
    const infection = state.infection || 'S';
    const emotion = state.emotion || 'unknown';
    const intensity = state.emotion_intensity || 0;
    const bdiGoal = state.bdi_goal || '';

    // 颜色：感染状态为基础，情绪为发光色
    const baseColor = INFECTION_COLORS[infection] || INFECTION_COLORS['S'];
    const emoColor = EMOTION_COLORS[emotion] || EMOTION_COLORS['unknown'];

    // 根据情绪强度调整大小
    const sizeMult = state.size_multiplier || (1 + intensity * 0.3);
    const baseSize = NODES_STATIC.find(n => n.id === node.id)?.base_size || 15;

    // 构建标签
    let label = node.label;
    if (bdiGoal) {{
      const icon = BDI_ICONS[bdiGoal] || BDI_ICONS['unknown'];
      label = icon + ' ' + label;
    }}

    updates.push({{
      id: node.id,
      color: {{
        background: baseColor,
        border: emoColor,
        highlight: {{ background: emoColor, border: '#fff' }},
      }},
      value: baseSize * sizeMult,
      label: label,
      borderWidth: intensity > 0.5 ? 3 : 1,
      shadow: {{
        enabled: true,
        color: emoColor,
        size: 8 + intensity * 15,
      }},
      // 保存当前状态到节点供筛选使用
      _infection: infection,
      _emotion: emotion,
      _bdi_goal: bdiGoal,
      _bdi_intention: state.bdi_intention || '',
    }});
  }});

  nodesDS.update(updates);

  // 更新 KPI 显示
  updateKPI(kpi);

  // 更新控件
  document.getElementById('time-slider').value = step;
  document.getElementById('time-label').textContent = kpi.time_label || ('Step ' + step);

  // 如果侧边栏打开，更新节点详情
  if (selectedNodeId) {{
    updateSidebar(selectedNodeId);
  }}
}}

function updateKPI(kpi) {{
  document.getElementById('kpi-step').textContent = kpi.step ?? currentStep;

  const total = (kpi.s_count || 0) + (kpi.i_count || 0) + (kpi.r_count || 0);
  const sPct = total > 0 ? (kpi.s_count / total * 100).toFixed(1) : 0;
  const iPct = total > 0 ? (kpi.i_count / total * 100).toFixed(1) : 0;
  const rPct = total > 0 ? (kpi.r_count / total * 100).toFixed(1) : 0;

  document.getElementById('sir-s').style.width = sPct + '%';
  document.getElementById('sir-i').style.width = iPct + '%';
  document.getElementById('sir-r').style.width = rPct + '%';
  document.getElementById('sir-label').textContent = `S:${{kpi.s_count||0}} I:${{kpi.i_count||0}} R:${{kpi.r_count||0}}`;

  document.getElementById('kpi-polar').textContent = (kpi.polarization_index || 0).toFixed(2);
  document.getElementById('kpi-threads').textContent = kpi.active_threads || 0;

  const emoEmoji = {{
    anger: '😡', anxiety: '😰', excitement: '🤩', calm: '😌', fatigue: '😴', unknown: '😐'
  }};
  document.getElementById('kpi-emotion').textContent = emoEmoji[kpi.dominant_emotion] || '😐';
  document.getElementById('kpi-emotion-name').textContent = kpi.dominant_emotion || '-';
}}

// ═══════════════════════════════════════════════════════════════
// 播放控制
// ═══════════════════════════════════════════════════════════════
function togglePlay() {{
  const btn = document.getElementById('play-btn');
  if (isPlaying) {{
    pauseAnimation();
    btn.textContent = '▶';
  }} else {{
    playAnimation();
    btn.textContent = '⏸';
  }}
}}

function playAnimation() {{
  if (isPlaying) return;
  isPlaying = true;
  const delay = 1000 / playSpeed;
  playInterval = setInterval(() => {{
    if (currentStep >= TIMELINE.length - 1) {{
      pauseAnimation();
      document.getElementById('play-btn').textContent = '▶';
      return;
    }}
    renderStep(currentStep + 1);
  }}, delay);
}}

function pauseAnimation() {{
  isPlaying = false;
  if (playInterval) {{
    clearInterval(playInterval);
    playInterval = null;
  }}
}}

function resetAnimation() {{
  pauseAnimation();
  document.getElementById('play-btn').textContent = '▶';
  renderStep(0);
}}

function stepPrev() {{
  pauseAnimation();
  document.getElementById('play-btn').textContent = '▶';
  renderStep(currentStep - 1);
}}

function stepNext() {{
  if (currentStep >= TIMELINE.length - 1) return;
  pauseAnimation();
  document.getElementById('play-btn').textContent = '▶';
  renderStep(currentStep + 1);
}}

function seekTo(step) {{
  pauseAnimation();
  document.getElementById('play-btn').textContent = '▶';
  renderStep(parseInt(step));
}}

function setSpeed(val) {{
  playSpeed = parseFloat(val);
  if (isPlaying) {{
    pauseAnimation();
    playAnimation();
  }}
}}

// ═══════════════════════════════════════════════════════════════
// 节点详情侧边栏
// ═══════════════════════════════════════════════════════════════
function selectNode(nodeId) {{
  selectedNodeId = nodeId;
  updateSidebar(nodeId);
  document.getElementById('sidebar').classList.add('active');
}}

function closeSidebar() {{
  selectedNodeId = null;
  document.getElementById('sidebar').classList.remove('active');
}}

function updateSidebar(nodeId) {{
  const node = NODES_STATIC.find(n => n.id === nodeId);
  if (!node) return;

  const tdata = TIMELINE[currentStep] || {{ nodes: {{}} }};
  const state = tdata.nodes[nodeId] || {{}};
  const emoColor = EMOTION_COLORS[state.emotion] || EMOTION_COLORS['unknown'];
  const infectColor = INFECTION_COLORS[state.infection] || INFECTION_COLORS['S'];
  const platColor = PLATFORM_COLORS[node.platform] || PLATFORM_COLORS['unknown'];

  let html = '';

  // 基本信息
  html += '<div class="sb-section">';
  html += '<div class="sb-section-title">基本信息</div>';
  html += '<div class="sb-card">';
  html += `<div class="sb-row"><span class="sb-key">ID</span><span class="sb-val">${{node.id.substring(0,16)}}</span></div>`;
  html += `<div class="sb-row"><span class="sb-key">平台</span><span class="sb-val" style="color:${{platColor}}">${{node.platform || '-'}}</span></div>`;
  html += `<div class="sb-row"><span class="sb-key">MBTI</span><span class="sb-val">${{node.mbti || '-'}}</span></div>`;
  html += `<div class="sb-row"><span class="sb-key">职业</span><span class="sb-val">${{node.occupation || '-'}}</span></div>`;
  html += `<div class="sb-row"><span class="sb-key">KOL</span><span class="sb-val">${{node.is_kol ? '⭐ 是' : '否'}}</span></div>`;
  html += '</div></div>';

  // 当前状态
  html += '<div class="sb-section">';
  html += '<div class="sb-section-title">当前状态 (Step ' + currentStep + ')</div>';
  html += '<div class="sb-card">';
  html += `<div class="sb-row"><span class="sb-key">感染状态</span><span class="sb-val" style="color:${{infectColor}}">${{state.infection || 'S'}}</span></div>`;
  html += `<div class="sb-row"><span class="sb-key">情绪</span><span class="sb-val" style="color:${{emoColor}}">${{state.emotion || 'unknown'}} (${{(state.emotion_intensity||0).toFixed(2)}})</span></div>`;
  html += `<div class="sb-row"><span class="sb-key">活跃度</span><span class="sb-val">${{state.is_active !== false ? '活跃' : '休眠'}}</span></div>`;
  html += '</div></div>';

  // BDI 决策链
  if (state.bdi_goal) {{
    html += '<div class="sb-section">';
    html += '<div class="sb-section-title">BDI 决策链</div>';
    html += '<div class="sb-card">';
    html += '<div class="bdi-chain">';
    html += `<span class="bdi-item">${{BDI_ICONS[state.bdi_goal]||'❓'}} ${{state.bdi_goal}}</span>`;
    html += '<span class="bdi-arrow">→</span>';
    html += `<span class="bdi-item">${{state.bdi_intention || '?'}}</span>`;
    html += '</div></div></div>';
  }}

  // 情绪占比（该节点在整个时间线的情绪分布）
  const emotionCounts = {{}};
  TIMELINE.forEach(t => {{
    const s = t.nodes[nodeId];
    if (s && s.emotion) emotionCounts[s.emotion] = (emotionCounts[s.emotion]||0)+1;
  }});
  if (Object.keys(emotionCounts).length > 0) {{
    html += '<div class="sb-section">';
    html += '<div class="sb-section-title">情绪分布</div>';
    html += '<div class="sb-card">';
    const totalE = Object.values(emotionCounts).reduce((a,b)=>a+b,0);
    Object.entries(emotionCounts).forEach(([emo, cnt]) => {{
      const pct = (cnt/totalE*100).toFixed(0);
      html += `<div class="sb-row"><span class="sb-key">${{emo}}</span><span class="sb-val" style="color:${{EMOTION_COLORS[emo]||'#aaa'}}">${{pct}}%</span></div>`;
    }});
    html += '</div></div>';
  }}

  document.getElementById('sb-content').innerHTML = html;
  document.getElementById('sb-title').textContent = node.label;
}}

// ═══════════════════════════════════════════════════════════════
// 筛选
// ═══════════════════════════════════════════════════════════════
function applyFilters() {{
  const showS = document.getElementById('filter-S').checked;
  const showI = document.getElementById('filter-I').checked;
  const showR = document.getElementById('filter-R').checked;
  const onlyBridge = document.getElementById('filter-bridge').checked;

  const tdata = TIMELINE[currentStep] || {{ nodes: {{}} }};
  const updates = [];

  nodesDS.forEach(node => {{
    const state = tdata.nodes[node.id] || {{}};
    const infection = state.infection || 'S';
    let visible = true;

    if (infection === 'S' && !showS) visible = false;
    if (infection === 'I' && !showI) visible = false;
    if (infection === 'R' && !showR) visible = false;
    if (onlyBridge && !node._is_bridge) visible = false;

    updates.push({{ id: node.id, hidden: !visible }});
  }});

  nodesDS.update(updates);
}}

// ═══════════════════════════════════════════════════════════════
// 启动
// ═══════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', initNetwork);
</script>
</body>
</html>'''
    return html


def save_dynamic_graph_html(
    filepath: str,
    simulator_result,
    simulator,
    top_n_nodes: int = 200,
    title: str = "Sentifox 动态认知图谱",
) -> str:
    """提取数据并保存动态图谱 HTML 文件"""
    nodes_static, edges_static, timeline_updates, kpi_series = extract_simulation_graph_data(
        simulator_result, simulator, top_n_nodes=top_n_nodes
    )
    html = create_dynamic_graph_html(
        nodes_static, edges_static, timeline_updates, kpi_series, title=title
    )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    return filepath
