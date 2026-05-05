"""
网络图可视化组件
使用 PyVis 生成交互式 HTML 网络图
支持 Zep 风格动态认知图谱（新增）
"""
import os
import tempfile
from typing import Dict, Any, List, Optional
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# 动态图谱引擎
from graph_analysis.dynamic_graph_viz import (
    create_dynamic_graph_html,
    extract_simulation_graph_data,
    save_dynamic_graph_html,
)


def create_pyvis_network(
    G: nx.DiGraph,
    height: str = "500px",
    width: str = "100%",
    bgcolor: str = "#ffffff",
    font_color: str = "#333333",
    directed: bool = True,
) -> Network:
    """创建 PyVis 网络对象"""
    net = Network(
        height=height,
        width=width,
        bgcolor=bgcolor,
        font_color=font_color,
        directed=directed,
        notebook=False,
    )
    net.toggle_physics(True)
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 150}
      },
      "nodes": {
        "font": {"size": 14}
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
        "color": {"inherit": "from", "opacity": 0.4},
        "smooth": {"type": "continuous"}
      }
    }
    """)
    return net


def add_nodes_to_network(
    net: Network,
    G: nx.DiGraph,
    size_attr: str = "engagement",
    color_attr: str = "sentiment",
    label_attr: str = "label",
) -> None:
    """将 NetworkX 节点添加到 PyVis 网络"""
    sentiment_colors = {
        "positive": "#2ecc71",
        "negative": "#e74c3c",
        "neutral": "#95a5a6",
    }
    platform_colors = {
        "微博": "#e6162d",
        "知乎": "#0084ff",
        "小红书": "#ff2442",
        "抖音": "#000000",
        "新闻": "#f39c12",
        "论坛": "#9b59b6",
    }

    for node_id, data in G.nodes(data=True):
        label = data.get(label_attr, str(node_id))[:15]

        # 节点大小
        size_val = data.get(size_attr, 10)
        if isinstance(size_val, (int, float)):
            size = min(50, max(10, 10 + size_val / 20))
        else:
            size = 15

        # 节点颜色
        if color_attr == "sentiment":
            color = sentiment_colors.get(data.get("sentiment", "neutral"), "#95a5a6")
        elif color_attr == "platform":
            color = platform_colors.get(data.get("platform", ""), "#3498db")
        else:
            color = "#3498db"

        title = f"ID: {node_id}<br>"
        title += f"作者: {data.get('label', '')}<br>"
        title += f"平台: {data.get('platform', '')}<br>"
        title += f"情感: {data.get('sentiment', '')}<br>"
        title += f"帖子数: {data.get('post_count', 0)}<br>"
        title += f"互动量: {data.get('engagement', 0)}"

        net.add_node(
            node_id,
            label=label,
            title=title,
            color=color,
            size=size,
        )


def add_edges_to_network(
    net: Network,
    G: nx.DiGraph,
    width_attr: str = "weight",
) -> None:
    """将 NetworkX 边添加到 PyVis 网络"""
    for u, v, data in G.edges(data=True):
        weight = data.get(width_attr, 1.0)
        width = min(5, max(1, weight))

        title = f"关系: {data.get('relation_type', '')}<br>"
        title += f"权重: {weight:.2f}<br>"
        if data.get("count", 1) > 1:
            title += f"次数: {data.get('count', 1)}"

        net.add_edge(u, v, width=width, title=title)


def render_network(
    G: nx.DiGraph,
    height: str = "600px",
    size_attr: str = "engagement",
    color_attr: str = "sentiment",
    key: str = "network",
) -> None:
    """
    在 Streamlit 中渲染交互式网络图
    """
    if G.number_of_nodes() == 0:
        st.warning("图中无节点")
        return

    net = create_pyvis_network(G, height=height, directed=True)
    add_nodes_to_network(net, G, size_attr=size_attr, color_attr=color_attr)
    add_edges_to_network(net, G)

    # 保存为临时 HTML
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
        net.save_graph(f.name)
        html_path = f.name

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    components.html(html_content, height=int(height.replace("px", "")), scrolling=False)

    # 清理临时文件
    try:
        os.remove(html_path)
    except Exception:
        pass


def create_propagation_animation_html(
    G: nx.DiGraph,
    timeline: List[Dict[str, Any]],
    seed_nodes: List[str],
    height: str = "600px",
) -> str:
    """
    生成传播动画 HTML（时间轴逐步显示节点和边）
    返回 HTML 字符串，供 Streamlit 嵌入
    """
    # 简化版本：生成一个多帧动画，使用 vis.js 的 timeline 功能
    # 由于 PyVis 不直接支持动画，我们生成一个自定义 HTML 页面

    nodes_data = []
    for node_id, data in G.nodes(data=True):
        nodes_data.append({
            "id": node_id,
            "label": data.get("label", str(node_id))[:10],
            "group": data.get("sentiment", "neutral"),
            "value": data.get("engagement", 10),
        })

    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            "from": u,
            "to": v,
            "value": data.get("weight", 1),
        })

    # 为每步生成激活节点集合
    step_nodes = []
    active = set(seed_nodes)
    for step in timeline:
        step_nodes.append(list(active))
        new = step.get("new_infected", [])
        if isinstance(new, int):
            # SIR 模型中 new_infected 是数量，需要随机选
            available = [n for n in G.nodes() if n not in active]
            if available and new > 0:
                import random
                selected = random.sample(available, min(new, len(available)))
                active.update(selected)
        else:
            active.update(new)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>body {{ margin: 0; padding: 0; }} #mynetwork {{ width: 100%; height: {height}; border: 1px solid lightgray; }}</style>
    </head>
    <body>
        <div>
            <button onclick="playAnimation()">▶ 播放传播动画</button>
            <button onclick="resetGraph()">⟲ 重置</button>
            <span id="stepInfo">Step: 0 / {len(timeline)}</span>
        </div>
        <div id="mynetwork"></div>
        <script>
            var nodes = new vis.DataSet({nodes_data});
            var edges = new vis.DataSet({edges_data});
            var container = document.getElementById('mynetwork');
            var data = {{nodes: nodes, edges: edges}};
            var options = {{
                nodes: {{
                    shape: 'dot',
                    scaling: {{min: 10, max: 30}},
                    font: {{size: 12}}
                }},
                edges: {{
                    width: 0.5,
                    color: {{opacity: 0.3}}
                }},
                physics: {{
                    stabilization: false
                }}
            }};
            var network = new vis.Network(container, data, options);

            var stepNodes = {step_nodes};
            var allNodeIds = nodes.getIds();
            var currentStep = 0;
            var animationInterval = null;

            function resetGraph() {{
                currentStep = 0;
                document.getElementById('stepInfo').innerText = 'Step: 0 / ' + stepNodes.length;
                nodes.update(allNodeIds.map(id => ({{id: id, color: '#95a5a6', opacity: 0.3}})));
                edges.update(edges.getIds().map(id => ({{id: id, color: {{opacity: 0.1}}}})));
                // 高亮种子节点
                var seeds = {seed_nodes};
                nodes.update(seeds.map(id => ({{id: id, color: '#f1c40f', opacity: 1}})));
            }}

            function playAnimation() {{
                if (animationInterval) clearInterval(animationInterval);
                resetGraph();
                currentStep = 0;
                animationInterval = setInterval(function() {{
                    if (currentStep >= stepNodes.length) {{
                        clearInterval(animationInterval);
                        return;
                    }}
                    var active = stepNodes[currentStep];
                    var colors = {{'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}};
                    var updates = active.map(function(id) {{
                        var node = nodes.get(id);
                        var color = colors[node.group] || '#3498db';
                        return {{id: id, color: color, opacity: 1}};
                    }});
                    nodes.update(updates);
                    document.getElementById('stepInfo').innerText = 'Step: ' + (currentStep + 1) + ' / ' + stepNodes.length;
                    currentStep++;
                }}, 800);
            }}

            // 初始状态
            resetGraph();
        </script>
    </body>
    </html>
    """
    return html


def render_propagation_animation(
    G: nx.DiGraph,
    timeline: List[Dict[str, Any]],
    seed_nodes: List[str],
    height: int = 600,
    key: str = "propagation",
) -> None:
    """在 Streamlit 中渲染传播动画"""
    import streamlit.components.v1 as components
    html = create_propagation_animation_html(G, timeline, seed_nodes, height=f"{height}px")
    components.html(html, height=height + 40, scrolling=False)


# ═══════════════════════════════════════════════════════════════
# Zep 风格动态认知图谱接口
# ═══════════════════════════════════════════════════════════════

def render_dynamic_graph(
    simulator_result,
    simulator,
    height: int = 700,
    top_n_nodes: int = 200,
    key: str = "dynamic_graph",
) -> None:
    """
    在 Streamlit 中渲染 Zep 风格动态认知图谱
    """
    nodes_static, edges_static, timeline_updates, kpi_series = extract_simulation_graph_data(
        simulator_result, simulator, top_n_nodes=top_n_nodes
    )
    html = create_dynamic_graph_html(
        nodes_static, edges_static, timeline_updates, kpi_series,
        height=f"{height}px", title="Sentifox 动态认知图谱"
    )
    components.html(html, height=height, scrolling=False)


# 兼容导入
import streamlit as st
