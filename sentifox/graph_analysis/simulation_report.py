#!/usr/bin/env python3
"""
Sentifox 仿真 HTML 报告生成器
生成包含时态图谱动态可视化、感染曲线、Agent 画像、干预对比的 HTML 报告
"""
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from graph_analysis.sentifox_simulator import SimulationResult


def generate_simulation_report(result: SimulationResult, 
                               temporal_graph_data: Dict[str, Any],
                               cognitive_data: Optional[Dict[str, Any]] = None,
                               output_path: Optional[str] = None,
                               simulator=None) -> str:
    """
    生成仿真 HTML 报告
    
    Args:
        result: 仿真结果
        temporal_graph_data: 时态图谱数据（用于可视化）
        cognitive_data: 认知层数据（BDI决策链、情绪状态等）
        output_path: 输出路径（默认自动生成）
        simulator: SentifoxSimulator 实例（可选，用于生成动态认知图谱）
    
    Returns:
        生成的 HTML 文件路径
    """
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"sentifox_simulation_report_{ts}.html"
    
    # 准备数据
    timeline_data = []
    for snap in result.timeline:
        timeline_data.append({
            "step": snap.step,
            "hour": snap.hour,
            "infected": snap.infected_count,
            "susceptible": snap.susceptible_count,
            "recovered": snap.recovered_count,
            "new_actions": snap.new_actions,
            "new_relations": snap.new_relations,
            "polarization": getattr(snap, "polarization_index", 0.0),
            "echo_chambers": getattr(snap, "echo_chamber_count", 0),
            "discussions": getattr(snap, "discussion_count", 0),
        })
    
    # 提取干预事件
    intervention_events = [e for e in result.events if e.get("type") == "intervention"]
    
    # 可选：生成动态认知图谱
    dynamic_graph_path = None
    if simulator is not None:
        try:
            from graph_analysis.dynamic_graph_viz import save_dynamic_graph_html
            graph_file = output_path.replace(".html", "_graph.html")
            save_dynamic_graph_html(graph_file, result, simulator, top_n_nodes=200)
            dynamic_graph_path = os.path.basename(graph_file)
        except Exception as e:
            print(f"[WARN] 动态图谱生成失败: {e}")
    
    # 构建 HTML
    html = _build_html(timeline_data, temporal_graph_data, intervention_events, result, cognitive_data or {}, dynamic_graph_path)
    
    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return os.path.abspath(output_path)


def _build_html(timeline_data: List[Dict], graph_data: Dict,
                interventions: List[Dict], result: SimulationResult,
                cognitive_data: Dict[str, Any],
                dynamic_graph_path: Optional[str] = None) -> str:
    """构建 HTML 内容"""
    
    timeline_json = json.dumps(timeline_data, ensure_ascii=False)
    graph_json = json.dumps(graph_data, ensure_ascii=False)
    intervention_json = json.dumps(interventions, ensure_ascii=False)
    cognitive_json = json.dumps(cognitive_data, ensure_ascii=False)
    
    # 统计摘要
    if timeline_data:
        final = timeline_data[-1]
        peak_infected = max(t["infected"] for t in timeline_data)
        total_steps = len(timeline_data)
    else:
        final = {"infected": 0, "susceptible": 0, "recovered": 0}
        peak_infected = 0
        total_steps = 0
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentifox Sentifox 传播仿真报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .header {{
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-radius: 16px;
            margin-bottom: 30px;
            border: 1px solid #334155;
        }}
        .header h1 {{ font-size: 2.5em; color: #f97316; margin-bottom: 10px; }}
        .header p {{ color: #94a3b8; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #1e293b;
            padding: 24px;
            border-radius: 12px;
            border: 1px solid #334155;
            text-align: center;
        }}
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #f97316;
        }}
        .stat-card .label {{ color: #94a3b8; margin-top: 8px; }}
        .panel {{
            background: #1e293b;
            border-radius: 12px;
            border: 1px solid #334155;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .panel h2 {{
            color: #f97316;
            margin-bottom: 20px;
            font-size: 1.3em;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
            width: 100%;
        }}
        .intervention-line {{
            position: absolute;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #ef4444;
            z-index: 10;
        }}
        .intervention-label {{
            position: absolute;
            top: -25px;
            transform: translateX(-50%);
            background: #ef4444;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            white-space: nowrap;
        }}
        .timeline {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        .timeline-item {{
            display: flex;
            align-items: flex-start;
            gap: 16px;
            padding: 12px;
            background: #0f172a;
            border-radius: 8px;
            border-left: 3px solid #334155;
        }}
        .timeline-item.intervention {{ border-left-color: #ef4444; }}
        .timeline-item.infection {{ border-left-color: #f97316; }}
        .timeline-item.post {{ border-left-color: #3b82f6; }}
        .timeline-step {{
            min-width: 60px;
            color: #64748b;
            font-size: 0.9em;
        }}
        .timeline-content {{ flex: 1; }}
        .timeline-type {{
            font-weight: bold;
            margin-bottom: 4px;
        }}
        .footer {{
            text-align: center;
            padding: 40px;
            color: #64748b;
            border-top: 1px solid #334155;
            margin-top: 40px;
        }}
        .two-col {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 24px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦊 Sentifox Sentifox</h1>
            <p>舆情动态传播仿真报告</p>
            <p style="margin-top:10px;color:#64748b;font-size:0.9em;">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <!-- 统计卡片 -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{final['infected']}</div>
                <div class="label">最终感染数</div>
            </div>
            <div class="stat-card">
                <div class="value">{final['susceptible']}</div>
                <div class="label">最终易感数</div>
            </div>
            <div class="stat-card">
                <div class="value">{final['recovered']}</div>
                <div class="label">最终恢复数</div>
            </div>
            <div class="stat-card">
                <div class="value">{peak_infected}</div>
                <div class="label">峰值感染数</div>
            </div>
            <div class="stat-card">
                <div class="value">{total_steps}</div>
                <div class="label">仿真步数</div>
            </div>
            <div class="stat-card">
                <div class="value">{len(interventions)}</div>
                <div class="label">干预措施</div>
            </div>
            <div class="stat-card">
                <div class="value">{round(max((t.get('polarization', 0) for t in timeline_data), default=0), 2)}</div>
                <div class="label">峰值极化指数</div>
            </div>
            <div class="stat-card">
                <div class="value">{sum(t.get('echo_chambers', 0) for t in timeline_data) // max(1, len(timeline_data) // 10)}</div>
                <div class="label">平均回音室数</div>
            </div>
        </div>
        
        <!-- 传播曲线 -->
        <div class="panel">
            <h2>📈 传播动力学曲线</h2>
            <div class="chart-container">
                <canvas id="infectionChart"></canvas>
            </div>
        </div>
        
        <!-- 认知层指标曲线 -->
        <div class="two-col">
            <div class="panel">
                <h2>⚡ 群体极化指数</h2>
                <div class="chart-container">
                    <canvas id="polarizationChart"></canvas>
                </div>
            </div>
            <div class="panel">
                <h2>💬 讨论与回音室动态</h2>
                <div class="chart-container">
                    <canvas id="discussionChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="two-col">
            <!-- 事件时间线 -->
            <div class="panel">
                <h2>📋 关键事件时间线</h2>
                <div class="timeline" id="timeline">
                    <!-- JS 填充 -->
                </div>
            </div>
            
            <!-- 干预记录 -->
            <div class="panel">
                <h2>🎛️ 干预措施记录</h2>
                <div class="timeline" id="interventionList">
                    <!-- JS 填充 -->
                </div>
            </div>
        </div>
        
        <!-- 情绪状态面板 -->
        <div class="panel">
            <h2>🎭 情绪状态时间线</h2>
            <div id="emotionTimeline" style="display:flex;flex-wrap:wrap;gap:12px;padding:8px;">
                <!-- JS 填充 -->
            </div>
        </div>
        
        <!-- BDI 决策链 -->
        <div class="panel">
            <h2>🧠 Agent BDI 决策链（采样）</h2>
            <div id="bdiChain" style="display:flex;flex-direction:column;gap:12px;">
                <!-- JS 填充 -->
            </div>
        </div>
        
        <!-- 桥接节点 -->
        <div class="panel">
            <h2>🔗 关键桥接节点</h2>
            <div id="bridgeNodes" style="display:flex;flex-wrap:wrap;gap:12px;">
                <!-- JS 填充 -->
            </div>
        </div>
        
        <!-- 动态认知图谱 -->
        {f'<div class="panel" style="padding:0;overflow:hidden;"><div style="padding:20px 20px 0;"><h2>🧬 动态认知图谱</h2><p style="color:#94a3b8;font-size:13px;margin:4px 0 0;">Zep 风格交互式时序图谱：播放按钮控制时间演进，点击节点查看 BDI 决策链与情绪状态</p></div><iframe src="{dynamic_graph_path}" style="width:100%;height:720px;border:none;"></iframe></div>' if dynamic_graph_path else ''}
        
        <!-- 图谱统计 -->
        <div class="panel">
            <h2>🕸️ 时态图谱统计</h2>
            <pre id="graphStats" style="background:#0f172a;padding:16px;border-radius:8px;overflow:auto;"></pre>
        </div>
    </div>
    
    <div class="footer">
        <p>Sentifox Sentifox — 多智能体舆情传播模拟引擎</p>
    </div>
    
    <script>
        // 数据
        const timelineData = {timeline_json};
        const graphData = {graph_json};
        const interventions = {intervention_json};
        const cognitiveData = {cognitive_json};
        
        // 传播曲线图表
        const ctx = document.getElementById('infectionChart').getContext('2d');
        const labels = timelineData.map(d => `Step ${{d.step}}`);
        
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{
                        label: '感染 I',
                        data: timelineData.map(d => d.infected),
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 2,
                    }},
                    {{
                        label: '易感 S',
                        data: timelineData.map(d => d.susceptible),
                        borderColor: '#64748b',
                        backgroundColor: 'rgba(100, 116, 139, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 2,
                    }},
                    {{
                        label: '恢复 R',
                        data: timelineData.map(d => d.recovered),
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 2,
                    }},
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false,
                }},
                plugins: {{
                    legend: {{
                        labels: {{ color: '#e2e8f0' }}
                    }}
                }},
                scales: {{
                    x: {{
                        grid: {{ color: '#334155' }},
                        ticks: {{ color: '#94a3b8' }}
                    }},
                    y: {{
                        grid: {{ color: '#334155' }},
                        ticks: {{ color: '#94a3b8' }}
                    }}
                }}
            }}
        }});
        
        // 填充时间线
        const timelineEl = document.getElementById('timeline');
        const significantEvents = timelineData.filter(d => d.new_actions > 0 || d.new_relations > 0).slice(-20);
        
        significantEvents.forEach(event => {{
            const item = document.createElement('div');
            item.className = 'timeline-item post';
            item.innerHTML = `
                <div class="timeline-step">Step ${{event.step}}</div>
                <div class="timeline-content">
                    <div class="timeline-type">动作记录</div>
                    <div>新动作: ${{event.new_actions}}, 新关系: ${{event.new_relations}}</div>
                </div>
            `;
            timelineEl.appendChild(item);
        }});
        
        // 填充干预记录
        const interventionEl = document.getElementById('interventionList');
        if (interventions.length === 0) {{
            interventionEl.innerHTML = '<div class="timeline-item"><div class="timeline-content">无干预措施</div></div>';
        }} else {{
            interventions.forEach(inv => {{
                const item = document.createElement('div');
                item.className = 'timeline-item intervention';
                item.innerHTML = `
                    <div class="timeline-step">Step ${{inv.step}}</div>
                    <div class="timeline-content">
                        <div class="timeline-type">' + (inv.intervention ? inv.intervention : inv.type) + '</div>
                        <div>${{JSON.stringify(inv.effect)}}</div>
                    </div>
                `;
                interventionEl.appendChild(item);
            }});
        }}
        
        // 极化指数图表
        const polarCtx = document.getElementById('polarizationChart').getContext('2d');
        new Chart(polarCtx, {{
            type: 'line',
            data: {{
                labels: timelineData.map(d => 'Step ' + d.step),
                datasets: [{{
                    label: '极化指数',
                    data: timelineData.map(d => d.polarization || 0),
                    borderColor: '#a855f7',
                    backgroundColor: 'rgba(168, 85, 247, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 2,
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#e2e8f0' }} }} }},
                scales: {{
                    x: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }},
                    y: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }}, min: 0, max: 1 }}
                }}
            }}
        }});
        
        // 讨论与回音室图表
        const discCtx = document.getElementById('discussionChart').getContext('2d');
        new Chart(discCtx, {{
            type: 'line',
            data: {{
                labels: timelineData.map(d => 'Step ' + d.step),
                datasets: [
                    {{
                        label: '回音室数',
                        data: timelineData.map(d => d.echo_chambers || 0),
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 2,
                    }},
                    {{
                        label: '讨论线程',
                        data: timelineData.map(d => d.discussions || 0),
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 2,
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#e2e8f0' }} }} }},
                scales: {{
                    x: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }},
                    y: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }}
                }}
            }}
        }});
        
        // 情绪状态时间线
        const emotionEl = document.getElementById('emotionTimeline');
        if (cognitiveData.emotion_samples) {{
            const emotionEmojis = {{ anger: '😠', anxiety: '😰', excitement: '🤩', calm: '😌', fatigue: '😴' }};
            cognitiveData.emotion_samples.forEach(sample => {{
                const card = document.createElement('div');
                card.style.cssText = 'background:#0f172a;padding:12px 16px;border-radius:8px;border:1px solid #334155;min-width:140px;';
                const emoji = emotionEmojis[sample.emotion] || '😐';
                card.innerHTML = `<div style="font-size:1.5em;text-align:center;">${{emoji}}</div><div style="text-align:center;color:#94a3b8;font-size:0.85em;margin-top:4px;">${{sample.agent_name || sample.agent_id}}</div><div style="text-align:center;color:#e2e8f0;font-size:0.9em;">${{sample.emotion}}(${{sample.intensity}})</div>`;
                emotionEl.appendChild(card);
            }});
        }} else {{
            emotionEl.innerHTML = '<div style="color:#64748b;">无情绪数据</div>';
        }}
        
        // BDI 决策链
        const bdiEl = document.getElementById('bdiChain');
        if (cognitiveData.bdi_samples && cognitiveData.bdi_samples.length > 0) {{
            cognitiveData.bdi_samples.forEach(sample => {{
                const card = document.createElement('div');
                card.style.cssText = 'background:#0f172a;padding:12px 16px;border-radius:8px;border-left:3px solid #f97316;';
                const reasoning = sample.reasoning ? sample.reasoning.join(' → ') : 'N/A';
                card.innerHTML = `<div style="color:#f97316;font-weight:bold;">${{sample.agent_name || sample.agent_id}} @ Step ${{sample.step}}</div><div style="color:#94a3b8;font-size:0.9em;margin-top:4px;">情绪: ${{sample.emotion}} | 目标: ${{sample.desire}} | 行动: ${{sample.intention}}</div><div style="color:#64748b;font-size:0.85em;margin-top:4px;">推理: ${{reasoning}}</div>`;
                bdiEl.appendChild(card);
            }});
        }} else {{
            bdiEl.innerHTML = '<div style="color:#64748b;">无 BDI 决策数据</div>';
        }}
        
        // 桥接节点
        const bridgeEl = document.getElementById('bridgeNodes');
        if (cognitiveData.bridge_nodes && cognitiveData.bridge_nodes.length > 0) {{
            cognitiveData.bridge_nodes.forEach(node => {{
                const card = document.createElement('div');
                card.style.cssText = 'background:#0f172a;padding:12px 16px;border-radius:8px;border:1px solid #334155;';
                card.innerHTML = `<div style="color:#22c55e;font-weight:bold;">🔗 ${{node.name || node.id}}</div><div style="color:#94a3b8;font-size:0.85em;">桥接分数: ${{node.score ? node.score.toFixed(3) : 'N/A'}}</div>`;
                bridgeEl.appendChild(card);
            }});
        }} else {{
            bridgeEl.innerHTML = '<div style="color:#64748b;">未检测到显著桥接节点</div>';
        }}
        
        // 图谱统计
        document.getElementById('graphStats').textContent = JSON.stringify(graphData.stats || {{}}, null, 2);
    </script>
</body>
</html>"""
    
    return html
