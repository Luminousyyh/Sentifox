# Sentifox 🦊

**认知增强的多智能体舆情分析与传播仿真系统**

Sentifox 是一个面向中文社交媒体的舆情分析平台，集成了情感分析、话题聚类、传播图谱可视化、RAG 智能问答，以及基于**多智能体时态图谱**的舆情传播仿真引擎。

> 区别于传统的 SIR / Agent-based 传播模型，Sentifox 的仿真引擎让每个 Agent 具备独立的 **BDI 认知架构**（信念-欲望-意图）、**情绪系统**和**记忆机制**，在**时态图谱**上自主感知、决策与交互，使传播成为群体智能的涌现结果。

---

## ✨ 核心特性

### 1. 多源舆情采集
- 支持微博、知乎、小红书、抖音、新闻、论坛等平台
- 内置高保真模拟数据生成器（无需真实账号即可体验完整功能）
- 预留真实爬虫接口，继承 `BaseCrawler` 即可接入新平台

### 2. 情感分析与话题聚类
- 基于 HuggingFace Transformers 预训练中文情感模型
- jieba 分词 + TF-IDF + KMeans 自动发现热点话题
- 实时情感指数计算与异常检测

### 3. 传播图谱与关键节点识别
- 交互式网络可视化（PyVis + 暗色主题）
- PageRank、Betweenness Centrality、社区发现
- KOL 识别与负面情绪放大器检测

### 4. 🔬 Sentifox 多智能体传播仿真（核心亮点）

**四层架构：**

| 层级 | 能力 | 说明 |
|------|------|------|
| **时态图谱层** | TemporalGraph | 关系随时间衰减、过期，作为 Agent 共享的"世界记忆" |
| **人格 Agent 层** | PersonaAgent | 每个 Agent 有独立人格、MBTI、职业、平台特性、活跃时段 |
| **认知层** | CognitiveAgent (BDI+情绪) | 信念-欲望-意图决策链、情绪状态（唤起/效价/支配）、语义记忆与情景记忆 |
| **通信层** | AgentCommunication | 讨论线程、信息级联、回音室检测、情绪传染 |

**仿真每步循环：**
1. 干预措施注入（可选）
2. Agent 环境感知（热点事件、社区情绪、近期关系）
3. 独立决策（认知 Agent 走 BDI 决策链）
4. 动作执行（转发/发帖/评论/点赞 → 修改时态图谱）
5. 通信层处理（观察邻居行为 + 处理收件箱消息）
6. 传播动力学（感染扩散 + 情绪传染）
7. 环境演化（关系衰减、过期清理）
8. 记录快照（SIR 状态、情绪分布、极化指数、回音室数量等）

**可视化输出：**
- Zep 风格动态认知图谱（vis.js）：时序播放、情绪多维编码、KPI 仪表盘、节点详情侧边栏
- HTML 仿真报告（Chart.js）：SIR 曲线、极化/讨论趋势、情绪卡片、BDI 决策链、桥接节点

### 5. RAG 智能问答
- 基于 ChromaDB 向量检索 + LLM 生成
- 支持平台过滤、情感过滤、自然语言查询

### 6. 告警与报告
- 负面情感占比阈值告警
- 提及量突增检测
- 自动导出 Word / HTML 格式报告

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

> 首次运行时会自动下载情感分析模型（~500MB），请确保网络畅通。

### 启动 Web 界面

```bash
streamlit run app.py
```

浏览器将自动打开 `http://localhost:8501`

### 一键运行完整流水线（CLI）

```bash
python cli.py pipeline --keywords "品牌,产品" --simulate-steps 48 --simulate-report
```

9 步流水线：数据采集 → 情感分析 → 话题聚类 → 向量同步 → 传播图谱 → RAG 洞察 → 生成报告 → **Sentifox 仿真** → **仿真报告**

---

## 📁 项目结构

```
sentifox/
├── app.py                          # Streamlit Web UI 主入口
├── cli.py                          # 命令行工具（9步完整流水线）
├── config.py                       # 全局配置（API Key 从环境变量读取）
├── requirements.txt
├── pyproject.toml
├── README.md
├── crawlers/                       # 数据采集
│   ├── base.py                     # 爬虫基类与数据模型
│   ├── mock_generator.py           # 模拟数据生成器
│   └── manager.py                  # 多平台爬虫管理
├── analysis/                       # 分析引擎
│   ├── sentiment.py                # 情感分析（Transformers）
│   ├── topic_clustering.py         # 话题聚类
│   ├── trend_analysis.py           # 趋势分析
│   └── insight_generator.py        # 洞察生成
├── graph_analysis/                 # 图谱与仿真（核心）
│   ├── sentifox_simulator.py       # ⭐ Sentifox 多智能体仿真引擎
│   ├── temporal_graph.py           # 时态图谱
│   ├── temporal_engine.py          # 时间引擎
│   ├── persona_agent.py            # 人格 Agent
│   ├── cognitive_agent.py          # 认知 Agent（BDI+情绪+记忆）
│   ├── agent_communication.py      # Agent 通信层
│   ├── platform_dynamics.py        # 平台动力学规则
│   ├── interventions.py            # 干预机制
│   ├── dynamic_graph_viz.py        # Zep 风格动态图谱 HTML 生成
│   ├── graph_viz.py                # PyVis 网络可视化
│   ├── graph_builder.py            # 图构建
│   ├── influencer_detection.py     # KOL 识别
│   └── simulation_report.py        # 仿真 HTML 报告生成
├── rag/                            # RAG 问答
│   ├── rag_engine.py
│   └── document_processor.py
├── visualization/                  # 图表组件
│   └── charts.py
├── reports/                        # 报告生成
│   └── generator.py
├── scheduler/                      # 定时任务
│   └── tasks.py
└── utils/                          # 工具
    ├── database.py
    └── stopwords.txt
```

---

## 🔧 配置说明

在 `config.py` 中调整参数，或**通过环境变量注入敏感信息**：

```bash
export VOLCES_API_KEY="your-api-key"
export WEIBO_COOKIE="your-weibo-cookie"
export ZHIHU_COOKIE="your-zhihu-cookie"
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `SENTIMENT_MODEL` | 情感分析模型路径 | `models/sentiment/` |
| `TOPIC_CLUSTER_NUM` | 话题聚类数 | 5 |
| `ALERT_CONFIG` | 告警阈值 | 负面 60%, 突增 3x |
| `LLM_CONFIG` | LLM API 配置 | 火山引擎 Doubao |
| `RAG_CONFIG` | 向量库配置 | ChromaDB |

---

## 🧪 扩展开发

### 接入真实爬虫

继承 `crawlers.base.BaseCrawler`：

```python
from crawlers.base import BaseCrawler, Post, GraphEdge

class MyCrawler(BaseCrawler):
    def crawl(self, max_posts=100) -> List[Post]:
        # 实现采集逻辑
        pass

    def extract_edges(self, posts: List[Post]) -> List[GraphEdge]:
        # 提取传播关系
        pass
```

### 自定义仿真干预

```python
from graph_analysis.interventions import Intervention

class MyIntervention(Intervention):
    def apply(self, graph, agents, step):
        # 在指定步数对图谱/Agent 施加干预
        for agent in agents.values():
            agent.sentiment_tendency += 0.2
        return {"affected": len(agents)}

simulator.add_intervention(MyIntervention(name="正面引导", step=10))
```

---

## 🛠️ 技术栈

- **Python 3.10+**
- **Streamlit** — Web UI
- **Transformers + PyTorch** — 情感分析
- **scikit-learn** — 聚类
- **NetworkX + PyVis + vis.js** — 图分析与可视化
- **Plotly** — 交互式图表
- **ChromaDB** — 向量检索
- **python-docx** — Word 报告
- **APScheduler** — 定时任务
- **Rich** — CLI 美化

---

## 📄 License

[MIT](LICENSE)
