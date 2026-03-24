# Social Town — 社会动力实验平台 设计规格

**版本**：1.1
**日期**：2026-03-24
**状态**：已批准，待实现（spec review v2 通过）

---

## 1. 项目定位

**目标**：构建一个可配置的 LLM 驱动社会动力实验平台，在封闭小镇中模拟信息传播、社区形成、观点极化和社会事件响应等社会现象。

**参考**：Stanford Generative Agents（Park et al., UIST '23），在其基础上新增社会网络层、实验注入接口和量化评估仪表盘。

**底层模型**：Qwen2.5（开发期 0.5b，生产期 7B），通过 `MODEL_NAME` 环境变量切换，代码零修改。

**交付形式**：Docker Compose 一键启动，包含 Ollama 服务 + 仿真引擎 + Web 仪表盘（端口 8080）。

---

## 2. 整体架构

### 2.1 目录结构

```
social_town/
├── agents/
│   ├── persona.py                # Agent 基类，认知循环入口
│   ├── memory/
│   │   ├── memory_stream.py      # 记忆流：观察/反思/计划统一存储
│   │   └── retrieval.py          # 三分量检索（时效/重要性/相关性）
│   ├── cognitive/
│   │   ├── reflect.py            # 反思引擎
│   │   └── plan.py               # 三层规划引擎（日→时→微）
│   └── social/
│       └── dialogue.py           # 对话生成
├── world/
│   ├── town.py                   # 世界状态、位置与对象树
│   ├── social_graph.py           # 关系图（亲密度/信任度）
│   └── event_injector.py         # 实验性事件注入器
├── core/
│   └── simulation.py             # 主循环（时间步协调器）
├── llm/
│   └── client.py                 # Ollama 客户端，MODEL_NAME 可配置
├── evaluation/
│   ├── metrics.py                # 四维指标计算
│   └── reporter.py               # 实验报告生成（JSON + Markdown）
├── webapp/
│   ├── server.py                 # FastAPI + SSE 实时推送
│   └── static/
│       └── index.html            # 纯 HTML/Chart.js 仪表盘
├── config/
│   └── params.py                 # 全局参数（Agent数/模型/步长/阈值）
├── tests/
│   ├── stage1_memory/
│   ├── stage2_llm/
│   ├── stage3_planning/
│   ├── stage4_social/
│   ├── stage5_simulation/
│   ├── stage6_evaluation/
│   ├── stage7_docker/
│   └── stage8_webapp/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── ollama-entrypoint.sh      # Ollama 启动 + 模型自动拉取
├── data/                         # 运行时数据持久化（gitignore）
├── scripts/
│   ├── run.sh
│   ├── test.sh
│   └── setup.sh
├── main.py
└── CLAUDE.md
```

### 2.2 数据流

```
时间步 t (= 5分钟模拟时间):
  ① PERCEIVE   : Agent 扫描视野 → 新增观察到记忆流
  ② RETRIEVE   : 三分量评分 → 取 top-k 记忆进入上下文
  ③ PLAN/REACT : LLM 决策（继续L3行动 or 响应新事件）
  ④ ACT        : 输出自然语言行动 → 更新位置/对象状态
  ⑤ COMMUNICATE: 若附近有其他 Agent → 触发对话 → 更新社会图
  ⑥ REFLECT    : 若重要性累积 > 100 → 触发反思循环（异步）
  ⑦ PLAN-DAY   : 每模拟天开始 → 生成 L1 日程大纲
  ⑧ METRICS    : 计算四维指标 → SSE 推送至 Web 仪表盘
```

---

## 3. 核心模块规格

### 3.1 记忆系统

**MemoryObject 数据结构**：

```python
@dataclass
class MemoryObject:
    id: str                   # UUID
    content: str              # 自然语言描述
    memory_type: str          # "observation" | "reflection" | "plan"
    created_at: int           # 模拟时间步
    last_accessed: int        # 最近访问时间步
    importance: float         # 1.0-10.0，LLM 评分（创建时调用一次）
    embedding: list[float]    # sentence-transformers 本地嵌入
    source_agent: str | None  # 信息来源 Agent ID（None = 直接观察）
    credibility: float        # 0.0-1.0，信息可信度（直接观察 = 1.0）
```

**credibility 传播规则**（多跳衰减）：
- 直接观察/事件注入：`credibility = 注入时指定值`（直接感知 = 1.0）
- A 传给 B（B 接受后写入记忆）：`credibility_B = credibility_A * trust(A→B)`
- 多跳累积衰减确保远距离谣言可信度趋近于 0

**三分量检索评分**（均归一化至 [0,1] 后等权相加）：

| 分量 | 公式 | 说明 |
|------|------|------|
| 时效性 | `exp(-λ * elapsed_steps)` | λ=0.005，约 200步（~16小时）半衰 |
| 重要性 | `importance / 10.0` | 仅创建时评分，节省 LLM 调用 |
| 相关性 | `cosine_similarity(query_emb, mem_emb)` | sentence-transformers all-MiniLM-L6 |

**反思触发阈值**：累积重要性 > 100（比论文 150 低，适配 0.5b 偏低的评分倾向）

### 3.2 三层规划引擎

| 层级 | 粒度 | LLM 调用 | 触发时机 |
|------|------|----------|----------|
| L1 日程大纲 | 全天 4-6 个活动块 | 1次/模拟天 | 每天开始 |
| L2 时段计划 | 当前块内 3-5 个活动 | 1次/进入新块 | 懒加载 |
| L3 当前行动 | 单个微行动（5-15分钟） | 1次/时间步 | 每步（含环境感知） |

**L1 Prompt 模板**（严控 500 tokens 以内）：
```
你是{name}，{seed_description}。
今天是模拟第{day}天。近期要点：{top3_memories}
用4个活动块描述今天计划，格式：时段|活动|地点，每块一行。
```

**L3 Prompt 模板**（严控 200 tokens 以内）：
```
你是{name}。当前时段：{l2_current_block}。
环境（最近3个实体）：{perceived_entities_top3}。
用一句话描述你此刻的行动：
```

`perceived_entities_top3`：按距离排序，最多取 3 个，每个格式为"[名称]正在[动作]"，确保总 tokens ≤ 60。

**动态重规划**：若 L3 决策为"响应新事件"，则丢弃当前 L2 块，重新生成。

### 3.3 社会网络层

```python
@dataclass
class SocialEdge:
    from_agent: str
    to_agent: str
    intimacy: float       # 0.0-1.0，每次对话后 +0.05（上限 1.0）
    trust: float          # 0.0-1.0，影响信息传播接收率
    interaction_count: int
    last_interaction: int # 时间步
```

**信息传播规则**：
- Agent A 通过对话向 B 传递信息 M（M.credibility = c_A）
- B 接受概率 = `c_A * trust(A→B)`；若 random() < 接受概率，则接受
- 接受后写入 B 记忆流，新记忆的 `credibility = c_A * trust(A→B)`，`source_agent = A.id`
- 多跳传播自动通过 credibility 字段衰减，B 再传 C 时使用 B 记忆中的 credibility

**Persona 意见模型**：
```python
@dataclass
class OpinionVector:
    # 每个维度代表对一类议题的态度（-1=强烈反对, 0=中立, +1=强烈支持）
    values: dict[str, float]  # e.g., {"market_reform": 0.3, "public_health": -0.1}
```
- 初始化：从 seed_description 中 LLM 推断，或随机均匀分布在 [-1, 1]
- 更新：每次接受信息 M 时，若 M 涉及某议题，opinion[议题] += 0.1 * M.credibility * sign(M影响方向)，clip 至 [-1, 1]
- 极化指数 BC 计算基于所有 Agent 对同一议题的 opinion 值分布

### 3.4 事件注入器

```python
injector.inject_event(
    event_type: str,          # "rumor" | "breaking_news" | "public_health" | "election"
    content: str,             # 事件自然语言描述
    seed_agents: list[str],   # 初始知情者 ID 列表
    credibility: float,       # 0.0-1.0，影响传播接收率
    step: int,                # 注入时间步（None = 立即）
)
```

---

## 4. 评估指标

| 指标 | 计算方法 | 正常范围 |
|------|----------|----------|
| **信息扩散率** | 知晓事件 Agent 数 / 总 Agent 数 × 100% | 0%-100% |
| **网络密度** | 实际边数 / (N×(N-1)/2) | 0.0-1.0 |
| **意见极化指数** | Bimodality Coefficient (BC) on opinion values | 0.0-1.0（>0.55 视为极化） |
| **社会响应延迟** | 事件注入步 → 50% Agent 知晓的步数差 | 依 Agent 数量而定 |

每 10 个时间步计算一次，通过 SSE 推送至 Web 仪表盘。

---

## 5. Web 仪表盘规格

**技术栈**：FastAPI（SSE 推送）+ 纯 HTML/Chart.js（无前端框架）

**界面布局**：
```
┌─────────────────────────────────────────────────────┐
│  Social Town Dashboard              Step: 042 Day: 3 │
├──────────────┬──────────────┬────────────────────────┤
│ 信息扩散折线图 │ 网络密度折线图 │  Agent 状态列表         │
│              │              │  [name] [location] [act]│
├──────────────┴──────────────┤                        │
│     极化指数柱状图            │                        │
├─────────────────────────────┴────────────────────────┤
│  事件注入控制面板 [类型] [内容] [可信度] [注入] 按钮   │
└─────────────────────────────────────────────────────┘
```

**端口**：8080（Docker 映射）

---

## 6. Docker 环境

```yaml
# docker-compose.yml
services:
  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
      - ./docker/ollama-entrypoint.sh:/entrypoint.sh
    entrypoint: ["/bin/sh", "/entrypoint.sh"]
    ports:
      - "11434:11434"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 10s
      timeout: 5s
      retries: 12         # 最多等待 2 分钟（含模型拉取时间）
      start_period: 30s

  social-town:
    build: .
    depends_on:
      ollama:
        condition: service_healthy   # 等待 Ollama healthcheck 通过
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - MODEL_NAME=qwen2.5:0.5b
      - NUM_AGENTS=10
      - SIMULATION_DAYS=3
      - HF_HOME=/app/.cache/huggingface  # sentence-transformers 模型缓存
    volumes:
      - ./data:/app/data
      - hf_cache:/app/.cache/huggingface  # 持久化 HF 模型，避免重复下载
    ports:
      - "8080:8080"

volumes:
  ollama_data:
  hf_cache:
```

**`docker/ollama-entrypoint.sh`**（Ollama 启动 + 自动拉取模型）：
```bash
#!/bin/sh
# 启动 Ollama 服务（后台）
ollama serve &
SERVE_PID=$!

# 等待 API 就绪
until curl -sf http://localhost:11434/api/tags; do
  sleep 2
done

# 拉取模型（若已存在则跳过）
MODEL_NAME=${MODEL_NAME:-qwen2.5:0.5b}
ollama pull "$MODEL_NAME"

# 保持 Ollama 服务前台运行
wait $SERVE_PID
```

---

## 7. 开发阶段与测试门控（深度工作法）

每个阶段完成所有测试方可进入下一阶段。

| 阶段 | 核心内容 | 测试门控标准 |
|------|----------|--------------|
| **Stage 1** | MemoryStream + 三分量检索 | 单元测试覆盖率 ≥ 90%；检索 top-3 精度 ≥ 0.8（手工标注测试集） |
| **Stage 2** | LLM Client + Prompt 解析 | 10次调用成功率 ≥ 95%；响应格式解析成功率 ≥ 90% |
| **Stage 3** | 三层规划引擎（L1/L2/L3） | L1 生成格式合法；重规划触发可验证；无 None/空输出 |
| **Stage 4** | 社会网络 + 对话 + 信息传播 | 对话生成非空；关系图更新正确；信息传播接收率与 credibility 正相关 |
| **Stage 5** | 主仿真循环 | 10个 Agent 运行 50步无崩溃；记忆/计划/反思三者均在日志中出现 |
| **Stage 6** | 评估指标计算 | 四维指标计算值在合理范围；事件注入后扩散率单调递增 |
| **Stage 7** | Docker Compose 集成 | `docker-compose up` 一键启动；Ollama 模型自动拉取；仿真成功运行 |
| **Stage 8** | Web 仪表盘 | 浏览器访问 localhost:8080；四维图表实时更新；事件注入按钮功能正常 |

---

## 8. 技术约束（CLAUDE.md 边界）

见 `CLAUDE.md` 文件。

---

## 9. 配置参数（config/params.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NUM_AGENTS` | 10 | Agent 数量（5-50） |
| `MODEL_NAME` | `qwen2.5:0.5b` | Ollama 模型名 |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API 地址 |
| `STEPS_PER_DAY` | 288 | 每模拟天的时间步数（5分钟×288=24小时） |
| `REFLECTION_THRESHOLD` | 100 | 触发反思的累积重要性阈值 |
| `MEMORY_TOP_K` | 10 | 检索时选取的记忆数量 |
| `SOCIAL_GRAPH_DECAY` | 0.01 | 亲密度每步自然衰减值 |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers 模型 |
| `METRICS_UPDATE_INTERVAL` | 10 | 评估指标更新间隔（步） |

---

## 10. 依赖项

```
# Python
fastapi>=0.100
uvicorn>=0.22
sentence-transformers>=2.2
networkx>=3.1        # 社会图分析
numpy>=1.24
httpx>=0.24          # Ollama 客户端
pytest>=7.4
pytest-asyncio>=0.21
scipy>=1.11          # BC 极化指数计算

# Docker
ollama/ollama:latest
python:3.11-slim
```
