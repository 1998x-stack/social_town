# Social Town — CLAUDE.md
# 技术边界、编码规范与评估标准

---

## 项目概述

一个基于 LLM 的社会动力实验平台，模拟封闭小镇中的信息传播、社区形成、观点极化和社会事件响应。
底层使用 Qwen2.5（via Ollama），Docker Compose 一键部署。

---

## 1. 技术边界（硬约束）

### 1.1 模型与 LLM 调用

- **唯一 LLM 接口**：所有 LLM 调用必须通过 `llm/client.py` 的 `OllamaClient`，禁止在其他模块直接调用 Ollama API
- **模型切换**：通过 `MODEL_NAME` 环境变量，代码中禁止硬编码模型名
- **Prompt 长度限制**：
  - L1 规划 Prompt ≤ 500 tokens
  - L2 规划 Prompt ≤ 400 tokens
  - L3 行动 Prompt ≤ 200 tokens
  - 反思 Prompt ≤ 600 tokens
  - 重要性评分 Prompt ≤ 150 tokens
- **超时设置**：单次 LLM 调用超时 30 秒，超时后返回 fallback 值（不崩溃）
- **错误处理**：LLM 调用失败必须有降级策略（重要性评分 fallback=5.0，行动 fallback="继续当前活动"）

### 1.2 嵌入计算

- **嵌入模型**：`sentence-transformers/all-MiniLM-L6-v2`（本地运行，无需 GPU）
- **禁止**：用 Ollama 做嵌入计算（避免混用接口）
- **缓存**：嵌入向量计算后缓存至 `MemoryObject.embedding`，禁止重复计算

### 1.3 数据持久化

- **运行时数据**：所有仿真数据存入 `./data/`（Docker volume 映射），不写入 `src/` 目录
- **Agent 状态**：每 10 步自动快照到 `data/snapshots/step_{n}.json`
- **禁止**：使用数据库（SQLite/PostgreSQL 等）——纯文件系统存储

### 1.4 Web 仪表盘

- **禁止前端框架**：仅用原生 HTML + Chart.js CDN，不引入 React/Vue/Svelte
- **实时通信**：使用 SSE（Server-Sent Events），禁止 WebSocket（复杂度边界）
- **端口**：仪表盘固定 8080，Ollama 固定 11434

---

## 2. 编码规范

### 2.1 代码风格

- Python 3.11+，强制 type hints
- 每个模块有 `__all__` 导出列表
- 数据类使用 `@dataclass`，不用 dict 传递结构化数据
- 禁止全局可变状态（`global` 关键字禁用）
- 函数/方法最大行数：50 行（超过则拆分）
- 文件最大行数：300 行（超过则拆分模块）

### 2.2 错误处理原则

- LLM 调用：必须 try/except，降级不崩溃
- 文件 I/O：必须 try/except，失败时记录日志不退出
- 业务逻辑错误：抛出自定义异常（`SocialTownError` 子类），不使用裸 `Exception`

### 2.3 日志规范

```python
import logging
logger = logging.getLogger(__name__)

# 必须记录的事件：
logger.info(f"[Step {step}] Agent {agent.name}: {action}")         # 每步行动
logger.info(f"[Reflect] {agent.name}: 触发反思（累积={score}）")    # 反思触发
logger.debug(f"[Memory] 检索 top-{k}: {[m.id for m in memories]}")  # 记忆检索
logger.warning(f"[LLM] 调用超时，使用 fallback: {fallback_value}")   # 降级
logger.error(f"[LLM] 调用失败: {e}")                                # 错误
```

### 2.4 测试规范

- 每个阶段测试目录：`tests/stage{N}_{name}/`
- 测试文件命名：`test_{module}.py`
- 测试类命名：`Test{ModuleName}`
- 必须有 `conftest.py` 提供 fixtures
- LLM 相关测试：提供 mock 模式（`--mock-llm` flag）和真实模式
- 每个阶段通过标准见"阶段测试门控"章节

---

## 3. 阶段测试门控（深度工作法）

**规则**：每个阶段所有测试通过后，方可进入下一阶段。不允许跳过测试门控。

### Stage 1 — 记忆系统
**文件**：`agents/memory/memory_stream.py`, `agents/memory/retrieval.py`

通过标准：
- [ ] `pytest tests/stage1_memory/ -v` 全部通过
- [ ] 单元测试覆盖率 ≥ 90%（`pytest --cov=agents/memory`）
- [ ] 检索精度测试：使用 `tests/stage1_memory/fixtures/retrieval_cases.json`（内置5组标注测试用例），top-3 命中率 ≥ 80%
  - fixture 格式：`[{"memories": [...], "query": "...", "expected_top3_ids": [...]}]`
- [ ] 三分量得分均在 [0,1] 范围内
- [ ] 反思阈值触发测试：累积 > 100 时触发标志位置 True

### Stage 2 — LLM 客户端
**文件**：`llm/client.py`

通过标准：
- [ ] `pytest tests/stage2_llm/ -v` 全部通过（mock 模式）
- [ ] 真实 Ollama 可用时：10 次调用成功率 ≥ 90%
- [ ] 响应格式解析成功率 ≥ 90%（格式错误时有 fallback）
- [ ] 超时 30 秒后正确返回 fallback 值
- [ ] `MODEL_NAME` 环境变量切换测试通过

### Stage 3 — 规划引擎
**文件**：`agents/cognitive/plan.py`

通过标准：
- [ ] `pytest tests/stage3_planning/ -v` 全部通过
- [ ] L1 生成 4-6 个活动块，格式合法（时段|活动|地点）
- [ ] L2 在 L1 块内生成 3-5 个活动
- [ ] L3 在 L2 活动内生成单个行动描述，非空
- [ ] 重规划（re-plan）触发后 L2 正确重置
- [ ] 连续 5 天规划多样性：对同一 Agent 提取 L1 地点字段，计算 Shannon entropy > 0.5
  - 地点集合例：["家", "图书馆", "咖啡馆", "广场"]；若每天都是"家"则 entropy=0，不通过

### Stage 4 — 社会网络 + 对话
**文件**：`world/social_graph.py`, `agents/social/dialogue.py`, `world/event_injector.py`

通过标准：
- [ ] `pytest tests/stage4_social/ -v` 全部通过
- [ ] 对话生成非空，字数 ≥ 5
- [ ] 对话后亲密度更新（intimacy += 0.05，不超过 1.0）
- [ ] 信息传播边界测试：credibility=1.0 且 trust=1.0 时接受率 = 1.0；credibility=0.0 时接受率 = 0.0（无论 trust 值）
- [ ] 事件注入后 seed_agents 记忆流中可找到该事件
- [ ] 社会图序列化/反序列化无信息丢失

### Stage 5 — 主仿真循环
**文件**：`core/simulation.py`, `agents/persona.py`

通过标准：
- [ ] `pytest tests/stage5_simulation/ -v` 全部通过
- [ ] 10 个 Agent 运行 50 步无崩溃（含 LLM mock）
- [ ] 50 步内三者均触发：记忆存储 ≥ 50 条、规划 ≥ 1 次、反思 ≥ 1 次
- [ ] 注入事件后 10 步内至少 1 个 Agent 知晓（传播测试）
- [ ] 快照文件正确写入 `data/snapshots/step_{n}.json`，包含完整可恢复状态，schema 见下：
  ```json
  {
    "step": 42,
    "agents": [
      {
        "id": "agent_01",
        "name": "Alice",
        "location": "图书馆",
        "current_action": "阅读",
        "opinion": {"market_reform": 0.3},
        "l1_plan": ["早晨|早餐|家", "上午|学习|图书馆"],
        "l2_current_block": "上午|学习|图书馆",
        "reflection_accumulator": 45.0,
        "memories": [
          {
            "id": "m-uuid-1",
            "content": "Alice 今天去了图书馆",
            "memory_type": "observation",
            "created_at": 40,
            "last_accessed": 42,
            "importance": 3.0,
            "source_agent": null,
            "credibility": 1.0
          }
        ]
      }
    ],
    "social_graph": {
      "edges": [{"from": "agent_01", "to": "agent_02", "intimacy": 0.35, "trust": 0.4, "interaction_count": 3}]
    }
  }
  ```
  注：embedding 向量不写入快照（体积大且可从 content 重新计算），恢复时重新嵌入。
- [ ] 快照恢复测试：加载 step_20 快照 → 继续运行 10 步 → step 编号从 21 继续（非从 0 开始）

### Stage 6 — 评估指标
**文件**：`evaluation/metrics.py`, `evaluation/reporter.py`

通过标准：
- [ ] `pytest tests/stage6_evaluation/ -v` 全部通过
- [ ] 信息扩散率：注入事件后 20 步内单调递增（或持平）
- [ ] 网络密度：值在 [0, 1]，随对话次数增加
- [ ] 极化指数（BC）：随机意见分布 BC < 0.55；双极分布 BC > 0.55
- [ ] 社会响应延迟：数值为正整数，credibility 高时延迟更短
- [ ] Markdown 报告生成无 KeyError

### Stage 7 — Docker 集成
**文件**：`docker/Dockerfile`, `docker/docker-compose.yml`

通过标准：
- [ ] `docker-compose build` 无报错
- [ ] `docker-compose up` 后 Ollama 服务健康（curl localhost:11434）
- [ ] 模型在容器首次启动时自动拉取
- [ ] 仿真引擎容器等 Ollama 就绪后再启动（healthcheck）
- [ ] 5 个 Agent 运行 10 步无崩溃（容器内）
- [ ] `docker-compose down && docker-compose up` 数据持久化验证

### Stage 8 — Web 仪表盘
**文件**：`webapp/server.py`, `webapp/static/index.html`

通过标准：
- [ ] `pytest tests/stage8_webapp/ -v` 全部通过
- [ ] 浏览器访问 localhost:8080 页面正常加载
- [ ] SSE 连接建立，10 步内收到至少 1 次数据推送
- [ ] 四个图表（扩散率/网络密度/极化/延迟）均有数据点渲染
- [ ] Agent 状态列表实时更新（每步刷新）
- [ ] 事件注入按钮：填写表单点击后，仿真中出现对应事件（端到端测试）

---

## 4. 评估标准（实验质量）

### 4.1 信息传播合理性
- credibility=1.0 的事件，2天内扩散率应 > 50%
- credibility=0.3 的事件，2天内扩散率应 < 30%
- 扩散路径应沿社会图高信任边传播（可从日志验证）

### 4.2 社区形成合理性
- 2天后社会图聚类系数 > 0.3（说明出现社区结构）
- 聚类数量 ≈ Agent数量 / 4（自然小组大小）

### 4.3 极化可诱导性
- 注入两个相互矛盾的事件（credibility 均高），3天后 BC > 0.55

### 4.4 Agent 行为一致性（可信度）
- 同一 Agent 连续 3 天的行动风格应与其 seed_description 吻合（人工抽检）
- Agent 的记忆不应包含从未发生的事件（幻觉率 < 5%）

---

## 4.5 补充说明

**message_bus.py 已移除**：Agent 间通信通过直接调用 `dialogue.py` 和写入对方记忆流实现，不使用消息总线（YAGNI）。

**Persona.opinion 字段**：类型 `dict[str, float]`，键为议题名（字符串），值为 [-1.0, 1.0] 的态度值。
- 初始化：从 `seed_description` 提取关键词推断，若无关键词则随机均匀分布
- 更新：每次 Agent 接受新信息时，自动更新相关议题的 opinion 值（见规格 Section 3.3）

**sentence-transformers 模型缓存**：Docker 中通过 `hf_cache` volume 持久化，本地开发时自动缓存至 `~/.cache/huggingface`，不需要手动管理。

---

## 5. 禁止事项（不得违反）

- **禁止跳过测试门控**：任何阶段测试未通过，不得合并代码进入下一阶段
- **禁止硬编码 Agent 数量**：所有涉及 Agent 数量的代码必须读 `NUM_AGENTS` 配置
- **禁止同步阻塞 LLM 调用在主线程**：LLM 调用必须有超时保护
- **禁止删除快照文件**：`data/` 目录内容不得在代码中主动删除
- **禁止前端框架**：Web 仪表盘只用原生 HTML + Chart.js
- **禁止全局变量**：跨模块共享状态通过依赖注入传递，不使用模块级全局变量

---

## 6. 快捷命令

```bash
# 开发运行
python main.py --agents 10 --days 3

# 测试（按阶段）
bash scripts/test.sh --stage 1
bash scripts/test.sh --stage 2
# ... 直到 stage 8

# 全量测试
bash scripts/test.sh

# 快速测试（mock LLM）
bash scripts/test.sh --mock-llm

# Docker 启动
docker-compose -f docker/docker-compose.yml up

# 查看仪表盘
open http://localhost:8080
```
