# Social Town — LLM 社会动力实验平台

基于 Qwen2.5（Ollama）的多智能体社会模拟系统，模拟信息传播、社区形成、观点极化和社会事件响应。

## 快速开始

### 本地开发
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --agents 5 --days 2
open http://localhost:8080
```

### Docker（推荐）
```bash
docker-compose -f docker/docker-compose.yml up
open http://localhost:8080
```

## 测试
```bash
bash scripts/test.sh              # 全量测试
bash scripts/test.sh --stage 1   # 单阶段测试
bash scripts/test.sh --mock-llm  # 跳过真实 LLM 调用
bash scripts/test.sh --coverage  # 含覆盖率报告
```

## 配置
通过环境变量或 `.env` 文件：
- `MODEL_NAME` — Ollama 模型（默认 qwen2.5:0.5b）
- `NUM_AGENTS` — Agent 数量（默认 10，支持 5-50）
- `OLLAMA_HOST` — Ollama API 地址（默认 http://localhost:11434）
- `SIMULATION_DAYS` — 模拟天数（默认 3）

## 架构
见 `docs/superpowers/specs/2026-03-24-social-town-design.md`
