# NIH Stage Model AI Chatbot（本地开发 + Docker 部署）

当前版本只使用 OpenAI 兼容推理接口（推荐 vLLM）。  
后端统一入口：`POST /chat`。

---

## 1) 本地开发（你现在最适合）

适合 Mac 或无 GPU 场景：本地只跑 FastAPI，请求转发到你的推理模型接口（本机或远程）。

### 1.1 准备环境

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 1.2 配置 `.env`

```bash
cp .env.local.example .env
```

编辑 `.env` 关键项：

```env
VLLM_BASE_URL=http://127.0.0.1:8001/v1
LLM_MODEL=my-inference-model
```

说明：
- `VLLM_BASE_URL`：你的推理服务地址（OpenAI 兼容）
- `LLM_MODEL`：你的模型服务名（和推理端一致）
- 如果推理服务需要鉴权，填 `LLM_API_KEY`
- 如果要启用 Redis 持久化会话记忆，配置：
  - `REDIS_URL=redis://127.0.0.1:6379/0`
  - `STATE_TTL_SECONDS=604800`
  - `REDIS_KEY_PREFIX=nih_chatbot`
- 混合记忆（短期+长期）可调参数：
  - `SHORT_TERM_LIMIT=20`
  - `SUMMARY_THRESHOLD=10`
  - `SUMMARY_REFRESH_EVERY_TURNS=6`
  - `LONG_TERM_MEMORY_WINDOW=50`
  - `LONG_TERM_MEMORY_MAX_LINES=8`
  - `MEMORY_CONTEXT_MAX_CHARS=6000`

### 1.3 启动 FastAPI

```bash
./run_local_dev.sh
```

或直接：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 1.4 验证

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo",
    "message": "请解释 NIH Stage Model 的阶段划分"
  }'
```

---

## 2) GPU 服务器 Docker 部署（可选）

这个模式用于把你的本地训练权重封装成服务并对外使用。

### 2.1 机器要求

- Linux + NVIDIA GPU
- Docker + Docker Compose
- NVIDIA Driver + nvidia-container-toolkit

### 2.2 准备 `.env`

```bash
cp .env.vllm.example .env
```

本地权重常见配置：

```env
LOCAL_MODEL_DIR=/opt/models
VLLM_MODEL=/models/my-ft-model
VLLM_SERVED_MODEL_NAME=my-ft-model
```

### 2.3 启动

```bash
./run_vllm_local.sh
```

或：

```bash
docker compose --env-file .env -f docker-compose.vllm.yml up -d --build
```

### 2.4 验证

```bash
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:8000/health
```

---

## 3) 常见问题

- 本地没有 GPU：用第 1 节，本地 API + 外部推理服务
- 超时：调大 `LLM_TIMEOUT_SECONDS`
- `404 /v1/chat/completions`：检查推理服务是否真的是 OpenAI 兼容接口
- 返回模型不存在：检查 `LLM_MODEL` 名称与推理服务注册名一致
