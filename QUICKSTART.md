# 

## 1. 

```bash
# （）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 
pip install -r requirements.txt
```

## 2. 

```bash
# 1: 
./run.sh

# 2:  uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 3:  Python 
python -m app.main
```

## 3.  API

###  curl

```bash
# 
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test_session",
    "message": " NIH Stage Model"
  }'

# 
curl "http://localhost:8000/sessions/test_session"
```

###  Python 

```bash
# （ requests）
pip install requests
python example_usage.py
```

### 

 http://localhost:8000/docs  Swagger UI ， API。

## 4. 

```
app/
  ├── main.py              # FastAPI ， API 
  ├── config.py            # 
  │
  ├── core/                # 
  │   ├── types.py         # （Pydantic）
  │   ├── state_store.py   # 
  │   ├── router.py        # 
  │   ├── orchestrator.py  # 
  │   ├── memory.py        # 
  │   └── guardrails.py    # 
  │
  ├── agents/              # Agent 
  │   ├── base.py          # Agent 
  │   ├── intent_agent.py  # 
  │   ├── stage_agent.py   # Stage 
  │   ├── planner_agent.py # 
  │   └── responder_agent.py # 
  │
  ├── tools/               # 
  │   ├── base.py          # 
  │   ├── db_tool.py       # 
  │   └── vector_tool.py   # （RAG）
  │
  └── prompts/             # Prompt 
      ├── intent.md
      ├── stage.md
      ├── planner.md
      └── responder.md
```

## 5. 

1. **** → `POST /chat`
2. **Router ** →  Agent
3. **Agent ** → （Intent → Stage → Planner → Responder）
4. **** → ，
5. **** → Responder 
6. **** → 

## 6. 

✅ ****：
- 
- 4  Agent （）
- Router 
- Memory 
- Tools 
- FastAPI API 

⏳ ****：
- LLM （）
- 
- （Pinecone/Weaviate）
- 

## 7. 

1. ** LLM**：
   -  `agents/`  LLM 
   -  `prompts/` 

2. ****：
   -  `tools/db_tool.py` 
   -  NIH Stage Model 

3. ****：
   -  `tools/vector_tool.py` 
   -  RAG（）

4. ****：
   -  `eval/metrics.py` 
   -  `eval/test_cases.yaml` 

## 8. 

**Q:  Agent ？**
A:  Agent （ `agents/intent_agent.py`）， `run()` 。

**Q: ？**
A:  `tools/` ， `BaseTool`， `tools/__init__.py` 。

**Q: ？**
A:  `core/router.py`  `decide()` 。

**Q: ？**
A:  `DEBUG=True`，API  `debug` 。

## 9. 

，：
- `README.md` - 
- `app/prompts/` - Prompt 
- API ：http://localhost:8000/docs
