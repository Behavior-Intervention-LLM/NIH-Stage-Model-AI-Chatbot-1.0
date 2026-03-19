# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run
```bash
./start.sh                                      # Backend + Frontend together
./run.sh                                        # Backend only (port 8000)
streamlit run frontend_streamlit.py             # Frontend only (port 8501)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000  # Backend (dev mode)
```

### Document Ingestion
```bash
python load_documents.py   # Scan data/documents/, chunk, index into TF-IDF vector store
```

### Test / Verify
```bash
python test_structure.py   # Import smoke tests for all core modules
curl http://127.0.0.1:8000/health
python example_usage.py    # Example API usage
```

### API
- Backend: `http://localhost:8000` (single endpoint: `POST /chat`)
- Swagger docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:8501`

## Configuration

All settings live in `app/config.py` (pydantic-settings, loaded from environment):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | LLM backend |
| `LLM_MODEL` | `qwen2.5:3b-instruct` | Model name |
| `LLM_TEMPERATURE` | `0.3` | Generation temp |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `VECTOR_STORE_PATH` | `data/vector_store` | TF-IDF index location |
| `DOCUMENTS_DIR` | `data/documents` | Source docs for RAG |
| `SHORT_TERM_LIMIT` | `20` | Messages kept in session |
| `SUMMARY_THRESHOLD` | `10` | When to summarize history |

## Architecture

### Overview
Single `POST /chat` endpoint → LangGraph orchestrator → multi-agent pipeline → response. State is in-memory (lost on restart). The default LLM is local Ollama.

### Request/Response
```
ChatRequest  { session_id, message }
ChatResponse { reply, session_id, debug_trace? }
```

### LangGraph Orchestration (`app/core/orchestrator.py`)
14-node state machine with conditional routing:

```
load_state → intent → [route by intent]
  ├─ stage flow: stage_reason → [workflow agent] → rag_plan → run_tools → react_judge
  └─ rag flow:  rag_plan → run_tools → react_judge
       ↓
  [react_judge: continue loop OR stop]
       ↓
  guardrails → responder → finalize
```

**Key routing rules:**
- After `intent`: routes to `stage_reason` (stage/study/grant/mechanism/measure workflows) or `rag_plan` (general Q&A)
- After `stage_reason`: confidence < 0.75 or stage is None → `clarify_only_gate` (returns clarification without RAG)
- After `react_judge`: up to 3 ReAct iterations, then forced stop

### Agent Layer (`app/agents/`)
Each agent wraps an LLM call with a markdown prompt template from `app/prompts/`:

| Agent | Role |
|---|---|
| `IntentAgent` | Classifies intent, workflow type, language |
| `StageAgent` | NIH Stage 0–V classification with confidence + reasoning |
| `RAGAgent` | Plans tool calls for retrieval |
| `ResponderAgent` | Final natural language response |
| `PlannerAgent` | Next-step guidance |
| `MechanismCoachAgent` | Mechanism ranking and manipulation hints |
| `StudyBuilderAgent` | Stage-aware study design matrix |
| `MeasureFinderAgent` | Construct-to-measure shortlist |
| `GrantPartnerAgent` | Grant writing and reviewer feedback |

### Tool Layer (`app/tools/`)
- `ToolRegistry`: plugin-based registration and dispatch
- `VectorTool` / `VersionedRAGTool`: TF-IDF retrieval from `data/vector_store/`
- `DBTool`: structured database lookups
- `SimpleVectorStore` (`vector_store.py`): sklearn TF-IDF backend; no external vector DB

### State & Memory (`app/core/`)
- `state_store.py`: in-memory dict of `SessionState` keyed by `session_id`
- `memory.py`: short-term message buffer + rolling summary when over `SUMMARY_THRESHOLD`
- `types.py`: all Pydantic models — `SessionState`, `StageSlots`, `ChatRequest`, `ChatResponse`, `AgentOutput`, `ToolCall`, `Citation`

### Prompts (`app/prompts/`)
Markdown files loaded at agent init. `stage.md` contains the full NIH Stage 0–V decision tree and is the most domain-critical file.

### Frontend (`frontend_streamlit.py`)
Streamlit chat UI; calls `POST /chat` on the backend. Includes an expandable debug panel showing the LangGraph execution trace when `DEBUG=true`.
