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
python load_documents.py            # Rebuild the TF-IDF index from data/documents/ (from scratch)
python load_documents.py --append   # Index only documents not already present
```

### Test / Verify
```bash
curl http://127.0.0.1:8000/health

# NOTE: test_structure.py and example_usage.py are referenced by older docs but
# no longer exist in the repo (removed in ff71420). There is currently no test suite.
```

### API
- Backend: `http://localhost:8000`
  - `POST /chat` — main endpoint. Returns `turn_uid`, which identifies the answer for rating.
  - `POST /feedback/rating` — thumbs up/down (`rating`: `1`, `-1`, or `null` to withdraw) plus optional `comment`, keyed by `turn_uid`. Overwrites on repeat; owner-checked.
  - `GET /feedback/rating/{turn_uid}` — current rating for a turn the caller owns.
  - `GET /analytics/*` — admin-only (`ANALYTICS_ADMIN_USERS`), incl. `/analytics/ratings`.
- Swagger docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:8501`

## Configuration

All settings live in `app/config.py` (pydantic-settings, loaded from environment):

## Architecture

### Overview
Single `POST /chat` endpoint → LangGraph orchestrator → multi-agent pipeline → response. State is in-memory (lost on restart). The default LLM is local Ollama.

### Request/Response
```
ChatRequest  { session_id, message }
ChatResponse { reply, session_id, debug_trace? }
```

### LangGraph Orchestration (`app/core/orchestrator.py`)

```
load_state → intent → [route by intent]
  ├─ stage flow: stage_reason → rag_plan
  └─ rag flow:                  rag_plan
                                   ↓
                    rag_plan → run_tools → assess_evidence
                        ↑                       ↓
                        └──── retry ────────────┤
                                                ↓
                                    responder → finalize
```

**Key routing rules:**
- After `intent`: routes to `stage_reason` (stage/workflow questions) or `rag_plan` (general Q&A). Definition queries and chit_chat/admin skip the stage agent.
- After `assess_evidence`: if the retrieved passages clear neither relevance floor and the attempt budget (`RAG_MAX_ATTEMPTS`, default 2) is not spent, loop back to `rag_plan`, which reformulates the query — via the cheap LLM, falling back to a deterministic keyword rewrite. Otherwise continue to `responder`.
- Low stage confidence (< 0.75) sets `stage_uncertain_hint`; the responder decides tone and follow-ups. There is no separate clarify gate.

**Evidence handling:** `state.artifacts` holds only the *current* turn's tool
observations — `_load_state` clears it. When the final verdict is
insufficient, artifacts are dropped entirely and the responder is told it has
no grounding, so it answers from general knowledge instead of citing passages
that do not support the answer.

Agents commented out of the graph (`PlannerAgent`, `MechanismCoachAgent`,
`StudyBuilderAgent`, `MeasureFinderAgent`, `GrantPartnerAgent`) still exist in
`app/agents/` but are not wired in. `ChatRequest.workflow` is accepted by the
API and currently ignored by the graph.

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
- `VectorTool` / `VersionedRAGTool`: TF-IDF retrieval from `data/vector_store/`.
  Only `VersionedRAGTool` is reached from the graph; `RAGAgent` hardcodes it.
- `SimpleVectorStore` (`vector_store.py`): hand-rolled TF-IDF over numpy (not sklearn); no external vector DB.
  Stopwords are dropped at both index and query time, and "Stage II" is
  normalized to a single `stage_ii` token — without that, the pronoun "I" and
  "Stage I" collide and question words outrank subject terms.

### Stage taxonomy (`app/core/stage_model.py`)
**Single source of truth for the NIH Stage Model.** Stage labels, descriptions,
classification keywords, progression order (`NEXT_STAGE`), the completed-feature
ladder, the confidence thresholds, `normalize_stage()` / `find_stage_in_text()` /
`is_definition_query()`. Never hardcode a stage definition anywhere else — this
module exists because five hand-maintained copies drifted, four of them onto a
taxonomy that does not exist (III as effectiveness, IV as implementation, V as
sustainability).

The correct model, matching `data/documents/def.docx`: 0 Basic Science ·
I Intervention Generation and Refinement (**IA** generation/manualization, then
**IB** feasibility/pilot) · II Efficacy in Research Settings · III Efficacy in
Community Settings · IV Effectiveness Research · V Implementation and
Dissemination. `STAGES` is the six top-level stages; `CLASSIFICATION_STAGES`
(`0, IA, IB, II, III, IV, V`) is what the classifier may emit.

Caveat: the Onken 1997/1998 PDFs in the corpus describe the earlier NIDA
*three*-stage model, where Stage III meant transportability to the community.
Both prompts warn against blending the two.

### State & Memory (`app/core/`)
- `state_store.py`: in-memory dict of `SessionState` keyed by `session_id`
- `memory.py`: short-term message buffer + rolling summary when over `SUMMARY_THRESHOLD`
- `types.py`: all Pydantic models — `SessionState`, `StageSlots`, `ChatRequest`, `ChatResponse`, `AgentOutput`, `ToolCall`, `Citation`

### Prompts (`app/prompts/`)
Markdown files loaded at agent init. `stage.md` contains the full NIH Stage 0–V decision tree and is the most domain-critical file; its stage definitions must stay in step with `app/core/stage_model.py`.

### Frontend (`frontend_streamlit.py`)
Streamlit chat UI; calls `POST /chat` on the backend. Branding comes from
`visuals/web/` — `st.logo()` puts the BID lockup top-left (the square badge
when the sidebar is collapsed) as the app's only brand mark, the badge is also
the favicon, and PNG icons
replace the emoji on the About / Change Password / Log Out widgets and the
Example Questions heading. Since `st.button`/`st.expander` take only emoji or
Material names for `icon=`, image icons are injected as CSS `::before`
backgrounds keyed on `st-key-<key>`. Where no PNG exists (sidebar nav, New
Chat/Delete, System Status, the "In development" card captions) Streamlit's
Material icons stand in, tinted to `BID_TEAL`. Never load from `visuals/` directly: the
two logo originals there are PSDs misnamed `.png`. See
`visuals/web/README.md` to regenerate or to add another icon. The About content sits
in a collapsed expander on the chat page (not a separate nav page), and the
Auto workflow card carries the usage guidance. The debug-JSON panel and
thinking-trace panel were removed from the UI; `debug_trace` is still
returned by the API and is still used in the frontend for `turn_uid`.
