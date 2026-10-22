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
  - `POST /chat` — main endpoint for API callers (the Streamlit UI does not use it — see Frontend). Returns `turn_uid`, which identifies the answer for rating.
  - `POST /feedback/rating` — thumbs up/down (`rating`: `1`, `-1`, or `null` to withdraw) plus optional `comment`, keyed by `turn_uid`. Overwrites on repeat; owner-checked.
  - `GET /feedback/rating/{turn_uid}` — current rating for a turn the caller owns.
  - `GET /analytics/*` — admin-only (`ANALYTICS_ADMIN_USERS`), incl. `/analytics/ratings`.
- Swagger docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:8501`

## Configuration

All settings live in `app/config.py` (pydantic-settings, loaded from environment).

LLM: `LLM_PROVIDER` defaults to `openai` (`LLM_MODEL=gpt-5.5`), with the
cheap non-reasoning `LLM_INTENT_MODEL` (`gpt-4.1-mini`) for intent
classification and RAG query reformulation. The `model=` override is honoured
only by the OpenAI provider, and so is true token streaming — the others
return the reply as one chunk. `app/core/llm.py` still has Ollama / Anthropic
/ Groq branches, but their settings (`OLLAMA_BASE_URL`, `ANTHROPIC_*`,
`GROQ_*`) are commented out in `config.py`, so selecting those providers
fails until they are restored.

Retrieval knobs: `RAG_TOP_K` (5), `RAG_MAX_ATTEMPTS` (2), `RAG_MIN_RELEVANCE`
(0.10 raw TF-IDF cosine), `RAG_MIN_COVERAGE` (0.25 query-term coverage),
`RAG_LLM_REFORMULATION`. Output caps: `LLM_MAX_TOKENS` (2000),
`LLM_COMPOSE_MAX_TOKENS` (4000).

## Architecture

### Overview
Two entry points into one `Orchestrator.process_message()`: the Streamlit UI
calls it in-process, API callers go through `POST /chat`. Either way:
guardrails → LangGraph orchestrator → multi-agent pipeline → response.
Session state is in-memory (lost on restart); chat history and attachments
are persisted via `chat_history.py`.

Guardrails (`app/core/guardrails.py`) run before the graph, in whichever
entry point is used: length/XSS validation, then a keyword topic pre-filter
that is **default-allow** — it blocks only clear off-topic keywords with no
domain keyword. The actual off-topic refusal is rule 0 of `responder.md`.
`sanitize_response` caps replies at 24k chars.

### Request/Response
```
ChatRequest  { session_id, message, workflow?, document_text?, document_name? }
ChatResponse { session_id, reply, debug, citations, next_question?, turn_uid? }
```
`process_message()` also takes `attachments=[{name, text}]` (used by the UI)
and a `stream_handler` callable; `POST /chat` exposes only the single
`document_text` blob and does not stream.

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
- After `intent`: routes to `stage_reason` (stage/workflow questions) or `rag_plan` (general Q&A). Definition queries, compose queries, and chit_chat/admin skip the stage agent.
- Compose queries (`query_type: "compose"`, detected by `is_compose_query()` in `stage_model.py` — "write an essay/summary/report…") are generation tasks: they never re-run stage classification, and the responder gets a task instruction (`user_instruction_compose` in `responder.md`) to produce the full deliverable using the already-committed `slots.stage`, with the larger `LLM_COMPOSE_MAX_TOKENS` output cap.
- After `assess_evidence`: if the retrieved passages clear neither relevance floor and the attempt budget (`RAG_MAX_ATTEMPTS`, default 2) is not spent, loop back to `rag_plan`, which reformulates the query — via the cheap LLM, falling back to a deterministic keyword rewrite. Otherwise continue to `responder`.
- Low stage confidence (< 0.75) sets `stage_uncertain_hint`; the responder decides tone and follow-ups. There is no separate clarify gate.
- Accepted offers: `_finalize` stores the offer at the tail of a reply ("Would you like me to…") as `pending_offer` for one turn. If the next message is a bare acceptance ("Yes please!", "2 please"), `_load_state` turns it into `accepted_offer` and `_intent` skips the intent LLM, routing it as `compose` (`app/core/followup.py`).

**What each agent sees:** routing agents get `context` from
`memory_manager.get_context_for_agent()` (summary, slots, last 5 messages
trimmed to head/tail, plus attachment *names* only). `IntentAgent` uses the
tail 2000 chars, `StageAgent` the head 1200, and the responder the tail 3000
— plus the full attachment text, retrieval evidence and a slots dump, built
in `ResponderAgent._build_general_context`. The rolling summary is created
once, when the session reaches `SUMMARY_THRESHOLD` messages; it is a
truncation of messages, not an LLM summary, and is not refreshed afterwards.

**Retrieval:** attempt 1 searches the user's message, with
`" NIH Stage {slots.stage}"` appended when a stage is committed. A passage is
usable only if its raw semantic score ≥ `RAG_MIN_RELEVANCE` **and** its term
coverage ≥ `RAG_MIN_COVERAGE`. Turns with attachments never retry. Retrieval
is skipped only for `chit_chat` / `admin` / `debug` intents.

**Per-turn state:** `_load_state` clears the per-turn stage clarification
fields (`missing_info`, `clarifying_question`, `stage_uncertain_hint`) so a
stale clarifying question cannot resurface on later turns; the committed
`slots.stage`/`stage_confidence` persist for the session, and
`reasoning_summary` persists only while a stage is committed. The
`stage_uncertain_hint` is only set on turns where the stage agent actually ran.

**Evidence handling:** `state.artifacts` holds only the *current* turn's tool
observations — `_load_state` clears it. When the final verdict is
insufficient, artifacts are dropped entirely and the responder is told it has
no grounding, so it answers from general knowledge instead of citing passages
that do not support the answer.

Agents commented out of the graph (`PlannerAgent`, `MechanismCoachAgent`,
`StudyBuilderAgent`, `MeasureFinderAgent`, `GrantPartnerAgent`) still exist in
`app/agents/` but are not wired in. `ChatRequest.workflow` is accepted by the
API and currently ignored by the graph (only logged as `workflow_override`).

After the graph returns, `process_message()` calls `feedback.observe_turn()`,
which mints the `turn_uid` and records implicit-feedback telemetry on a
background thread. Learned per-source weights from that loop feed back into
`VersionedRAGTool` ranking (`FEEDBACK_ADAPTIVE_RETRIEVAL`).

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

**Attachments** (files uploaded in chat) are conversation working context, never
corpus material — the reference index is built separately and deliberately by
`load_documents.py`. They are the *subject* of a turn ("rewrite my aims page"),
so they go to the model whole rather than being sampled:
`ATTACHMENT_MAX_CHARS` is sized to fit a full grant document (measured median
26k / max 84k chars over `data/documents/`; the former 24k cap silently
head-cut more than half of them). Because `state_store` is in-process only,
`chat_history.py` persists attachment text per conversation
(`save_attachments` / `load_attachments` / `clear_attachments`, ownership on
the row) and `_load_state` rehydrates when the in-memory state is empty — so a
restart mid-conversation cannot silently empty the document and leave the model
composing from chat memory. `debug_trace` carries `attachments_restored` and
`attachments_truncated` for monitoring.

### Prompts (`app/prompts/`)
Markdown files read at call time. Only three are live: `stage.md`
(`StageAgent` system prompt), `responder.md` (split on `## section` headings;
each section overrides the matching `_FALLBACK_SECTIONS` entry in
`responder_agent.py`), and `judge.md` (feedback judge). `IntentAgent` and the
RAG reformulation prompt are inline strings in code — editing `intent.md`
changes nothing; `planner.md` and `grant.md` belong to unwired agents.
`stage.md` contains the full NIH Stage 0–V decision tree and is the most domain-critical file; its stage definitions, and the ones inlined in `responder.md`, must stay in step with `app/core/stage_model.py`.

### Frontend (`frontend_streamlit.py`)
Streamlit chat UI. It does **not** call `POST /chat`: it imports the
orchestrator and runs `process_message()` in-process (cached with
`st.cache_resource`) on a worker thread, streaming responder tokens back
through a queue into `st.write_stream`. It runs its own copy of the
guardrail checks and parses uploaded PDF/DOCX/TXT files client-side,
caching the result per (name, size). Changes to `app/main.py`'s `/chat`
handler therefore do not affect the UI. Branding comes from
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
