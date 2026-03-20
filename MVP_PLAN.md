# MVP Plan — NIH Stage Model AI Chatbot

**Goal:** A publicly accessible web app (Streamlit) that helps students, professors, and researchers
navigate the NIH Stage Model (Stages 0–V) for behavioral intervention research.

---

## Access Requirements
- Hosted on Streamlit Community Cloud
- Password-protected (link + password = access)
- No login accounts needed — simple shared password

---

## Status Legend
- [ ] Not started
- [~] Partially done (exists but needs work)
- [x] Done

---

## 1. CRITICAL — Must work for MVP

### 1a. Authentication (Password Gate)
- [x] Add password screen before any content loads (`st.secrets["APP_PASSWORD"]`)
- [x] Store password in Streamlit secrets (not in code)
- [x] Session persists once authenticated (no re-prompting)
- [x] If `APP_PASSWORD` not set → open access (local dev mode)

### 1b. Cloud LLM (replace local Ollama)
- [x] Add Groq provider to `llm.py` and `config.py`
- [ ] Set `LLM_PROVIDER=groq` and `GROQ_API_KEY` in Streamlit secrets
- [ ] Set `GROQ_MODEL=llama-3.3-70b-versatile` (free, fast, capable)
- [ ] Test that all 9 agents work with Groq backend
- **Why:** Ollama requires a local GPU/machine. Groq is free and fast.
- **Get key at:** https://console.groq.com

### 1c. Deployment Config
- [ ] `requirements.txt` audit — remove unused deps, pin versions
- [x] Create `.streamlit/config.toml` for theme/server settings
- [x] Create `.streamlit/secrets.toml.example` (template, not real secrets)
- [x] `.streamlit/secrets.toml` added to `.gitignore`
- [x] `data/vector_store/` files present and committed
- [ ] Add `packages.txt` if system packages needed (e.g. for pytesseract/OCR)

### 1d. Stage Classification Works End-to-End
- [~] Stage Agent classifies Stage 0–V → works but depends on LLM prompt quality
- [ ] Review `app/prompts/stage.md` — confirm decision tree is accurate per General Capability.md
- [ ] Test each stage path manually before launch

---

## 2. IMPORTANT — Core capability per General Capability.md

### 2a. Mechanism Examination at Every Stage
- **Gap:** Currently only triggers via `mechanism_coach` workflow (user must select it)
- [ ] After every stage classification, proactively surface 1–2 mechanism questions
- [ ] Add mechanism prompt injection into `ResponderAgent` when stage is determined
- **Ref:** General Capability.md line 20: "Examination of mechanisms of behavior change is encouraged on every stage"

### 2b. Visual Stage Indicator
- **Gap:** Stage result is buried in text — no visual feedback
- [ ] Show a stage badge/progress bar in the UI after classification
  - Example: `Stage II` shown as a highlighted pill or progress indicator (Stage 0 → I → **II** → III → IV → V)
- [ ] Show confidence score visually (low/medium/high)
- **Ref:** General Capability.md line 17: "would be nice to showcase visually going into"

### 2c. "How to Improve" Guidance Per Stage
- [~] PlannerAgent provides next steps — but not structured per General Capability questions
- [ ] Ensure each stage response answers:
  - Stage 1: What counts as pilot testing?
  - Stage 2: How to improve methodology?
  - Stage 3: What counts as real-world providers?
  - Stage 4: What makes effective research?
  - Stage 5: What does dissemination look like?

### 2d. Thinking Trace / Underlying Thought
- [~] "Show Thinking Trace" toggle exists but shows raw internal logs
- [ ] Clean up trace display — make it human-readable (not agent codes)
- [ ] Label it "How I reasoned this" instead of "Thinking Trace"

---

## 3. NICE TO HAVE — Polish before wider sharing

### 3a. Audience Adaptation
- [ ] Detect or ask user role: Student / Researcher / Professor / Practitioner
- [ ] Adjust response depth accordingly (simpler language for students, technical for researchers)

### 3b. Better Error Messages
- [ ] If LLM fails → show friendly message, not raw exception
- [ ] If stage can't be determined → clearly ask for more info (already partially done via clarifying questions)

### 3c. Citations Display
- [~] Citations returned in API but not clearly shown in UI
- [ ] Show source document name and excerpt below each response

### 3d. Mobile / Readability
- [ ] Test on mobile browser — Streamlit is responsive but workflow card buttons may overflow
- [ ] Reduce sidebar clutter for first-time users

### 3e. Conversation Persistence
- **Current state:** Conversations live in browser session only — refresh = lost
- [ ] Optional: Use Streamlit's session or a lightweight DB (SQLite) to persist across refreshes
- **Note:** Not critical for MVP if users know to not refresh

---

## 4. NOT IN MVP — Future

- Multi-user accounts / login system
- Embeddings-based vector search (replace TF-IDF)
- Admin dashboard (usage stats, session logs)
- Export conversation as PDF/report
- Integration with NIH grant databases
- Email/Slack notifications

---

## Deployment Checklist (before going live)

- [ ] Run `python load_documents.py` to rebuild vector index
- [ ] Commit `data/vector_store/chunks.json` and `metadata.json`
- [ ] Set Streamlit secrets: `APP_PASSWORD`, `ANTHROPIC_API_KEY`, `LLM_PROVIDER=anthropic`
- [ ] Push to GitHub (repo must be public or connected to Streamlit Cloud)
- [ ] Deploy on [share.streamlit.io](https://share.streamlit.io)
- [ ] Test full flow: password → chat → stage classification → mechanism question → response
- [ ] Share link + password with pilot users

---

## Priority Order for Dev Work

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 1 | Password gate | Small | Unblocks sharing |
| 2 | Switch to Anthropic LLM | Small | Unblocks cloud deploy |
| 3 | Streamlit Cloud deployment config | Small | Unblocks hosting |
| 4 | Visual stage indicator in UI | Medium | Big UX win |
| 5 | Mechanism prompt at every stage | Medium | Matches spec |
| 6 | Clean up thinking trace display | Small | Matches spec |
| 7 | Stage prompt review/accuracy | Medium | Core correctness |
| 8 | Per-stage "how to" guidance | Medium | Matches spec |