"""NIH Stage Model AI Chatbot - Streamlit Frontend"""
import io
import importlib
import importlib.util
import os
import uuid
from datetime import datetime

import streamlit as st

# Inject Streamlit secrets into environment variables before importing app modules.
# Required for Streamlit Cloud: secrets are not automatically available as os.environ.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

# Enables tool to utilize agents
from app.core.orchestrator import Orchestrator
from app.core.guardrails import Guardrails
from app.core.state_store import state_store
from app.tools import tool_registry
import chat_history


# How is it storing/logging information - do we need to set up a "database"
# How does streamlit cache-resource work
@st.cache_resource(show_spinner="Loading AI system...")
def get_orchestrator():
    orch = Orchestrator(tool_registry=tool_registry)
    return orch

st.set_page_config(
    page_title="NIH Stage Model AI Chatbot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Access Authentication (users.db-backed, see auth.py)
def _require_auth():
    """Block access until valid credentials are entered.
    Set AUTH_DISABLED=true to skip the gate (local dev mode only)."""
    if os.environ.get("AUTH_DISABLED", "").lower() == "true":
        return
    if st.session_state.get("authenticated"):
        return

    import hmac as _hmac

    from auth import create_user, seed_users_from_env, verify_login

    # First-deploy bootstrap: creates accounts from SEED_USERS (Streamlit
    # secrets or env) if set. No-op when unset or users already exist.
    if not st.session_state.get("_seed_checked"):
        seed_users_from_env()
        st.session_state._seed_checked = True

    st.title("🔬 NIH Stage Model AI Chatbot")
    st.markdown("This tool is for authorized users only. Sign in to continue.")

    # Self-signup is disabled unless an invite code is configured.
    signup_code = st.secrets.get("SIGNUP_CODE", os.environ.get("SIGNUP_CODE", ""))

    login_tab, signup_tab = st.tabs(["Sign In", "Create Account"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username", key="_auth_user")
            password = st.text_input("Password", type="password", key="_auth_pw")
            submitted = st.form_submit_button("Login", type="primary")
        if submitted:
            ok, msg = verify_login(username, password)
            if ok:
                st.session_state.authenticated = True
                st.session_state.auth_username = username.strip().lower()
                st.rerun()
            else:
                st.error(msg)

    with signup_tab:
        if not signup_code:
            st.info("Account creation is disabled. Contact the administrator for access.")
        else:
            st.caption("You need an invite code from the administrator to create an account.")
            with st.form("signup_form"):
                new_username = st.text_input("Username (3-32 chars: a-z, 0-9, . _ -)")
                new_password = st.text_input("Password (min 8 characters)", type="password")
                confirm_password = st.text_input("Confirm password", type="password")
                invite = st.text_input("Invite code", type="password")
                signup_submitted = st.form_submit_button("Create Account", type="primary")
            if signup_submitted:
                if not _hmac.compare_digest(invite.strip(), signup_code):
                    st.error("Invalid invite code.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    try:
                        create_user(new_username, new_password)
                        st.session_state.authenticated = True
                        st.session_state.auth_username = new_username.strip().lower()
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

    st.stop()


_require_auth()


# What is happening here

# How might sessions state help with the overall program
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "show_thinking_trace" not in st.session_state:
    st.session_state.show_thinking_trace = True
if "selected_workflow" not in st.session_state:
    st.session_state.selected_workflow = "auto"
if "current_page" not in st.session_state:
    st.session_state.current_page = "chat"
def _history_username() -> str:
    """Username that owns persisted chat history (matches backend auth)."""
    if os.environ.get("AUTH_DISABLED", "").lower() == "true":
        return "anonymous"
    return st.session_state.get("auth_username") or "anonymous"


if "conversations" not in st.session_state:
    initial_id = st.session_state.session_id
    conversations = {
        initial_id: {
            "session_id": initial_id,
            "title": "New Chat",
            "messages": st.session_state.messages,
            "created_at": datetime.now().isoformat(),
        }
    }
    # Restore this user's persisted conversations from the auth database.
    # Messages are lazy-loaded (None) until a conversation is opened.
    try:
        for conv in chat_history.list_conversations(_history_username()):
            conversations[conv["id"]] = {
                "session_id": conv["id"],
                "title": conv["title"],
                "messages": None,
                "created_at": conv["created_at"],
            }
    except Exception:
        pass  # history is best-effort; never block the chat UI
    st.session_state.conversations = conversations
if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = next(iter(st.session_state.conversations.keys()))


def _ensure_messages_loaded(conv: dict) -> None:
    """Fetch a DB-backed conversation's messages on first open."""
    if conv.get("messages") is not None:
        return
    try:
        rows = chat_history.get_messages(_history_username(), conv["session_id"]) or []
    except Exception:
        rows = []
    conv["messages"] = [
        {"role": r["role"], "content": r["content"], "timestamp": r["created_at"]}
        for r in rows
    ]


def _record_exchange(user_text: str, assistant_text: str) -> None:
    """Persist one turn to the database; failures never break the chat."""
    try:
        chat_history.record_exchange(
            _history_username(), st.session_state.session_id, user_text, assistant_text
        )
    except Exception:
        pass


# What is happening here
def create_new_conversation(title: str = "New Chat") -> str:
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {
        "session_id": conv_id,
        "title": title,
        "messages": [],
        "created_at": datetime.now().isoformat(),
    }
    st.session_state.active_conversation_id = conv_id
    st.session_state.session_id = conv_id
    st.session_state.messages = []
    return conv_id


def get_active_conversation() -> dict:
    conv_id = st.session_state.active_conversation_id
    if conv_id not in st.session_state.conversations:
        create_new_conversation()
        conv_id = st.session_state.active_conversation_id
    conv = st.session_state.conversations[conv_id]
    _ensure_messages_loaded(conv)
    return conv


def sync_active_conversation_messages():
    active = get_active_conversation()
    active["messages"] = st.session_state.messages


def human_title(title: str) -> str:
    return title if title and title.strip() else "Untitled Chat"


MAX_PDF_PAGES = 60


# Extracting PDF
def _extract_text_from_pdf(file_bytes: bytes) -> tuple[str, str]:
    """Return (text, status). Status is shown to the user, so it must explain
    a failure rather than leaving them to assume the file was read."""
    py_pdf2 = importlib.util.find_spec("PyPDF2")  # type: ignore
    if py_pdf2 is None:
        return "", "PDF support is not installed (PyPDF2 missing)."
    PyPDF2 = importlib.import_module("PyPDF2")
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:
        return "", f"Could not open the PDF ({type(exc).__name__}). It may be corrupt or password-protected."

    total_pages = len(reader.pages)
    pages = []
    for i, page in enumerate(reader.pages[:MAX_PDF_PAGES], 1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append(f"[Page {i}]\n{text.strip()}")

    if not pages:
        # An image-only scan extracts nothing. Saying so is the difference
        # between the user re-attaching a text PDF and believing we read it.
        return "", (
            f"No text layer found in this PDF ({total_pages} pages). It is most likely a "
            "scan or image export — OCR is not available here, so please attach a "
            "text-based PDF or paste the text."
        )

    status = f"Read {len(pages)} of {total_pages} page(s)."
    if total_pages > MAX_PDF_PAGES:
        status += f" Only the first {MAX_PDF_PAGES} pages were read."
    return "\n\n".join(pages).strip(), status


# Extracting DOCX
def _extract_text_from_docx(file_bytes: bytes) -> tuple[str, str]:
    docx_spec = importlib.util.find_spec("docx")
    if docx_spec is None:
        return "", "DOCX support is not installed (python-docx missing)."
    Document = importlib.import_module("docx").Document
    try:
        doc = Document(io.BytesIO(file_bytes))
    except Exception as exc:
        return "", f"Could not open the DOCX ({type(exc).__name__}). If this is an older .doc file, save it as .docx first."

    lines = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    # Tables carry study designs and measures matrices; dropping them loses
    # exactly the content people attach these documents to discuss.
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
            if cells:
                lines.append(" | ".join(cells))

    if not lines:
        return "", "The DOCX contains no readable text (it may hold only images)."
    return "\n".join(lines).strip(), f"Read {len(lines)} paragraph(s)."


# Extracting TXT
def _extract_text_from_txt(file_bytes: bytes) -> tuple[str, str]:
    """Decode text, actually honouring the encoding.

    The previous loop passed errors="ignore", so the utf-8 attempt always
    "succeeded" and the utf-16/latin-1 branches were unreachable — a utf-16
    file decoded to text interleaved with NUL bytes.
    """
    if file_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        candidates = ("utf-16", "utf-8", "latin-1")
    else:
        candidates = ("utf-8-sig", "utf-16", "latin-1")

    for enc in candidates:
        try:
            text = file_bytes.decode(enc).strip()
        except (UnicodeDecodeError, UnicodeError):
            continue
        # latin-1 decodes any byte sequence, so a binary file "succeeds" as
        # control-character soup. Reject that rather than feed it to the model.
        printable = sum(1 for c in text if c.isprintable() or c.isspace())
        if text and printable / len(text) > 0.9:
            return text, f"Read {len(text)} characters ({enc})."

    return "", "Could not decode this file as text."


# An image OCR branch used to live here. It was unreachable: the uploader
# accepts only pdf/docx/txt, and pytesseract is not a dependency. Removed
# rather than left as a branch that looks supported but never runs.


MAX_ATTACHMENT_BYTES = 15 * 1024 * 1024


def parse_uploaded_files(uploaded_files) -> tuple[list[dict], list[str]]:
    """Extract text from each attached file.

    Returns (files, failures) where `files` is [{"name", "text"}] and
    `failures` are user-facing messages for anything that could not be read.

    No truncation happens here. Files are handed to the orchestrator whole and
    the prompt budget (ATTACHMENT_MAX_CHARS) is applied once, at the point of
    use — the old chain of per-file/merged/session/prompt caps discarded up to
    two thirds of a multi-file upload without telling anyone.
    """
    if not uploaded_files:
        return [], []

    files: list[dict] = []
    failures: list[str] = []

    for up in uploaded_files:
        name = up.name
        lower = name.lower()
        file_bytes = up.getvalue()

        if len(file_bytes) > MAX_ATTACHMENT_BYTES:
            failures.append(
                f"**{name}** is {len(file_bytes) / 1024 / 1024:.1f} MB, over the "
                f"{MAX_ATTACHMENT_BYTES // 1024 // 1024} MB limit — it was not attached."
            )
            continue

        if lower.endswith(".pdf"):
            text, status = _extract_text_from_pdf(file_bytes)
        elif lower.endswith(".docx"):
            text, status = _extract_text_from_docx(file_bytes)
        elif lower.endswith(".txt"):
            text, status = _extract_text_from_txt(file_bytes)
        else:
            text, status = "", "Unsupported file type (supported: pdf, docx, txt)."

        if text:
            files.append({"name": name, "text": text, "status": status})
        else:
            failures.append(f"**{name}** could not be read. {status}")

    return files, failures


# What is happening here
def check_backend_health() -> bool:
    try:
        get_orchestrator()
        return True
    except Exception:
        return False


def render_thinking_trace(debug_info: dict):
    """Render lightweight chain-of-thought-like trace in muted gray UI."""
    if not debug_info:
        return

    route_mode = debug_info.get("route_mode")
    route_notes = debug_info.get("route_notes", "")
    agents_called = debug_info.get("agents_called", [])
    trace = debug_info.get("execution_trace", []) or []

    lines = []
    lines.append(f"Route: {route_mode}")
    if route_notes:
        lines.append(f"Router notes: {route_notes}")
    if agents_called:
        lines.append(f"Agents called: {', '.join(agents_called)}")
    if debug_info.get("workflow_resolution"):
        lines.append(f"Workflow resolution: {debug_info.get('workflow_resolution')}")
    if debug_info.get("workflow_correction"):
        lines.append(f"Workflow correction: {debug_info.get('workflow_correction')}")

    for step in trace:
        kind = step.get("kind")
        name = step.get("name", "unknown")
        if kind == "agent":
            conf = step.get("confidence")
            analysis = step.get("analysis", "")
            tool_actions = step.get("tool_actions", [])
            decision_preview = step.get("decision_preview", {}) or {}
            summary = f"• Agent `{name}`"
            if conf is not None:
                summary += f" (conf={conf})"
            if analysis:
                summary += f": {analysis}"
            if decision_preview:
                summary += f" | decision={decision_preview}"
            if tool_actions:
                summary += f" | tool_actions={tool_actions}"
            lines.append(summary)
        elif kind == "tool":
            success = step.get("success", True)
            sources = step.get("sources", [])
            err = step.get("error")
            summary = f"• Tool `{name}` -> {'ok' if success else 'error'}"
            if sources:
                summary += f" | sources={sources}"
            if err:
                summary += f" | error={err}"
            lines.append(summary)
        elif kind == "react":
            summary = f"• ReAct `{name}`"
            if step.get("step") is not None:
                summary += f" (step={step.get('step')})"
            if step.get("analysis"):
                summary += f": {step.get('analysis')}"
            if step.get("reason"):
                summary += f" | reason={step.get('reason')}"
            if step.get("planned_tools") is not None:
                summary += f" | planned_tools={step.get('planned_tools')}"
            if step.get("successful_results") is not None:
                summary += f" | successful_results={step.get('successful_results')}"
            lines.append(summary)
        elif kind == "guardrail":
            lines.append(f"• Guardrail `{name}` | warnings_count={step.get('warnings_count', 0)}")
        elif kind == "gate":
            lines.append(
                f"• Gate `{name}` | triggered={step.get('triggered')} | reason={step.get('reason', '')}"
            )

    html = "<br/>".join(line.replace("<", "&lt;").replace(">", "&gt;") for line in lines)
    st.markdown(
        f"""
            <div style="
            background: #f6f7f8;
            border: 1px solid #e6e8eb;
            border-radius: 8px;
            padding: 10px 12px;
            color: #6b7280;
            font-size: 12px;
            line-height: 1.45;">
            <div style="font-weight: 600; color: #9ca3af; margin-bottom: 6px;">Thinking Trace</div>
            {html}
            </div>
        """,
        unsafe_allow_html=True,
    )


def render_analytics_page():
    """Admin view over the implicit feedback system (app/feedback/).

    Reads the feedback tables directly rather than calling /analytics/*, for
    the same reason the chat page calls the orchestrator in-process: this
    frontend is often deployed without a separate backend.
    """
    from app.feedback import adaptation as fb_adaptation
    from app.feedback import judge as fb_judge
    from app.feedback import rankings as fb_rankings
    from app.feedback import store as fb_store

    st.title("📊 Feedback & Usage Analytics")
    st.caption(
        "Quality fuses three streams: the 👍/👎 users leave on an answer, what they did "
        "next (re-asked, corrected, thanked, built on it), and an LLM judge grading each "
        "answer against the passages it actually retrieved. A stated rating outweighs "
        "both inferred streams but does not replace them — ratings are self-selected, so "
        "coverage is reported next to every rate. Scale is −1 to +1."
    )

    overview = fb_rankings.overview()
    if not overview["total_turns"]:
        st.info("No turns observed yet. Have a conversation, then come back.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Turns observed", overview["total_turns"])
    c2.metric("Mean quality", f"{overview['mean_quality']:+.3f}")
    c3.metric("Good / bad turns", f"{overview['good_turns']} / {overview['bad_turns']}")
    c4.metric("Mean latency", f"{overview['mean_latency_ms']} ms")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions", overview["total_sessions"])
    c2.metric("Users", overview["total_users"])
    c3.metric("Judge coverage", f"{overview['judge_coverage']:.0%}")
    c4.metric("Behavioural coverage", f"{overview['behavioural_coverage']:.0%}")

    c1, c2, c3, c4 = st.columns(4)
    sat = overview.get("satisfaction_rate")
    c1.metric("👍 Thumbs up", overview.get("thumbs_up", 0))
    c2.metric("👎 Thumbs down", overview.get("thumbs_down", 0))
    c3.metric("Satisfaction", f"{sat:.0%}" if sat is not None else "—")
    c4.metric("Rating coverage", f"{overview.get('rating_coverage', 0.0):.0%}")
    if overview.get("rated_turns"):
        st.caption(
            f"Satisfaction is over the {overview['rated_turns']} turns that were rated "
            f"({overview.get('rating_coverage', 0.0):.0%} of all turns). Ratings are "
            "self-selected, so read the rate together with its coverage — a high rate "
            "over a handful of turns is not a measurement of anything."
        )

    if overview["scored_turns"] < overview["total_turns"]:
        st.caption(
            f"{overview['scored_turns']} of {overview['total_turns']} turns carry enough "
            "evidence to be scored. The rest are the newest turns and session-final "
            "turns, which have no follow-up to learn from yet."
        )

    tabs = st.tabs(
        ["Responses", "Ratings", "Features", "Users", "Documents", "Inferred needs", "Knowledge gaps"]
    )

    with tabs[0]:
        st.subheader("Ranking of API responses")
        order = st.radio(
            "Order", ["worst", "best"], horizontal=True, label_visibility="collapsed"
        )
        rows = fb_rankings.response_ranking(top_n=25, order=order)
        if not rows:
            st.info("No turns have enough evidence to rank yet.")
        for row in rows:
            header = (
                f"{row['quality']:+.2f} · {row['question'][:70] or '(empty)'}"
            )
            with st.expander(header):
                st.markdown(f"**Question:** {row['question']}")
                st.markdown(f"**Answer (preview):** {row['reply_preview']}")
                if row.get("inferred_user_need"):
                    st.markdown(f"**What they actually wanted:** {row['inferred_user_need']}")
                if row.get("rationale"):
                    st.markdown(f"**Judge rationale:** {row['rationale']}")
                meta = st.columns(4)
                meta[0].metric("Quality", f"{row['quality']:+.2f}")
                meta[1].metric(
                    "Behavioural",
                    "—" if row["behavioral_score"] is None else f"{row['behavioral_score']:+.2f}",
                )
                meta[2].metric(
                    "Judge",
                    "—" if row["judge_overall"] is None else f"{row['judge_overall']:.2f}",
                )
                meta[3].metric("Latency", f"{row['latency_ms']} ms")
                flags = []
                if row["rephrased"]:
                    flags.append("user re-asked")
                if row["corrected"]:
                    flags.append("user corrected the system")
                if flags:
                    st.warning(" · ".join(flags))
                if row["sources"]:
                    st.caption("Retrieved from: " + ", ".join(str(s) for s in row["sources"]))

    with tabs[1]:
        st.subheader("Explicit user ratings")
        summary = fb_rankings.rating_summary()
        if not summary["rated_turns"]:
            st.info(
                "No ratings yet. The 👍/👎 buttons under each answer in the chat feed "
                "this tab."
            )
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rated turns", summary["rated_turns"])
            c2.metric("👍 / 👎", f"{summary['thumbs_up']} / {summary['thumbs_down']}")
            sat = summary["satisfaction_rate"]
            c3.metric("Satisfaction", f"{sat:.0%}" if sat is not None else "—")
            c4.metric("With comments", summary["with_comments"])

            st.markdown("**Where thumbs-down concentrates**")
            st.caption(
                "A global satisfaction number is not actionable; the split is. "
                "Low-volume buckets are shown with their counts rather than hidden, "
                "since at these volumes a suppressed bucket misleads more than a small one."
            )
            for label, key in [
                ("By query type", "by_query_type"),
                ("By workflow", "by_workflow"),
                ("By stage", "by_stage"),
            ]:
                rows = [r for r in summary[key] if r["rated"]]
                if rows:
                    st.markdown(f"*{label}*")
                    st.dataframe(rows, hide_index=True, use_container_width=True)

            st.markdown("**What people wrote**")
            comments = fb_store.rating_rows(limit=100, only_with_comments=True)
            if not comments:
                st.caption("No written comments yet.")
            for row in comments:
                thumb = "👍" if int(row["rating"]) > 0 else "👎"
                question = (row.get("user_message") or "(turn not recorded)")[:80]
                with st.expander(f"{thumb} {question}"):
                    st.markdown(f"**Comment:** {row['comment']}")
                    st.caption(
                        f"{row['username']} · {row['updated_at']} · "
                        f"workflow={row.get('workflow')} · query_type={row.get('query_type')}"
                    )
                    if row.get("reply"):
                        st.text_area(
                            "Answer that was rated",
                            value=row["reply"],
                            height=140,
                            disabled=True,
                            key=f"rated_reply_{row['turn_uid']}",
                        )
                    sources = [s.get("source") for s in (row.get("sources") or [])]
                    if sources:
                        st.caption(f"Sources: {', '.join(str(s) for s in sources)}")

    with tabs[2]:
        st.subheader("Ranking of use")
        features = fb_rankings.feature_ranking()
        for label, key in [
            ("Workflows", "workflows"),
            ("Intents", "intents"),
            ("Query types", "query_types"),
            ("Stages classified", "stages"),
        ]:
            st.markdown(f"**{label}**")
            st.dataframe(features[key], use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("Ranking of user usage")
        st.dataframe(fb_rankings.user_ranking(), use_container_width=True, hide_index=True)

    with tabs[4]:
        st.subheader("Document standing (the closed loop)")
        st.caption(
            "`weight` multiplies each document's retrieval ranking score. It stays at "
            "1.00 until a document has enough scored turns, and is clamped so learned "
            "preference breaks near-ties without overriding a strong semantic match."
        )
        sources = fb_rankings.source_ranking()
        if sources:
            st.dataframe(sources, use_container_width=True, hide_index=True)
        else:
            st.info("No document weights learned yet.")
        if st.button("🔄 Recompute weights and gaps now"):
            st.json(fb_adaptation.recompute_all())
        if st.button("⚖️ Judge any ungraded turns"):
            st.success(f"Graded {fb_judge.judge.judge_pending(limit=50)} turn(s).")

    with tabs[5]:
        st.subheader("What the system thinks users want")
        st.caption(
            "Reconstructed by the judge from each exchange — nobody was asked. "
            "Low mean quality here means a need the system recognises but serves badly."
        )
        needs = fb_rankings.inferred_user_needs()
        if needs:
            st.dataframe(needs, use_container_width=True, hide_index=True)
        else:
            st.info("No inferred needs recorded yet (the judge may be disabled).")

    with tabs[6]:
        st.subheader("Knowledge gaps")
        st.caption("Recurring questions answered badly — the ingestion to-do list.")
        gaps = fb_rankings.knowledge_gaps()
        if gaps:
            st.dataframe(gaps, use_container_width=True, hide_index=True)
        else:
            st.success("No recurring failure topics detected.")


def render_about_page():
    """Static About page (content from 'About the NIH Stage Model Chatbot')."""
    st.title("ℹ️ About the NIH Stage Model Chatbot")
    st.markdown(
        """
        The NIH Stage Model Chatbot is an AI assistant designed to help behavioral
        scientists apply the NIH Stage Model throughout the intervention development
        process. Whether you are generating new research ideas, developing an NIH
        grant application, writing a manuscript, preparing a presentation, or
        planning the next phase of an intervention program, the chatbot can provide
        stage-specific guidance grounded in the NIH Stage Model.

        **The chatbot can help you:**
        - Determine the most appropriate NIH Stage for your intervention or research project.
        - Clarify what evidence is needed to justify progression to the next Stage.
        - Identify potential mechanisms of behavior change and recommend validated measures to assess them.
        - Suggest study designs, comparison groups, outcomes, and fidelity assessments that align with a given Stage.
        - Anticipate common reviewer concerns and identify gaps in a research plan.
        - Explain how hybrid effectiveness-implementation designs, optimization trials, and implementation studies fit within the NIH Stage Model.
        - Recommend strategies for reporting intervention development studies in manuscripts and presentations.
        """
    )

    st.header("Example Questions")
    example_sections = [
        (
            "📝 Developing a grant",
            [
                '"I have an intervention that showed promising feasibility results. '
                'What NIH Stage am I in, and what should my next R01 study look like?"',
                '"What mechanisms of behavior change should I measure in this Stage Ib trial?"',
                '"What control group would be appropriate for a Stage II efficacy trial?"',
            ],
        ),
        (
            "🔬 Planning a study",
            [
                '"How large should my pilot study be?"',
                '"Should I conduct another pilot study or move directly to an efficacy trial?"',
                '"How should I evaluate intervention fidelity at this Stage?"',
            ],
        ),
        (
            "📄 Writing a manuscript",
            [
                '"Does this study fit the NIH Stage Model?"',
                '"How should I describe my intervention development process using the NIH Stage Model?"',
                '"What limitations should I discuss based on my current Stage?"',
            ],
        ),
        (
            "📊 Preparing presentations",
            [
                '"Help me explain why our intervention is ready to advance to the next Stage."',
                '"Create a figure illustrating our progression through the NIH Stage Model."',
                '"Summarize the rationale for our study using NIH Stage Model terminology."',
            ],
        ),
        (
            "🎓 Learning the NIH Stage Model",
            [
                '"Explain the difference between Stage Ia and Stage Ib."',
                '"When should implementation outcomes first be measured?"',
                '"How do optimization trials fit within the NIH Stage Model?"',
            ],
        ),
    ]
    for section_title, questions in example_sections:
        with st.expander(section_title):
            for q in questions:
                st.markdown(f"- {q}")

    st.info(
        "**Tip:** The chatbot works best when you provide context. Paste a draft "
        "Specific Aims page, manuscript abstract, study design, or intervention "
        "description, and ask the chatbot to critique it through the lens of the "
        "NIH Stage Model. It can identify your current Stage, highlight strengths "
        "and gaps, suggest mechanisms to measure, and recommend logical next steps "
        "in intervention development."
    )
    st.warning(
        "**Important:** The chatbot is intended as a scientific planning and "
        "educational resource. Its recommendations should complement—not "
        "replace—careful review of the NIH Stage Model literature, relevant "
        "methodological guidance, and consultation with collaborators, mentors, "
        "and NIH program staff."
    )


def render_workflow_cards():
    # st.markdown("### Guided Workflows")
    # st.caption("Choose Auto for intent-driven routing, or pick one of the three specialized workflows.")

    options = [
        ("auto", "Auto", "Intent-driven routing", True),
        ("mechanism_coach", "Mechanism Coach", "Mechanism ranking + validation", False),
        ("study_builder", "Study Builder", "Stage-specific design matrix", False),
        ("grant_partner", "Grant Partner", "Specific aims + reviewer critique", False),
        ("measure_finder", "Measure Finder", "Construct-to-measure shortlist", False),
    ]

    cols = st.columns(len(options))
    for col, (value, title, subtitle, enabled) in zip(cols, options):
        with col:
            is_active = st.session_state.selected_workflow == value
            if enabled:
                if st.button(title, key=f"workflow_{value}", use_container_width=True, type="primary" if is_active else "secondary"):
                    st.session_state.selected_workflow = value
                    st.rerun()
                st.caption(subtitle)
            else:
                st.button(title, key=f"workflow_{value}", use_container_width=True, disabled=True)
                st.caption("🚧 In development")

    # st.info(f"Current workflow mode: **{st.session_state.selected_workflow}**")


with st.sidebar:
    st.title("🔬 NIH Stage Model")
    if st.session_state.get("authenticated"):
        st.caption(f"Signed in as **{st.session_state.get('auth_username', 'user')}**")
        if st.button("🚪 Log Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.auth_username = None
            # Drop per-user state so the next login reloads its own history.
            for _key in ("conversations", "active_conversation_id", "messages", "session_id"):
                st.session_state.pop(_key, None)
            st.rerun()
        with st.expander("🔑 Change Password"):
            with st.form("change_password_form"):
                current_pw = st.text_input("Current password", type="password")
                new_pw = st.text_input("New password", type="password")
                confirm_pw = st.text_input("Confirm new password", type="password")
                pw_submitted = st.form_submit_button("Update")
            if pw_submitted:
                from auth import change_password

                if new_pw != confirm_pw:
                    st.error("New passwords do not match.")
                else:
                    ok, msg = change_password(
                        st.session_state.get("auth_username", ""), current_pw, new_pw
                    )
                    (st.success if ok else st.error)(msg)
    st.markdown("---")

    # Analytics expose per-user activity, so the tab only appears for admins
    # (ANALYTICS_ADMIN_USERS, or any user when AUTH_DISABLED).
    from app.feedback import is_admin as _is_admin

    _pages = ["chat", "about"]
    if _is_admin(_history_username()):
        _pages.append("analytics")
    if st.session_state.current_page not in _pages:
        st.session_state.current_page = "chat"

    nav_choice = st.radio(
        "Page",
        options=_pages,
        index=_pages.index(st.session_state.current_page),
        format_func=lambda p: {
            "chat": "💬 Chat",
            "about": "ℹ️ About",
            "analytics": "📊 Analytics",
        }[p],
        horizontal=True,
        label_visibility="collapsed",
    )
    if nav_choice != st.session_state.current_page:
        st.session_state.current_page = nav_choice
        st.rerun()
    st.markdown("---")

    st.subheader("Conversations")
    if st.button("➕ New Chat", use_container_width=True):
        create_new_conversation()
        st.rerun()

    conversation_ids = list(st.session_state.conversations.keys())
    selected_conv_id = st.radio(
        "History",
        options=conversation_ids,
        index=conversation_ids.index(st.session_state.active_conversation_id)
        if st.session_state.active_conversation_id in conversation_ids
        else 0,
        format_func=lambda cid: human_title(st.session_state.conversations[cid].get("title", "New Chat")),
        label_visibility="collapsed",
    )
    if selected_conv_id != st.session_state.active_conversation_id:
        st.session_state.active_conversation_id = selected_conv_id
        st.session_state.session_id = selected_conv_id
        selected_conv = st.session_state.conversations[selected_conv_id]
        _ensure_messages_loaded(selected_conv)
        st.session_state.messages = selected_conv.get("messages", [])
        st.rerun()

    active_conv = get_active_conversation()
    # st.caption(f"Session ID: `{active_conv['session_id'][:8]}...`")

    if st.button("🗑️ Delete Current Chat", use_container_width=True):
        _deleted_id = st.session_state.active_conversation_id
        try:
            chat_history.delete_conversation(_history_username(), _deleted_id)
        except Exception:
            pass
        st.session_state.conversations.pop(_deleted_id, None)
        remaining = list(st.session_state.conversations.keys())
        if remaining:
            st.session_state.active_conversation_id = remaining[0]
            st.session_state.session_id = remaining[0]
            next_conv = st.session_state.conversations[remaining[0]]
            _ensure_messages_loaded(next_conv)
            st.session_state.messages = next_conv.get("messages", [])
        else:
            create_new_conversation()
        st.rerun()

    st.markdown("---")
    st.subheader("Settings")
    st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
    st.session_state.show_thinking_trace = st.checkbox(
        "Show Thinking Trace",
        value=st.session_state.show_thinking_trace,
    )

    st.markdown("---")
    st.subheader("System Status")
    backend_ok = check_backend_health()
    if backend_ok:
        st.success("✅ System Ready")
    else:
        st.error("❌ System failed to initialize")

    st.markdown("---")
    with st.expander("📖 Usage"):
        st.markdown(
            """
            **How to use:**
            1. Type your question
            2. The system detects intent/stage automatically
            3. Review answer, reasoning, and references

            **Example prompts:**
            - "What is NIH Stage Model?"
            - "Our study is a pilot feasibility trial. Which stage is it?"
            - "What are Stage I requirements?"
            """
        )

if st.session_state.current_page == "about":
    render_about_page()
    st.stop()

if st.session_state.current_page == "analytics":
    render_analytics_page()
    st.stop()

def render_rating_controls(message: dict) -> None:
    """Thumbs up/down plus an optional comment, under one assistant answer.

    Renders nothing without a turn_uid — that is the case for guardrail
    rejections, for answers produced while FEEDBACK_ENABLED was off, and for
    conversations reloaded from the database (history stores only role and
    content, so the id is not recoverable).

    The rating is written straight to app.feedback, for the same reason the
    analytics page reads the tables directly: this frontend is often deployed
    without a separate backend.
    """
    turn_uid = message.get("turn_uid")
    if not turn_uid:
        return

    from app import feedback as fb

    current = message.get("rating")
    saved_comment = message.get("rating_comment") or ""

    def _write(rating, comment=None):
        try:
            fb.record_rating(
                turn_uid=turn_uid,
                username=_history_username(),
                rating=rating,
                comment=comment,
            )
            message["rating"] = rating
            message["rating_comment"] = comment
            sync_active_conversation_messages()
        except Exception as exc:
            st.warning(f"Could not save feedback: {exc}")

    up_col, down_col, status_col = st.columns([1, 1, 10])
    with up_col:
        # Clicking the active thumb again withdraws the rating.
        if st.button(
            "👍",
            key=f"rate_up_{turn_uid}",
            type="primary" if current == 1 else "secondary",
            help="This answer was useful",
        ):
            _write(None if current == 1 else 1, saved_comment or None)
            st.rerun()
    with down_col:
        if st.button(
            "👎",
            key=f"rate_down_{turn_uid}",
            type="primary" if current == -1 else "secondary",
            help="This answer was not useful",
        ):
            _write(None if current == -1 else -1, saved_comment or None)
            st.rerun()
    with status_col:
        if current == 1:
            st.caption("Thanks — recorded as helpful.")
        elif current == -1:
            st.caption("Thanks — recorded as not helpful. A comment helps us fix it.")

    if current is not None:
        with st.expander("💬 Add a comment (optional)", expanded=False):
            text = st.text_area(
                "What was good or wrong about this answer?",
                value=saved_comment,
                key=f"rate_comment_{turn_uid}",
                height=90,
                label_visibility="collapsed",
                placeholder="e.g. it cited the 1997 paper but my question was about the 2025 revision",
            )
            if st.button("Submit comment", key=f"rate_comment_submit_{turn_uid}"):
                _write(current, text)
                st.success("Comment saved.")


st.title("🔬 NIH Stage Model AI Chatbot")
st.markdown("A multi-agent assistant for NIH Stage Model guidance.")
render_workflow_cards()

active_conv = get_active_conversation()
st.session_state.session_id = active_conv["session_id"]
st.session_state.messages = active_conv.get("messages", [])

if not backend_ok:
    st.error("⚠️ System failed to initialize. Check your environment variables and dependencies.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if st.session_state.debug_mode and message.get("debug"):
            with st.expander("🔍 Debug Info"):
                st.json(message["debug"])
        if st.session_state.show_thinking_trace and message.get("debug"):
            render_thinking_trace(message.get("debug") or {})
        if message["role"] == "assistant":
            render_rating_controls(message)

uploaded_files = st.file_uploader(
    "Attach Files Here",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help=(
        "Attached files are working context for this conversation only. They are "
        "never added to the reference corpus and are discarded when the session ends."
    ),
)

# Extraction is the expensive part of a turn and Streamlit hands back the same
# file objects on every rerun, so results are cached per (name, size). Without
# this a PDF is re-parsed on every message sent while it stays attached.
if "_attachment_cache" not in st.session_state:
    st.session_state._attachment_cache = {}


def parse_uploaded_files_cached(files):
    if not files:
        return [], []
    cache = st.session_state._attachment_cache
    parsed, failures, to_parse = [], [], []
    for up in files:
        key = (up.name, up.size)
        if key in cache:
            hit = cache[key]
            (parsed if hit["ok"] else failures).append(hit["value"])
        else:
            to_parse.append((key, up))

    if to_parse:
        fresh, fresh_failures = parse_uploaded_files([up for _, up in to_parse])
        by_name = {f["name"]: f for f in fresh}
        for key, up in to_parse:
            if up.name in by_name:
                cache[key] = {"ok": True, "value": by_name[up.name]}
                parsed.append(by_name[up.name])
        for msg in fresh_failures:
            failures.append(msg)
    return parsed, failures


_attached_files, _attach_failures = parse_uploaded_files_cached(uploaded_files)

# Failures are shown in the page, not folded into a collapsed expander: a user
# whose scanned PDF yielded nothing needs to know before they ask about it.
for _msg in _attach_failures:
    st.warning(f"📎 {_msg}")

# Files bound to the session on an earlier turn: the uploader forgets them
# once the widget is cleared, but the conversation still has them in context.
_bound_state = None
_bound = []
try:
    _bound_state = state_store.get_state(st.session_state.session_id)
    _bound = list(_bound_state.attachments) if _bound_state else []
except Exception:
    _bound_state, _bound = None, []

if _attached_files or _bound:
    names = {a.name for a in _bound} | {f["name"] for f in _attached_files}
    cols = st.columns([6, 1])
    cols[0].caption(
        "📎 In context for this conversation: "
        + ", ".join(sorted(names))
        + " — not added to the reference corpus."
    )
    if cols[1].button("Clear", key="clear_attachments", help="Detach all files"):
        if _bound_state:
            _bound_state.clear_attachments()
        st.session_state._attachment_cache = {}
        st.rerun()

user_input = st.chat_input("Enter your question...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "timestamp": datetime.now().isoformat()}
    )
    if active_conv.get("title") == "New Chat":
        short_title = user_input.strip().replace("\n", " ")
        active_conv["title"] = (short_title[:36] + "...") if len(short_title) > 36 else short_title
    sync_active_conversation_messages()

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            workflow_value = st.session_state.selected_workflow
            payload = {"session_id": st.session_state.session_id, "message": user_input}
            payload["workflow"] = workflow_value
            payload["attachments"] = [
                {"name": f["name"], "text": f["text"]} for f in _attached_files
            ]
            if _attached_files:
                with st.expander(
                    f"📎 {len(_attached_files)} attached file(s) in context", expanded=False
                ):
                    for f in _attached_files:
                        st.markdown(f"**{f['name']}** — {len(f['text'])} chars. {f['status']}")
                        st.text(f["text"][:800] + ("..." if len(f["text"]) > 800 else ""))
            try:
                is_valid, error_msg = Guardrails.validate_message(payload["message"])
                if not is_valid:
                    st.error(f"❌ {error_msg}")
                elif not Guardrails.is_behavioral_science_related(payload["message"]):
                    reply = Guardrails.rejection_message()
                    st.markdown(reply)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply,
                        "timestamp": datetime.now().isoformat(),
                        "debug": {},
                    })
                    sync_active_conversation_messages()
                    _record_exchange(user_input, reply)
                else:
                    orchestrator = get_orchestrator()
                    # Read on the main thread: the worker below has no access
                    # to st.session_state.
                    _username = _history_username()

                    # Stream the response: the orchestrator runs in a worker
                    # thread and pushes responder tokens into a queue that
                    # st.write_stream drains in the main thread.
                    import queue
                    import threading

                    token_queue: queue.Queue = queue.Queue()
                    _DONE = object()
                    result_holder = {}

                    def _run_orchestrator():
                        try:
                            result_holder["result"] = orchestrator.process_message(
                                session_id=payload["session_id"],
                                user_message=payload["message"],
                                workflow_override=payload.get("workflow"),
                                attachments=payload.get("attachments"),
                                stream_handler=token_queue.put,
                                username=_username,
                            )
                        except Exception as worker_exc:
                            result_holder["error"] = worker_exc
                        finally:
                            token_queue.put(_DONE)

                    worker = threading.Thread(target=_run_orchestrator, daemon=True)
                    worker.start()

                    def _token_gen():
                        while True:
                            item = token_queue.get()
                            if item is _DONE:
                                break
                            yield item

                    st.write_stream(_token_gen())
                    worker.join()

                    if "error" in result_holder:
                        raise result_holder["error"]
                    reply, debug_info = result_holder["result"]
                    reply = Guardrails.sanitize_response(reply)

                    debug_info = debug_info or {}
                    if debug_info and st.session_state.debug_mode:
                        with st.expander("🔍 Debug Info", expanded=False):
                            st.json(debug_info)
                            if debug_info.get("stage"):
                                st.metric("Detected Stage", debug_info.get("stage"))
                            if debug_info.get("stage_confidence") is not None:
                                st.metric(
                                    "Stage Confidence",
                                    f"{debug_info.get('stage_confidence'):.2f}",
                                )
                            if debug_info.get("route_mode"):
                                st.text(f"Route Mode: {debug_info.get('route_mode')}")
                            if debug_info.get("agents_called"):
                                st.text(
                                    f"Called Agents: {', '.join(debug_info.get('agents_called', []))}"
                                )
                    if debug_info and st.session_state.show_thinking_trace:
                        render_thinking_trace(debug_info)

                    assistant_message = {
                        "role": "assistant",
                        "content": reply,
                        "timestamp": datetime.now().isoformat(),
                        "debug": debug_info,
                        # Identifies this turn for an explicit rating.
                        "turn_uid": debug_info.get("turn_uid"),
                    }
                    st.session_state.messages.append(assistant_message)
                    sync_active_conversation_messages()
                    _record_exchange(user_input, reply)
                    # This message was appended after the history loop ran, so
                    # its controls are drawn here; later reruns draw them in
                    # the loop instead.
                    render_rating_controls(assistant_message)

            except Exception as exc:
                st.error(f"❌ Error: {str(exc)}")
                if st.session_state.debug_mode:
                    st.exception(exc)

# st.markdown("---")
# st.caption("NIH Stage Model AI Chatbot | Built with Streamlit")
