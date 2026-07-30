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


# Extracting PDF
def _extract_text_from_pdf(file_bytes: bytes) -> str:
    py_pdf2 = importlib.util.find_spec("PyPDF2") # type: ignore
    if py_pdf2 is None:
        return ""
    PyPDF2 = importlib.import_module("PyPDF2")
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages[:30], 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {i}]\n{text.strip()}")
        return "\n\n".join(pages).strip()
    except Exception:
        return ""


# Extracting DOCX
def _extract_text_from_docx(file_bytes: bytes) -> str:
    docx_spec = importlib.util.find_spec("docx")
    if docx_spec is None:
        return ""
    Document = importlib.import_module("docx").Document
    try:
        doc = Document(io.BytesIO(file_bytes))
        lines = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(lines).strip()
    except Exception:
        return ""


# Extracting TXT
def _extract_text_from_txt(file_bytes: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return file_bytes.decode(enc, errors="ignore").strip()
        except Exception:
            continue
    return ""


def _extract_text_from_image(file_bytes: bytes) -> tuple[str, str]:
    """Try OCR; return (text, status_msg)."""
    try:
        from PIL import Image
    except Exception:
        return "", "Pillow not available, image OCR skipped."
    pyt_spec = importlib.util.find_spec("pytesseract")
    if pyt_spec is None:
        return "", "pytesseract not installed, image OCR skipped."
    pytesseract = importlib.import_module("pytesseract")
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image).strip()
        if text:
            return text, "Image OCR succeeded."
        return "", "Image OCR completed but no text found."
    except Exception:
        return "", "Image OCR failed."



def parse_uploaded_files(uploaded_files) -> tuple[str, list[str]]:
    if not uploaded_files:
        return "", []
    parsed_parts = []
    parse_logs = []
    for up in uploaded_files:
        name = up.name
        lower = name.lower()
        file_bytes = up.getvalue()
        text = ""
        status = ""

        if lower.endswith(".pdf"):
            text = _extract_text_from_pdf(file_bytes)
            status = "PDF parsed" if text else "PDF parse failed or empty"
        elif lower.endswith(".docx"):
            text = _extract_text_from_docx(file_bytes)
            status = "DOCX parsed" if text else "DOCX parse failed or empty"
        elif lower.endswith(".txt"):
            text = _extract_text_from_txt(file_bytes)
            status = "TXT parsed" if text else "TXT parse failed or empty"
        elif lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
            text, status = _extract_text_from_image(file_bytes)
        else:
            status = "Unsupported file type (supported: pdf/docx/txt/images)"

        if text:
            parsed_parts.append(f"[Source: {name}]\n{text[:3500]}")
        parse_logs.append(f"{name}: {status}")

    merged = "\n\n".join(parsed_parts).strip()
    return merged[:12000], parse_logs


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

    nav_choice = st.radio(
        "Page",
        options=["chat", "about"],
        index=0 if st.session_state.current_page == "chat" else 1,
        format_func=lambda p: {"chat": "💬 Chat", "about": "ℹ️ About"}[p],
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

uploaded_files = st.file_uploader(
    "Attach Files Here",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="Parsed text will be appended as uploaded context for the next message.",
)

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
            parsed_context_text, parse_logs = parse_uploaded_files(uploaded_files)
            if parsed_context_text:
                payload["document_text"] = parsed_context_text
                with st.expander("📎 Parsed Upload Context", expanded=False):
                    st.caption(f"Parsed chars: {len(parsed_context_text)}")
                    st.text(parsed_context_text[:1200] + ("..." if len(parsed_context_text) > 1200 else ""))
            if parse_logs:
                with st.expander("🧾 Upload Parse Logs", expanded=False):
                    for item in parse_logs:
                        st.text(f"- {item}")
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
                                uploaded_context_text=payload.get("document_text"),
                                stream_handler=token_queue.put,
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
                    }
                    st.session_state.messages.append(assistant_message)
                    sync_active_conversation_messages()
                    _record_exchange(user_input, reply)

            except Exception as exc:
                st.error(f"❌ Error: {str(exc)}")
                if st.session_state.debug_mode:
                    st.exception(exc)

# st.markdown("---")
# st.caption("NIH Stage Model AI Chatbot | Built with Streamlit")
