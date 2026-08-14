"""
Implicit feedback system.

The system infers how well it is serving people instead of asking them. There
is no thumbs up/down anywhere in the UI, and none is planned: explicit ratings
are given by a self-selecting minority, mostly when they are annoyed, and they
train the model to be agreeable rather than useful.

Two evidence streams replace the rating widget:

    signals.py   what the user did next (re-asked, corrected, thanked, built
                 on the answer, left) -- sparse but it is ground truth
    judge.py     an LLM grading each answer against the passages that answer
                 actually retrieved -- dense, but a model marking its own work

scoring.py fuses them into one quality number per turn. rankings.py aggregates
those into the four rankings. adaptation.py feeds them back: documents that
keep appearing in bad answers get demoted in retrieval, and topics the system
keeps fumbling get recorded as knowledge gaps.

Entry point: `observe_turn()`, called once per turn by the orchestrator. It
returns immediately and does all its work on a daemon thread, so a failure
here can slow down or break exactly nothing in the chat path.

Note on imports: submodules are imported lazily inside functions. judge.py and
adaptation.py import `from app.feedback import ...`, so a module-level import
of them here would be circular.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from app.config import settings
from app.logging_config import logger


def is_enabled() -> bool:
    return bool(getattr(settings, "FEEDBACK_ENABLED", True))


def is_admin(username: str) -> bool:
    """Analytics expose per-user activity, so they are admin-only.

    Allow-list comes from ANALYTICS_ADMIN_USERS (comma-separated). When it is
    unset, access is granted only with AUTH_DISABLED -- i.e. local development.
    Deny-by-default in any deployment that has authentication switched on.
    """
    raw = (getattr(settings, "ANALYTICS_ADMIN_USERS", "") or "").strip()
    if not raw:
        return bool(getattr(settings, "AUTH_DISABLED", False))
    allowed = {u.strip().lower() for u in raw.split(",") if u.strip()}
    return (username or "").strip().lower() in allowed


def _observe_sync(
    *,
    session_id: str,
    username: str,
    user_message: str,
    reply: str,
    debug_info: Dict[str, Any],
    sources: List[Dict[str, Any]],
    tool_errors: int,
    latency_ms: int,
) -> Optional[str]:
    """Record, derive, judge, adapt. Runs off the request path."""
    from app.feedback import adaptation, judge as judge_module, signals, store

    turn_uid = store.record_turn(
        session_id=session_id,
        username=username,
        user_message=user_message,
        reply=reply,
        workflow=debug_info.get("workflow"),
        intent_label=debug_info.get("intent_label"),
        query_type=debug_info.get("intent_query_type"),
        need_stage=bool(debug_info.get("need_stage")),
        stage=debug_info.get("stage"),
        stage_confidence=float(debug_info.get("stage_confidence") or 0.0),
        intent_confidence=float(debug_info.get("intent_confidence") or 0.0),
        agents_called=debug_info.get("agents_called") or [],
        sources=sources,
        tool_errors=tool_errors,
        latency_ms=latency_ms,
    )

    # Recompute the whole session: this turn's arrival is what reveals whether
    # the *previous* turn landed, so earlier rows get rewritten too.
    try:
        turns = store.session_turns(session_id)
        for uid, signal in signals.derive_for_session(turns).items():
            store.save_signals(uid, signal)
    except Exception:
        logger.warning("Failed deriving feedback signals for %s", session_id, exc_info=True)

    # Judge inline on this same background thread so the adaptation step below
    # already sees the fresh judgement.
    if getattr(settings, "FEEDBACK_JUDGE_ENABLED", True):
        try:
            judge_module.judge.judge_turn(turn_uid)
        except Exception:
            logger.warning("Feedback judge failed for turn %s", turn_uid, exc_info=True)

    try:
        adaptation.note_turn_and_maybe_recompute()
    except Exception:
        logger.warning("Feedback adaptation step failed", exc_info=True)

    return turn_uid


def observe_turn(
    *,
    session_id: str,
    username: str,
    user_message: str,
    reply: str,
    debug_info: Optional[Dict[str, Any]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    tool_errors: int = 0,
    latency_ms: int = 0,
    blocking: bool = False,
) -> None:
    """Observe one completed turn. Never raises, never blocks by default.

    Set blocking=True only in tests, where a daemon thread would race the
    assertions.
    """
    if not is_enabled():
        return

    kwargs = dict(
        session_id=session_id,
        username=username or "anonymous",
        user_message=user_message or "",
        reply=reply or "",
        debug_info=debug_info or {},
        sources=sources or [],
        tool_errors=int(tool_errors or 0),
        latency_ms=int(latency_ms or 0),
    )

    def _run():
        try:
            _observe_sync(**kwargs)
        except Exception:
            # Telemetry is never worth an error in the chat path.
            logger.warning("Feedback observation failed", exc_info=True)

    if blocking:
        _run()
        return

    threading.Thread(target=_run, daemon=True, name="feedback-observe").start()


__all__ = ["observe_turn", "is_enabled", "is_admin"]
