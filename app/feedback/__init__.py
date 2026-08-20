"""
Feedback system.

How well the system is serving people is measured from three streams:

    ratings.     an explicit thumbs up/down with an optional comment, written
                 by the user -- the only stream that is stated rather than
                 inferred, and by far the sparsest
    signals.py   what the user did next (re-asked, corrected, thanked, built
                 on the answer, left) -- sparse but it is ground truth
    judge.py     an LLM grading each answer against the passages that answer
                 actually retrieved -- dense, but a model marking its own work

The inferred streams exist because ratings are self-selected: they are left
mostly by people who are annoyed or delighted, so treating them as the only
measure both undersamples ordinary turns and pulls toward agreeableness. They
are therefore weighted to decide the sign of a turn's quality without erasing
the other two (see scoring.EXPLICIT_WEIGHT), and rating coverage is reported
alongside every aggregate so the sampling bias stays visible.

scoring.py fuses them into one quality number per turn. rankings.py aggregates
those into the rankings. adaptation.py feeds them back: documents that keep
appearing in bad answers get demoted in retrieval, and topics the system keeps
fumbling get recorded as knowledge gaps.

Entry points:
    observe_turn()   called once per turn by the orchestrator. Returns the
                     turn_uid immediately and does all its work on a daemon
                     thread, so a failure here can slow down or break exactly
                     nothing in the chat path. The turn_uid is what a client
                     later quotes to attach a rating.
    record_rating()  called when a user clicks a thumb.

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
    turn_uid: str,
) -> Optional[str]:
    """Record, derive, judge, adapt. Runs off the request path."""
    from app.feedback import adaptation, judge as judge_module, signals, store

    turn_uid = store.record_turn(
        turn_uid=turn_uid,
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
) -> Optional[str]:
    """Observe one completed turn. Never raises, never blocks by default.

    Returns the turn_uid so the caller can pass it to the client, which needs
    it to attach a rating later. The id is minted here rather than by the
    background write, so it is available immediately and stays stable.

    Set blocking=True only in tests, where a daemon thread would race the
    assertions.
    """
    if not is_enabled():
        return None

    from app.feedback import store

    turn_uid = store.new_turn_uid()

    kwargs = dict(
        session_id=session_id,
        username=username or "anonymous",
        user_message=user_message or "",
        reply=reply or "",
        debug_info=debug_info or {},
        sources=sources or [],
        tool_errors=int(tool_errors or 0),
        latency_ms=int(latency_ms or 0),
        turn_uid=turn_uid,
    )

    def _run():
        try:
            _observe_sync(**kwargs)
        except Exception:
            # Telemetry is never worth an error in the chat path.
            logger.warning("Feedback observation failed", exc_info=True)

    if blocking:
        _run()
        return turn_uid

    threading.Thread(target=_run, daemon=True, name="feedback-observe").start()
    return turn_uid


def record_rating(
    *,
    turn_uid: str,
    username: str,
    rating: Optional[int],
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach (or clear) an explicit rating for a turn.

    `rating` is +1, -1, or None to withdraw a previous rating. Raises
    PermissionError if the turn is owned by someone else, and ValueError on a
    bad rating value — callers map those onto 403/400.

    The ownership check passes when the turn row does not exist yet: the turn
    is written on a background thread, so a fast click can arrive first. The
    turn_uid is an unguessable uuid4 handed only to the user who produced the
    turn, so it is what authorises the write.
    """
    from app.feedback import store

    if not is_enabled():
        raise RuntimeError("Feedback system is disabled (FEEDBACK_ENABLED=false).")

    owner = store.turn_owner(turn_uid)
    if owner is not None and owner != store._normalize_username(username or "anonymous"):
        raise PermissionError("That turn belongs to another user.")

    if rating is None:
        removed = store.clear_rating(turn_uid)
        return {"turn_uid": turn_uid, "rating": None, "comment": None, "cleared": removed}

    return store.save_rating(turn_uid, username, int(rating), comment)


def get_rating(turn_uid: str) -> Optional[Dict[str, Any]]:
    from app.feedback import store

    return store.get_rating(turn_uid)


__all__ = [
    "observe_turn",
    "record_rating",
    "get_rating",
    "is_enabled",
    "is_admin",
]
