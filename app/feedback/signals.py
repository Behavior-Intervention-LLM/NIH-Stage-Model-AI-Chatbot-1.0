"""
Behavioural signal derivation -- the replacement for thumbs up/down.

The premise: a user who got a good answer behaves differently from one who
did not, and that difference is visible in the transcript without ever asking
them to rate anything. Concretely, for turn N we look at what the user did on
turn N+1:

    they re-ask the same thing        -> turn N failed
    they say "no, I meant ..."        -> turn N failed
    they say "thanks, that helps"     -> turn N landed
    they ask a *deeper* follow-up     -> turn N landed (it moved them forward)
    they read for a while, then go on -> mild positive
    they leave right after complaining-> turn N failed and they gave up

Each turn therefore gets a score in [-1, 1] plus a confidence in [0, 1]
recording how much evidence there actually was. A final turn with no
follow-up yields confidence 0, not a zero rating -- absence of evidence is
not evidence of mediocrity, and the scoring layer must be able to tell the
two apart.

Signals are recomputed for a whole session each time a turn lands, which is
cheap (sessions are short) and self-healing: a late-arriving turn fixes the
score of the turn before it.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

# A rephrase of the previous question shares most of its content words.
REPEAT_SIMILARITY = 0.60
# Below this, the user changed subject rather than drilling in.
TOPIC_SWITCH_SIMILARITY = 0.12
# A repeat sent this fast means they did not even finish reading.
IMPATIENT_SECONDS = 90
# Long enough to have actually read the answer before continuing.
ENGAGED_DWELL_SECONDS = 25

# Anchored to the start of the message, or addressed at the assistant, so that
# a legitimate question like "what's wrong with Stage II designs?" is not read
# as a complaint.
CORRECTION_PATTERNS = [
    r"^\s*(no|nope|nah)\b",
    r"\bi (meant|asked|said)\b",
    r"\bthat'?s (not|wrong|incorrect)\b",
    r"\bthat is (not|wrong|incorrect)\b",
    r"\byou (misunderstood|missed|didn'?t answer|did not answer|are wrong)\b",
    r"\bnot what i (asked|meant|wanted)\b",
    r"\bdoesn'?t answer\b",
    r"\b(still|again)\s+(doesn'?t|does not|not)\b",
    r"^\s*(try again|rephrase|answer the question)\b",
    r"\bthis is (useless|unhelpful|irrelevant)\b",
]

ACCEPTANCE_PATTERNS = [
    r"^\s*(thanks|thank you|ty|cheers)\b",
    r"\b(that|this) (helps|helped|makes sense|is helpful|is what i needed)\b",
    r"^\s*(great|perfect|excellent|awesome|nice)\b",
    r"\bgot it\b",
    r"\bexactly\b",
]

# Follow-ups that build on an answer rather than retry it.
DEEPENING_PATTERNS = [
    r"^\s*(and|also|then)\b",
    r"\bwhat about\b",
    r"\bhow (do|would|should) (i|we)\b",
    r"\bcan you (also|now)\b",
    r"\bnext step\b",
    r"\bgive me an example\b",
]

_STOPWORDS: Set[str] = {
    "a", "about", "an", "and", "any", "are", "as", "at", "be", "but", "by",
    "can", "do", "does", "for", "from", "get", "give", "has", "have", "how",
    "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "please", "should", "so", "some", "tell", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "to", "up", "us", "was",
    "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your",
}

_WORD_RE = re.compile(r"[a-z0-9]+")

# Score contributions. Negative evidence is weighted harder than positive
# because complaining is a deliberate act while continuing is often just
# momentum.
_CONTRIB_CORRECTION = -0.60
_CONTRIB_REPEAT = -0.45
_CONTRIB_IMPATIENT = -0.15
_CONTRIB_ABANDON_AFTER_COMPLAINT = -0.30
_CONTRIB_ACCEPTANCE = 0.50
_CONTRIB_DEEPENING = 0.25
_CONTRIB_ENGAGED_DWELL = 0.10
_CONTRIB_MOVED_ON = 0.05


def content_tokens(text: str) -> Set[str]:
    """Lowercased content words, stopwords removed."""
    return {
        w for w in _WORD_RE.findall((text or "").lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _matches(patterns: List[str], text: str) -> bool:
    low = (text or "").lower()
    return any(re.search(p, low) for p in patterns)


def _parse_time(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _seconds_between(earlier: Any, later: Any) -> Optional[float]:
    a, b = _parse_time(earlier), _parse_time(later)
    if a is None or b is None:
        return None
    try:
        return max(0.0, (b - a).total_seconds())
    except Exception:
        # Mixed naive/aware timestamps: not worth guessing.
        return None


def derive_for_session(turns: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Derive signals for every turn in a session.

    Args:
        turns: session turns in chronological order (see store.session_turns).

    Returns:
        {turn_uid: signal dict} ready for store.save_signals.
    """
    results: Dict[str, Dict[str, Any]] = {}

    for i, turn in enumerate(turns):
        nxt = turns[i + 1] if i + 1 < len(turns) else None
        this_tokens = content_tokens(turn.get("user_message", ""))

        score = 0.0
        evidence = 0
        detail: Dict[str, Any] = {}
        rephrased = corrected = accepted = followed_up = abandoned = False
        seconds_to_next: Optional[float] = None

        if nxt is not None:
            followed_up = True
            next_message = nxt.get("user_message", "")
            next_tokens = content_tokens(next_message)
            similarity = jaccard(this_tokens, next_tokens)
            seconds_to_next = _seconds_between(turn.get("created_at"), nxt.get("created_at"))
            detail["similarity_to_next"] = round(similarity, 3)

            corrected = _matches(CORRECTION_PATTERNS, next_message)
            accepted = _matches(ACCEPTANCE_PATTERNS, next_message)
            rephrased = similarity >= REPEAT_SIMILARITY and not accepted

            if corrected:
                score += _CONTRIB_CORRECTION
                evidence += 1
                detail["corrected"] = True
            if rephrased:
                score += _CONTRIB_REPEAT
                evidence += 1
                detail["rephrased"] = True
                if seconds_to_next is not None and seconds_to_next < IMPATIENT_SECONDS:
                    score += _CONTRIB_IMPATIENT
                    detail["impatient_repeat"] = True
            if accepted:
                score += _CONTRIB_ACCEPTANCE
                evidence += 1
                detail["accepted"] = True

            deepening = (
                not rephrased
                and not corrected
                and (
                    _matches(DEEPENING_PATTERNS, next_message)
                    or TOPIC_SWITCH_SIMILARITY <= similarity < REPEAT_SIMILARITY
                )
            )
            if deepening:
                score += _CONTRIB_DEEPENING
                evidence += 1
                detail["deepening_followup"] = True
            elif not rephrased and not corrected and similarity < TOPIC_SWITCH_SIMILARITY:
                # Clean subject change: weak evidence the previous thread closed.
                score += _CONTRIB_MOVED_ON
                detail["moved_on"] = True

            if (
                seconds_to_next is not None
                and seconds_to_next >= ENGAGED_DWELL_SECONDS
                and not rephrased
                and not corrected
            ):
                score += _CONTRIB_ENGAGED_DWELL
                detail["engaged_dwell"] = True

        else:
            # Terminal turn. Leaving is ambiguous -- a satisfied user and a
            # defeated one both stop typing. Only treat it as negative when
            # the user had already signalled the thread was going wrong.
            prev = turns[i - 1] if i > 0 else None
            if prev is not None and _matches(
                CORRECTION_PATTERNS, turn.get("user_message", "")
            ):
                abandoned = True
                score += _CONTRIB_ABANDON_AFTER_COMPLAINT
                evidence += 1
                detail["abandoned_after_complaint"] = True

        # A retrieval that came back empty is a weak negative on its own: the
        # responder had nothing grounded to work from.
        if int(turn.get("retrieval_count") or 0) == 0 and turn.get("need_stage"):
            score -= 0.10
            detail["no_retrieval"] = True
        if int(turn.get("tool_errors") or 0) > 0:
            score -= 0.10
            detail["tool_errors"] = int(turn["tool_errors"])

        results[turn["turn_uid"]] = {
            "behavioral_score": max(-1.0, min(1.0, round(score, 4))),
            # Two independent observations is treated as full confidence.
            "confidence": round(min(1.0, evidence / 2.0), 4),
            "rephrased": rephrased,
            "corrected": corrected,
            "accepted": accepted,
            "followed_up": followed_up,
            "abandoned": abandoned,
            "seconds_to_next": seconds_to_next,
            "detail": detail,
        }

    return results


def topic_key(text: str, max_terms: int = 5) -> tuple[str, List[str]]:
    """A stable key for grouping semantically similar questions.

    Longest content words, alphabetised -- crude but deterministic, and good
    enough to cluster "which stage is my feasibility pilot" with "what stage
    for a pilot feasibility study".
    """
    tokens = sorted(content_tokens(text), key=len, reverse=True)[:max_terms]
    terms = sorted(tokens)
    return "|".join(terms), terms
