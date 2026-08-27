"""Pending-offer tracking for follow-up turns.

The responder often ends a reply with an offer ("Would you like me to write
it for you?", "I can next help you with: 1. …, 2. …"). Users accept with a
bare "Yes please!" or "2 please" — messages that carry none of the keywords
the intent layer classifies on, so without this module the turn re-entered
stage classification and the user got their stage restated instead of the
thing they just accepted (observed repeatedly in production logs).

Three pieces:
- `extract_pending_offer(reply)` — pull the offer block off the tail of an
  assistant reply, to be stashed in session state for exactly one turn.
- `is_affirmation(msg)` / `extract_option_choice(msg)` — recognize the short
  acceptance messages themselves.
- `resolve_accepted_offer(pending_offer, msg)` — combine the two: what, if
  anything, did the user just accept?
"""
from __future__ import annotations

import re
from typing import Optional

# Offers live in the closing sentences of a reply. Cues are lowercase; the
# earliest match in the tail window starts the offer block.
_OFFER_CUES = (
    "would you like",
    "do you want me to",
    "want me to",
    "shall i",
    "if you want",
    "if you'd like",
    "if you would like",
    "i can next help",
    "i can help you with",
    "i can also",
    "i could",
    "let me know if",
)

# How far back from the end of a reply an offer can start. Offers observed in
# production sit in the last ~400 chars; 900 leaves margin for a trailing
# numbered list.
_OFFER_TAIL_WINDOW = 900
_OFFER_MAX_CHARS = 700

# "1.", "2)", "**3.**" at line start — the numbered-options format the
# responder produces.
_NUMBERED_ITEM_RE = re.compile(r"^\s*(?:\*\*)?(\d{1,2})[.)]\s*(.+)$")

# Bare option picks: "1", "2 please", "option 3", "#2.", "the second one".
_OPTION_CHOICE_RE = re.compile(
    r"^\s*(?:option\s+)?#?(\d{1,2})\s*(?:please|pls|plz|thanks|thank you)?\s*[.!]*\s*$",
    re.IGNORECASE,
)
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
}
_ORDINAL_CHOICE_RE = re.compile(
    r"^\s*(?:the\s+)?(first|second|third|fourth|fifth)\s+(?:one|option)?\s*"
    r"(?:please|pls|plz|thanks|thank you)?\s*[.!]*\s*$",
    re.IGNORECASE,
)

# An affirmation must contain at least one strong token and nothing outside
# the allowed vocabulary — "yes, but what about Stage III" is a new question,
# not an acceptance.
_AFFIRM_STRONG = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "absolutely",
    "definitely", "certainly", "gladly",
}
_AFFIRM_FILLER = {
    "please", "pls", "plz", "thanks", "thank", "you", "that", "would", "be",
    "great", "good", "sounds", "do", "it", "go", "ahead", "kindly", "why",
    "not", "lets", "let's",
}
_AFFIRM_PHRASES = {"go ahead", "do it", "sounds good", "why not", "please do"}
_NEGATIVE_TOKENS = {"no", "nope", "nah", "dont", "don't", "never"}


def _normalize(msg: str) -> str:
    return re.sub(r"[^\w\s']", " ", (msg or "").lower()).strip()


def is_affirmation(user_message: str) -> bool:
    """True for a short, unqualified acceptance ("Yes please!", "sure, go
    ahead"). Anything that adds content of its own is not an affirmation."""
    normalized = _normalize(user_message)
    if not normalized:
        return False
    tokens = normalized.split()
    if len(tokens) > 6:
        return False
    if any(t in _NEGATIVE_TOKENS for t in tokens):
        # "why not" is idiomatic assent, not a refusal.
        if normalized not in _AFFIRM_PHRASES:
            return False
    has_strong = any(t in _AFFIRM_STRONG for t in tokens) or any(
        p in normalized for p in _AFFIRM_PHRASES
    )
    if not has_strong:
        return False
    return all(t in _AFFIRM_STRONG or t in _AFFIRM_FILLER for t in tokens)


def extract_option_choice(user_message: str) -> Optional[int]:
    """The option number when the whole message is a pick ("2 please",
    "option 1", "the second one"); None otherwise."""
    m = _OPTION_CHOICE_RE.match(user_message or "")
    if m:
        return int(m.group(1))
    m = _ORDINAL_CHOICE_RE.match(user_message or "")
    if m:
        return _ORDINAL_WORDS[m.group(1).lower()]
    return None


def is_short_acknowledgement(user_message: str) -> bool:
    """Affirmation or bare option pick — a message that answers the
    assistant's previous turn rather than asking anything new. Such a turn
    must never be routed into stage classification."""
    return is_affirmation(user_message) or extract_option_choice(user_message) is not None


def extract_pending_offer(reply: str) -> Optional[str]:
    """The offer block at the tail of an assistant reply, or None.

    Looks for the earliest offer cue in the last `_OFFER_TAIL_WINDOW` chars
    and keeps everything from that cue to the end (which is where any
    numbered options list lives)."""
    if not reply:
        return None
    tail = reply[-_OFFER_TAIL_WINDOW:]
    lowered = tail.lower()
    positions = [p for p in (lowered.find(cue) for cue in _OFFER_CUES) if p != -1]
    if not positions:
        return None
    offer = tail[min(positions):].strip()
    return offer[:_OFFER_MAX_CHARS] or None


def _numbered_items(offer: str) -> dict[int, str]:
    """Map option number → item text for a numbered list inside an offer.
    Continuation lines are folded into the preceding item."""
    items: dict[int, str] = {}
    current: Optional[int] = None
    for line in offer.splitlines():
        m = _NUMBERED_ITEM_RE.match(line)
        if m:
            current = int(m.group(1))
            items[current] = m.group(2).strip()
        elif current is not None and line.strip():
            items[current] += " " + line.strip()
    return items


def resolve_accepted_offer(
    pending_offer: Optional[str], user_message: str
) -> Optional[str]:
    """What the user just accepted, phrased for the responder prompt — or
    None when this turn is not an acceptance of the pending offer."""
    if not pending_offer:
        return None
    choice = extract_option_choice(user_message)
    if choice is not None:
        item = _numbered_items(pending_offer).get(choice)
        if item:
            return f"Option {choice} of your offer: {item}"
        return pending_offer
    if is_affirmation(user_message):
        return pending_offer
    return None
