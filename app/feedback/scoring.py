"""
Fusing the two evidence streams into one quality number, and assigning credit
for that number back to the documents that produced the answer.

Three sources disagree about scale and about coverage:

    explicit     in {-1, +1}, very sparse, but stated outright by the user
    behavioural  in [-1, 1],  sparse, carries its own confidence
    judge        in [0, 1],   dense, but it is a model grading a model

They are mapped onto a common [-1, 1] axis and combined by availability, so a
turn with only a judge score is still usable and a turn with a loud
behavioural signal is not drowned out by a bland judge score.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Behaviour is trusted more than self-evaluation when both are present and
# the behavioural signal is confident: a user re-asking the question is
# ground truth in a way a model's self-grade is not.
BEHAVIOURAL_MAX_WEIGHT = 0.65
JUDGE_WEIGHT = 0.35

# An explicit thumbs up/down outweighs both inferred streams combined
# (0.65 + 0.35 = 1.0) without erasing them: the user's stated verdict should
# decide the sign, while the other streams still shade the magnitude.
#
# It is not weighted higher still, because ratings are self-selected — mostly
# left by people who are annoyed or delighted — so a handful of them must not
# be able to swing document weights on their own. Everything downstream scales
# by `evidence`, and rating coverage is reported in the overview so the
# sampling bias stays visible rather than being silently averaged in.
EXPLICIT_WEIGHT = 1.20

# Turns above/below these are counted as clear wins/losses in the rankings.
GOOD_THRESHOLD = 0.25
BAD_THRESHOLD = -0.15


def judge_to_axis(judge_overall: Optional[float]) -> Optional[float]:
    """Map a 0..1 judge score onto the shared -1..1 quality axis."""
    if judge_overall is None:
        return None
    return (2.0 * float(judge_overall)) - 1.0


def explicit_to_axis(rating: Optional[Any]) -> Optional[float]:
    """Map a stored +1/-1 thumb onto the shared -1..1 quality axis."""
    if rating in (None, ""):
        return None
    try:
        value = float(rating)
    except (TypeError, ValueError):
        return None
    if value == 0:
        return None
    return 1.0 if value > 0 else -1.0


def fuse(
    behavioral_score: Optional[float],
    behavioral_confidence: Optional[float],
    judge_overall: Optional[float],
    explicit_rating: Optional[Any] = None,
) -> Dict[str, float]:
    """Combine the streams.

    Returns quality in [-1, 1] and evidence in [0, 1], where evidence is how
    much this number should be trusted. A turn with no stream at all returns
    quality 0.0 and evidence 0.0 -- callers must check evidence before acting.
    """
    judge_axis = judge_to_axis(judge_overall)
    explicit_axis = explicit_to_axis(explicit_rating)
    confidence = max(0.0, min(1.0, float(behavioral_confidence or 0.0)))

    behaviour_weight = BEHAVIOURAL_MAX_WEIGHT * confidence if behavioral_score is not None else 0.0
    judge_weight = JUDGE_WEIGHT if judge_axis is not None else 0.0
    explicit_weight = EXPLICIT_WEIGHT if explicit_axis is not None else 0.0
    total = behaviour_weight + judge_weight + explicit_weight

    if total <= 0.0:
        return {"quality": 0.0, "evidence": 0.0, "rated": False}

    quality = (
        (float(behavioral_score or 0.0) * behaviour_weight)
        + ((judge_axis or 0.0) * judge_weight)
        + ((explicit_axis or 0.0) * explicit_weight)
    ) / total

    # Evidence is high only when at least one stream is speaking clearly. A
    # stated rating is the strongest single thing we can know about a turn.
    evidence = min(
        1.0,
        confidence * 0.7
        + (0.5 if judge_axis is not None else 0.0)
        + (0.9 if explicit_axis is not None else 0.0),
    )

    return {
        "quality": round(max(-1.0, min(1.0, quality)), 4),
        "evidence": round(evidence, 4),
        "rated": explicit_axis is not None,
    }


def score_turn(turn: Dict[str, Any]) -> Dict[str, float]:
    """Fuse the streams already joined onto a row by store.scored_turns()."""
    return fuse(
        turn.get("behavioral_score"),
        turn.get("behavioral_confidence"),
        turn.get("judge_overall"),
        turn.get("explicit_rating"),
    )


def annotate(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach quality/evidence to each row in place and return the list."""
    for turn in turns:
        turn.update(score_turn(turn))
    return turns


def credit_for_sources(
    sources: List[Dict[str, Any]], quality: float, evidence: float
) -> Dict[str, float]:
    """Split a turn's quality across the documents that fed it.

    Credit is proportional to each source's share of the turn's total
    retrieval score, so the passage that dominated the answer carries most of
    the blame or most of the praise. Scaled by evidence, so a guess about a
    turn barely moves any document's standing.
    """
    if not sources or evidence <= 0.0:
        return {}

    scores = {}
    for src in sources:
        name = str(src.get("source") or "").strip()
        if not name:
            continue
        # Retrieval scores are non-negative in practice; guard anyway.
        scores[name] = scores.get(name, 0.0) + max(0.0, float(src.get("score") or 0.0))

    total = sum(scores.values())
    if total <= 0.0:
        # No usable scores: split evenly rather than discarding the signal.
        share = 1.0 / len(scores) if scores else 0.0
        return {name: quality * evidence * share for name in scores}

    return {
        name: quality * evidence * (score / total) for name, score in scores.items()
    }
