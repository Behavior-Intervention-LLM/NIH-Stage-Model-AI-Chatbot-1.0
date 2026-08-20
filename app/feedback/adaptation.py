"""
The closed loop: turning accumulated feedback into behaviour change.

Two things are learned and written back:

1. **Source weights.** Every document that gets retrieved accumulates credit
   from the turns it contributed to. A document that keeps showing up in
   answers users re-ask gets demoted; one that keeps showing up in answers
   users build on gets promoted. VersionedRAGTool multiplies its final ranking
   score by that weight, so retrieval drifts toward what actually works.

2. **Knowledge gaps.** Questions the system keeps answering badly are
   clustered by topic and recorded, so ingestion has a list of what the corpus
   is missing rather than a guess.

Safety properties that matter here:

- A source needs FEEDBACK_MIN_OBSERVATIONS turns before its weight moves off
  1.0 at all, so one bad day cannot bury a document.
- Weights are clamped to [1 - span, 1 + span] (default 0.6-1.4), so learned
  preference can reorder near-ties but can never override a strong semantic
  match with a weak one.
- Weights are recomputed from the full history every time, not nudged
  incrementally. Recomputation is idempotent, and fixing the scoring rules
  fixes every historical weight on the next run.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.feedback import scoring, signals, store
from app.logging_config import logger

# Gap clustering: a topic must fail this often, this badly, to get recorded.
GAP_MIN_OCCURRENCES = 2
GAP_QUALITY_CEILING = -0.10

_weight_cache: Dict[str, float] = {}
_weight_cache_at: float = 0.0
_WEIGHT_CACHE_TTL_SECONDS = 60.0

_turns_since_recompute = 0


# ==================== source weights ====================


def recompute_source_weights() -> Dict[str, Any]:
    """Rebuild every document's weight from the full scored history."""
    min_observations = int(getattr(settings, "FEEDBACK_MIN_OBSERVATIONS", 3))
    span = float(getattr(settings, "FEEDBACK_WEIGHT_SPAN", 0.4))
    gain = float(getattr(settings, "FEEDBACK_WEIGHT_GAIN", 1.5))

    turns = scoring.annotate(store.scored_turns())

    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    good: Dict[str, int] = {}
    bad: Dict[str, int] = {}

    for turn in turns:
        evidence = float(turn.get("evidence") or 0.0)
        if evidence <= 0.0:
            continue
        quality = float(turn.get("quality") or 0.0)
        credit = scoring.credit_for_sources(turn.get("sources") or [], quality, evidence)
        for source, value in credit.items():
            totals[source] = totals.get(source, 0.0) + value
            counts[source] = counts.get(source, 0) + 1
            if quality >= scoring.GOOD_THRESHOLD:
                good[source] = good.get(source, 0) + 1
            elif quality <= scoring.BAD_THRESHOLD:
                bad[source] = bad.get(source, 0) + 1

    updated = 0
    for source, total in totals.items():
        observations = counts.get(source, 0)
        mean_credit = total / observations if observations else 0.0
        if observations < min_observations:
            weight = 1.0  # Not enough evidence to move it yet.
        else:
            weight = max(1.0 - span, min(1.0 + span, 1.0 + gain * mean_credit))
        store.upsert_source_weight(
            source,
            weight=weight,
            observations=observations,
            good_turns=good.get(source, 0),
            bad_turns=bad.get(source, 0),
            mean_credit=mean_credit,
        )
        updated += 1

    invalidate_weight_cache()
    return {
        "sources_updated": updated,
        "turns_considered": len(turns),
        "turns_with_evidence": sum(1 for t in turns if float(t.get("evidence") or 0) > 0),
    }


def source_weight_lookup() -> Dict[str, float]:
    """Cached weight table for the retrieval hot path.

    Retrieval runs on every turn; the weights change on the order of hours.
    A short TTL cache keeps this from being a database round trip per query,
    and any failure degrades to "no learned preference" rather than an error.
    """
    global _weight_cache, _weight_cache_at

    if not getattr(settings, "FEEDBACK_ADAPTIVE_RETRIEVAL", True):
        return {}

    now = time.monotonic()
    if _weight_cache_at and (now - _weight_cache_at) < _WEIGHT_CACHE_TTL_SECONDS:
        return _weight_cache

    try:
        _weight_cache = store.all_source_weights()
    except Exception:
        logger.debug("Feedback source weights unavailable; using neutral weights.", exc_info=True)
        _weight_cache = {}
    _weight_cache_at = now
    return _weight_cache


def invalidate_weight_cache() -> None:
    global _weight_cache_at
    _weight_cache_at = 0.0


# ==================== knowledge gaps ====================


def refresh_knowledge_gaps() -> Dict[str, Any]:
    """Cluster badly-answered questions into a to-fix list for ingestion."""
    turns = scoring.annotate(store.scored_turns())

    clusters: Dict[str, Dict[str, Any]] = {}
    for turn in turns:
        if float(turn.get("evidence") or 0.0) <= 0.0:
            continue
        quality = float(turn.get("quality") or 0.0)
        if quality > GAP_QUALITY_CEILING:
            continue
        key, terms = signals.topic_key(turn.get("user_message") or "")
        if not key:
            continue
        bucket = clusters.setdefault(
            key,
            {
                "terms": terms,
                "qualities": [],
                "example": turn.get("user_message") or "",
                # An unmet need the judge articulated is a better description
                # of the gap than the raw question.
                "need": turn.get("inferred_user_need") or "",
            },
        )
        bucket["qualities"].append(quality)
        if not bucket["need"] and turn.get("inferred_user_need"):
            bucket["need"] = turn["inferred_user_need"]

    recorded = 0
    for key, bucket in clusters.items():
        occurrences = len(bucket["qualities"])
        if occurrences < GAP_MIN_OCCURRENCES:
            continue
        store.upsert_gap(
            key,
            terms=bucket["terms"],
            occurrences=occurrences,
            mean_quality=sum(bucket["qualities"]) / occurrences,
            example_query=bucket["need"] or bucket["example"],
        )
        recorded += 1

    return {"gaps_recorded": recorded, "clusters_seen": len(clusters)}


# ==================== scheduling ====================


def recompute_all() -> Dict[str, Any]:
    """Run every learning step. Safe to call from an endpoint or a cron."""
    weights = recompute_source_weights()
    gaps = refresh_knowledge_gaps()
    return {**weights, **gaps}


def note_turn_and_maybe_recompute() -> Optional[Dict[str, Any]]:
    """Called once per turn; recomputes every FEEDBACK_RECOMPUTE_EVERY_TURNS.

    Recomputation walks the whole history, so it is deliberately infrequent
    rather than per-turn. It runs on whichever thread already finished a turn,
    which is the background telemetry thread, never the request thread.
    """
    global _turns_since_recompute

    interval = int(getattr(settings, "FEEDBACK_RECOMPUTE_EVERY_TURNS", 25))
    if interval <= 0:
        return None

    _turns_since_recompute += 1
    if _turns_since_recompute < interval:
        return None

    _turns_since_recompute = 0
    try:
        result = recompute_all()
        logger.info("Feedback adaptation recomputed: %s", result)
        return result
    except Exception:
        logger.warning("Feedback adaptation recompute failed", exc_info=True)
        return None
