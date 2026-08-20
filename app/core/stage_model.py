"""Canonical NIH Stage Model taxonomy — the single source of truth.

Every stage label, description and keyword in the application is defined here
and imported from here. It previously lived in five hand-maintained copies
(app/prompts/stage.md, app/prompts/responder.md, the since-deleted
DBTool.STAGE_DEFINITIONS, StageAgent's keyword_map, ResponderAgent's rule-path
answer) and four of the five had drifted onto a taxonomy that does not exist —
Stage III as effectiveness, Stage IV as implementation, Stage V as
sustainability.

The definitions below follow the corpus (`data/documents/def.docx`) and
Onken et al., which agree: Stage III is *efficacy in community settings* and
effectiveness does not arrive until Stage IV.

Note on the two Onken PDFs in the corpus (1997/1998): they describe the
earlier NIDA three-stage behavioral therapy development model, where Stage III
meant transportability. That is a different framework, not a variant of this
one, and retrieval can surface it. This module always describes the modern
six-stage model.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# The six top-level stages, in order. Roman numerals: this is what NIH
# publications, app/prompts/stage.md and StageSlots.stage all use. `def.docx`
# writes them as Arabic ("Stage 1"); `normalize_stage` accepts either.
# Use this for definitions and overviews — "the model has six stages".
STAGES: tuple[str, ...] = ("0", "I", "II", "III", "IV", "V")

# Stage I has two ordered sub-stages: IA (generation and refinement) comes
# first, IB (feasibility and pilot testing) second.
SUBSTAGES: tuple[str, ...] = ("IA", "IB")

# Every value the classifier may put in StageSlots.stage, in progression
# order. Stage I resolves to its sub-stages here: when the evidence shows
# which half of Stage I a project is in, that is the more useful answer, and
# bare "I" remains valid for when it does not.
CLASSIFICATION_STAGES: tuple[str, ...] = ("0", "IA", "IB", "II", "III", "IV", "V")

# Sub-stage -> the top-level stage it belongs to.
PARENT_STAGE: Dict[str, str] = {"IA": "I", "IB": "I"}


class Stage:
    """One stage of the model."""

    def __init__(
        self,
        key: str,
        title: str,
        summary: str,
        description: str,
        key_question: str,
        keywords: List[str],
    ):
        self.key = key
        self.title = title            # "Stage III: Efficacy in Community Settings"
        self.summary = summary        # one line, for compact listings
        self.description = description
        self.key_question = key_question
        self.keywords = keywords      # rule-based classification signals

    @property
    def label(self) -> str:
        return f"Stage {self.key}"


STAGE_MODEL: Dict[str, Stage] = {
    "0": Stage(
        key="0",
        title="Stage 0: Basic Science",
        summary="Basic science on the mechanisms and determinants underlying the target behavior; no intervention yet.",
        description=(
            "Stage 0 focuses on fundamental mechanisms underlying health-related behaviors "
            "and psychological processes. Research at this stage investigates causal pathways, "
            "risk and protective factors, and biological, cognitive, social, or environmental "
            "determinants of behavior. Studies are typically mechanistic and may include "
            "laboratory experiments, longitudinal observational research, or neurobiological "
            "investigations. No intervention is yet implemented; the objective is to generate "
            "theoretically grounded knowledge that informs subsequent intervention design."
        ),
        key_question="What mechanisms explain the target problem?",
        keywords=[
            "basic research", "basic science", "mechanism", "mechanistic", "hypothesis",
            "causal pathway", "determinant", "risk factor", "protective factor",
            "observational", "predictor", "preliminary",
        ],
    ),
    "I": Stage(
        key="I",
        title="Stage I: Intervention Generation and Refinement",
        summary="Creating, adapting and refining the intervention, culminating in feasibility and pilot testing.",
        description=(
            "Stage I involves the development, adaptation, and preliminary testing of "
            "interventions informed by Stage 0 findings. Researchers design intervention "
            "protocols, manuals, and delivery systems, often conducting pilot studies to "
            "assess feasibility, acceptability, and preliminary signals of efficacy. "
            "Iterative refinement is common, and component testing may be used to identify "
            "active ingredients. Stage IA covers generation and refinement; Stage IB covers "
            "feasibility and pilot testing."
        ),
        key_question="What is the intervention, and how should it be structured?",
        keywords=[
            "feasibility", "pilot", "manual", "manualization", "manualized",
            "acceptability", "usability", "refinement", "adaptation", "adapt",
            "intervention development", "component testing", "small sample",
            "proof of concept",
        ],
    ),
    "IA": Stage(
        key="IA",
        title="Stage IA: Intervention Generation and Refinement",
        summary="The first half of Stage I: creating, adapting and manualizing the intervention, before any pilot testing.",
        description=(
            "Stage IA is the first of Stage I's two sub-stages. It covers generating a new "
            "intervention, or modifying, adapting or refining an existing one, informed by "
            "Stage 0 findings. The work is design work: intervention protocols, treatment "
            "manuals, delivery systems and training materials, along with component testing "
            "to identify active ingredients. Pilot testing has not begun."
        ),
        key_question="What is the intervention, and how should it be structured?",
        keywords=[
            "refinement", "refine", "adaptation", "adapt", "manual", "manualization",
            "manualized", "intervention development", "protocol development",
            "component testing", "treatment development", "training materials",
            "design the intervention",
        ],
    ),
    "IB": Stage(
        key="IB",
        title="Stage IB: Feasibility and Pilot Testing",
        summary="The second half of Stage I: pilot testing the developed intervention for feasibility, acceptability and early signals.",
        description=(
            "Stage IB is the second of Stage I's two sub-stages and follows Stage IA. The "
            "intervention already exists in manualized form; Stage IB pilots it to assess "
            "feasibility, acceptability, usability and preliminary signals of efficacy, "
            "usually in a small sample. Findings often send the work back to Stage IA for "
            "another round of refinement."
        ),
        key_question="Is the intervention feasible and acceptable as delivered?",
        keywords=[
            "feasibility", "feasible", "pilot", "pilot study", "pilot testing",
            "acceptability", "acceptable", "usability", "small sample",
            "proof of concept", "preliminary signal", "open trial",
        ],
    ),
    "II": Stage(
        key="II",
        title="Stage II: Efficacy in Research Settings",
        summary="Efficacy testing under highly controlled conditions with research-based providers.",
        description=(
            "Stage II evaluates whether the intervention produces desired outcomes under "
            "highly controlled conditions. Randomized controlled trials are typically "
            "conducted in research settings with trained, research-based providers and high "
            "intervention fidelity. The emphasis is on internal validity and causal inference, "
            "including tests of the mechanisms through which the intervention is hypothesized "
            "to work."
        ),
        key_question="Can the intervention work under ideal conditions?",
        keywords=[
            "efficacy", "randomized", "rct", "randomised", "controlled trial",
            "internal validity", "research setting", "laboratory", "fidelity",
            "research staff", "trained providers", "causal inference",
        ],
    ),
    "III": Stage(
        key="III",
        title="Stage III: Efficacy in Community Settings",
        summary="Efficacy testing extended to community settings with community providers, while keeping methodological control.",
        description=(
            "Stage III extends efficacy testing to community or real-world service settings "
            "while retaining a strong emphasis on methodological rigor. Interventions are "
            "delivered by community practitioners rather than research staff, and variability "
            "in implementation is expected. This stage examines whether previously established "
            "efficacy generalizes beyond laboratory conditions; it is sometimes described as a "
            "hybrid efficacy-effectiveness stage."
        ),
        key_question="Does the intervention retain its effects when delivered in real-world contexts?",
        keywords=[
            "community setting", "community practitioner", "community-based",
            "real world", "real-world", "community provider", "generalize",
            "generalise", "hybrid", "practice setting", "clinic setting",
        ],
    ),
    "IV": Stage(
        key="IV",
        title="Stage IV: Effectiveness Research",
        summary="Effectiveness at scale across diverse populations and settings, maximizing external validity.",
        description=(
            "Stage IV investigates the intervention's performance at scale across diverse "
            "populations and settings. Studies may involve multi-site trials, pragmatic trial "
            "designs, and comparisons with usual care. Greater heterogeneity in participants, "
            "providers, and environments enhances external validity."
        ),
        key_question="Does the intervention work at the population level?",
        keywords=[
            "effectiveness", "pragmatic", "multi-site", "multisite", "multicenter",
            "multi-center", "usual care", "external validity", "diverse population",
            "heterogeneity", "population level", "large scale", "large-scale",
        ],
    ),
    "V": Stage(
        key="V",
        title="Stage V: Implementation and Dissemination",
        summary="Implementation strategies, dissemination, adoption and sustained integration into systems of care.",
        description=(
            "Stage V focuses on sustainable implementation, system integration, and "
            "dissemination strategies. Research at this stage examines training models, policy "
            "integration, cost-effectiveness, adaptation processes, and equity considerations. "
            "The objective is long-term maintenance and institutionalization of evidence-based "
            "practices. Implementation strategies themselves can be developed and tested "
            "through the same staged logic (a Stage 0-IV progression nested within Stage V)."
        ),
        key_question="How can the intervention be sustainably implemented and disseminated within systems of care?",
        keywords=[
            "implementation", "dissemination", "adoption", "uptake", "scale-up",
            "scale up", "sustainability", "sustainable", "maintenance",
            "institutionalization", "policy", "cost-effectiveness", "training model",
            "de-implementation", "system integration",
        ],
    ),
}

# Progression order. Used for "what comes next" guidance. Stage 0 leads into
# Stage IA, IA into IB, and IB out of Stage I into Stage II; bare "I" is
# treated as the whole of Stage I and leads to Stage II.
NEXT_STAGE: Dict[str, Optional[str]] = {
    "0": "IA", "IA": "IB", "IB": "II", "I": "II",
    "II": "III", "III": "IV", "IV": "V", "V": None,
}

# Which StageSlots booleans are *established* when a project is at a given
# stage — i.e. the work of every earlier stage is behind it.
#
# The flag for a stage's own work is deliberately NOT set: a project at
# Stage IV is *conducting* effectiveness research, it has not finished it.
# The previous ladder set efficacy_tested at Stage III and effectiveness_tested
# at Stage IV, which marked in-progress work as complete — an artefact of the
# incorrect taxonomy, where III was believed to be effectiveness.
COMPLETED_FEATURES_AT_STAGE: Dict[str, Dict[str, bool]] = {
    "0": {},
    # Stage IA is where the intervention gets defined and manualized, so
    # neither flag is set yet. Reaching IB means IA produced both.
    "IA": {},
    "IB": {"intervention_defined": True, "manualized": True},
    "I": {},
    "II": {"intervention_defined": True, "manualized": True},
    "III": {"intervention_defined": True, "manualized": True, "mechanism_tested": True,
            "efficacy_tested": True},
    "IV": {"intervention_defined": True, "manualized": True, "mechanism_tested": True,
           "efficacy_tested": True},
    "V": {"intervention_defined": True, "manualized": True, "mechanism_tested": True,
          "efficacy_tested": True, "effectiveness_tested": True},
}

# Confidence bands, named once so the prompt, the classifier and the responder
# gate on the same numbers. These used to be three different literals (0.5 in
# stage.md, 0.58 in StageAgent, 0.75 in StageAgent and the orchestrator),
# leaving a 0.50-0.58 band where the agent returned a stage that was then
# never written to the slots — the responder saw "stage unknown" next to
# reasoning that named a stage.
#
# COMMIT: below this the stage is not trustworthy enough to record or report.
# CLARIFY: below this the answer is given, but with missing info and a
# follow-up question attached.
STAGE_COMMIT_CONFIDENCE = 0.58
STAGE_CLARIFY_CONFIDENCE = 0.75


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

# Arabic aliases: `def.docx` and many users write "Stage 2" for "Stage II",
# and "Stage 1a" for "Stage IA".
_ARABIC_TO_ROMAN = {
    "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V",
    "1A": "IA", "1B": "IB",
}

# Alternation is ordered longest-first so "IV" is not consumed as "I" and
# "IB" is not consumed as "I".
_STAGE_TOKEN_RE = re.compile(r"\bstages?\s*(0|IV|III|II|IB|IA|I|V|[1-5][AB]?)\b", re.IGNORECASE)

_VALID_KEYS = frozenset(STAGES) | frozenset(SUBSTAGES)


def normalize_stage(value: object) -> Optional[str]:
    """Coerce a model- or user-supplied stage into a canonical key.

    Accepts "II", "2", "Stage 2", "stage ii", 2, "IB", "1a". Sub-stages are
    preserved as "IA"/"IB" rather than collapsed into "I" — use
    `parent_stage` when you need the top-level stage. Returns None when the
    value names no stage in the model.
    """
    if value is None:
        return None

    text = str(value).strip().upper()
    if not text:
        return None

    # Strip a leading "STAGE" word if present.
    text = re.sub(r"^STAGES?\s*", "", text)
    # Drop anything after the identifier ("II (efficacy)" -> "II").
    match = re.match(r"^(0|IV|III|II|IB|IA|I|V|[1-5][AB]?)\b", text)
    if not match:
        return None
    token = match.group(1)

    if token in _ARABIC_TO_ROMAN:
        return _ARABIC_TO_ROMAN[token]
    if token in _VALID_KEYS:
        return token
    return None


def parent_stage(key: Optional[str]) -> Optional[str]:
    """The top-level stage a key belongs to: "IB" -> "I", "III" -> "III"."""
    if not key:
        return None
    return PARENT_STAGE.get(key, key)


def find_stage_in_text(text: str) -> Optional[str]:
    """Return the canonical stage named in free text, if any."""
    match = _STAGE_TOKEN_RE.search(text or "")
    return normalize_stage(match.group(1)) if match else None


# --------------------------------------------------------------------------
# Definition-query detection
# --------------------------------------------------------------------------

# "Is the user asking what the model *is*?" was implemented four times with
# four drifting keyword lists, so "explain the NIH stage model" counted as a
# definition query in three of them and not the fourth. One implementation,
# using whole-word matching: plain substring tests matched "hi" inside
# "something" and similar.
_DEFINITION_VERBS = (
    "what is", "what's", "what are", "define", "explain", "introduce",
    "how many stages", "number of stages", "list stages",
)
_DEFINITION_SUBJECTS = ("nih stage model", "nih stage", "stage model")

# A composition request pairs a producing verb with a named deliverable
# ("write an essay", "draft a summary"). The verb alone is not enough: "how
# do I write my grant" is a question about writing, not a request to write.
_COMPOSE_VERBS = (
    "write", "compose", "draft", "rewrite", "paraphrase",
    "summarize", "summarise", "put together", "turn this into",
)
_COMPOSE_DELIVERABLES = (
    "essay", "summary", "report", "paragraph", "paragraphs", "abstract",
    "overview", "write-up", "writeup", "narrative", "outline",
)
_COMPOSE_PHRASES = ("write about", "write on it", "write up", "write something")


def _has_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def is_definition_query(user_message: str) -> bool:
    """True when the user is asking what the NIH Stage Model is, rather than
    asking to be classified into it."""
    lowered = (user_message or "").lower()
    return any(_has_phrase(lowered, v) for v in _DEFINITION_VERBS) and any(
        _has_phrase(lowered, s) for s in _DEFINITION_SUBJECTS
    )


def is_compose_query(user_message: str) -> bool:
    """True when the user is asking the assistant to produce a piece of
    writing (essay, summary, report) rather than asking a question. Such a
    turn should be answered from what is already known — including the
    committed stage in session state — not re-routed into stage
    classification, whose clarifying questions would displace the deliverable."""
    lowered = (user_message or "").lower()
    if any(_has_phrase(lowered, p) for p in _COMPOSE_PHRASES):
        return True
    return any(_has_phrase(lowered, v) for v in _COMPOSE_VERBS) and any(
        _has_phrase(lowered, d) for d in _COMPOSE_DELIVERABLES
    )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def stage_summary_lines() -> List[str]:
    """One line per top-level stage, with Stage I's two sub-stages listed
    under it in order — for prompts and compact answers."""
    lines: List[str] = []
    for key in STAGES:
        lines.append(f"{STAGE_MODEL[key].title} — {STAGE_MODEL[key].summary}")
        if key == "I":
            for sub in SUBSTAGES:
                lines.append(f"    {STAGE_MODEL[sub].title} — {STAGE_MODEL[sub].summary}")
    return lines


def stage_overview() -> str:
    """Full prose overview of all six stages, including Stage I's sub-stages."""
    parts = [
        f"The NIH Stage Model has {len(STAGES)} stages "
        f"({', '.join('Stage ' + k for k in STAGES)}). "
        f"Stage I divides into two ordered sub-stages, "
        f"{' then '.join('Stage ' + s for s in SUBSTAGES)}.",
        "",
    ]
    for key in STAGES:
        stage = STAGE_MODEL[key]
        parts.append(f"{stage.title}: {stage.description}")
        parts.append(f"Key question: {stage.key_question}")
        parts.append("")
        if key == "I":
            for sub in SUBSTAGES:
                substage = STAGE_MODEL[sub]
                parts.append(f"{substage.title}: {substage.description}")
                parts.append(f"Key question: {substage.key_question}")
                parts.append("")
    return "\n".join(parts).strip()
