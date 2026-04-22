"""
Guardrails: input validation, output sanitization, and topic enforcement.
"""
import re
from typing import Optional

from app.core.types import Message  # noqa: F401 (kept for potential future use)

# Keywords that signal behavioral science / intervention relevance.
# A message matching any of these is treated as on-topic.
_BEHAVIORAL_SCIENCE_KEYWORDS = [
    # NIH stage model
    "nih stage", "stage model", "stage 0", "stage i", "stage ii",
    "stage iii", "stage iv", "stage v", "stage 1", "stage 2",
    "stage 3", "stage 4", "stage 5",
    # Intervention design
    "intervention", "behavioral intervention", "behaviour", "behavior",
    "treatment arm", "control group", "comparator", "active comparator",
    "pilot", "feasibility", "rct", "randomized", "randomised",
    "clinical trial", "efficacy", "effectiveness", "dissemination",
    "implementation", "sustainability", "scale-up", "scale up",
    "fidelity", "adherence", "dose", "dosage",
    # Mechanisms & theory
    "mechanism", "mediator", "moderator", "pathway", "causal",
    "theoretical model", "theory of change", "behavior change",
    "behaviour change", "self-efficacy", "motivation", "cognitive",
    "psychosocial", "social cognitive", "health belief",
    "transtheoretical", "theory of planned behavior",
    # Measurement & methodology
    "measure", "scale", "instrument", "psychometric", "questionnaire",
    "survey", "construct", "outcome measure", "validated",
    "study design", "design matrix", "sample size", "power",
    "recruitment", "enrollment", "retention", "dropout",
    # Grant / funding
    "grant", "specific aims", "aim 1", "aim 2", "r01", "r21", "r34",
    "reviewer", "resubmission", "revision plan", "nih", "nimh",
    "nida", "nhlbi", "nci", "ahrq",
    # Population / settings
    "patient", "participant", "subject", "community", "clinical",
    "health promotion", "prevention", "mental health", "substance",
    "smoking", "obesity", "physical activity", "diet", "nutrition",
    "stress", "anxiety", "depression",
]

# Keywords that almost certainly indicate a completely off-topic message.
_OFF_TOPIC_KEYWORDS = [
    "stock market", "crypto", "bitcoin", "recipe", "weather forecast",
    "sports score", "movie review", "song lyrics", "video game",
    "tax advice", "legal advice", "real estate",
]

_REJECTION_MESSAGE = (
    "Sorry, this isn't behavioral science related, therefore I cannot help you."
)


class Guardrails:
    """Input validation, output sanitization, and topic enforcement."""

    MAX_MESSAGE_LENGTH = 5000
    MAX_RESPONSE_LENGTH = 2000
    FORBIDDEN_PATTERNS = [
        r"<script",
        r"javascript:",
        r"onerror\s*=",
    ]

    # ------------------------------------------------------------------ #
    # Input validation                                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def validate_message(cls, message: str) -> tuple[bool, Optional[str]]:
        """Check length and XSS patterns."""
        if len(message) > cls.MAX_MESSAGE_LENGTH:
            return False, f"Message exceeds the maximum length of {cls.MAX_MESSAGE_LENGTH} characters."

        for pattern in cls.FORBIDDEN_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return False, "Message contains disallowed content."

        return True, None

    # ------------------------------------------------------------------ #
    # Topic enforcement                                                    #
    # ------------------------------------------------------------------ #

    @classmethod
    def is_behavioral_science_related(cls, message: str) -> bool:
        """
        Return True if the message appears to be about behavioral science
        or intervention research; False otherwise.

        Uses keyword heuristics only — the intent agent runs immediately after
        and provides full LLM-based topic classification, so a second LLM call
        here would be redundant.
        """
        return cls._keyword_topic_check(message)

    # Greetings and admin-style tokens that should pass through even with no
    # behavioral keyword (the intent agent handles these gracefully).
    _GREETING_TOKENS = {
        "hi", "hello", "hey", "thanks", "thank", "bye", "goodbye",
        "help", "start", "reset", "clear",
    }

    @classmethod
    def _keyword_topic_check(cls, message: str) -> bool:
        """Rule-based fallback: match against known behavioral science keywords."""
        lower = message.lower()

        # Admin slash-commands are always allowed.
        if lower.strip().startswith("/"):
            return True

        # Pure greetings (short + only greeting tokens) are allowed.
        words = lower.split()
        if len(words) <= 5 and all(w.strip(".,!?") in cls._GREETING_TOKENS for w in words):
            return True

        # Require at least one behavioral science keyword to pass.
        if any(kw in lower for kw in _BEHAVIORAL_SCIENCE_KEYWORDS):
            return True

        return False

    @classmethod
    def rejection_message(cls) -> str:
        return _REJECTION_MESSAGE

    # ------------------------------------------------------------------ #
    # Output sanitization                                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def sanitize_response(cls, response: str) -> str:
        """Truncate overly long responses."""
        if len(response) > cls.MAX_RESPONSE_LENGTH:
            response = response[:cls.MAX_RESPONSE_LENGTH] + "..."
        return response

    @classmethod
    def check_content_policy(cls, content: str) -> bool:
        """Placeholder for future content policy checks."""
        return True
