"""
Memory manager: hybrid short-term + long-term (Redis-backed via SessionState persistence).
"""
from __future__ import annotations

from typing import List, Optional

from app.config import settings
from app.core.llm import llm_client
from app.core.types import Message, MessageRole, SessionState


class MemoryManager:
    """Hybrid memory manager with summary-first + long-term + short-term context."""

    def __init__(
        self,
        short_term_limit: int = 20,
        summary_threshold: int = 10,
        summary_refresh_every_turns: int = 6,
        long_term_window: int = 50,
        long_term_max_lines: int = 8,
        context_max_chars: int = 6000,
    ):
        self.short_term_limit = short_term_limit
        self.summary_threshold = summary_threshold
        self.summary_refresh_every_turns = max(1, summary_refresh_every_turns)
        self.long_term_window = max(0, long_term_window)
        self.long_term_max_lines = max(0, long_term_max_lines)
        self.context_max_chars = max(1000, context_max_chars)

    def get_short_term_memory(self, state: SessionState) -> List[Message]:
        return state.get_recent_messages(self.short_term_limit)

    def get_summary(self, state: SessionState) -> Optional[str]:
        return state.summary

    def should_summarize(self, state: SessionState) -> bool:
        msg_count = len(state.messages)
        if msg_count < self.summary_threshold:
            return False
        if not state.summary:
            return True
        last_count = int(state.slots.extracted_features.get("summary_last_msg_count", 0) or 0)
        return (msg_count - last_count) >= self.summary_refresh_every_turns

    def _create_summary_rule_fallback(self, state: SessionState) -> str:
        recent_messages = state.get_recent_messages(min(self.short_term_limit, 12))
        summary_parts = []
        for msg in recent_messages:
            if msg.role == MessageRole.USER:
                summary_parts.append(f"user: {msg.content[:140]}")
            elif msg.role == MessageRole.ASSISTANT:
                summary_parts.append(f"assistant: {msg.content[:140]}")
        return "\n".join(summary_parts)[:1800]

    def create_summary(self, state: SessionState) -> str:
        """Create/update conversation summary using LLM; fallback to rule summary."""
        recent_messages = state.get_recent_messages(min(len(state.messages), 30))
        transcript = "\n".join([f"{m.role.value}: {m.content[:500]}" for m in recent_messages])
        existing_summary = state.summary or "None"

        if llm_client.is_enabled():
            system_prompt = (
                "You are a conversation memory compressor for a NIH Stage Model assistant.\n"
                "Write a concise structured summary for future turns.\n"
                "Include sections:\n"
                "1) User goal\n"
                "2) Confirmed study facts\n"
                "3) Current stage hypothesis + confidence signal\n"
                "4) Missing information\n"
                "5) Best next question\n"
                "Do not invent facts. Keep under 1600 characters."
            )
            user_prompt = (
                f"Existing summary:\n{existing_summary}\n\n"
                f"Recent transcript:\n{transcript}\n\n"
                "Return plain text summary with the 5 sections."
            )
            text = llm_client.chat_text(system_prompt=system_prompt, user_prompt=user_prompt)
            if text and text.strip():
                return text.strip()[:1800]

        return self._create_summary_rule_fallback(state)

    def update_summary(self, state: SessionState, summary: str):
        state.summary = summary
        state.slots.extracted_features["summary_last_msg_count"] = len(state.messages)
        if summary.strip():
            state.summary_history.append(summary.strip())
            state.summary_history = state.summary_history[-12:]

    def _get_long_term_memory_lines(self, state: SessionState) -> List[str]:
        """
        Build long-term memory lines from older turns.
        Since SessionState is persisted in Redis, these lines are effectively Redis-backed memory.
        """
        if self.long_term_max_lines <= 0 or self.long_term_window <= 0:
            return []
        if len(state.messages) <= self.short_term_limit:
            return []

        older_msgs = state.messages[:-self.short_term_limit]
        older_msgs = older_msgs[-self.long_term_window :]

        # Down-sample to keep context compact.
        if len(older_msgs) <= self.long_term_max_lines:
            selected = older_msgs
        else:
            step = max(1, len(older_msgs) // self.long_term_max_lines)
            selected = older_msgs[::step][: self.long_term_max_lines]

        return [f"{m.role.value}: {m.content[:180]}" for m in selected]

    def get_context_for_agent(self, state: SessionState, include_summary: bool = True) -> str:
        """Compose agent context: summary first, then slots, then long-term, then recent turns."""
        context_parts: List[str] = []

        if include_summary and state.summary:
            context_parts.append(f"[Summary]\n{state.summary}")

        slots_info = []
        if state.slots.need_stage is not None:
            slots_info.append(f"Need stage flow: {state.slots.need_stage}")
        if state.slots.stage:
            slots_info.append(f"Current stage hypothesis: {state.slots.stage} (confidence: {state.slots.stage_confidence:.2f})")
        if state.slots.user_goal:
            slots_info.append(f"User goal: {state.slots.user_goal}")
        if slots_info:
            context_parts.append("[Slots]\n" + "\n".join(slots_info))

        long_term_lines = self._get_long_term_memory_lines(state)
        if long_term_lines:
            context_parts.append("[Long-term Memory from Redis session history]\n" + "\n".join(long_term_lines))

        recent = self.get_short_term_memory(state)
        if recent:
            messages_text = "\n".join([f"{msg.role.value}: {msg.content[:300]}" for msg in recent[-6:]])
            context_parts.append("[Recent Turns]\n" + messages_text)

        combined = "\n\n".join(context_parts)
        return combined[: self.context_max_chars]


# Global memory manager singleton
memory_manager = MemoryManager(
    short_term_limit=settings.SHORT_TERM_LIMIT,
    summary_threshold=settings.SUMMARY_THRESHOLD,
    summary_refresh_every_turns=settings.SUMMARY_REFRESH_EVERY_TURNS,
    long_term_window=settings.LONG_TERM_MEMORY_WINDOW,
    long_term_max_lines=settings.LONG_TERM_MEMORY_MAX_LINES,
    context_max_chars=settings.MEMORY_CONTEXT_MAX_CHARS,
)
