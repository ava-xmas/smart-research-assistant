"""
Composable termination conditions for the AIOrchestrator (Ch. 7).

Each condition implements `should_terminate(history) -> bool`. Conditions
combine with `|` (OR) so the orchestrator stops as soon as any one of them
fires - exactly like the book's `MaxMessageTermination(20) | TextMentionTermination(...)`.
"""

from __future__ import annotations

from typing import Callable, Dict, List


class TerminationCondition:
    def should_terminate(self, history: List[Dict[str, str]]) -> bool:
        raise NotImplementedError

    def __or__(self, other: "TerminationCondition") -> "OrTermination":
        return OrTermination(self, other)


class OrTermination(TerminationCondition):
    def __init__(self, *conditions: TerminationCondition):
        self.conditions = conditions

    def should_terminate(self, history: List[Dict[str, str]]) -> bool:
        return any(c.should_terminate(history) for c in self.conditions)

    def __or__(self, other: "TerminationCondition") -> "OrTermination":
        return OrTermination(*self.conditions, other)


class MaxMessageTermination(TerminationCondition):
    """Stop once the conversation reaches N messages (a hard cost/runaway guard)."""

    def __init__(self, max_messages: int):
        self.max_messages = max_messages

    def should_terminate(self, history: List[Dict[str, str]]) -> bool:
        return len(history) >= self.max_messages


class TextMentionTermination(TerminationCondition):
    """Stop as soon as any message contains a given phrase (e.g. 'REPORT_COMPLETE')."""

    def __init__(self, phrase: str):
        self.phrase = phrase.lower()

    def should_terminate(self, history: List[Dict[str, str]]) -> bool:
        return any(self.phrase in (m.get("content") or "").lower() for m in history)
