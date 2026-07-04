"""
Thin wrapper around the Anthropic Messages API.

Centralizing all model calls here means:
  1. Every call can be traced (tokens, latency, cost) in one place (Ch. 14 - cost awareness).
  2. Agents / steps don't need to know API details - they just call `.chat()` or
     `.structured()`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import anthropic
from pydantic import BaseModel

# Rough, illustrative per-million-token prices (USD). Update as needed -
# the point of the project is to *track* cost, not to be a pricing oracle.
PRICING_PER_MILLION_TOKENS = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.8, "output": 4.0},
    "claude-opus-4-8": {"input": 15.0, "output": 75.0},
}
DEFAULT_PRICING = {"input": 3.0, "output": 15.0}


@dataclass
class ModelCallRecord:
    """A single record of one API call, used for tracing/cost tracking."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    estimated_cost_usd: float
    purpose: str = ""


@dataclass
class AnthropicModelClient:
    """Wraps anthropic.Anthropic and exposes plain-chat and structured-output helpers."""

    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1500
    temperature: float = 0.4
    api_key: Optional[str] = None
    call_log: List[ModelCallRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or export ANTHROPIC_API_KEY in your shell."
            )
        self._client = anthropic.Anthropic(api_key=key)

    # ------------------------------------------------------------------ #
    # Cost / bookkeeping
    # ------------------------------------------------------------------ #
    def _record(self, usage, latency_s: float, purpose: str) -> ModelCallRecord:
        prices = PRICING_PER_MILLION_TOKENS.get(self.model, DEFAULT_PRICING)
        cost = (
            usage.input_tokens / 1_000_000 * prices["input"]
            + usage.output_tokens / 1_000_000 * prices["output"]
        )
        record = ModelCallRecord(
            model=self.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_s=latency_s,
            estimated_cost_usd=round(cost, 6),
            purpose=purpose,
        )
        self.call_log.append(record)
        return record

    def total_cost(self) -> float:
        return round(sum(r.estimated_cost_usd for r in self.call_log), 6)

    # ------------------------------------------------------------------ #
    # Plain chat (with optional tool use)
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: List[Dict[str, Any]],
        system: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        purpose: str = "chat",
        max_tokens: Optional[int] = None,
    ):
        """Single raw call to the Messages API. Returns the raw Anthropic response."""
        start = time.time()
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = self._client.messages.create(**kwargs)
        self._record(response.usage, time.time() - start, purpose)
        return response

    # ------------------------------------------------------------------ #
    # Structured output via a forced tool call (Ch. 4 - structured output)
    # ------------------------------------------------------------------ #
    def structured(
        self,
        prompt: str,
        schema_model: Type[BaseModel],
        system: str = "",
        purpose: str = "structured_output",
    ) -> BaseModel:
        """
        Force the model to respond with JSON matching `schema_model` by giving it
        a single tool whose input_schema is the Pydantic model's JSON schema, and
        forcing tool_choice to that tool. This is far more reliable than asking
        the model to "please respond in JSON".
        """
        tool_name = f"emit_{schema_model.__name__.lower()}"
        tool = {
            "name": tool_name,
            "description": f"Emit a {schema_model.__name__} object.",
            "input_schema": schema_model.model_json_schema(),
        }
        start = time.time()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": prompt}],
        )
        self._record(response.usage, time.time() - start, purpose)

        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return schema_model.model_validate(block.input)

        raise ValueError("Model did not return the expected structured tool call.")
