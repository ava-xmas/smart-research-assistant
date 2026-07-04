"""
Thin wrapper around Groq's OpenAI-compatible chat completions API
(https://api.groq.com/openai/v1).

Centralizing all model calls here means:
  1. Every call can be traced (tokens, latency, cost) in one place (Ch. 14 - cost awareness).
  2. Agents / steps don't need to know API details - they just call `.chat()` or
     `.structured()`.

Note: Groq's endpoint speaks the OpenAI chat-completions protocol, so this
client uses the `openai` Python package pointed at Groq's base_url. This
also means it works unmodified against plain OpenAI, or any other
OpenAI-compatible endpoint - just change `base_url` and `model`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from openai import OpenAI
from pydantic import BaseModel

# Rough, illustrative per-million-token prices (USD) for Groq-hosted models.
# Update as needed - the point of the project is to *track* cost, not to be
# a pricing oracle. See https://groq.com/pricing for current rates.
PRICING_PER_MILLION_TOKENS = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.75},
    "openai/gpt-oss-20b": {"input": 0.10, "output": 0.50},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},  # deprecated by Groq June 2026
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},  # deprecated by Groq June 2026
}
DEFAULT_PRICING = {"input": 0.15, "output": 0.75}

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


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
class OpenAIModelClient:
    """
    Generic client for OpenAI-compatible chat completion APIs. Defaults to
    Groq. Exposes plain-chat and structured-output helpers used by Agent,
    AIOrchestrator, and the evaluation harness.
    """

    model: str = "openai/gpt-oss-120b"
    max_tokens: int = 1500
    temperature: float = 0.4
    api_key: Optional[str] = None
    base_url: str = GROQ_BASE_URL
    call_log: List[ModelCallRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or export GROQ_API_KEY in your shell. Get a free key at "
                "https://console.groq.com/keys"
            )
        self._client = OpenAI(api_key=key, base_url=self.base_url)

    # ------------------------------------------------------------------ #
    # Cost / bookkeeping
    # ------------------------------------------------------------------ #
    def _record(self, usage, latency_s: float, purpose: str) -> ModelCallRecord:
        prices = PRICING_PER_MILLION_TOKENS.get(self.model, DEFAULT_PRICING)
        # OpenAI-compatible usage objects (Groq included) use prompt_tokens /
        # completion_tokens, not Anthropic's input_tokens / output_tokens.
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost = (
            input_tokens / 1_000_000 * prices["input"]
            + output_tokens / 1_000_000 * prices["output"]
        )
        record = ModelCallRecord(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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
        """
        Single call to the chat completions endpoint. `messages` should
        already be in OpenAI format (role/content, plus optional
        tool_calls / tool role messages - see agent.py). Returns the raw
        OpenAI-SDK ChatCompletion response.
        """
        openai_messages: List[Dict[str, Any]] = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(messages)

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
            messages=openai_messages,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        start = time.time()
        response = self._client.chat.completions.create(**kwargs)
        self._record(response.usage, time.time() - start, purpose)
        return response

    # ------------------------------------------------------------------ #
    # Structured output via a forced function call (Ch. 4 - structured output)
    # ------------------------------------------------------------------ #
    def structured(
        self,
        prompt: str,
        schema_model: Type[BaseModel],
        system: str = "",
        purpose: str = "structured_output",
    ) -> BaseModel:
        """
        Force the model to respond with JSON matching `schema_model` by giving
        it a single function tool whose parameters are the Pydantic model's
        JSON schema, and forcing tool_choice to that function. This is far
        more reliable than asking the model to "please respond in JSON".
        """
        tool_name = f"emit_{schema_model.__name__.lower()}"
        tool = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": f"Emit a {schema_model.__name__} object.",
                "parameters": schema_model.model_json_schema(),
            },
        }

        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )
        self._record(response.usage, time.time() - start, purpose)

        choice = response.choices[0]
        if choice.message.tool_calls:
            call = choice.message.tool_calls[0]
            args = json.loads(call.function.arguments)
            return schema_model.model_validate(args)

        raise ValueError("Model did not return the expected structured tool call.")
