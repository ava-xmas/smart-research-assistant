"""
Agent: the core execution loop (Ch. 4).

An Agent has instructions (a system prompt), a model client, and an optional
set of tools. `run()` implements the standard agent loop:

    1. send messages + tool definitions to the model
    2. if the model asks to use one or more tools -> execute them locally,
       append the tool results, and go back to step 1
    3. if the model returns plain text -> that's the agent's final answer

This uses the OpenAI-compatible tool-calling protocol (used by Groq, OpenAI,
and most other providers behind an OpenAI-shaped API):
  - the assistant's tool-requesting turn is a message with a `tool_calls`
    list (each with an `id`, and `function.name` / `function.arguments`)
  - each tool's result is then sent back as its OWN message with
    `role="tool"` and a matching `tool_call_id` (NOT bundled into a single
    user message the way Anthropic's API does it).

This is intentionally minimal (no planning/reflection loop) so the
*orchestration* pattern in orchestrator.py is where the multi-agent behavior
actually lives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .model_client import OpenAIModelClient
from .tracing import Tracer


@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    func: Callable[..., str]

    def to_api_schema(self) -> Dict[str, Any]:
        """OpenAI/Groq-style function-tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class Agent:
    name: str
    instructions: str
    model_client: OpenAIModelClient
    tools: List[Tool] = field(default_factory=list)
    max_tool_iterations: int = 5

    def _tool_by_name(self, name: str) -> Optional[Tool]:
        return next((t for t in self.tools if t.name == name), None)

    def run(
        self,
        user_message: str,
        conversation_context: Optional[List[Dict[str, Any]]] = None,
        tracer: Optional[Tracer] = None,
    ) -> str:
        """
        Run the agent on a single user_message (optionally with prior
        conversation_context prepended, so multiple agents can share history
        in the orchestrator). Returns the agent's final text reply.
        """
        messages: List[Dict[str, Any]] = list(conversation_context or [])
        messages.append({"role": "user", "content": user_message})

        tool_schemas = [t.to_api_schema() for t in self.tools] or None

        for _ in range(self.max_tool_iterations):
            span_ctx = tracer.span(f"agent:{self.name}") if tracer else None
            span = span_ctx.__enter__() if span_ctx else None

            response = self.model_client.chat(
                messages=messages,
                system=self.instructions,
                tools=tool_schemas,
                purpose=f"agent:{self.name}",
            )

            if span is not None:
                usage = self.model_client.call_log[-1]
                tracer.add_attributes(
                    span,
                    model=usage.model,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    estimated_cost_usd=usage.estimated_cost_usd,
                )
                span_ctx.__exit__(None, None, None)

            choice = response.choices[0]
            message = choice.message

            if message.tool_calls:
                # 1. Append the assistant's tool-calling turn exactly as
                #    the API needs to see it echoed back.
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ],
                    }
                )

                # 2. Execute each requested tool and append ONE "tool" role
                #    message per call, matched by tool_call_id.
                for tc in message.tool_calls:
                    tool = self._tool_by_name(tc.function.name)
                    if tool is None:
                        result_text = f"Error: unknown tool '{tc.function.name}'"
                    else:
                        try:
                            tool_args = json.loads(tc.function.arguments or "{}")
                            result_text = tool.func(**tool_args)
                        except Exception as exc:  # tool failures shouldn't crash the run
                            result_text = f"Error running tool '{tc.function.name}': {exc}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(result_text)[:8000],
                        }
                    )
                continue

            # Plain text -> final answer
            return (message.content or "").strip()

        return "[Agent stopped: exceeded max tool iterations without a final answer]"
