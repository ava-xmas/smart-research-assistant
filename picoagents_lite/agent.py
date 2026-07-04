"""
Agent: the core execution loop (Ch. 4).

An Agent has instructions (a system prompt), a model client, and an optional
set of tools. `run()` implements the standard agent loop:

    1. send messages + tool definitions to the model
    2. if the model asks to use a tool -> execute it locally, append the
       tool_result, and go back to step 1
    3. if the model returns plain text -> that's the agent's final answer

This is intentionally minimal (no planning/reflection loop) so the
*orchestration* pattern in orchestrator.py is where the multi-agent behavior
actually lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .model_client import AnthropicModelClient
from .tracing import Tracer


@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    func: Callable[..., str]

    def to_api_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class Agent:
    name: str
    instructions: str
    model_client: AnthropicModelClient
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

            if response.stop_reason == "tool_use":
                # Append the assistant's tool-use turn, execute each tool call,
                # append results, and loop back so the model can continue.
                assistant_content = [block.model_dump() for block in response.content]
                messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    tool = self._tool_by_name(block.name)
                    if tool is None:
                        result_text = f"Error: unknown tool '{block.name}'"
                    else:
                        try:
                            result_text = tool.func(**block.input)
                        except Exception as exc:  # tool failures shouldn't crash the run
                            result_text = f"Error running tool '{block.name}': {exc}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result_text)[:8000],
                        }
                    )
                messages.append({"role": "user", "content": tool_results})
                continue

            # Plain text (or end_turn) -> final answer
            text_parts = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_parts).strip()

        return "[Agent stopped: exceeded max tool iterations without a final answer]"
