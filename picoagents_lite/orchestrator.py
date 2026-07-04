"""
AIOrchestrator: the autonomous multi-agent pattern (Ch. 7).

Unlike the deterministic Workflow (fixed step order), the orchestrator asks
an LLM "router" call, after every turn, which agent should speak next (or
whether the team is done). This is what makes the pattern *autonomous*:
the control flow is decided at run time by a model, not hard-coded by us.

Design notes for the interview:
  - The orchestrator keeps ONE shared conversation history that every agent
    sees, so agents can build on each other's work.
  - A separate lightweight "router" LLM call (cheap model, tiny output)
    picks the next speaker - this keeps the expensive reasoning inside the
    specialist agents and the routing decision cheap (cost-awareness, Ch. 14).
  - Termination conditions are checked before every routing decision so we
    never pay for a routing call we don't need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .agent import Agent
from .model_client import OpenAIModelClient
from .termination import TerminationCondition
from .tracing import Tracer


class RouteDecision(BaseModel):
    next_agent: str = Field(description="Name of the agent who should speak next.")
    instruction: str = Field(
        description="A short, specific instruction telling that agent what to do next."
    )
    is_done: bool = Field(
        description="True if the team has produced a final, complete result and should stop."
    )


@dataclass
class AIOrchestrator:
    agents: List[Agent]
    termination: TerminationCondition
    model_client: OpenAIModelClient  # used for routing decisions
    max_rounds: int = 10

    history: List[Dict[str, str]] = field(default_factory=list, init=False)

    def _agent_by_name(self, name: str) -> Optional[Agent]:
        return next((a for a in self.agents if a.name.lower() == name.lower()), None)

    def _history_as_text(self) -> str:
        return "\n\n".join(f"[{m['role']}] {m['content']}" for m in self.history)

    def _route(self, task: str, tracer: Optional[Tracer]) -> RouteDecision:
        agent_descriptions = "\n".join(
            f"- {a.name}: {a.instructions.splitlines()[0]}" for a in self.agents
        )
        prompt = f"""
You are coordinating a team of specialist agents working on this task:

TASK: {task}

TEAM:
{agent_descriptions}

CONVERSATION SO FAR:
{self._history_as_text() or "(nothing yet)"}

Decide which agent should act next and what specifically they should do.
Typical flow: Researcher gathers facts -> Writer drafts the report -> Critic
reviews it -> Writer revises if the Critic requested changes -> once the
Critic explicitly approves, set is_done=true.
"""
        span_ctx = tracer.span("orchestrator:route") if tracer else None
        span = span_ctx.__enter__() if span_ctx else None

        decision = self.model_client.structured(
            prompt=prompt,
            schema_model=RouteDecision,
            system="You are a precise, terse multi-agent task router.",
            purpose="orchestrator_routing",
        )

        if span is not None:
            usage = self.model_client.call_log[-1]
            tracer.add_attributes(
                span,
                model=usage.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                estimated_cost_usd=usage.estimated_cost_usd,
                next_agent=decision.next_agent,
            )
            span_ctx.__exit__(None, None, None)

        return decision

    def run(self, task: str, tracer: Optional[Tracer] = None) -> List[Dict[str, str]]:
        """Runs the team until a termination condition fires. Returns full history."""
        self.history = [{"role": "user", "content": task}]

        for round_num in range(self.max_rounds):
            if self.termination.should_terminate(self.history):
                break

            decision = self._route(task, tracer)
            if decision.is_done:
                self.history.append(
                    {"role": "system", "content": "REPORT_COMPLETE (orchestrator ended team run)"}
                )
                break

            agent = self._agent_by_name(decision.next_agent)
            if agent is None:
                # Router hallucinated a name - fall back to round robin.
                agent = self.agents[round_num % len(self.agents)]

            reply = agent.run(
                user_message=decision.instruction,
                conversation_context=[
                    {"role": m["role"] if m["role"] in ("user", "assistant") else "user",
                     "content": m["content"]}
                    for m in self.history
                ],
                tracer=tracer,
            )
            self.history.append({"role": "assistant", "content": f"[{agent.name}] {reply}"})

        return self.history
