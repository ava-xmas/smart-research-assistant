"""
Workflow: the deterministic pipeline pattern (Ch. 6).

Use this for steps where you want reliability, predictable cost, and
easy debugging - as opposed to the AIOrchestrator, which is for the
parts of the problem that genuinely benefit from autonomous, adaptive
collaboration.

Each step is a plain Python callable: `step(context: dict, tracer) -> dict`.
It receives the shared context dict, mutates/returns it with new keys, and
the Workflow passes the updated context to the next step. This makes steps
easy to unit test in isolation (just call them with a fake context).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .tracing import Tracer

Step = Callable[[Dict[str, Any], Optional[Tracer]], Dict[str, Any]]


@dataclass
class WorkflowMetadata:
    name: str
    description: str = ""


@dataclass
class Workflow:
    metadata: WorkflowMetadata
    steps: List[Step] = field(default_factory=list)

    def chain(self, *steps: Step) -> "Workflow":
        self.steps.extend(steps)
        return self

    def run(self, initial_context: Dict[str, Any], tracer: Optional[Tracer] = None) -> Dict[str, Any]:
        context = dict(initial_context)
        for step in self.steps:
            step_name = getattr(step, "__name__", str(step))
            if tracer:
                with tracer.span(f"workflow_step:{step_name}"):
                    context = step(context, tracer)
            else:
                context = step(context, tracer)
        return context
