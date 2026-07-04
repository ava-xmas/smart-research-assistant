"""
Step 3 - multi_agent_research_step (Ch. 2 & 7: autonomous multi-agent pattern).

This step is itself a node inside the deterministic Workflow, but internally
it hands control to an AIOrchestrator, which lets an LLM decide - turn by
turn - which specialist agent (Researcher / Writer / Critic) should act
next, until the Critic approves the draft or a safety limit is hit.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from picoagents_lite import AIOrchestrator, MaxMessageTermination, TextMentionTermination
from agents import build_critic, build_researcher, build_writer


def multi_agent_research_step(context: Dict[str, Any], tracer: Optional[Any] = None) -> Dict[str, Any]:
    model_client = context["model_client"]
    plan = context["plan"]
    filtered_sources = context.get("filtered_sources", [])

    researcher = build_researcher(model_client)
    writer = build_writer(model_client)
    critic = build_critic(model_client)

    orchestrator = AIOrchestrator(
        agents=[researcher, writer, critic],
        termination=MaxMessageTermination(context.get("max_messages", 12))
        | TextMentionTermination("REPORT_COMPLETE"),
        model_client=model_client,
        max_rounds=context.get("max_rounds", 8),
    )

    task = (
        f"Main theme: {plan.main_theme}\n"
        f"Sub-questions to cover:\n"
        + "\n".join(f"  - {q}" for q in plan.sub_questions)
        + "\n\nPre-filtered candidate sources (use these as a starting point, "
        "the Researcher may search for more if gaps remain):\n"
        + "\n".join(filtered_sources[:10])
    )

    history = orchestrator.run(task, tracer=tracer)
    context["conversation_history"] = history
    context["orchestrator_cost_usd"] = model_client.total_cost()
    return context
