"""
Step 1 - parse_query_step (Ch. 4: structured output).

Turns a free-text user query into a structured ResearchPlan using a forced
tool-call (see OpenAIModelClient.structured). Structured output here
means every downstream step can rely on a stable schema instead of parsing
free text.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    main_theme: str = Field(description="One-sentence summary of what the user wants researched.")
    sub_questions: List[str] = Field(description="3-6 specific sub-questions to investigate.")
    search_domains: List[str] = Field(
        description="Which kinds of sources to consult, e.g. 'academic papers', 'industry blogs', 'news'."
    )
    time_range: str = Field(description="Relevant time window, e.g. 'last 12 months', '2024-2026'.")


def parse_query_step(context: Dict[str, Any], tracer: Optional[Any] = None) -> Dict[str, Any]:
    model_client = context["model_client"]
    query = context["query"]

    plan = model_client.structured(
        prompt=(
            f"Break the following research request into a structured plan.\n\n"
            f"REQUEST: {query}"
        ),
        schema_model=ResearchPlan,
        system=(
            "You are a research planner. Produce specific, non-overlapping "
            "sub-questions that together fully cover the request."
        ),
        purpose="parse_query",
    )
    context["plan"] = plan
    return context
