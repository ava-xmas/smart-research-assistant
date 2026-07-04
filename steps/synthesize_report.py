"""
Step 4 - synthesize_report_step.

Takes the raw multi-agent conversation transcript and makes one final,
cheap-ish LLM call to format it into a clean Markdown report: title,
executive summary, sections per sub-question, and a sources list.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def synthesize_report_step(context: Dict[str, Any], tracer: Optional[Any] = None) -> Dict[str, Any]:
    model_client = context["model_client"]
    plan = context["plan"]
    history = context["conversation_history"]

    transcript = "\n\n".join(f"[{m['role']}] {m['content']}" for m in history)

    prompt = f"""
Using the multi-agent conversation transcript below, produce a final,
polished Markdown research report on: "{plan.main_theme}"

Structure:
# <Title>
## Executive Summary  (3-5 sentences)
## Findings by Sub-Question   (one ## subsection per sub-question, with inline
   source URLs as citations)
## Key Challenges / Risks
## Recommendations
## Sources   (deduplicated bullet list of all URLs referenced above)

Only use facts that actually appear in the transcript below. Do not invent
sources or statistics.

TRANSCRIPT:
{transcript}
"""
    response = model_client.chat(
        messages=[{"role": "user", "content": prompt}],
        system="You are a precise technical editor producing a final client-ready report.",
        purpose="synthesize_report",
        max_tokens=2000,
    )
    report_text = "\n".join(b.text for b in response.content if b.type == "text")
    context["report"] = report_text
    context["total_cost_usd"] = model_client.total_cost()
    return context
