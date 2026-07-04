"""
Step 2 - filter_sources_step (Ch. 14: cost-aware pre-filtering).

Key idea: don't burn an expensive LLM call deciding whether a search result
is relevant. Instead:
  1. Call cheap, free, deterministic tools (arXiv API, web search) directly
     in Python - zero LLM cost.
  2. Use plain keyword overlap (no LLM) to drop obviously irrelevant results
     before anything is shown to an agent.

Only the *survivors* of this cheap filter are handed to the (expensive)
Researcher agent later, which keeps token spend down.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from tools import arxiv_search_tool, web_search_tool


def _keywords(text: str) -> set:
    return set(re.findall(r"[a-zA-Z]{4,}", text.lower()))


def _is_relevant(candidate_text: str, keyword_set: set, min_overlap: int = 1) -> bool:
    candidate_keywords = _keywords(candidate_text)
    return len(candidate_keywords & keyword_set) >= min_overlap


def filter_sources_step(context: Dict[str, Any], tracer: Optional[Any] = None) -> Dict[str, Any]:
    plan = context["plan"]
    keyword_set = _keywords(plan.main_theme) | set().union(
        *[_keywords(q) for q in plan.sub_questions]
    )

    domains = [d.lower() for d in plan.search_domains] or ["academic papers", "industry blogs"]
    raw_candidates: List[str] = []

    if any("academic" in d or "paper" in d or "arxiv" in d for d in domains):
        raw_candidates.append(arxiv_search_tool.func(query=plan.main_theme, max_results=6))

    if any(
        d not in ("academic papers",) for d in domains
    ) or not raw_candidates:
        raw_candidates.append(web_search_tool.func(query=plan.main_theme, max_results=6))

    # Split each tool's block output back into per-result chunks (they're '- ' bulleted)
    all_chunks: List[str] = []
    for block in raw_candidates:
        chunks = re.split(r"\n(?=- )", block)
        all_chunks.extend(c for c in chunks if c.strip())

    filtered = [c for c in all_chunks if _is_relevant(c, keyword_set)]
    # Fall back to unfiltered results if the cheap filter was too aggressive
    if not filtered:
        filtered = all_chunks

    context["filtered_sources"] = filtered
    context["filter_stats"] = {
        "candidates_found": len(all_chunks),
        "candidates_kept": len(filtered),
    }
    return context
