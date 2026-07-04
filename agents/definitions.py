"""
Factory functions for the three specialist agents used inside the
multi_agent_research_step. Kept as factories (rather than module-level
singletons) so tests / evaluation can spin up fresh agents with different
model clients (e.g. a cheaper model for the baseline comparison).
"""

from __future__ import annotations

from picoagents_lite import Agent, AnthropicModelClient
from tools import arxiv_search_tool, web_search_tool


def build_researcher(model_client: AnthropicModelClient) -> Agent:
    return Agent(
        name="Researcher",
        instructions=(
            "You are an expert research assistant.\n"
            "Use the web_search and arxiv_search tools to find specific, current, "
            "and credible information relevant to the assigned sub-question. "
            "Always cite the URL for every fact you report. Report findings as a "
            "concise bullet list: fact -> source URL. Do not editorialize."
        ),
        model_client=model_client,
        tools=[web_search_tool, arxiv_search_tool],
    )


def build_writer(model_client: AnthropicModelClient) -> Agent:
    return Agent(
        name="Writer",
        instructions=(
            "You are a technical writer.\n"
            "Synthesize the Researcher's findings into clear, well-organized "
            "prose with section headings. Preserve source URLs as inline "
            "citations. Do not invent facts that weren't reported by the "
            "Researcher. When revising after Critic feedback, address every "
            "point the Critic raised explicitly."
        ),
        model_client=model_client,
        tools=[],
    )


def build_critic(model_client: AnthropicModelClient) -> Agent:
    return Agent(
        name="Critic",
        instructions=(
            "You are a senior reviewer.\n"
            "Evaluate the Writer's draft for factual grounding (are claims "
            "backed by a cited source?), completeness (does it cover all "
            "sub-questions?), and clarity. If it is good enough to ship, say so "
            "explicitly and include the exact phrase REPORT_COMPLETE. Otherwise "
            "list specific, actionable revisions the Writer should make."
        ),
        model_client=model_client,
        tools=[],
    )
