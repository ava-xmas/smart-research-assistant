"""
Single-agent baseline (Ch. 10: you need a baseline to prove the multi-agent
system actually adds value). This is just one Agent with the same tools as
the Researcher, asked to answer the query directly in one shot - no
planning step, no Writer, no Critic.
"""

from __future__ import annotations

from picoagents_lite import Agent, OpenAIModelClient
from tools import arxiv_search_tool, web_search_tool


def run_single_agent_baseline(query: str, model_client: OpenAIModelClient) -> str:
    agent = Agent(
        name="BaselineAgent",
        instructions=(
            "You are a helpful research assistant. Use the web_search and "
            "arxiv_search tools as needed, then answer the user's question "
            "directly and completely in well-organized Markdown, with source "
            "URLs cited inline."
        ),
        model_client=model_client,
        tools=[web_search_tool, arxiv_search_tool],
        max_tool_iterations=6,
    )
    return agent.run(user_message=query)
