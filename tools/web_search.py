"""
General web search tool.

Default backend: the `ddgs` package (DuckDuckGo Search), which needs no API
key at all - good for a project people can clone and run immediately.

If you have a SERPAPI_KEY set in your environment, this will use SerpAPI's
Google Search results instead (usually higher quality), so the tool is easy
to upgrade later without changing any calling code.
"""

from __future__ import annotations

import os

import requests


def _search_serpapi(query: str, max_results: int) -> str:
    api_key = os.environ["SERPAPI_KEY"]
    resp = requests.get(
        "https://serpapi.com/search",
        params={"q": query, "num": max_results, "api_key": api_key},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("organic_results", [])[:max_results]
    if not results:
        return f"No web results found for query: {query}"
    lines = []
    for r in results:
        lines.append(
            f"- {r.get('title', '')}\n  URL: {r.get('link', '')}\n  {r.get('snippet', '')}"
        )
    return "\n".join(lines)


def _search_duckduckgo(query: str, max_results: int) -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        return (
            "Web search backend not installed. Run `pip install ddgs` "
            "(or set SERPAPI_KEY) to enable the web_search tool."
        )

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:  # network errors, rate limits, etc.
        return f"Web search failed: {exc}"

    if not results:
        return f"No web results found for query: {query}"

    lines = []
    for r in results:
        title = r.get("title", "")
        url = r.get("href", "")
        snippet = r.get("body", "")
        lines.append(f"- {title}\n  URL: {url}\n  {snippet}")
    return "\n".join(lines)


def _search(query: str, max_results: int = 5) -> str:
    if os.environ.get("SERPAPI_KEY"):
        return _search_serpapi(query, max_results)
    return _search_duckduckgo(query, max_results)


from picoagents_lite.agent import Tool  # noqa: E402  (avoids circular import at module load)

web_search_tool = Tool(
    name="web_search",
    description=(
        "Search the general web for a query. Returns titles, URLs, and short "
        "snippets. Use for industry news, blog posts, and non-academic sources."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search terms"},
            "max_results": {"type": "integer", "description": "How many results to return (default 5)"},
        },
        "required": ["query"],
    },
    func=_search,
)
