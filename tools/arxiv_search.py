"""
arXiv search tool - uses arXiv's free public Atom API (no API key needed).
Docs: https://info.arxiv.org/help/api/user-manual.html
"""

from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET

import requests

from picoagents_lite.agent import Tool

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


def _search_arxiv(query: str, max_results: int = 5) -> str:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return f"arXiv search failed: {exc}"

    root = ET.fromstring(resp.text)
    entries = root.findall(f"{ATOM_NS}entry")
    if not entries:
        return f"No arXiv results found for query: {query}"

    lines = []
    for entry in entries:
        title = entry.findtext(f"{ATOM_NS}title", default="").strip().replace("\n", " ")
        summary = entry.findtext(f"{ATOM_NS}summary", default="").strip().replace("\n", " ")
        link = entry.findtext(f"{ATOM_NS}id", default="").strip()
        published = entry.findtext(f"{ATOM_NS}published", default="")[:10]
        lines.append(
            f"- \"{title}\" ({published})\n  URL: {link}\n  Abstract: {summary[:400]}..."
        )
    return "\n".join(lines)


arxiv_search_tool = Tool(
    name="arxiv_search",
    description=(
        "Search arXiv.org for academic papers relevant to a query. "
        "Returns titles, publish dates, links, and short abstracts."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search terms, e.g. 'multi-agent LLM orchestration'"},
            "max_results": {"type": "integer", "description": "How many papers to return (default 5)"},
        },
        "required": ["query"],
    },
    func=_search_arxiv,
)
