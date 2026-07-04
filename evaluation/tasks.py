"""
Task suite for the evaluation harness (Ch. 10).

Each task is a research question plus a short description of what a
"good" answer should cover. The judge model uses `expected_coverage` as a
rubric hint - it does not need to be an exact answer key, since research
questions rarely have one exact answer.
"""

TASKS = [
    {
        "id": "task_01",
        "query": "What are the key trends in autonomous AI agents for 2026, and what are the main challenges?",
        "expected_coverage": [
            "current adoption trends of autonomous agents",
            "at least one concrete challenge (reliability, cost, safety, evaluation)",
            "forward-looking recommendation",
        ],
    },
    {
        "id": "task_02",
        "query": "How are companies approaching evaluation of LLM-based multi-agent systems?",
        "expected_coverage": [
            "mention of LLM-as-judge or human evaluation",
            "mention of benchmark or task-suite based evaluation",
            "at least one named framework, paper, or company practice",
        ],
    },
    {
        "id": "task_03",
        "query": "What are the main cost-optimization strategies used in production LLM agent systems?",
        "expected_coverage": [
            "model routing / cheap-vs-expensive model selection",
            "caching or pre-filtering",
            "concrete numeric or qualitative example",
        ],
    },
    {
        "id": "task_04",
        "query": "What are the current best practices for observability and tracing in multi-agent AI systems?",
        "expected_coverage": [
            "mention of tracing/spans",
            "mention of a standard or tool (OpenTelemetry, Jaeger, LangSmith, etc.)",
            "explanation of why observability matters for debugging",
        ],
    },
    {
        "id": "task_05",
        "query": "What security risks are unique to autonomous multi-agent AI systems compared to single-agent chatbots?",
        "expected_coverage": [
            "prompt injection or tool-misuse risk",
            "risk from agent-to-agent delegation / cascading errors",
            "at least one mitigation strategy",
        ],
    },
]
