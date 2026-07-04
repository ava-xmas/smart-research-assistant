"""
Smart Research & Synthesis Assistant - CLI entry point.

Usage:
    python main.py "What are the key trends in autonomous AI agents for 2026?"
    python main.py "..." --model openai/gpt-oss-120b --trace
    python main.py --evaluate      # runs the full baseline-vs-multi-agent evaluation

Requires GROQ_API_KEY to be set (see .env.example).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from picoagents_lite import OpenAIModelClient, Tracer, Workflow, WorkflowMetadata
from steps import (
    filter_sources_step,
    multi_agent_research_step,
    parse_query_step,
    synthesize_report_step,
)


def build_workflow() -> Workflow:
    return (
        Workflow(metadata=WorkflowMetadata(name="Smart Research Assistant"))
        .chain(parse_query_step, filter_sources_step, multi_agent_research_step, synthesize_report_step)
    )


def run(query: str, model: str, max_rounds: int, show_trace: bool) -> None:
    model_client = OpenAIModelClient(model=model)
    tracer = Tracer(run_name="smart_research_assistant")

    workflow = build_workflow()
    context = workflow.run(
        {
            "query": query,
            "model_client": model_client,
            "max_rounds": max_rounds,
        },
        tracer=tracer,
    )

    print("\n" + "=" * 80)
    print(context["report"])
    print("=" * 80)

    print(f"\nSources found: {context['filter_stats']['candidates_found']}  "
          f"| kept after cheap filter: {context['filter_stats']['candidates_kept']}")
    print(f"Total estimated cost: ${context['total_cost_usd']:.5f}")

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/last_report.md").write_text(context["report"])
    tracer.save("outputs/last_trace.json")

    if show_trace:
        tracer.print_tree()

    print("\nSaved report to outputs/last_report.md and trace to outputs/last_trace.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Research & Synthesis Assistant")
    parser.add_argument("query", nargs="?", help="Research question to investigate")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Groq model name (see https://console.groq.com/docs/models)")
    parser.add_argument("--max-rounds", type=int, default=8, help="Max orchestrator rounds")
    parser.add_argument("--trace", action="store_true", help="Print the observability trace tree")
    parser.add_argument("--evaluate", action="store_true", help="Run the full evaluation harness")
    args = parser.parse_args()

    if args.evaluate:
        from evaluation.run_comparison import main as run_eval
        run_eval()
        return

    if not args.query:
        parser.error("Please provide a research query, or use --evaluate")
        sys.exit(1)

    run(args.query, args.model, args.max_rounds, args.trace)


if __name__ == "__main__":
    main()
