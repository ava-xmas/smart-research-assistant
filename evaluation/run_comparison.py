"""
Runs the full evaluation: single-agent baseline vs. the multi-agent
workflow, across the task suite in tasks.py, then saves a CSV/JSON of raw
scores plus a bar chart PNG comparing the two systems on each dimension.

Usage:
    python -m evaluation.run_comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from picoagents_lite import OpenAIModelClient, Tracer, Workflow, WorkflowMetadata
from evaluation.baseline import run_single_agent_baseline
from evaluation.evaluator import EvalRunner
from evaluation.tasks import TASKS
from steps import (
    filter_sources_step,
    multi_agent_research_step,
    parse_query_step,
    synthesize_report_step,
)


def run_multi_agent_system(task: dict) -> str:
    model_client = OpenAIModelClient(model="openai/gpt-oss-120b")
    workflow = (
        Workflow(metadata=WorkflowMetadata(name="Smart Research Assistant"))
        .chain(parse_query_step, filter_sources_step, multi_agent_research_step, synthesize_report_step)
    )
    tracer = Tracer(run_name=f"eval:{task['id']}")
    context = workflow.run({"query": task["query"], "model_client": model_client}, tracer=tracer)
    return context["report"]


def run_baseline_system(task: dict) -> str:
    model_client = OpenAIModelClient(model="openai/gpt-oss-120b")
    return run_single_agent_baseline(task["query"], model_client)


def main() -> None:
    judge_client = OpenAIModelClient(model="openai/gpt-oss-120b", temperature=0)
    runner = EvalRunner(judge_model_client=judge_client)

    runner.evaluate_system("single_agent_baseline", run_baseline_system, TASKS)
    runner.evaluate_system("multi_agent_workflow", run_multi_agent_system, TASKS)

    runner.print_report()
    runner.save_csv("outputs/eval_results.csv")
    runner.save_json("outputs/eval_results.json")
    print("\nSaved outputs/eval_results.csv and outputs/eval_results.json")

    _plot_results(runner)


def _plot_results(runner: EvalRunner) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping chart. `pip install matplotlib` to enable it.")
        return

    dims = ["accuracy", "completeness", "clarity", "actionability"]
    systems = list(runner.summary_by_system().keys())

    means = {sys_name: [] for sys_name in systems}
    for sys_name in systems:
        rows = [r for r in runner.results if r.system_name == sys_name]
        for dim in dims:
            vals = [getattr(r, dim) for r in rows]
            means[sys_name].append(sum(vals) / len(vals))

    import numpy as np
    x = np.arange(len(dims))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, sys_name in enumerate(systems):
        ax.bar(x + i * width, means[sys_name], width, label=sys_name)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(dims)
    ax.set_ylabel("Average judge score (1-5)")
    ax.set_title("Single-Agent Baseline vs. Multi-Agent Workflow")
    ax.legend()
    ax.set_ylim(0, 5)
    fig.tight_layout()

    out_path = "outputs/eval_comparison_chart.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


if __name__ == "__main__":
    main()
