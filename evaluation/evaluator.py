"""
EvalRunner (Ch. 10): runs a task suite against a system and scores each
output with an LLM-as-judge on accuracy, completeness, clarity, and
actionability (1-5 each). This is what lets us make an evidence-based claim
like "the multi-agent workflow scored X% higher than the single-agent
baseline" instead of an anecdotal one.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List

from pydantic import BaseModel, Field

from picoagents_lite import OpenAIModelClient


class JudgeScore(BaseModel):
    accuracy: int = Field(description="1-5: are claims plausible/well-supported, no fabrication?")
    completeness: int = Field(description="1-5: does it cover the expected_coverage points?")
    clarity: int = Field(description="1-5: is it well organized and easy to follow?")
    actionability: int = Field(description="1-5: does it give concrete, usable recommendations?")
    justification: str = Field(description="1-2 sentence rationale for the scores.")

    @property
    def average(self) -> float:
        return round((self.accuracy + self.completeness + self.clarity + self.actionability) / 4, 2)


@dataclass
class EvalResult:
    task_id: str
    system_name: str
    query: str
    output_preview: str
    accuracy: int
    completeness: int
    clarity: int
    actionability: int
    average: float
    justification: str


class EvalRunner:
    def __init__(self, judge_model_client: OpenAIModelClient):
        self.judge = judge_model_client
        self.results: List[EvalResult] = []

    def _judge(self, task: Dict, output: str) -> JudgeScore:
        prompt = f"""
You are grading a research assistant's output.

QUESTION: {task['query']}

EXPECTED COVERAGE (a good answer should touch on these points, not
necessarily verbatim):
{chr(10).join('- ' + c for c in task['expected_coverage'])}

ASSISTANT'S OUTPUT:
{output[:6000]}

Score the output from 1 (poor) to 5 (excellent) on each dimension.
"""
        return self.judge.structured(
            prompt=prompt,
            schema_model=JudgeScore,
            system="You are a strict, consistent evaluator of research report quality.",
            purpose="llm_judge",
        )

    def evaluate_system(
        self,
        system_name: str,
        run_fn: Callable[[Dict], str],
        tasks: List[Dict],
        verbose: bool = True,
    ) -> List[EvalResult]:
        """
        run_fn takes a task dict and returns the system's text output for
        that task's query.
        """
        system_results = []
        for task in tasks:
            if verbose:
                print(f"[{system_name}] running {task['id']}...")
            output = run_fn(task)
            score = self._judge(task, output)
            result = EvalResult(
                task_id=task["id"],
                system_name=system_name,
                query=task["query"],
                output_preview=output[:200].replace("\n", " "),
                accuracy=score.accuracy,
                completeness=score.completeness,
                clarity=score.clarity,
                actionability=score.actionability,
                average=score.average,
                justification=score.justification,
            )
            system_results.append(result)
            self.results.append(result)
            if verbose:
                print(f"  -> avg score: {result.average}/5  ({score.justification})")
        return system_results

    def summary_by_system(self) -> Dict[str, float]:
        totals: Dict[str, List[float]] = {}
        for r in self.results:
            totals.setdefault(r.system_name, []).append(r.average)
        return {name: round(sum(vals) / len(vals), 3) for name, vals in totals.items()}

    def save_csv(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.results[0]).keys()))
            writer.writeheader()
            for r in self.results:
                writer.writerow(asdict(r))

    def save_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps([asdict(r) for r in self.results], indent=2))

    def print_report(self) -> None:
        print("\n=== Evaluation Summary ===")
        for name, avg in self.summary_by_system().items():
            print(f"{name:<20} avg score: {avg}/5")
