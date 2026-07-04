"""
Lightweight, dependency-free tracer that mimics what an OpenTelemetry
GenAI-semantic-conventions setup would give you: a span per step/agent
call, with duration, token usage, and cost.

Why not just wire up real OpenTelemetry + Jaeger here? Because that requires
standing up extra infrastructure (a Jaeger collector) that most people
evaluating this project on their laptop won't have running. Instead this
module:
  1. Produces the same *shape* of trace data (spans with attributes),
  2. Writes it to a local JSON file you can inspect or graph,
  3. Is trivially swappable for real OTel - see `to_otel_spans()` below and
     the README section "Wiring this up to real OpenTelemetry / Jaeger".
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Span:
    span_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return round(self.end_time - self.start_time, 4)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration_s"] = self.duration_s
        return d


class Tracer:
    """Collects spans for one run of the pipeline."""

    def __init__(self, run_name: str = "run"):
        self.run_name = run_name
        self.spans: List[Span] = []
        self._stack: List[str] = []

    @contextmanager
    def span(self, name: str, **attributes: Any):
        span = Span(
            span_id=str(uuid.uuid4())[:8],
            name=name,
            start_time=time.time(),
            attributes=attributes,
            parent_id=self._stack[-1] if self._stack else None,
        )
        self._stack.append(span.span_id)
        self.spans.append(span)
        try:
            yield span
        finally:
            span.end_time = time.time()
            self._stack.pop()

    def add_attributes(self, span: Span, **attributes: Any) -> None:
        span.attributes.update(attributes)

    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        total_cost = sum(
            s.attributes.get("estimated_cost_usd", 0) or 0 for s in self.spans
        )
        total_tokens = sum(
            (s.attributes.get("input_tokens", 0) or 0)
            + (s.attributes.get("output_tokens", 0) or 0)
            for s in self.spans
        )
        total_time = None
        if self.spans:
            start = min(s.start_time for s in self.spans)
            end = max(s.end_time or s.start_time for s in self.spans)
            total_time = round(end - start, 3)
        return {
            "run_name": self.run_name,
            "num_spans": len(self.spans),
            "total_estimated_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "wall_clock_s": total_time,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary(),
            "spans": [s.to_dict() for s in self.spans],
        }
        Path(path).write_text(json.dumps(payload, indent=2, default=str))

    def print_tree(self) -> None:
        print(f"\n--- Trace: {self.run_name} ---")
        for s in self.spans:
            indent = "  " if s.parent_id else ""
            extra = ""
            if "estimated_cost_usd" in s.attributes:
                extra = f" | ${s.attributes['estimated_cost_usd']:.5f}"
            print(f"{indent}{s.name:<30} {s.duration_s:>7}s{extra}")
        summ = self.summary()
        print(
            f"TOTAL: {summ['wall_clock_s']}s | "
            f"{summ['total_tokens']} tokens | "
            f"${summ['total_estimated_cost_usd']:.5f}"
        )

    def to_otel_spans(self) -> List[Dict[str, Any]]:
        """
        Convert to a shape close to the OpenTelemetry GenAI semantic conventions
        (gen_ai.system, gen_ai.request.model, gen_ai.usage.*). Feed each dict to
        an OTLP exporter if you wire up a real collector.
        """
        otel_spans = []
        for s in self.spans:
            otel_spans.append(
                {
                    "name": s.name,
                    "start_time_unix_nano": int(s.start_time * 1e9),
                    "end_time_unix_nano": int((s.end_time or s.start_time) * 1e9),
                    "attributes": {
                        "gen_ai.system": "anthropic",
                        "gen_ai.request.model": s.attributes.get("model"),
                        "gen_ai.usage.input_tokens": s.attributes.get("input_tokens"),
                        "gen_ai.usage.output_tokens": s.attributes.get("output_tokens"),
                        **s.attributes,
                    },
                }
            )
        return otel_spans
