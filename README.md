# Smart Research & Synthesis Assistant

A hybrid multi-agent system that takes a research question, plans it, gathers
sources with cost-aware filtering, runs a team of specialist agents to
research and draft a report, and evaluates its own quality against a
single-agent baseline — with built-in cost/latency tracing throughout.

It's a working, from-scratch implementation of a small multi-agent framework
(`picoagents_lite/`) on top of the Anthropic Messages API — no external
agent framework required. Given a question like:

> "What are the key trends in autonomous AI agents for 2026 and what are the main challenges?"

it produces a structured Markdown report with citations, in a few minutes,
for a few cents.

---

## Why this architecture (the 30-second interview answer)

> "I used a **hybrid architecture**. A deterministic **Workflow** handles
> the reliable, cheap, predictable steps — parsing the query and
> pre-filtering sources. Inside that workflow, one step hands off to an
> **autonomous multi-agent team** (Researcher → Writer → Critic) coordinated
> by an LLM router, for the part of the problem that actually benefits from
> adaptive collaboration. I proved the multi-agent approach is worth its
> extra cost with an **LLM-as-judge evaluation harness** against a
> single-agent baseline, and instrumented the whole thing with tracing so I
> can see exactly where time and money go."

---

## Architecture

```
                         ┌─────────────────────────────────────────┐
                         │              Workflow (Ch.6)             │
                         │         deterministic, cheap, reliable   │
                         └─────────────────────────────────────────┘
   User query                     │
       │                          ▼
       │                 ┌──────────────────┐
       └────────────────▶│ parse_query_step │  structured output (Pydantic)
                          │                  │  -> ResearchPlan
                          └──────────────────┘
                                   │
                                   ▼
                          ┌────────────────────┐
                          │ filter_sources_step │  cheap, LLM-free:
                          │                     │  arXiv API + web search,
                          │                     │  keyword-overlap filter
                          └────────────────────┘
                                   │
                                   ▼
              ┌───────────────────────────────────────────┐
              │        multi_agent_research_step            │
              │  ┌─────────────────────────────────────┐   │
              │  │       AIOrchestrator (Ch.7)          │   │
              │  │   autonomous, LLM-routed turn-taking │   │
              │  │                                       │   │
              │  │   Researcher ──▶ Writer ──▶ Critic    │   │
              │  │       ▲______________________|        │   │
              │  │        (revise until approved)         │   │
              │  └─────────────────────────────────────┘   │
              └───────────────────────────────────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ synthesize_report_step │  final formatting pass
                       └────────────────────────┘
                                   │
                                   ▼
                          Markdown research report
```

Every step and every agent call is wrapped in a **Tracer** span (`Ch.4 –
observability`), and every model call records token counts + an estimated
dollar cost (`Ch.14 – cost awareness`).

---

## Project layout

```
smart-research-assistant/
├── README.md
├── requirements.txt
├── .env.example
├── config.yaml
├── main.py                        # CLI entry point
├── picoagents_lite/                # the mini agent framework, built from scratch
│   ├── agent.py                    #   Agent: system prompt + tool-use loop
│   ├── model_client.py             #   AnthropicModelClient: chat() + structured()
│   ├── orchestrator.py             #   AIOrchestrator: LLM-routed multi-agent turns
│   ├── termination.py              #   MaxMessageTermination | TextMentionTermination
│   ├── workflow.py                 #   Workflow: deterministic step chaining
│   └── tracing.py                  #   Tracer: spans, cost, OTel-shaped export
├── tools/
│   ├── arxiv_search.py             # free arXiv API, no key needed
│   └── web_search.py               # free DuckDuckGo search, or SerpAPI if configured
├── agents/
│   └── definitions.py              # Researcher / Writer / Critic factories
├── steps/
│   ├── parse_query.py              # -> ResearchPlan (structured output)
│   ├── filter_sources.py           # cheap keyword pre-filter
│   ├── multi_agent_research.py     # runs the AIOrchestrator
│   └── synthesize_report.py        # final report formatting
├── evaluation/
│   ├── tasks.py                    # 5-question task suite
│   ├── baseline.py                 # single-agent baseline for comparison
│   ├── evaluator.py                # EvalRunner + LLM-as-judge (JudgeScore)
│   └── run_comparison.py           # runs baseline vs. multi-agent, saves chart
└── outputs/                        # reports, traces, and eval results land here
```

---

## Setup

```bash
git clone <this repo>
cd smart-research-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

You only need `ANTHROPIC_API_KEY`. `SERPAPI_KEY` is optional — without it,
web search falls back to the free `ddgs` (DuckDuckGo) backend automatically.
arXiv search never needs a key.

## Usage

**Run a single research query:**

```bash
python main.py "What are the key trends in autonomous AI agents for 2026 and what are the main challenges?"
```

Add `--trace` to print the observability trace tree (per-step/agent timing
and cost):

```bash
python main.py "How are companies evaluating multi-agent LLM systems?" --trace
```

This writes `outputs/last_report.md` (the final report) and
`outputs/last_trace.json` (the full trace).

**Run the full evaluation harness (baseline vs. multi-agent):**

```bash
python main.py --evaluate
# or directly:
python -m evaluation.run_comparison
```

This runs all 5 tasks in `evaluation/tasks.py` through both the
single-agent baseline and the full multi-agent workflow, scores every
output with an LLM-as-judge on **accuracy, completeness, clarity, and
actionability**, and writes:

- `outputs/eval_results.csv` / `.json` — raw per-task scores
- `outputs/eval_comparison_chart.png` — bar chart comparing the two systems

> Note: `--evaluate` makes ~10-20 real API calls (2 systems × 5 tasks ×
> multiple agent/judge calls) and costs on the order of $0.50-$2 depending
> on model choice. Trim `evaluation/tasks.py` if you want a cheaper smoke test.

---

## How each piece maps to core multi-agent concepts

| Concept | Where it lives | What it demonstrates |
|---|---|---|
| **Structured output** | `steps/parse_query.py`, `AnthropicModelClient.structured()` | Forcing a tool call with a Pydantic JSON schema instead of hoping the model formats JSON correctly |
| **Deterministic workflow** | `picoagents_lite/workflow.py` | Reliability and predictable cost for steps that don't need autonomy |
| **Autonomous multi-agent orchestration** | `picoagents_lite/orchestrator.py` | An LLM decides turn-by-turn who speaks next and when the team is done, rather than a hard-coded sequence |
| **Cost-aware engineering** | `steps/filter_sources.py`, `model_client.py` cost tracking | Cheap, LLM-free filtering before expensive agent calls; per-call cost logged everywhere |
| **Tool use / agent execution loop** | `picoagents_lite/agent.py` | The standard loop: call model → detect `tool_use` → execute tool → feed result back → repeat |
| **Observability** | `picoagents_lite/tracing.py` | Span-based tracing with duration/tokens/cost per step and per agent call; shaped to map onto OpenTelemetry GenAI semantic conventions |
| **Evaluation / LLM-as-judge** | `evaluation/evaluator.py`, `evaluation/tasks.py` | A repeatable task suite + judge model scoring, used to prove the multi-agent system beats a single-agent baseline |

---

## Wiring this up to real OpenTelemetry / Jaeger

`Tracer` in `picoagents_lite/tracing.py` already collects spans in a shape
close to the OpenTelemetry GenAI semantic conventions
(`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*`). To export to a
real collector:

1. `pip install opentelemetry-sdk opentelemetry-exporter-otlp`
2. Stand up a local Jaeger instance (`docker run -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one`)
3. In `Tracer.save()` (or a new `export_otel()` method), take the output of
   `to_otel_spans()` and feed each span into an OTLP exporter instead of / in
   addition to writing the local JSON file.

This was left as a local JSON tracer by default so the project runs with
zero extra infrastructure, while still being straightforward to upgrade.

---

## Extending the project

- **Swap models per agent** — pass a cheaper `AnthropicModelClient(model="claude-haiku-4-5-20251001")`
  to the Researcher (high tool-call volume, less reasoning-heavy) and keep a
  stronger model for the Writer/Critic — a natural next cost-optimization step.
- **Add more tools** — e.g. a Wikipedia tool or an internal document search —
  by adding a new `Tool(...)` in `tools/` and wiring it into `agents/definitions.py`.
- **Add checkpointing** — persist `context` to disk after each `Workflow` step
  so a long-running research task can resume after a crash.
- **Grow the eval suite** — add more tasks to `evaluation/tasks.py`; the
  harness scales to any number automatically.

---

## Talking points for an interview

1. **You understand the fundamentals** — you built a full agent execution
   loop (`agent.py`) and structured-output handling from the raw Messages
   API, not just called a high-level SDK.
2. **You can design and reason about architectures** — you knew when to use
   a deterministic `Workflow` (reliability/cost) vs. an autonomous
   `AIOrchestrator` (adaptive collaboration), and combined them.
3. **You're cost-aware** — every model call is logged with token counts and
   estimated dollar cost, and cheap filtering happens before expensive
   agent calls.
4. **You care about proving quality, not just building things** — the
   evaluation harness with an LLM-as-judge is how you show the multi-agent
   system is measurably better than a naive baseline, on real numbers.
5. **You can debug and monitor production systems** — span-based tracing
   with a clear upgrade path to OpenTelemetry/Jaeger.
