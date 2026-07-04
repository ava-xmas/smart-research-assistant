"""
picoagents_lite
================
A small, self-contained multi-agent framework (Agent, Workflow, Orchestrator,
Termination conditions, Tracing) built directly on the Anthropic Messages API.

This is a from-scratch, runnable implementation of the concepts described in
the "Smart Research & Synthesis Assistant" project plan:
  - Chapter 4  -> agent execution loop (agent.py) + observability (tracing.py)
  - Chapter 6  -> deterministic Workflow (workflow.py)
  - Chapter 7  -> autonomous multi-agent orchestration (orchestrator.py)
  - Chapter 10 -> evaluation harness (see evaluation/)
  - Chapter 14 -> cost-aware design (tracing.py records token/cost estimates)
"""

from .agent import Agent
from .model_client import AnthropicModelClient
from .orchestrator import AIOrchestrator
from .termination import MaxMessageTermination, TextMentionTermination
from .workflow import Workflow, WorkflowMetadata
from .tracing import Tracer

__all__ = [
    "Agent",
    "AnthropicModelClient",
    "AIOrchestrator",
    "MaxMessageTermination",
    "TextMentionTermination",
    "Workflow",
    "WorkflowMetadata",
    "Tracer",
]
