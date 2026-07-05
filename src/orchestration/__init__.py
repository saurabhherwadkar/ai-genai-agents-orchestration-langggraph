"""LangGraph Multi-Agent Orchestration Framework.

This package implements a multi-agent orchestration system using LangGraph,
demonstrating patterns for building stateful, cyclical agent workflows with
conditional routing, tool integration, and checkpointing.

Architecture Overview:
    - State: TypedDict schemas define the shared state passed between agents
    - Agents: Researcher, Writer, and Reviewer nodes process and transform state
    - Graph: StateGraph wires agents together with conditional edges for routing
    - Tools: Custom tools provide agents with capabilities (search, file I/O)
    - Config: Centralized configuration management for API keys and settings
    - Checkpointing: Persistent state for fault tolerance and resumability
"""

from orchestration.config import OrchestratorConfig
from orchestration.graph import build_orchestration_graph
from orchestration.state import AgentState, MessageEntry, WorkflowStatus

__all__ = [
    "AgentState",
    "MessageEntry",
    "OrchestratorConfig",
    "WorkflowStatus",
    "build_orchestration_graph",
]
