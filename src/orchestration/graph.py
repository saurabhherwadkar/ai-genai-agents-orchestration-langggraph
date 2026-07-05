"""LangGraph StateGraph construction and routing logic.

This module builds the complete orchestration graph that wires together
agent nodes with conditional edges. It demonstrates key LangGraph patterns:

    - StateGraph: The core graph structure with typed state
    - add_node: Registering agent functions as graph nodes
    - add_edge: Static transitions between nodes
    - add_conditional_edges: Dynamic routing based on state
    - compile: Producing a runnable graph with optional checkpointing

Graph Topology:
    START -> researcher -> writer -> reviewer -> [conditional]
                                                    |-> writer (revision needed)
                                                    |-> finalizer (approved) -> END
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from orchestration.agents import (
    finalizer_node,
    researcher_node,
    reviewer_node,
    writer_node,
)
from orchestration.checkpointing import create_checkpointer
from orchestration.config import get_config
from orchestration.state import AgentState


def should_revise_or_finalize(state: AgentState) -> Literal["writer", "finalizer"]:
    """Routing function: decide whether to revise content or finalize.

    This function implements the conditional edge logic after the reviewer node.
    It checks:
    1. Whether the reviewer approved the content
    2. Whether we've exceeded the maximum revision count

    Args:
        state: Current workflow state with review feedback and revision count.

    Returns:
        "writer" if revision is needed, "finalizer" if content is approved or
        max revisions reached.
    """
    review = state.get("review_feedback")
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 3)

    # If approved, finalize
    if review and review.approved:
        return "finalizer"

    # If max revisions reached, finalize anyway
    if revision_count >= max_revisions:
        return "finalizer"

    # Otherwise, send back to writer for revision
    return "writer"


def should_research_or_skip(state: AgentState) -> Literal["researcher", "writer"]:
    """Routing function: decide whether research is needed.

    If research findings already exist (e.g., from a resumed checkpoint),
    skip directly to the writer node.

    Args:
        state: Current workflow state.

    Returns:
        "researcher" if research is needed, "writer" if findings exist.
    """
    if state.get("research_findings") is not None:
        return "writer"
    return "researcher"


def build_orchestration_graph(enable_checkpointing: bool | None = None) -> StateGraph:
    """Build and compile the multi-agent orchestration StateGraph.

    Constructs the full graph topology with nodes for each agent and
    conditional edges for routing decisions. Optionally enables checkpointing
    for state persistence.

    Args:
        enable_checkpointing: Whether to enable checkpointing. If None,
            uses the value from config.

    Returns:
        A compiled StateGraph ready for invocation.

    Example:
        >>> graph = build_orchestration_graph()
        >>> initial_state = create_initial_state("Write about AI agents")
        >>> result = await graph.ainvoke(initial_state)
        >>> print(result["final_output"])
    """
    # Create the StateGraph with our typed state schema
    workflow = StateGraph(AgentState)

    # --- Register Nodes ---
    # Each node is an async function that takes state and returns partial updates
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("finalizer", finalizer_node)

    # --- Define Edges ---

    # Entry point: conditionally route to researcher or writer
    workflow.add_conditional_edges(
        START,
        should_research_or_skip,
        {
            "researcher": "researcher",
            "writer": "writer",
        },
    )

    # After research, always proceed to writer
    workflow.add_edge("researcher", "writer")

    # After writing, always proceed to reviewer
    workflow.add_edge("writer", "reviewer")

    # After review, conditionally route to revision or finalization
    workflow.add_conditional_edges(
        "reviewer",
        should_revise_or_finalize,
        {
            "writer": "writer",
            "finalizer": "finalizer",
        },
    )

    # Finalizer leads to END
    workflow.add_edge("finalizer", END)

    # --- Compile the Graph ---
    config = get_config()
    use_checkpointing = enable_checkpointing if enable_checkpointing is not None else config.enable_checkpointing

    if use_checkpointing:
        checkpointer = create_checkpointer()
        compiled = workflow.compile(checkpointer=checkpointer)
    else:
        compiled = workflow.compile()

    return compiled


def build_simple_graph() -> StateGraph:
    """Build a minimal graph for testing without checkpointing.

    Useful for unit tests and quick demonstrations where persistence
    is not needed.

    Returns:
        A compiled StateGraph without checkpointing.
    """
    return build_orchestration_graph(enable_checkpointing=False)
