"""Tests for graph construction and routing logic.

Validates the StateGraph topology, conditional routing functions,
and graph compilation without requiring actual LLM calls.
"""

from __future__ import annotations

import pytest

from orchestration.graph import (
    build_orchestration_graph,
    build_simple_graph,
    should_research_or_skip,
    should_revise_or_finalize,
)
from orchestration.state import (
    AgentState,
    ResearchFindings,
    ReviewFeedback,
    WorkflowStatus,
    create_initial_state,
)


class TestShouldReviseOrFinalize:
    """Tests for the reviewer -> writer/finalizer routing logic."""

    def test_approved_content_goes_to_finalizer(self) -> None:
        """When review is approved, route to finalizer."""
        state = create_initial_state("Test")
        state["review_feedback"] = ReviewFeedback(approved=True, score=8.0)
        state["revision_count"] = 1

        result = should_revise_or_finalize(state)
        assert result == "finalizer"

    def test_rejected_content_goes_to_writer(self) -> None:
        """When review is not approved and revisions remain, route to writer."""
        state = create_initial_state("Test")
        state["review_feedback"] = ReviewFeedback(approved=False, score=4.0, issues=["Too short"])
        state["revision_count"] = 1

        result = should_revise_or_finalize(state)
        assert result == "writer"

    def test_max_revisions_forces_finalizer(self) -> None:
        """When max revisions reached, route to finalizer regardless of approval."""
        state = create_initial_state("Test", max_revisions=3)
        state["review_feedback"] = ReviewFeedback(approved=False, score=5.0)
        state["revision_count"] = 3

        result = should_revise_or_finalize(state)
        assert result == "finalizer"

    def test_no_feedback_goes_to_writer(self) -> None:
        """When no feedback exists (edge case), route to writer."""
        state = create_initial_state("Test")
        state["revision_count"] = 0

        result = should_revise_or_finalize(state)
        assert result == "writer"

    def test_exactly_at_max_revisions(self) -> None:
        """Boundary: revision_count equals max_revisions should finalize."""
        state = create_initial_state("Test", max_revisions=2)
        state["review_feedback"] = ReviewFeedback(approved=False, score=6.0)
        state["revision_count"] = 2

        result = should_revise_or_finalize(state)
        assert result == "finalizer"


class TestShouldResearchOrSkip:
    """Tests for the start -> researcher/writer routing logic."""

    def test_no_findings_routes_to_researcher(self) -> None:
        """When no research findings exist, route to researcher."""
        state = create_initial_state("Test")
        assert state["research_findings"] is None

        result = should_research_or_skip(state)
        assert result == "researcher"

    def test_existing_findings_skips_to_writer(self) -> None:
        """When research findings already exist, skip to writer."""
        state = create_initial_state("Test")
        state["research_findings"] = ResearchFindings(
            topic="Test", key_points=["Point 1"], confidence=0.8
        )

        result = should_research_or_skip(state)
        assert result == "writer"


class TestBuildOrchestrationGraph:
    """Tests for graph construction and compilation."""

    def test_graph_compiles_without_checkpointing(self) -> None:
        """Graph should compile successfully without checkpointing."""
        graph = build_simple_graph()
        assert graph is not None

    def test_graph_compiles_with_checkpointing(self) -> None:
        """Graph should compile successfully with checkpointing enabled."""
        graph = build_orchestration_graph(enable_checkpointing=True)
        assert graph is not None

    def test_graph_has_expected_nodes(self) -> None:
        """Compiled graph should contain all expected node names."""
        graph = build_simple_graph()
        # LangGraph compiled graphs expose node names via get_graph()
        graph_repr = graph.get_graph()
        node_ids = {node.id for node in graph_repr.nodes}

        # Should contain our agent nodes (plus __start__ and __end__)
        assert "researcher" in node_ids
        assert "writer" in node_ids
        assert "reviewer" in node_ids
        assert "finalizer" in node_ids

    def test_graph_has_edges(self) -> None:
        """Compiled graph should have edges connecting nodes."""
        graph = build_simple_graph()
        graph_repr = graph.get_graph()
        edges = graph_repr.edges

        # Should have at least the static edges
        assert len(edges) > 0

    def test_build_simple_graph_no_checkpointer(self) -> None:
        """build_simple_graph should create graph without persistence."""
        graph = build_simple_graph()
        # Should be a compiled graph (CompiledStateGraph)
        assert hasattr(graph, "ainvoke")
        assert hasattr(graph, "invoke")
