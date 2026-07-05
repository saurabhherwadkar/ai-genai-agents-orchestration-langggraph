"""Tests for state management module.

Validates the state schemas, initial state creation, Pydantic models,
and state transition logic used by the orchestration graph.
"""

from __future__ import annotations

import pytest

from orchestration.state import (
    AgentState,
    MessageEntry,
    ResearchFindings,
    ReviewFeedback,
    WorkflowStatus,
    create_initial_state,
)


class TestWorkflowStatus:
    """Tests for the WorkflowStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Verify all expected workflow statuses are defined."""
        expected = {"pending", "researching", "writing", "reviewing", "revising", "complete", "failed"}
        actual = {s.value for s in WorkflowStatus}
        assert actual == expected

    def test_status_is_string_enum(self) -> None:
        """WorkflowStatus values should be usable as strings."""
        assert WorkflowStatus.PENDING == "pending"
        assert str(WorkflowStatus.COMPLETE) == "WorkflowStatus.COMPLETE"

    def test_status_from_value(self) -> None:
        """Should be able to construct status from string value."""
        status = WorkflowStatus("researching")
        assert status == WorkflowStatus.RESEARCHING


class TestMessageEntry:
    """Tests for the MessageEntry Pydantic model."""

    def test_create_message(self) -> None:
        """Should create a message with required fields."""
        msg = MessageEntry(role="researcher", content="Found 5 key points.")
        assert msg.role == "researcher"
        assert msg.content == "Found 5 key points."
        assert msg.metadata == {}

    def test_create_message_with_metadata(self) -> None:
        """Should support optional metadata."""
        msg = MessageEntry(
            role="writer",
            content="Draft complete.",
            metadata={"word_count": "500", "tone": "professional"},
        )
        assert msg.metadata["word_count"] == "500"
        assert msg.metadata["tone"] == "professional"

    def test_message_is_frozen(self) -> None:
        """MessageEntry should be immutable (frozen model)."""
        msg = MessageEntry(role="reviewer", content="Looks good.")
        with pytest.raises(Exception):  # ValidationError for frozen model
            msg.role = "writer"  # type: ignore[misc]


class TestResearchFindings:
    """Tests for the ResearchFindings model."""

    def test_create_findings(self) -> None:
        """Should create findings with all fields."""
        findings = ResearchFindings(
            topic="AI Agents",
            key_points=["Point 1", "Point 2"],
            sources=["https://example.com"],
            confidence=0.85,
        )
        assert findings.topic == "AI Agents"
        assert len(findings.key_points) == 2
        assert findings.confidence == 0.85

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        findings = ResearchFindings(topic="Test")
        assert findings.key_points == []
        assert findings.sources == []
        assert findings.confidence == 0.0

    def test_confidence_validation(self) -> None:
        """Confidence should be constrained between 0 and 1."""
        with pytest.raises(Exception):
            ResearchFindings(topic="Test", confidence=1.5)
        with pytest.raises(Exception):
            ResearchFindings(topic="Test", confidence=-0.1)


class TestReviewFeedback:
    """Tests for the ReviewFeedback model."""

    def test_create_feedback(self) -> None:
        """Should create feedback with all fields."""
        feedback = ReviewFeedback(
            approved=True,
            score=8.5,
            issues=[],
            suggestions=["Add more examples"],
        )
        assert feedback.approved is True
        assert feedback.score == 8.5
        assert len(feedback.suggestions) == 1

    def test_score_validation(self) -> None:
        """Score should be constrained between 0 and 10."""
        with pytest.raises(Exception):
            ReviewFeedback(score=11.0)
        with pytest.raises(Exception):
            ReviewFeedback(score=-1.0)

    def test_default_not_approved(self) -> None:
        """Default feedback should not be approved."""
        feedback = ReviewFeedback()
        assert feedback.approved is False
        assert feedback.score == 0.0


class TestCreateInitialState:
    """Tests for the create_initial_state factory function."""

    def test_creates_valid_state(self) -> None:
        """Should create a state with all required fields."""
        state = create_initial_state("Write about LangGraph")
        assert state["task"] == "Write about LangGraph"
        assert state["status"] == WorkflowStatus.PENDING
        assert state["messages"] == []
        assert state["research_findings"] is None
        assert state["draft_content"] == ""
        assert state["review_feedback"] is None
        assert state["revision_count"] == 0
        assert state["max_revisions"] == 3
        assert state["final_output"] == ""

    def test_custom_max_revisions(self) -> None:
        """Should accept custom max_revisions parameter."""
        state = create_initial_state("Test task", max_revisions=5)
        assert state["max_revisions"] == 5

    def test_state_is_dict(self) -> None:
        """State should be a plain dictionary (TypedDict)."""
        state = create_initial_state("Test")
        assert isinstance(state, dict)

    def test_state_keys_match_schema(self) -> None:
        """State should contain exactly the keys defined in AgentState."""
        state = create_initial_state("Test")
        expected_keys = {
            "messages",
            "task",
            "status",
            "research_findings",
            "draft_content",
            "review_feedback",
            "revision_count",
            "max_revisions",
            "final_output",
        }
        assert set(state.keys()) == expected_keys
