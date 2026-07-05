"""State schemas for multi-agent orchestration workflows.

This module defines TypedDict-based state schemas that serve as the shared
memory passed between agent nodes in the LangGraph StateGraph. The state
accumulates messages, tracks workflow progress, and carries metadata needed
for routing decisions.

Key Concepts:
    - AgentState: The primary state schema used by the orchestration graph.
    - MessageEntry: Structured message format for inter-agent communication.
    - WorkflowStatus: Enum-like tracking of overall workflow progress.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, TypedDict

from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class WorkflowStatus(str, Enum):
    """Tracks the current phase of the orchestration workflow."""

    PENDING = "pending"
    RESEARCHING = "researching"
    WRITING = "writing"
    REVIEWING = "reviewing"
    REVISING = "revising"
    COMPLETE = "complete"
    FAILED = "failed"


class MessageEntry(BaseModel):
    """Structured message exchanged between agents in the workflow.

    Each agent appends MessageEntry instances to the shared state to communicate
    findings, drafts, and feedback to downstream agents.
    """

    role: str = Field(description="The agent role that produced this message (researcher, writer, reviewer)")
    content: str = Field(description="The message content")
    metadata: dict[str, str] = Field(default_factory=dict, description="Optional metadata (sources, confidence, etc.)")

    model_config = {"frozen": True}


class ResearchFindings(BaseModel):
    """Structured output from the researcher agent."""

    topic: str = Field(description="The research topic")
    key_points: list[str] = Field(default_factory=list, description="Key findings from research")
    sources: list[str] = Field(default_factory=list, description="Sources consulted")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in findings (0-1)")


class ReviewFeedback(BaseModel):
    """Structured feedback from the reviewer agent."""

    approved: bool = Field(default=False, description="Whether the content is approved")
    score: float = Field(default=0.0, ge=0.0, le=10.0, description="Quality score (0-10)")
    issues: list[str] = Field(default_factory=list, description="List of issues found")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")


class AgentState(TypedDict):
    """Primary state schema for the multi-agent orchestration graph.

    This TypedDict defines the shape of state that flows through the LangGraph
    StateGraph. Each key represents a different aspect of the workflow state.

    The `messages` field uses LangGraph's add_messages reducer to automatically
    handle message accumulation across node invocations.

    Attributes:
        messages: Accumulated LLM messages (uses add_messages reducer for automatic merging)
        task: The original task/query that initiated the workflow
        status: Current workflow phase
        research_findings: Structured research output from the researcher agent
        draft_content: Written content produced by the writer agent
        review_feedback: Structured review from the reviewer agent
        revision_count: Number of revision cycles completed
        max_revisions: Maximum allowed revision cycles before forcing completion
        final_output: The approved final content
    """

    messages: Annotated[list, add_messages]
    task: str
    status: WorkflowStatus
    research_findings: ResearchFindings | None
    draft_content: str
    review_feedback: ReviewFeedback | None
    revision_count: int
    max_revisions: int
    final_output: str


def create_initial_state(task: str, max_revisions: int = 3) -> AgentState:
    """Create a fresh initial state for a new orchestration workflow.

    Args:
        task: The task description to be processed by the agent pipeline.
        max_revisions: Maximum revision cycles before auto-completing.

    Returns:
        A fully initialized AgentState ready to be passed to the graph.
    """
    return AgentState(
        messages=[],
        task=task,
        status=WorkflowStatus.PENDING,
        research_findings=None,
        draft_content="",
        review_feedback=None,
        revision_count=0,
        max_revisions=max_revisions,
        final_output="",
    )
