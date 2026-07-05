"""Agent node definitions for the multi-agent orchestration graph.

Each agent is implemented as a function that takes the current AgentState and
returns a partial state update. Agents use LangChain's ChatOpenAI with structured
prompts and optional tool bindings.

Agent Roles:
    - Researcher: Gathers information and produces structured findings
    - Writer: Transforms research findings into polished content
    - Reviewer: Evaluates written content and provides feedback/approval

Pattern Notes:
    - Each agent function is a LangGraph "node" — it receives state, returns updates
    - Agents communicate through the shared state, not directly
    - Tool usage is bound per-agent (researcher gets search, writer gets file tools)
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from orchestration.config import get_config
from orchestration.state import (
    AgentState,
    ResearchFindings,
    ReviewFeedback,
    WorkflowStatus,
)
from orchestration.tools import get_agent_tools


def _get_research_tools() -> list:
    """Get tools available to the researcher agent."""
    tools = get_agent_tools()
    # Researcher gets web_search and summarize_text
    return [t for t in tools if t.name in ("web_search", "summarize_text")]


def _get_writer_tools() -> list:
    """Get tools available to the writer agent."""
    tools = get_agent_tools()
    # Writer gets read_file and write_file
    return [t for t in tools if t.name in ("read_file", "write_file")]


RESEARCHER_SYSTEM_PROMPT = """You are a Research Agent in a multi-agent content pipeline.

Your role is to gather comprehensive, accurate information on the given topic.
You should:
1. Identify the key aspects of the topic that need research
2. Use available tools to search for current information
3. Synthesize findings into structured key points with sources
4. Assess your confidence level in the findings

Be thorough but concise. Focus on factual, verifiable information.
Always cite your sources when possible."""

WRITER_SYSTEM_PROMPT = """You are a Writer Agent in a multi-agent content pipeline.

Your role is to transform research findings into polished, well-structured content.
You should:
1. Review the research findings provided
2. Organize information into a logical flow
3. Write clear, engaging content appropriate for the target audience
4. Incorporate all key points from the research

If you receive revision feedback, carefully address each issue raised by the reviewer.
Maintain accuracy while improving readability and engagement."""

REVIEWER_SYSTEM_PROMPT = """You are a Reviewer Agent in a multi-agent content pipeline.

Your role is to critically evaluate written content for quality, accuracy, and completeness.
You should assess:
1. Factual accuracy against the research findings
2. Clarity and readability of the writing
3. Completeness — are all key research points covered?
4. Structure and logical flow
5. Grammar and style

Provide a score from 0-10 and specific, actionable feedback.
Only approve content (score >= 7) if it meets quality standards.
If not approved, list specific issues and suggestions for improvement."""


async def researcher_node(state: AgentState) -> dict:
    """Researcher agent node — gathers information on the given task.

    This node invokes an LLM with research-focused tools to gather
    information about the task topic. It produces structured ResearchFindings
    that downstream agents use.

    Args:
        state: Current workflow state containing the task description.

    Returns:
        Partial state update with research_findings and updated status.
    """
    config = get_config()
    llm = config.create_llm(temperature=0.3)

    # Bind research tools to the LLM
    research_tools = _get_research_tools()
    if research_tools:
        llm_with_tools = llm.bind_tools(research_tools)
    else:
        llm_with_tools = llm

    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT),
        HumanMessage(content=f"""Research the following topic thoroughly:

Topic: {state['task']}

Provide your findings as structured key points. Include sources where possible.
Rate your confidence in the findings from 0 to 1."""),
    ]

    response = await llm_with_tools.ainvoke(messages)

    # Parse response into structured findings
    findings = ResearchFindings(
        topic=state["task"],
        key_points=_extract_key_points(response.content if isinstance(response.content, str) else ""),
        sources=["LLM-generated research"],
        confidence=0.75,
    )

    return {
        "messages": [response],
        "status": WorkflowStatus.RESEARCHING,
        "research_findings": findings,
    }


async def writer_node(state: AgentState) -> dict:
    """Writer agent node — produces content based on research findings.

    Takes the structured research findings and transforms them into
    polished written content. Handles both initial drafting and revisions
    based on reviewer feedback.

    Args:
        state: Current workflow state with research findings and optional feedback.

    Returns:
        Partial state update with draft_content and updated status.
    """
    config = get_config()
    llm = config.create_llm(temperature=0.7)

    # Build context from research findings
    findings = state.get("research_findings")
    findings_text = ""
    if findings:
        points = "\n".join(f"- {p}" for p in findings.key_points)
        findings_text = f"Research Findings on '{findings.topic}':\n{points}"

    # Include revision feedback if this is a revision cycle
    feedback_text = ""
    review = state.get("review_feedback")
    if review and not review.approved:
        issues = "\n".join(f"- {i}" for i in review.issues)
        suggestions = "\n".join(f"- {s}" for s in review.suggestions)
        feedback_text = f"""

REVISION REQUIRED (Score: {review.score}/10):
Issues to address:
{issues}

Suggestions:
{suggestions}

Previous draft to revise:
{state.get('draft_content', '')}"""

    messages = [
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        HumanMessage(content=f"""{findings_text}{feedback_text}

Please write comprehensive, well-structured content on this topic.
{"This is a revision — address all reviewer feedback while maintaining quality." if feedback_text else "Create an initial draft based on the research findings."}"""),
    ]

    response = await llm.ainvoke(messages)
    content = response.content if isinstance(response.content, str) else ""

    return {
        "messages": [response],
        "status": WorkflowStatus.WRITING,
        "draft_content": content,
    }


async def reviewer_node(state: AgentState) -> dict:
    """Reviewer agent node — evaluates content quality and provides feedback.

    Critically assesses the written content against the original research
    findings. Produces a structured ReviewFeedback with approval status,
    score, issues, and suggestions.

    Args:
        state: Current workflow state with draft content and research findings.

    Returns:
        Partial state update with review_feedback, revision_count, and status.
    """
    config = get_config()
    llm = config.create_llm(temperature=0.2)

    draft = state.get("draft_content", "")
    findings = state.get("research_findings")

    findings_context = ""
    if findings:
        points = "\n".join(f"- {p}" for p in findings.key_points)
        findings_context = f"Original Research Points:\n{points}\n\n"

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(content=f"""{findings_context}Content to Review:
{draft}

Evaluate this content. Provide:
1. A quality score from 0-10
2. Whether you approve it (score >= 7 means approved)
3. Specific issues found (if any)
4. Suggestions for improvement (if any)"""),
    ]

    response = await llm.ainvoke(messages)
    response_text = response.content if isinstance(response.content, str) else ""

    # Parse review response into structured feedback
    score = _extract_score(response_text)
    approved = score >= config.review_threshold

    feedback = ReviewFeedback(
        approved=approved,
        score=score,
        issues=_extract_list_items(response_text, "issues"),
        suggestions=_extract_list_items(response_text, "suggestions"),
    )

    new_revision_count = state.get("revision_count", 0) + 1

    return {
        "messages": [response],
        "status": WorkflowStatus.REVIEWING,
        "review_feedback": feedback,
        "revision_count": new_revision_count,
    }


async def finalizer_node(state: AgentState) -> dict:
    """Finalizer node — marks the workflow as complete with approved content.

    This terminal node copies the approved draft content to final_output
    and sets the workflow status to COMPLETE.

    Args:
        state: Current workflow state with approved draft content.

    Returns:
        Partial state update with final_output and COMPLETE status.
    """
    return {
        "messages": [AIMessage(content="Workflow complete. Content has been approved.")],
        "status": WorkflowStatus.COMPLETE,
        "final_output": state.get("draft_content", ""),
    }


# --- Helper Functions ---


def _extract_key_points(text: str) -> list[str]:
    """Extract bullet points or numbered items from LLM response text."""
    points: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(("-", "*", "•")) and len(line) > 3:
            points.append(line.lstrip("-*• ").strip())
        elif len(line) > 2 and line[0].isdigit() and line[1] in (".", ")"):
            points.append(line[2:].strip())
    return points if points else [text[:200]] if text else ["No findings generated"]


def _extract_score(text: str) -> float:
    """Extract a numerical score from reviewer response text."""
    import re

    # Look for patterns like "Score: 7/10", "7/10", "score: 8"
    patterns = [
        r"[Ss]core[:\s]+(\d+(?:\.\d+)?)\s*/\s*10",
        r"(\d+(?:\.\d+)?)\s*/\s*10",
        r"[Ss]core[:\s]+(\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            score = float(match.group(1))
            return min(score, 10.0)
    return 5.0  # Default middle score if parsing fails


def _extract_list_items(text: str, section: str) -> list[str]:
    """Extract list items from a section of the reviewer's response."""
    items: list[str] = []
    in_section = False

    for line in text.split("\n"):
        lower_line = line.lower().strip()

        # Check if we're entering the target section
        if section.lower() in lower_line and (":" in lower_line or lower_line.endswith(section.lower())):
            in_section = True
            continue

        # Check if we're leaving the section (new header)
        if in_section and lower_line and not lower_line.startswith(("-", "*", "•", " ")) and ":" in lower_line:
            in_section = False
            continue

        # Collect items in the section
        if in_section:
            cleaned = line.strip().lstrip("-*• ").strip()
            if cleaned and len(cleaned) > 3:
                items.append(cleaned)

    return items
