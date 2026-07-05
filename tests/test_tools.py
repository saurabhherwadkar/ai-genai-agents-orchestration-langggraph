"""Tests for custom agent tools.

Validates tool functionality, error handling, and edge cases for
the tools provided to agents in the orchestration framework.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from orchestration.tools import (
    get_agent_tools,
    read_file,
    summarize_text,
    web_search,
    write_file,
)


class TestWebSearch:
    """Tests for the web_search tool."""

    def test_tool_has_correct_name(self) -> None:
        """Tool should be registered with the correct name."""
        assert web_search.name == "web_search"

    def test_tool_has_description(self) -> None:
        """Tool should have a non-empty description."""
        assert web_search.description
        assert len(web_search.description) > 10

    def test_returns_string_without_api_key(self) -> None:
        """Should return informative message when no API key is configured."""
        result = web_search.invoke({"query": "test query", "max_results": 3})
        assert isinstance(result, str)
        # Without a real API key, should get an error or no-key message
        assert "query" in result.lower() or "error" in result.lower() or "api" in result.lower()


class TestReadFile:
    """Tests for the read_file tool."""

    def test_read_existing_file(self) -> None:
        """Should successfully read an existing file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Hello, LangGraph!")
            temp_path = f.name

        try:
            result = read_file.invoke({"file_path": temp_path})
            assert "Hello, LangGraph!" in result
            assert "[File:" in result
        finally:
            Path(temp_path).unlink()

    def test_read_nonexistent_file(self) -> None:
        """Should return error message for missing files."""
        result = read_file.invoke({"file_path": "/nonexistent/path/file.txt"})
        assert "not found" in result.lower() or "error" in result.lower()

    def test_read_directory_not_file(self) -> None:
        """Should return error when path is a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = read_file.invoke({"file_path": tmpdir})
            assert "not a file" in result.lower() or "error" in result.lower()

    def test_read_file_size_limit(self) -> None:
        """Should reject files larger than the size limit."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            # Write more than 100KB
            f.write("x" * (101 * 1024))
            temp_path = f.name

        try:
            result = read_file.invoke({"file_path": temp_path})
            assert "too large" in result.lower()
        finally:
            Path(temp_path).unlink()


class TestWriteFile:
    """Tests for the write_file tool."""

    def test_write_new_file(self) -> None:
        """Should successfully write content to a new file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = str(Path(tmpdir) / "output.txt")
            result = write_file.invoke({"file_path": file_path, "content": "Test content"})

            assert "successfully" in result.lower()
            assert Path(file_path).read_text() == "Test content"

    def test_write_creates_parent_dirs(self) -> None:
        """Should create parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = str(Path(tmpdir) / "nested" / "deep" / "file.txt")
            result = write_file.invoke({"file_path": file_path, "content": "Nested!"})

            assert "successfully" in result.lower()
            assert Path(file_path).exists()
            assert Path(file_path).read_text() == "Nested!"

    def test_write_overwrites_existing(self) -> None:
        """Should overwrite existing file content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Original")
            temp_path = f.name

        try:
            write_file.invoke({"file_path": temp_path, "content": "Updated"})
            assert Path(temp_path).read_text() == "Updated"
        finally:
            Path(temp_path).unlink()


class TestSummarizeText:
    """Tests for the summarize_text tool."""

    def test_summarize_normal_text(self) -> None:
        """Should extract key points from normal text."""
        text = (
            "LangGraph is a framework for building stateful agents. "
            "It extends LangChain with graph-based orchestration. "
            "Nodes represent processing steps in the workflow. "
            "Edges define transitions between nodes. "
            "Conditional edges enable dynamic routing based on state."
        )
        result = summarize_text.invoke({"text": text, "max_points": 3})
        assert "[Summary" in result
        assert "key points" in result.lower() or "Summary" in result

    def test_summarize_empty_text(self) -> None:
        """Should handle empty text gracefully."""
        result = summarize_text.invoke({"text": "", "max_points": 3})
        assert "empty" in result.lower() or "Empty" in result

    def test_summarize_short_text(self) -> None:
        """Should handle text shorter than requested points."""
        result = summarize_text.invoke({"text": "Just one short sentence here.", "max_points": 5})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_respects_max_points(self) -> None:
        """Should not exceed the requested number of key points."""
        text = ". ".join([f"This is sentence number {i} with enough content to qualify" for i in range(20)])
        result = summarize_text.invoke({"text": text, "max_points": 3})
        # Count bullet points in output
        bullet_count = result.count("\n-")
        assert bullet_count <= 3


class TestGetAgentTools:
    """Tests for the tool registry function."""

    def test_returns_all_tools(self) -> None:
        """Should return all registered tools."""
        tools = get_agent_tools()
        assert len(tools) == 4

    def test_tools_have_names(self) -> None:
        """All tools should have valid names."""
        tools = get_agent_tools()
        names = {t.name for t in tools}
        expected = {"web_search", "read_file", "write_file", "summarize_text"}
        assert names == expected

    def test_tools_are_callable(self) -> None:
        """All tools should be callable (have invoke method)."""
        tools = get_agent_tools()
        for tool in tools:
            assert hasattr(tool, "invoke")
