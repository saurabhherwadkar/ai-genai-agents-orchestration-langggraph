"""Custom tools for LangGraph agents.

Defines tool functions that agents can invoke during execution. Tools follow
the LangChain tool abstraction pattern using the @tool decorator, making them
compatible with OpenAI function calling and LangGraph tool nodes.

Available Tools:
    - web_search: Search the web for current information using Tavily
    - read_file: Read contents of a local file
    - write_file: Write content to a local file
    - summarize_text: Summarize long text into key points
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information on a given query.

    Uses the Tavily search API to find current, relevant information.
    Returns formatted search results with titles, URLs, and snippets.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default: 5).

    Returns:
        Formatted string of search results.
    """
    try:
        from tavily import TavilyClient

        from orchestration.config import get_config

        config = get_config()
        if not config.tavily_api_key:
            return f"[Web Search] No Tavily API key configured. Query was: {query}"

        client = TavilyClient(api_key=config.tavily_api_key)
        response = client.search(query=query, max_results=max_results)

        results: list[str] = []
        for item in response.get("results", []):
            title = item.get("title", "No title")
            url = item.get("url", "")
            content = item.get("content", "")[:200]
            results.append(f"- **{title}**\n  URL: {url}\n  {content}")

        if not results:
            return f"[Web Search] No results found for: {query}"

        return f"[Web Search Results for '{query}']\n\n" + "\n\n".join(results)

    except ImportError:
        return f"[Web Search] Tavily client not installed. Query was: {query}"
    except Exception as e:
        return f"[Web Search] Error searching for '{query}': {e}"


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a local file.

    Safely reads a file with size limits and encoding handling.

    Args:
        file_path: Path to the file to read.

    Returns:
        The file contents as a string, or an error message.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"[Read File] File not found: {file_path}"
        if not path.is_file():
            return f"[Read File] Path is not a file: {file_path}"

        # Safety: limit file size to 100KB
        file_size = path.stat().st_size
        max_size = 100 * 1024  # 100KB
        if file_size > max_size:
            return f"[Read File] File too large ({file_size} bytes, max {max_size}): {file_path}"

        content = path.read_text(encoding="utf-8")
        return f"[File: {file_path}]\n{content}"

    except UnicodeDecodeError:
        return f"[Read File] Cannot read binary file: {file_path}"
    except PermissionError:
        return f"[Read File] Permission denied: {file_path}"
    except Exception as e:
        return f"[Read File] Error reading '{file_path}': {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a local file.

    Creates parent directories if they don't exist. Overwrites existing files.

    Args:
        file_path: Path where the file should be written.
        content: The content to write to the file.

    Returns:
        A success or error message.
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"[Write File] Successfully wrote {len(content)} characters to: {file_path}"

    except PermissionError:
        return f"[Write File] Permission denied: {file_path}"
    except Exception as e:
        return f"[Write File] Error writing to '{file_path}': {e}"


@tool
def summarize_text(text: str, max_points: int = 5) -> str:
    """Extract key points from a longer text.

    Performs simple extractive summarization by splitting text into sentences
    and selecting the most informative ones. For production use, this would
    delegate to an LLM for abstractive summarization.

    Args:
        text: The text to summarize.
        max_points: Maximum number of key points to extract.

    Returns:
        A bullet-point summary of the text.
    """
    if not text.strip():
        return "[Summarize] Empty text provided."

    # Simple sentence-based extraction (educational; production would use LLM)
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]

    if not sentences:
        return f"[Summary]\n- {text[:200]}"

    # Select sentences spread across the text for coverage
    step = max(1, len(sentences) // max_points)
    selected = sentences[::step][:max_points]

    points = "\n".join(f"- {s.strip()}." for s in selected)
    return f"[Summary ({len(selected)} key points)]\n{points}"


def get_agent_tools() -> list:
    """Return the list of all tools available to agents.

    Returns:
        List of tool functions decorated with @tool.
    """
    return [web_search, read_file, write_file, summarize_text]
