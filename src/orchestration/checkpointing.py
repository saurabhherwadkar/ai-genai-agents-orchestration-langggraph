"""Checkpoint and persistence management for graph state.

Implements checkpointing strategies that allow the orchestration graph to
persist its state between invocations. This enables:
    - Fault tolerance: resume from the last successful node on failure
    - Human-in-the-loop: pause for human approval, then resume
    - Debugging: inspect state at any point in the workflow
    - Long-running workflows: save progress across sessions

LangGraph supports multiple checkpointing backends. This module provides
factory functions to create the appropriate checkpointer based on config.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver


def create_checkpointer(backend: str = "memory") -> MemorySaver:
    """Create a checkpointer instance based on the specified backend.

    Currently supports in-memory checkpointing for development and testing.
    Production deployments would use persistent backends like SQLite or PostgreSQL.

    Args:
        backend: The checkpointing backend to use. Options:
            - "memory": In-memory storage (default, good for dev/test)

    Returns:
        A configured checkpointer instance compatible with LangGraph.

    Example:
        >>> checkpointer = create_checkpointer("memory")
        >>> graph = workflow.compile(checkpointer=checkpointer)
    """
    if backend == "memory":
        return MemorySaver()
    else:
        # Default to memory saver for unsupported backends
        return MemorySaver()


def ensure_checkpoint_directory(checkpoint_dir: str = ".checkpoints") -> Path:
    """Ensure the checkpoint directory exists.

    Creates the directory structure needed for file-based checkpointing.

    Args:
        checkpoint_dir: Path to the checkpoint directory.

    Returns:
        The Path object for the checkpoint directory.
    """
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


class CheckpointManager:
    """High-level manager for checkpoint operations.

    Provides a convenient interface for common checkpointing operations
    like listing available checkpoints, resuming from a specific point,
    and cleaning up old checkpoints.

    Attributes:
        checkpointer: The underlying LangGraph checkpointer instance.
        checkpoint_dir: Directory for any file-based storage.
    """

    def __init__(self, backend: str = "memory", checkpoint_dir: str = ".checkpoints") -> None:
        """Initialize the checkpoint manager.

        Args:
            backend: Checkpointing backend type.
            checkpoint_dir: Directory for file-based storage.
        """
        self.checkpointer = create_checkpointer(backend)
        self.checkpoint_dir = ensure_checkpoint_directory(checkpoint_dir)

    def get_checkpointer(self) -> MemorySaver:
        """Get the configured checkpointer instance.

        Returns:
            The MemorySaver (or other backend) instance.
        """
        return self.checkpointer

    def create_thread_config(self, thread_id: str) -> dict:
        """Create a configuration dict for a specific thread/conversation.

        LangGraph uses thread IDs to isolate checkpoint streams for
        different workflow instances.

        Args:
            thread_id: Unique identifier for this workflow thread.

        Returns:
            Configuration dict with the thread ID set.
        """
        return {"configurable": {"thread_id": thread_id}}

    def list_checkpoint_files(self) -> list[Path]:
        """List all checkpoint files in the checkpoint directory.

        Returns:
            List of Path objects for checkpoint files.
        """
        if self.checkpoint_dir.exists():
            return sorted(self.checkpoint_dir.glob("*.json"))
        return []

    def cleanup_old_checkpoints(self, keep_latest: int = 5) -> int:
        """Remove old checkpoint files, keeping only the most recent.

        Args:
            keep_latest: Number of recent checkpoints to keep.

        Returns:
            Number of checkpoint files removed.
        """
        files = self.list_checkpoint_files()
        if len(files) <= keep_latest:
            return 0

        to_remove = files[:-keep_latest]
        for f in to_remove:
            f.unlink()
        return len(to_remove)
