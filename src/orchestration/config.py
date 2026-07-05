"""Configuration management for the orchestration framework.

Centralizes all settings, API keys, and model parameters needed by the
multi-agent system. Uses Pydantic Settings for validation and environment
variable loading.

Usage:
    config = OrchestratorConfig()  # loads from .env and environment
    llm = config.create_llm()
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI
from pydantic import Field
from pydantic_settings import BaseSettings


class OrchestratorConfig(BaseSettings):
    """Central configuration for the orchestration framework.

    Loads settings from environment variables and .env files. All sensitive
    values (API keys) are read from the environment rather than hardcoded.

    Environment Variables:
        OPENAI_API_KEY: OpenAI API key for LLM calls
        TAVILY_API_KEY: Tavily API key for web search tool
        LANGSMITH_API_KEY: LangSmith API key for tracing (optional)
        LANGSMITH_PROJECT: LangSmith project name (optional)
    """

    # API Keys
    openai_api_key: str = Field(default="", description="OpenAI API key")
    tavily_api_key: str = Field(default="", description="Tavily API key for web search")
    langsmith_api_key: str = Field(default="", description="LangSmith API key for tracing")
    langsmith_project: str = Field(default="langgraph-orchestration", description="LangSmith project name")

    # Model Configuration
    model_name: str = Field(default="gpt-4o", description="Primary LLM model name")
    model_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    model_max_tokens: int = Field(default=4096, gt=0, description="Maximum tokens per LLM call")

    # Workflow Configuration
    max_revisions: int = Field(default=3, ge=1, le=10, description="Maximum revision cycles")
    research_depth: int = Field(default=3, ge=1, le=10, description="Number of research queries to run")
    review_threshold: float = Field(default=7.0, ge=0.0, le=10.0, description="Minimum score for approval")

    # Checkpointing
    checkpoint_dir: str = Field(default=".checkpoints", description="Directory for checkpoint storage")
    enable_checkpointing: bool = Field(default=True, description="Whether to enable state checkpointing")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def create_llm(self, temperature: float | None = None) -> ChatOpenAI:
        """Create a configured ChatOpenAI instance.

        Args:
            temperature: Override temperature (uses config default if None).

        Returns:
            A configured ChatOpenAI instance ready for use in agents.
        """
        return ChatOpenAI(
            model=self.model_name,
            temperature=temperature if temperature is not None else self.model_temperature,
            max_tokens=self.model_max_tokens,
            api_key=self.openai_api_key,  # type: ignore[arg-type]
        )


@lru_cache(maxsize=1)
def get_config() -> OrchestratorConfig:
    """Get the singleton configuration instance.

    Returns:
        The cached OrchestratorConfig instance.
    """
    return OrchestratorConfig()
