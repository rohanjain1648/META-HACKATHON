"""Configuration Manager — Loads, validates, and provides access to YAML config.

Satisfies NFR-03: All settings configurable via a single YAML file without code changes.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "google"
    model: str = "gemini-2.5-flash"
    temperature: float = 0.2
    max_tokens: int = 8192
    api_key_env: str = "GOOGLE_API_KEY"


class GuardrailsConfig(BaseModel):
    max_files_per_task: int = 8
    max_lines_per_file: int = 600
    blocked_commands: list[str] = Field(default_factory=lambda: ["rm -rf /", "del /s /q C:\\", "format", "mkfs"])
    require_approval_for: list[str] = Field(default_factory=lambda: ["database_schema_changes", "security_sensitive_patterns"])


class WorkflowConfig(BaseModel):
    max_retries: int = 3
    retry_delay_seconds: int = 2
    checkpoints: list[str] = Field(default_factory=lambda: ["after_specification", "after_architecture", "after_plan"])
    auto_approve_checkpoints: bool = False


class OutputConfig(BaseModel):
    project_dir: str = "./generated_project"
    log_file: str = "./forgeai_activity.log"
    spec_format: str = "yaml"
    enable_git: bool = True


class TestingConfig(BaseModel):
    framework: str = "pytest"
    timeout_seconds: int = 60
    coverage_threshold: int = 70


class DockerConfig(BaseModel):
    enabled: bool = False
    base_image: str = "python:3.11-slim"


class SecurityAuditConfig(BaseModel):
    enabled: bool = True
    scan_patterns: list[str] = Field(default_factory=lambda: [
        "hardcoded_secrets", "sql_injection", "command_injection",
        "path_traversal", "auth_bypass"
    ])


class WebDashboardConfig(BaseModel):
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8000


class ForgeAIConfig(BaseModel):
    """Root configuration model."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    testing: TestingConfig = Field(default_factory=TestingConfig)
    docker: DockerConfig = Field(default_factory=DockerConfig)
    security_audit: SecurityAuditConfig = Field(default_factory=SecurityAuditConfig)
    web_dashboard: WebDashboardConfig = Field(default_factory=WebDashboardConfig)


class ConfigManager:
    """Loads and manages ForgeAI configuration from YAML files."""

    _instance: Optional["ConfigManager"] = None
    _config: ForgeAIConfig

    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self._config = self._load_from_file(config_path)
        else:
            default_path = Path(__file__).parent / "default_config.yaml"
            self._config = self._load_from_file(str(default_path))

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> "ConfigManager":
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton (used in testing)."""
        cls._instance = None

    def _load_from_file(self, path: str) -> ForgeAIConfig:
        """Load configuration from a YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        return ForgeAIConfig(**raw)

    @property
    def config(self) -> ForgeAIConfig:
        return self._config

    @property
    def llm(self) -> LLMConfig:
        return self._config.llm

    @property
    def guardrails(self) -> GuardrailsConfig:
        return self._config.guardrails

    @property
    def workflow(self) -> WorkflowConfig:
        return self._config.workflow

    @property
    def output(self) -> OutputConfig:
        return self._config.output

    @property
    def testing(self) -> TestingConfig:
        return self._config.testing

    @property
    def docker(self) -> DockerConfig:
        return self._config.docker

    @property
    def security_audit(self) -> SecurityAuditConfig:
        return self._config.security_audit

    @property
    def web_dashboard(self) -> WebDashboardConfig:
        return self._config.web_dashboard

    def get_api_key(self) -> str:
        """Retrieve the LLM API key from the environment variable."""
        key = os.environ.get(self._config.llm.api_key_env, "")
        if not key:
            raise ValueError(
                f"API key not found. Set the '{self._config.llm.api_key_env}' "
                f"environment variable."
            )
        return key
