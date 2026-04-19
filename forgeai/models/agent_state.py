"""Agent State & Context — The data passed between agents during orchestration."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from forgeai.models.specification import StructuredSpecification
from forgeai.models.task import AtomicTask, ImplementationPlan


class AgentRole(str, Enum):
    INTAKE = "intake"
    ARCHITECT = "architect"
    PLANNER = "planner"
    QA = "qa"
    CODER = "coder"
    SECURITY = "security"
    RECOVERY = "recovery"
    OVERSIGHT = "oversight"      # Fleet AI: monitors and explains agent behavior
    REVIEWER = "reviewer"        # Snorkel AI: simulated expert-in-the-loop


class AgentContext(BaseModel):
    """Context provided to an agent when it is invoked.
    
    Contains everything the agent needs: the specification, current project 
    state, current task (if applicable), and accumulated conversation history.
    """
    role: AgentRole
    specification: Optional[StructuredSpecification] = None
    architecture: Optional[dict] = None
    implementation_plan: Optional[ImplementationPlan] = None
    current_task: Optional[AtomicTask] = None

    # File system state
    project_dir: str = ""
    existing_files: dict[str, str] = Field(default_factory=dict)  # path -> content

    # Error context for recovery
    error_message: str = ""
    error_traceback: str = ""
    previous_attempts: list[str] = Field(default_factory=list)

    # User interaction
    user_input: str = ""
    clarification_responses: dict[str, str] = Field(default_factory=dict)

    # Metadata
    retry_count: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)


class AgentResult(BaseModel):
    """Result returned by an agent after execution."""
    success: bool = False
    role: AgentRole = AgentRole.INTAKE
    
    # Output data — agents populate the relevant fields
    specification: Optional[StructuredSpecification] = None
    architecture: Optional[dict] = None
    implementation_plan: Optional[ImplementationPlan] = None
    generated_files: dict[str, str] = Field(default_factory=dict)  # path -> content
    test_results: Optional[dict] = None
    security_report: Optional[dict] = None

    # Communication
    message: str = ""
    clarifying_questions: list[str] = Field(default_factory=list)
    requires_human_input: bool = False
    
    # Diagnostics
    error: str = ""
    api_calls_made: int = 0
    tokens_used: int = 0
    duration_seconds: float = 0.0
