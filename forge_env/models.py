"""ForgeRL Models — OpenEnv-compatible Action, Observation, and State models.

These Pydantic models define the interface between the RL agent and the
ForgeEnvironment. They follow the OpenEnv specification for compatibility
with the Gymnasium-style reset/step/state API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Action Space ──────────────────────────────────────────────────────────────


class ActionType(str, Enum):
    """All possible orchestration actions the meta-agent can take."""

    # Phase transitions — delegate to sub-agents
    DELEGATE_INTAKE = "delegate_intake"
    DELEGATE_ARCHITECT = "delegate_architect"
    DELEGATE_PLANNER = "delegate_planner"
    DELEGATE_QA = "delegate_qa"
    DELEGATE_CODER = "delegate_coder"
    DELEGATE_RECOVERY = "delegate_recovery"
    DELEGATE_SECURITY = "delegate_security"
    DELEGATE_OVERSIGHT = "delegate_oversight"

    # Review decisions
    APPROVE_PLAN = "approve_plan"
    REJECT_PLAN = "reject_plan"
    APPROVE_CODE = "approve_code"
    REJECT_CODE = "reject_code"

    # Feedback and control
    PROVIDE_FEEDBACK = "provide_feedback"
    SKIP_TASK = "skip_task"
    RETRY_TASK = "retry_task"
    ESCALATE = "escalate"

    # Terminal actions
    FINALIZE = "finalize"


class ForgeAction(BaseModel):
    """An action taken by the meta-agent at each step.

    The meta-agent chooses which sub-agent to invoke, what feedback to provide,
    or whether to approve/reject intermediate outputs.
    """

    action_type: ActionType = Field(
        description="The type of orchestration action to perform."
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters (e.g., feedback text, task ID).",
    )
    reasoning: str = Field(
        default="",
        description="Chain-of-thought reasoning for this action choice.",
    )
    target_task_id: Optional[int] = Field(
        default=None,
        description="ID of the specific task this action targets (if applicable).",
    )


# ── Observation Space ─────────────────────────────────────────────────────────


class AgentOutputSummary(BaseModel):
    """Summary of the last sub-agent's output (partially observable)."""

    agent_name: str = ""
    success: bool = False
    message: str = ""
    files_produced: list[str] = Field(default_factory=list)
    error: str = ""
    duration_seconds: float = 0.0


class ProjectSnapshot(BaseModel):
    """Current state of the generated project."""

    files_generated: list[str] = Field(default_factory=list)
    total_files: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    test_pass_rate: float = 0.0
    has_specification: bool = False
    has_architecture: bool = False
    has_plan: bool = False
    total_api_calls: int = 0
    error_count: int = 0


class TaskProgressSummary(BaseModel):
    """Progress of the implementation plan."""

    total_tasks: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    in_progress: int = 0
    pending: int = 0
    percent_complete: float = 0.0
    current_task_title: str = ""
    current_task_id: int = 0


class OversightReport(BaseModel):
    """Quality report from the Oversight Agent."""

    issues_found: int = 0
    critical_issues: int = 0
    quality_score: float = 0.0  # 0.0 - 1.0
    hallucination_flags: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    agent_behavior_summary: str = ""


class ReviewerFeedback(BaseModel):
    """Feedback from the Simulated Expert Reviewer."""

    approved: bool = False
    satisfaction_score: float = 0.0  # 0.0 - 1.0
    feedback_text: str = ""
    preference_changes: list[str] = Field(default_factory=list)
    current_preferences: dict[str, str] = Field(default_factory=dict)


class ForgeObservation(BaseModel):
    """What the meta-agent observes after each step.

    This is intentionally partially observable — the agent sees sub-agent
    outputs but not their internal prompts, LLM reasoning, or raw API responses.
    """

    # Current state
    current_phase: str = "idle"
    step_count: int = 0
    max_steps: int = 300
    episode_reward: float = 0.0

    # Last agent output (partial observability)
    last_agent_output: AgentOutputSummary = Field(
        default_factory=AgentOutputSummary
    )

    # Project state
    project_state: ProjectSnapshot = Field(default_factory=ProjectSnapshot)

    # Task progress
    task_progress: TaskProgressSummary = Field(
        default_factory=TaskProgressSummary
    )

    # Valid actions in current state
    available_actions: list[str] = Field(default_factory=list)

    # Quality signals
    oversight_report: Optional[OversightReport] = None
    reviewer_feedback: Optional[ReviewerFeedback] = None

    # Specification context (what we're building)
    spec_summary: str = ""
    spec_tier: int = 1

    # Phase history for the agent to track trajectory
    phase_history: list[str] = Field(default_factory=list)

    # Current error context (if in recovery)
    error_context: str = ""


# ── Environment State (Internal Metadata) ─────────────────────────────────────


class ForgeState(BaseModel):
    """Internal environment metadata exposed via the state() API.

    Unlike ForgeObservation, this contains full ground-truth information
    about the environment's internal state — used for debugging and evaluation.
    """

    episode_id: str = ""
    step_count: int = 0
    total_reward: float = 0.0
    spec_tier: int = 1
    spec_text: str = ""
    is_terminal: bool = False
    termination_reason: str = ""

    # Ground-truth metrics
    true_test_pass_rate: float = 0.0
    true_code_quality_score: float = 0.0
    true_files_generated: int = 0
    total_api_calls: int = 0
    total_tokens_used: int = 0

    # Timing
    started_at: Optional[datetime] = None
    elapsed_seconds: float = 0.0

    # Episode history (full action-observation trace)
    action_history: list[dict] = Field(default_factory=list)
    reward_history: list[float] = Field(default_factory=list)
    phase_trace: list[str] = Field(default_factory=list)


# ── OpenEnv Result Types ──────────────────────────────────────────────────────


class ResetResult(BaseModel):
    """Result returned by environment.reset()."""

    observation: ForgeObservation
    info: dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    """Result returned by environment.step()."""

    observation: ForgeObservation
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = Field(default_factory=dict)
