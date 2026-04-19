"""Workflow State — Tracks the overall state of the ForgeAI pipeline."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from forgeai.models.specification import StructuredSpecification
from forgeai.models.task import ImplementationPlan


class WorkflowPhase(str, Enum):
    """Finite state machine phases for the orchestration pipeline."""
    IDLE = "idle"
    INTAKE = "intake"
    CLARIFICATION = "clarification"
    SPECIFICATION = "specification"
    ARCHITECTURE = "architecture"
    PLANNING = "planning"
    PLAN_REVIEW = "plan_review"          # Checkpoint: user approves plan
    EXECUTION = "execution"              # TDD loop over tasks
    TASK_QA = "task_qa"                  # QA writing tests for current task
    TASK_CODE = "task_code"              # Coder generating code
    TASK_TEST = "task_test"              # Running tests
    TASK_RECOVERY = "task_recovery"      # Recovery on failure
    OVERSIGHT = "oversight"              # Fleet AI: quality monitoring
    REVIEW = "review"                    # Snorkel AI: expert reviewer
    SECURITY_AUDIT = "security_audit"    # Extended: post-module scan
    SUMMARY = "summary"                  # Generating final report
    DONE = "done"
    ERROR = "error"


# Valid state transitions
VALID_TRANSITIONS: dict[WorkflowPhase, list[WorkflowPhase]] = {
    WorkflowPhase.IDLE: [WorkflowPhase.INTAKE],
    WorkflowPhase.INTAKE: [WorkflowPhase.CLARIFICATION, WorkflowPhase.SPECIFICATION],
    WorkflowPhase.CLARIFICATION: [WorkflowPhase.SPECIFICATION, WorkflowPhase.INTAKE],
    WorkflowPhase.SPECIFICATION: [WorkflowPhase.ARCHITECTURE, WorkflowPhase.OVERSIGHT],
    WorkflowPhase.ARCHITECTURE: [WorkflowPhase.PLANNING, WorkflowPhase.OVERSIGHT],
    WorkflowPhase.PLANNING: [WorkflowPhase.PLAN_REVIEW, WorkflowPhase.OVERSIGHT],
    WorkflowPhase.PLAN_REVIEW: [WorkflowPhase.EXECUTION, WorkflowPhase.PLANNING, WorkflowPhase.REVIEW],
    WorkflowPhase.EXECUTION: [WorkflowPhase.TASK_QA, WorkflowPhase.SECURITY_AUDIT, WorkflowPhase.SUMMARY, WorkflowPhase.OVERSIGHT, WorkflowPhase.REVIEW],
    WorkflowPhase.TASK_QA: [WorkflowPhase.TASK_CODE, WorkflowPhase.TASK_RECOVERY],
    WorkflowPhase.TASK_CODE: [WorkflowPhase.TASK_TEST, WorkflowPhase.TASK_RECOVERY, WorkflowPhase.REVIEW],
    WorkflowPhase.TASK_TEST: [WorkflowPhase.EXECUTION, WorkflowPhase.TASK_RECOVERY],
    WorkflowPhase.TASK_RECOVERY: [WorkflowPhase.TASK_QA, WorkflowPhase.TASK_CODE, WorkflowPhase.EXECUTION, WorkflowPhase.ERROR],
    WorkflowPhase.OVERSIGHT: [WorkflowPhase.EXECUTION, WorkflowPhase.SPECIFICATION, WorkflowPhase.ARCHITECTURE, WorkflowPhase.PLANNING, WorkflowPhase.SECURITY_AUDIT],
    WorkflowPhase.REVIEW: [WorkflowPhase.EXECUTION, WorkflowPhase.TASK_CODE, WorkflowPhase.PLANNING],
    WorkflowPhase.SECURITY_AUDIT: [WorkflowPhase.SUMMARY, WorkflowPhase.OVERSIGHT],
    WorkflowPhase.SUMMARY: [WorkflowPhase.DONE],
    WorkflowPhase.DONE: [],
    WorkflowPhase.ERROR: [WorkflowPhase.IDLE],
}


class WorkflowState(BaseModel):
    """Persistent state of the entire ForgeAI workflow."""
    phase: WorkflowPhase = WorkflowPhase.IDLE
    
    # Accumulated artifacts
    raw_specification: str = ""
    specification: Optional[StructuredSpecification] = None
    architecture: Optional[dict] = None
    implementation_plan: Optional[ImplementationPlan] = None
    
    # Execution tracking
    current_task_index: int = 0
    files_generated: list[str] = Field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0
    total_api_calls: int = 0
    total_tokens_used: int = 0
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Error tracking
    errors: list[str] = Field(default_factory=list)

    # ForgeRL: Oversight and reviewer tracking
    oversight_reports: list[dict] = Field(default_factory=list)
    review_feedback: list[dict] = Field(default_factory=list)
    
    def transition_to(self, new_phase: WorkflowPhase) -> bool:
        """Attempt to transition to a new phase. Returns True if valid."""
        valid = VALID_TRANSITIONS.get(self.phase, [])
        if new_phase in valid:
            self.phase = new_phase
            return True
        return False

    def get_summary(self) -> dict:
        """Generate the workflow summary report (NFR-06)."""
        plan_progress = self.implementation_plan.get_progress() if self.implementation_plan else {}
        return {
            "status": self.phase.value,
            "tasks_completed": plan_progress.get("passed", 0),
            "tasks_skipped": plan_progress.get("skipped", 0),
            "tasks_failed": plan_progress.get("failed", 0),
            "files_generated": len(self.files_generated),
            "file_list": self.files_generated,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "total_api_calls": self.total_api_calls,
            "total_tokens_used": self.total_tokens_used,
            "duration_seconds": (
                (self.completed_at - self.started_at).total_seconds()
                if self.started_at and self.completed_at else 0
            ),
            "errors": self.errors,
        }
