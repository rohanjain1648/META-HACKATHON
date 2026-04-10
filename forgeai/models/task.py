"""Atomic Task & Implementation Plan models.

An atomic task is a unit of work that produces a single, independently
verifiable change to the codebase (FR-05).
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    TESTS_WRITTEN = "tests_written"
    CODE_GENERATED = "code_generated"
    TESTING = "testing"
    PASSED = "passed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AtomicTask(BaseModel):
    """A single unit of work in the implementation plan."""
    id: int = 0
    title: str = ""
    description: str = ""
    target_files: list[str] = Field(default_factory=list)
    dependencies: list[int] = Field(default_factory=list)  # IDs of prerequisite tasks
    risk_level: RiskLevel = RiskLevel.LOW
    estimated_complexity: str = "simple"  # simple | moderate | complex
    status: TaskStatus = TaskStatus.PENDING
    is_checkpoint: bool = False  # Requires human approval after completion

    # Artifacts produced during execution
    test_files: list[str] = Field(default_factory=list)
    generated_code: dict[str, str] = Field(default_factory=dict)  # filename -> code
    test_results: Optional[dict] = None
    error_log: list[str] = Field(default_factory=list)
    retry_count: int = 0


class ImplementationPlan(BaseModel):
    """Ordered list of atomic tasks forming the complete implementation plan."""
    project_name: str = ""
    tasks: list[AtomicTask] = Field(default_factory=list)
    total_estimated_files: int = 0
    architecture_summary: str = ""

    def get_next_task(self) -> Optional[AtomicTask]:
        """Get the next pending task whose dependencies are all satisfied."""
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.PASSED}
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                if all(dep in completed_ids for dep in task.dependencies):
                    return task
        return None

    def get_progress(self) -> dict:
        """Return a progress summary."""
        total = len(self.tasks)
        passed = sum(1 for t in self.tasks if t.status == TaskStatus.PASSED)
        failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
        skipped = sum(1 for t in self.tasks if t.status == TaskStatus.SKIPPED)
        in_progress = sum(1 for t in self.tasks if t.status in (
            TaskStatus.IN_PROGRESS, TaskStatus.TESTS_WRITTEN,
            TaskStatus.CODE_GENERATED, TaskStatus.TESTING, TaskStatus.RETRYING
        ))
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "in_progress": in_progress,
            "pending": total - passed - failed - skipped - in_progress,
            "percent_complete": round((passed / total) * 100, 1) if total > 0 else 0,
        }
