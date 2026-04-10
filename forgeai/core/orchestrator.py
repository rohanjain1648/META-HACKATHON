"""Orchestrator Engine — The brain of ForgeAI.

Drives the entire agent pipeline through a finite state machine,
manages state handoffs, enforces guardrails, triggers checkpoints,
and produces the final summary report.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from forgeai.agents.intake_agent import IntakeAgent
from forgeai.agents.architect_agent import ArchitectAgent
from forgeai.agents.planner_agent import PlannerAgent
from forgeai.agents.qa_agent import QAAgent
from forgeai.agents.coder_agent import CoderAgent
from forgeai.agents.recovery_agent import RecoveryAgent
from forgeai.agents.security_agent import SecurityAgent
from forgeai.config.config_manager import ConfigManager
from forgeai.core.activity_logger import ActivityLogger
from forgeai.models.agent_state import AgentContext, AgentResult, AgentRole
from forgeai.models.specification import StructuredSpecification
from forgeai.models.task import AtomicTask, ImplementationPlan, TaskStatus
from forgeai.models.workflow_state import WorkflowPhase, WorkflowState
from forgeai.tools.file_manager import FileManager
from forgeai.tools.llm_gateway import LLMGateway
from forgeai.tools.test_runner import TestRunner


class Orchestrator:
    """Main orchestration engine coordinating all agents through the SDLC pipeline.
    
    Workflow:
        IDLE → INTAKE → CLARIFICATION → SPECIFICATION → ARCHITECTURE → PLANNING
        → PLAN_REVIEW (checkpoint) → EXECUTION (TDD loop per task) → SECURITY_AUDIT
        → SUMMARY → DONE
    """

    def __init__(self, config: ConfigManager):
        self.config = config
        self.state = WorkflowState()
        self.logger = ActivityLogger(config.output.log_file)

        # Initialize LLM Gateway
        api_key = config.get_api_key()
        self.llm = LLMGateway(
            provider=config.llm.provider,
            model=config.llm.model,
            api_key=api_key,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            logger=self.logger,
        )

        # Initialize tools
        self.file_manager = FileManager(config.output.project_dir, self.logger)
        self.test_runner = TestRunner(
            config.output.project_dir,
            timeout=config.testing.timeout_seconds,
            logger=self.logger,
        )

        # Initialize agents
        self.intake_agent = IntakeAgent(self.llm, self.logger)
        self.architect_agent = ArchitectAgent(self.llm, self.logger)
        self.planner_agent = PlannerAgent(self.llm, self.logger)
        self.qa_agent = QAAgent(self.llm, self.logger)
        self.coder_agent = CoderAgent(self.llm, self.logger)
        self.recovery_agent = RecoveryAgent(self.llm, self.logger)
        self.security_agent = SecurityAgent(self.llm, self.logger)

        # Callback hooks for UI (CLI/Web)
        self._on_phase_change: Optional[Callable] = None
        self._on_checkpoint: Optional[Callable] = None
        self._on_question: Optional[Callable] = None
        self._on_task_progress: Optional[Callable] = None
        self._on_diff_review: Optional[Callable] = None

    # ── UI Hooks ──────────────────────────────────────────────────────────

    def set_callbacks(self, on_phase_change=None, on_checkpoint=None,
                      on_question=None, on_task_progress=None, on_diff_review=None):
        """Register UI callbacks for interactive events."""
        self._on_phase_change = on_phase_change
        self._on_checkpoint = on_checkpoint
        self._on_question = on_question
        self._on_task_progress = on_task_progress
        self._on_diff_review = on_diff_review

    def _notify_phase(self, phase: WorkflowPhase):
        """Transition to a new phase and notify UI."""
        old = self.state.phase
        if self.state.transition_to(phase):
            self.logger.info("Orchestrator", f"Phase: {old.value} → {phase.value}")
            if self._on_phase_change:
                self._on_phase_change(phase)
        else:
            self.logger.error("Orchestrator",
                              f"Invalid transition: {old.value} → {phase.value}")

    # ── Main Pipeline ─────────────────────────────────────────────────────

    def run(self, raw_specification: str) -> dict:
        """Execute the full ForgeAI pipeline.
        
        Args:
            raw_specification: Natural-language project description.
            
        Returns:
            Workflow summary report dictionary.
        """
        self.state.started_at = datetime.now()
        self.state.raw_specification = raw_specification
        self.logger.info("Orchestrator", f"Starting ForgeAI pipeline")
        self.logger.info("Orchestrator", f"Raw spec: {raw_specification[:200]}...")

        try:
            # Phase 1: Intake & Clarification
            spec = self._phase_intake(raw_specification)
            if not spec:
                return self._finalize("Intake phase failed")

            # Phase 2: Architecture
            architecture = self._phase_architecture(spec)
            if not architecture:
                return self._finalize("Architecture phase failed")

            # Phase 3: Planning
            plan = self._phase_planning(spec, architecture)
            if not plan:
                return self._finalize("Planning phase failed")

            # Phase 4: Plan Review (Checkpoint)
            approved = self._phase_plan_review(plan)
            if not approved:
                return self._finalize("Plan rejected by user")

            # Phase 5: Execution (TDD loop)
            self._phase_execution(spec, architecture, plan)

            # Phase 6: Security Audit (Extended)
            if self.config.security_audit.enabled:
                self._phase_security_audit(spec)

            # Phase 7: Summary
            return self._finalize("Pipeline completed successfully")

        except Exception as e:
            self.logger.error("Orchestrator", f"Pipeline failed: {str(e)}")
            self.state.errors.append(str(e))
            return self._finalize(f"Pipeline error: {str(e)}")

    # ── Phase Implementations ─────────────────────────────────────────────

    def _phase_intake(self, raw_spec: str) -> Optional[StructuredSpecification]:
        """Phase 1: Analyze spec, ask questions, produce structured specification."""
        self._notify_phase(WorkflowPhase.INTAKE)

        # Step 1: Analyze and get clarifying questions
        context = AgentContext(
            role=AgentRole.INTAKE,
            user_input=raw_spec,
        )
        result = self.intake_agent.execute(context)
        self.state.total_api_calls += result.api_calls_made

        if not result.success:
            self.logger.error("Orchestrator", f"Intake failed: {result.error}")
            return None

        # Step 2: If questions, get answers from user
        if result.requires_human_input and result.clarifying_questions:
            self._notify_phase(WorkflowPhase.CLARIFICATION)
            answers = {}

            if self._on_question:
                answers = self._on_question(result.clarifying_questions)
            else:
                # Auto-answer mode — let LLM infer reasonable defaults
                self.logger.info("Orchestrator", "Auto-answering clarifying questions")
                for q in result.clarifying_questions:
                    answers[q] = "Use reasonable defaults based on common best practices."

            # Step 3: Produce structured specification with answers
            self._notify_phase(WorkflowPhase.SPECIFICATION)
            context2 = AgentContext(
                role=AgentRole.INTAKE,
                user_input=raw_spec,
                clarification_responses=answers,
            )
            result2 = self.intake_agent.execute(context2)
            self.state.total_api_calls += result2.api_calls_made

            if result2.success and result2.specification:
                self.state.specification = result2.specification
                self._save_artifact("structured_specification.yaml", result2.specification)
                self.logger.info("Orchestrator",
                                 f"Specification finalized: {result2.specification.project_name}")
                return result2.specification
            else:
                return None
        elif result.specification:
            self._notify_phase(WorkflowPhase.SPECIFICATION)
            self.state.specification = result.specification
            self._save_artifact("structured_specification.yaml", result.specification)
            return result.specification
        
        return None

    def _phase_architecture(self, spec: StructuredSpecification) -> Optional[dict]:
        """Phase 2: Design project architecture."""
        self._notify_phase(WorkflowPhase.ARCHITECTURE)

        context = AgentContext(
            role=AgentRole.ARCHITECT,
            specification=spec,
        )
        result = self.architect_agent.execute(context)
        self.state.total_api_calls += result.api_calls_made

        if result.success and result.architecture:
            self.state.architecture = result.architecture
            self._save_artifact("architecture.json", result.architecture)
            self.logger.info("Orchestrator", f"Architecture designed: {result.message}")

            # Checkpoint: after architecture (if configured)
            if "after_architecture" in self.config.workflow.checkpoints:
                if not self.config.workflow.auto_approve_checkpoints:
                    if self._on_checkpoint:
                        approved = self._on_checkpoint(
                            "Architecture Review",
                            f"Architecture designed:\n{json.dumps(result.architecture, indent=2)[:3000]}",
                        )
                        if not approved:
                            return None

            return result.architecture

        self.logger.error("Orchestrator", f"Architecture failed: {result.error}")
        return None

    def _phase_planning(self, spec: StructuredSpecification,
                        architecture: dict) -> Optional[ImplementationPlan]:
        """Phase 3: Decompose into atomic tasks."""
        self._notify_phase(WorkflowPhase.PLANNING)

        context = AgentContext(
            role=AgentRole.PLANNER,
            specification=spec,
            architecture=architecture,
        )
        result = self.planner_agent.execute(context)
        self.state.total_api_calls += result.api_calls_made

        if result.success and result.implementation_plan:
            self.state.implementation_plan = result.implementation_plan
            self._save_artifact("implementation_plan.json", result.implementation_plan)
            self.logger.info("Orchestrator", f"Plan created: {result.message}")
            return result.implementation_plan

        self.logger.error("Orchestrator", f"Planning failed: {result.error}")
        return None

    def _phase_plan_review(self, plan: ImplementationPlan) -> bool:
        """Phase 4: Human checkpoint — review and approve plan."""
        self._notify_phase(WorkflowPhase.PLAN_REVIEW)

        if self.config.workflow.auto_approve_checkpoints:
            self.logger.info("Orchestrator", "Auto-approving plan (config)")
            return True

        if self._on_checkpoint:
            plan_summary = self._format_plan_for_review(plan)
            return self._on_checkpoint("Implementation Plan Review", plan_summary)

        # No callback — auto-approve
        self.logger.info("Orchestrator", "No checkpoint callback — auto-approving plan")
        return True

    def _phase_execution(self, spec: StructuredSpecification,
                         architecture: dict, plan: ImplementationPlan):
        """Phase 5: TDD execution loop over all tasks."""
        self._notify_phase(WorkflowPhase.EXECUTION)
        self.file_manager.initialize_project()

        while True:
            task = plan.get_next_task()
            if not task:
                self.logger.info("Orchestrator", "All tasks completed or no more pending tasks")
                break

            self.logger.info("Orchestrator",
                             f"Starting task #{task.id}: {task.title}")
            task.status = TaskStatus.IN_PROGRESS
            self.state.current_task_index = task.id

            if self._on_task_progress:
                self._on_task_progress(task, plan.get_progress())

            success = self._execute_single_task(spec, architecture, task)

            if success:
                task.status = TaskStatus.PASSED
                self.logger.info("Orchestrator", f"Task #{task.id} PASSED ✓")
            else:
                task.status = TaskStatus.FAILED
                self.logger.error("Orchestrator", f"Task #{task.id} FAILED ✗")

            if self._on_task_progress:
                self._on_task_progress(task, plan.get_progress())

    def _execute_single_task(self, spec: StructuredSpecification,
                             architecture: dict, task: AtomicTask) -> bool:
        """Execute a single task using the TDD workflow: QA → Coder → Test → [Recovery]."""
        existing_files = self.file_manager.get_all_source_files()

        # ── Step 1: QA Agent writes failing tests ──
        self._notify_phase(WorkflowPhase.TASK_QA)
        qa_context = AgentContext(
            role=AgentRole.QA,
            specification=spec,
            architecture=architecture,
            current_task=task,
            project_dir=str(self.file_manager.project_dir),
            existing_files=existing_files,
        )
        qa_result = self.qa_agent.execute(qa_context)
        self.state.total_api_calls += qa_result.api_calls_made

        if not qa_result.success:
            self.logger.error("Orchestrator", f"QA failed for task #{task.id}: {qa_result.error}")
            return False

        # Write test files to disk
        for filepath, content in qa_result.generated_files.items():
            self.file_manager.write_file(filepath, content)
            task.test_files.append(filepath)
            self.state.files_generated.append(filepath)

        task.status = TaskStatus.TESTS_WRITTEN
        self.logger.info("Orchestrator",
                         f"Tests written for task #{task.id}: {list(qa_result.generated_files.keys())}")

        # ── Step 2: Coder Agent generates production code ──
        max_retries = self.config.workflow.max_retries
        for attempt in range(max_retries + 1):
            self._notify_phase(WorkflowPhase.TASK_CODE)

            # Refresh existing files (includes new test files)
            existing_files = self.file_manager.get_all_source_files()

            error_msg = ""
            prev_attempts = []
            if attempt > 0:
                error_msg = task.error_log[-1] if task.error_log else ""
                prev_attempts = task.error_log

            coder_context = AgentContext(
                role=AgentRole.CODER,
                specification=spec,
                architecture=architecture,
                current_task=task,
                project_dir=str(self.file_manager.project_dir),
                existing_files=existing_files,
                error_message=error_msg,
                previous_attempts=prev_attempts,
                retry_count=attempt,
            )
            coder_result = self.coder_agent.execute(coder_context)
            self.state.total_api_calls += coder_result.api_calls_made

            if not coder_result.success:
                task.error_log.append(f"Coder failed: {coder_result.error}")
                continue

            # Diff review (FR-09)
            if self._on_diff_review and not self.config.workflow.auto_approve_checkpoints:
                approved = self._on_diff_review(task, coder_result.generated_files)
                if not approved:
                    continue

            # Write production code
            for filepath, content in coder_result.generated_files.items():
                self.file_manager.write_file(filepath, content)
                if filepath not in self.state.files_generated:
                    self.state.files_generated.append(filepath)

            task.status = TaskStatus.CODE_GENERATED
            task.generated_code = coder_result.generated_files

            # ── Step 3: Run tests ──
            self._notify_phase(WorkflowPhase.TASK_TEST)
            test_result = self.test_runner.run_tests()

            if test_result.success:
                self.state.tests_passed += test_result.passed
                task.test_results = test_result.to_dict()
                self.logger.info("Orchestrator",
                                 f"Task #{task.id} tests PASSED: {test_result.passed} passed")
                return True
            else:
                self.state.tests_failed += test_result.failed
                error_info = (
                    f"Attempt {attempt + 1}: {test_result.failed} tests failed, "
                    f"{test_result.errors} errors.\n{test_result.output[-1000:]}"
                )
                task.error_log.append(error_info)
                task.retry_count = attempt + 1

                self.logger.warn("Orchestrator",
                                 f"Task #{task.id} tests FAILED (attempt {attempt + 1}/{max_retries + 1})")

                # ── Step 3b: Recovery ──
                if attempt < max_retries:
                    self._notify_phase(WorkflowPhase.TASK_RECOVERY)
                    recovery_context = AgentContext(
                        role=AgentRole.RECOVERY,
                        specification=spec,
                        current_task=task,
                        existing_files=self.file_manager.get_all_source_files(),
                        error_message=test_result.output[-2000:],
                        error_traceback=test_result.error_output[-1000:],
                        previous_attempts=task.error_log,
                        retry_count=attempt,
                    )
                    recovery_result = self.recovery_agent.execute(recovery_context)
                    self.state.total_api_calls += recovery_result.api_calls_made

                    if recovery_result.success and recovery_result.architecture:
                        strategy = recovery_result.architecture.get("strategy", "ESCALATE")
                        if strategy == "SKIP_TASK":
                            self.logger.info("Orchestrator",
                                             f"Recovery: Skipping task #{task.id}")
                            task.status = TaskStatus.SKIPPED
                            return False
                        elif strategy == "ESCALATE":
                            self.logger.warn("Orchestrator",
                                             f"Recovery: Escalating task #{task.id}")
                            return False

                        # Apply modified tests if provided
                        if recovery_result.generated_files:
                            for fp, content in recovery_result.generated_files.items():
                                self.file_manager.write_file(fp, content)

                    time.sleep(self.config.workflow.retry_delay_seconds)

        # All retries exhausted
        self.logger.error("Orchestrator",
                          f"Task #{task.id} failed after {max_retries + 1} attempts")
        return False

    def _phase_security_audit(self, spec: StructuredSpecification):
        """Phase 6: Security audit on all generated code."""
        self._notify_phase(WorkflowPhase.SECURITY_AUDIT)

        all_files = self.file_manager.get_all_source_files()
        context = AgentContext(
            role=AgentRole.SECURITY,
            specification=spec,
            existing_files=all_files,
        )
        result = self.security_agent.execute(context)
        self.state.total_api_calls += result.api_calls_made

        if result.success and result.security_report:
            self._save_artifact("security_report.json", result.security_report)
            self.logger.info("Orchestrator", f"Security audit: {result.message}")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _finalize(self, message: str) -> dict:
        """Finalize the workflow and produce the summary report."""
        self.state.completed_at = datetime.now()

        if self.state.phase != WorkflowPhase.ERROR:
            self._notify_phase(WorkflowPhase.SUMMARY)

        # Update stats from LLM gateway
        llm_stats = self.llm.get_stats()
        self.state.total_api_calls = llm_stats["total_api_calls"]
        self.state.total_tokens_used = llm_stats["total_tokens_used"]

        summary = self.state.get_summary()
        summary["message"] = message
        summary["file_stats"] = self.file_manager.get_stats()

        # Save summary report
        self._save_artifact("workflow_summary.json", summary)

        self.logger.info("Orchestrator", f"Pipeline finished: {message}")
        self.logger.info("Orchestrator", json.dumps(summary, indent=2, default=str))

        if self.state.phase == WorkflowPhase.SUMMARY:
            self._notify_phase(WorkflowPhase.DONE)

        return summary

    def _save_artifact(self, filename: str, data):
        """Save an intermediate artifact to the project directory."""
        artifacts_dir = Path(self.config.output.project_dir) / ".forgeai"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        filepath = artifacts_dir / filename
        if hasattr(data, "model_dump"):
            content = json.dumps(data.model_dump(), indent=2, default=str)
        elif isinstance(data, dict):
            content = json.dumps(data, indent=2, default=str)
        else:
            content = str(data)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        if self.logger:
            self.logger.file_write("Orchestrator", str(filepath))

    def _format_plan_for_review(self, plan: ImplementationPlan) -> str:
        """Format the implementation plan for human review."""
        lines = [
            f"# Implementation Plan: {plan.project_name}",
            f"Architecture: {plan.architecture_summary}",
            f"Total tasks: {len(plan.tasks)}",
            f"Estimated files: {plan.total_estimated_files}",
            "",
            "## Tasks:",
        ]
        for task in plan.tasks:
            risk = f"[{task.risk_level.value.upper()}]" if task.risk_level.value != "low" else ""
            deps = f" (depends on: {task.dependencies})" if task.dependencies else ""
            cp = " [CHECKPOINT]" if task.is_checkpoint else ""
            lines.append(
                f"  {task.id}. {task.title} {risk}{cp}{deps}\n"
                f"     Files: {', '.join(task.target_files)}\n"
                f"     {task.description}"
            )
        return "\n".join(lines)
