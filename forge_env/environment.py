"""ForgeRL Environment — The core OpenEnv-compatible RL environment.

This is the heart of the hackathon submission. It wraps ForgeAI's multi-agent
SDLC orchestrator as a Gymnasium-style environment where an LLM meta-agent
learns to coordinate sub-agents to build software autonomously.

Episode lifecycle:
  1. reset() — Sample a spec from the curriculum, initialize ForgeAI
  2. step(action) — Execute one orchestration decision, get observation + reward
  3. Repeat until terminal (all tasks done, max steps, or unrecoverable failure)

Themes addressed:
  - Multi-Agent Interactions: 7+ sub-agents with handoffs and partial observability
  - Long-Horizon Reasoning: 50-300+ step episodes with sparse terminal rewards
  - Professional Tasks: Real file system, pytest, Docker interactions
  - Self-Improvement: Adaptive difficulty curriculum across 5 tiers
  - Fleet AI Oversight: Oversight agent monitors/explains sub-agent behavior
  - Snorkel Expert-in-Loop: Simulated reviewers with changing preferences
  - Mercor Token Scaling: Rewards scale with useful token output
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from forge_env.curriculum import (
    AdaptiveCurriculum,
    EpisodeResult,
    ReviewerPersonality,
    REVIEWER_POOL,
    SpecVariation,
)
from forge_env.models import (
    ActionType,
    AgentOutputSummary,
    ForgeAction,
    ForgeObservation,
    ForgeState,
    OversightReport,
    ProjectSnapshot,
    ResetResult,
    ReviewerFeedback,
    StepResult,
    TaskProgressSummary,
)
from forge_env.reward import ForgeRewardCalculator, RewardSignal


class ForgeEnvironment:
    """Multi-Agent Software Engineering RL Environment.

    OpenEnv-compatible environment where an LLM meta-agent orchestrates
    a team of specialized sub-agents to build software from natural-language
    specifications. Implements the Gymnasium-style reset/step/state API.

    The environment is designed for training with GRPO (Group Relative Policy
    Optimization) using Unsloth + HuggingFace TRL.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_steps: int = 300,
        use_real_llm: bool = True,
        project_base_dir: Optional[str] = None,
        curriculum_config: Optional[dict] = None,
    ):
        """Initialize the ForgeRL environment.

        Args:
            api_key: Google Gemini API key. Falls back to GOOGLE_API_KEY env var.
            max_steps: Maximum steps per episode before truncation.
            use_real_llm: If True, use real LLM for sub-agents. If False, use
                simulated responses (for fast training/testing).
            project_base_dir: Base directory for generated projects. Uses temp
                dir if not specified.
            curriculum_config: Override curriculum params (promote_threshold, etc.)
        """
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._max_steps = max_steps
        self._use_real_llm = use_real_llm
        self._project_base_dir = project_base_dir

        # Curriculum for adaptive difficulty
        curriculum_params = curriculum_config or {}
        self._curriculum = AdaptiveCurriculum(**curriculum_params)

        # Reward calculator
        self._reward_calc = ForgeRewardCalculator()

        # Episode state (initialized on reset)
        self._episode_id: str = ""
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._is_terminal: bool = False
        self._termination_reason: str = ""
        self._started_at: Optional[datetime] = None

        # ForgeAI engine state
        self._orchestrator = None
        self._current_spec: Optional[SpecVariation] = None
        self._spec_text: str = ""
        self._project_dir: str = ""
        self._current_phase: str = "idle"

        # Observation tracking
        self._last_agent_output = AgentOutputSummary()
        self._phase_history: list[str] = []
        self._action_history: list[dict] = []
        self._reward_history: list[float] = []
        self._error_context: str = ""
        self._tokens_this_step: int = 0

        # Oversight and reviewer state
        self._oversight_report: Optional[OversightReport] = None
        self._reviewer: Optional[ReviewerPersonality] = None
        self._reviewer_feedback: Optional[ReviewerFeedback] = None

        # Internal tracking for reward computation
        self._prev_observation: Optional[ForgeObservation] = None
        self._files_generated: list[str] = []
        self._tests_passed: int = 0
        self._tests_failed: int = 0
        self._total_api_calls: int = 0
        self._total_tokens: int = 0

        # Task tracking
        self._tasks_total: int = 0
        self._tasks_completed: int = 0
        self._tasks_failed: int = 0
        self._tasks_skipped: int = 0
        self._current_task_id: int = 0
        self._current_task_title: str = ""

        # Specification/architecture/plan artifacts
        self._has_specification: bool = False
        self._has_architecture: bool = False
        self._has_plan: bool = False

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    async def reset(
        self, spec_text: Optional[str] = None, tier: Optional[int] = None
    ) -> ResetResult:
        """Reset the environment for a new episode.

        Args:
            spec_text: Optional override specification. If None, samples from
                the adaptive curriculum.
            tier: Optional override tier. If None, uses curriculum's current tier.

        Returns:
            ResetResult with initial observation.
        """
        # Generate episode ID
        self._episode_id = str(uuid.uuid4())[:8]
        self._step_count = 0
        self._total_reward = 0.0
        self._is_terminal = False
        self._termination_reason = ""
        self._started_at = datetime.now()

        # Reset reward calculator
        self._reward_calc.reset()

        # Sample or use provided spec
        if spec_text:
            self._spec_text = spec_text
            self._current_spec = SpecVariation(
                name="custom",
                tier=tier or 1,
                base_description=spec_text,
                max_steps=self._max_steps,
            )
        else:
            if tier:
                self._curriculum.current_tier = tier
            self._current_spec, self._spec_text = self._curriculum.sample_spec()

        # Set max steps based on tier
        self._max_steps = self._current_spec.max_steps

        # Create a fresh project directory
        if self._project_base_dir:
            self._project_dir = os.path.join(
                self._project_base_dir, f"project_{self._episode_id}"
            )
        else:
            self._project_dir = tempfile.mkdtemp(
                prefix=f"forgerl_{self._episode_id}_"
            )

        # Reset all tracking state
        self._current_phase = "idle"
        self._last_agent_output = AgentOutputSummary()
        self._phase_history = ["idle"]
        self._action_history = []
        self._reward_history = []
        self._error_context = ""
        self._tokens_this_step = 0
        self._files_generated = []
        self._tests_passed = 0
        self._tests_failed = 0
        self._total_api_calls = 0
        self._total_tokens = 0
        self._tasks_total = 0
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._tasks_skipped = 0
        self._current_task_id = 0
        self._current_task_title = ""
        self._has_specification = False
        self._has_architecture = False
        self._has_plan = False
        self._oversight_report = None
        self._reviewer_feedback = None

        # Select a random reviewer personality
        import random
        self._reviewer = random.choice(REVIEWER_POOL).model_copy(deep=True)

        # Initialize the ForgeAI orchestrator in step mode
        self._initialize_orchestrator()

        # Build initial observation
        obs = self._build_observation()
        self._prev_observation = obs

        return ResetResult(
            observation=obs,
            info={
                "episode_id": self._episode_id,
                "spec_name": self._current_spec.name,
                "tier": self._current_spec.tier,
                "max_steps": self._max_steps,
                "reviewer": self._reviewer.name if self._reviewer else "None",
            },
        )

    async def step(self, action: ForgeAction) -> StepResult:
        """Execute one orchestration step.

        The meta-agent chooses an action (delegate to sub-agent, approve/reject,
        provide feedback, etc.), and the environment executes it, returning
        the new observation and reward.

        Args:
            action: The orchestration action to execute.

        Returns:
            StepResult with observation, reward, termination flags, and info.
        """
        if self._is_terminal:
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                terminated=True,
                truncated=False,
                info={"error": "Episode already terminated"},
            )

        self._step_count += 1
        self._tokens_this_step = 0

        # Check if action is valid in current state
        available = self._get_available_actions()
        action_valid = action.action_type.value in available

        # Execute the action
        if action_valid:
            await self._execute_action(action)
        else:
            self._last_agent_output = AgentOutputSummary(
                agent_name="orchestrator",
                success=False,
                message=f"Invalid action '{action.action_type.value}' in phase '{self._current_phase}'",
                error=f"Valid actions: {available}",
            )

        # Check for simulated reviewer preference changes
        if self._reviewer:
            change = self._reviewer.maybe_change_preference()
            if change:
                key, old_val, new_val = change
                self._reviewer_feedback = ReviewerFeedback(
                    approved=False,
                    satisfaction_score=0.3,
                    feedback_text=f"Preference changed: {key} from '{old_val}' to '{new_val}'",
                    preference_changes=[f"{key}: {old_val} -> {new_val}"],
                    current_preferences=dict(self._reviewer.preferences),
                )

        # Build observation
        curr_obs = self._build_observation()

        # Check terminal conditions
        terminated = self._check_terminal()
        truncated = self._step_count >= self._max_steps

        if truncated and not terminated:
            self._is_terminal = True
            self._termination_reason = "max_steps_exceeded"

        if terminated:
            self._is_terminal = True

        # Compute reward
        reward_signal = self._reward_calc.compute_step_reward(
            action=action,
            prev_obs=self._prev_observation or curr_obs,
            curr_obs=curr_obs,
            action_valid=action_valid,
            is_terminal=self._is_terminal,
            tokens_generated=self._tokens_this_step,
        )

        reward = reward_signal.total
        self._total_reward += reward
        self._reward_history.append(reward)

        # Record action
        self._action_history.append({
            "step": self._step_count,
            "action": action.action_type.value,
            "valid": action_valid,
            "reward": reward,
            "phase": self._current_phase,
        })

        # Record episode result for curriculum if terminal
        if self._is_terminal and self._current_spec:
            self._curriculum.record_episode(
                EpisodeResult(
                    spec_name=self._current_spec.name,
                    tier=self._current_spec.tier,
                    success=self._tasks_completed > 0 and self._tasks_failed == 0,
                    total_reward=self._total_reward,
                    steps_taken=self._step_count,
                    tasks_completed=self._tasks_completed,
                    tasks_total=self._tasks_total,
                    test_pass_rate=(
                        self._tests_passed / max(self._tests_passed + self._tests_failed, 1)
                    ),
                )
            )

        self._prev_observation = curr_obs

        return StepResult(
            observation=curr_obs,
            reward=reward,
            terminated=terminated or truncated,
            truncated=truncated and not terminated,
            info={
                "reward_breakdown": {
                    "phase_progress": reward_signal.phase_progress,
                    "task_completion": reward_signal.task_completion,
                    "recovery_success": reward_signal.recovery_success,
                    "oversight_bonus": reward_signal.oversight_bonus,
                    "code_quality_terminal": reward_signal.code_quality_terminal,
                    "token_scaling": reward_signal.token_scaling,
                    "step_cost": reward_signal.step_cost,
                    "invalid_action_penalty": reward_signal.invalid_action_penalty,
                    "efficiency_bonus": reward_signal.efficiency_bonus,
                    "reviewer_satisfaction": reward_signal.reviewer_satisfaction,
                },
                "action_valid": action_valid,
                "curriculum_stats": self._curriculum.get_stats(),
            },
        )

    @property
    def state(self) -> ForgeState:
        """Return the full internal state (for debugging and evaluation)."""
        elapsed = 0.0
        if self._started_at:
            elapsed = (datetime.now() - self._started_at).total_seconds()

        return ForgeState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            total_reward=self._total_reward,
            spec_tier=self._current_spec.tier if self._current_spec else 1,
            spec_text=self._spec_text,
            is_terminal=self._is_terminal,
            termination_reason=self._termination_reason,
            true_test_pass_rate=(
                self._tests_passed / max(self._tests_passed + self._tests_failed, 1)
            ),
            true_code_quality_score=self._compute_quality_score(),
            true_files_generated=len(self._files_generated),
            total_api_calls=self._total_api_calls,
            total_tokens_used=self._total_tokens,
            started_at=self._started_at,
            elapsed_seconds=elapsed,
            action_history=self._action_history,
            reward_history=self._reward_history,
            phase_trace=self._phase_history,
        )

    # ── Synchronous Wrappers ──────────────────────────────────────────────────

    def reset_sync(self, **kwargs) -> ResetResult:
        """Synchronous wrapper for reset()."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.reset(**kwargs))
        finally:
            loop.close()

    def step_sync(self, action: ForgeAction) -> StepResult:
        """Synchronous wrapper for step()."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.step(action))
        finally:
            loop.close()

    # ── Internal: ForgeAI Orchestrator ────────────────────────────────────────

    def _initialize_orchestrator(self):
        """Initialize the ForgeAI orchestrator in step mode.

        Falls back to simulated mode if ForgeAI isn't available or if
        the API key is missing (common during training/evaluation).
        """
        if not self._use_real_llm:
            self._orchestrator = None
            return

        try:
            from forgeai.config.config_manager import ConfigManager
            from forgeai.core.orchestrator import Orchestrator

            # Reset singleton to get fresh config
            ConfigManager.reset()

            # Create a temporary config for this episode
            config = ConfigManager.get_instance()

            # Override project directory
            config._config.output.project_dir = self._project_dir
            config._config.workflow.auto_approve_checkpoints = True
            config._config.workflow.max_retries = 2

            self._orchestrator = Orchestrator(config)
            self._orchestrator._step_mode = True

        except (ImportError, ValueError, Exception) as e:
            # ForgeAI not available or API key missing — use simulated mode
            self._orchestrator = None
            self._use_real_llm = False

    async def _execute_action(self, action: ForgeAction):
        """Execute a single orchestration action and update internal state."""
        action_type = action.action_type
        prev_phase = self._current_phase

        try:
            if action_type == ActionType.DELEGATE_INTAKE:
                await self._do_intake()
            elif action_type == ActionType.DELEGATE_ARCHITECT:
                await self._do_architect()
            elif action_type == ActionType.DELEGATE_PLANNER:
                await self._do_planner()
            elif action_type == ActionType.DELEGATE_QA:
                await self._do_qa()
            elif action_type == ActionType.DELEGATE_CODER:
                await self._do_coder()
            elif action_type == ActionType.DELEGATE_RECOVERY:
                await self._do_recovery()
            elif action_type == ActionType.DELEGATE_SECURITY:
                await self._do_security()
            elif action_type == ActionType.DELEGATE_OVERSIGHT:
                await self._do_oversight()
            elif action_type == ActionType.APPROVE_PLAN:
                await self._do_approve_plan()
            elif action_type == ActionType.REJECT_PLAN:
                await self._do_reject_plan()
            elif action_type == ActionType.APPROVE_CODE:
                await self._do_approve_code()
            elif action_type == ActionType.REJECT_CODE:
                await self._do_reject_code()
            elif action_type == ActionType.PROVIDE_FEEDBACK:
                await self._do_provide_feedback(action.parameters)
            elif action_type == ActionType.SKIP_TASK:
                await self._do_skip_task()
            elif action_type == ActionType.RETRY_TASK:
                await self._do_retry_task()
            elif action_type == ActionType.ESCALATE:
                await self._do_escalate()
            elif action_type == ActionType.FINALIZE:
                await self._do_finalize()

        except Exception as e:
            self._last_agent_output = AgentOutputSummary(
                agent_name=action_type.value,
                success=False,
                message=f"Action execution error: {str(e)}",
                error=str(e),
            )
            self._error_context = str(e)

        # Track phase transitions
        if self._current_phase != prev_phase:
            self._phase_history.append(self._current_phase)

    # ── Sub-agent Delegation (Simulated or Real) ──────────────────────────────

    async def _do_intake(self):
        """Delegate to the Intake Agent for requirements analysis."""
        if self._current_phase not in ("idle", "intake"):
            return

        self._current_phase = "intake"

        if self._orchestrator and self._use_real_llm:
            try:
                from forgeai.models.agent_state import AgentContext, AgentRole
                context = AgentContext(
                    role=AgentRole.INTAKE,
                    user_input=self._spec_text,
                )
                result = self._orchestrator.intake_agent.execute(context)
                self._total_api_calls += result.api_calls_made
                self._tokens_this_step += result.tokens_used

                if result.success and result.specification:
                    self._orchestrator.state.specification = result.specification
                    self._has_specification = True
                    self._current_phase = "specification"
                    self._last_agent_output = AgentOutputSummary(
                        agent_name="intake",
                        success=True,
                        message=f"Specification created: {result.specification.project_name}",
                        files_produced=["structured_specification.yaml"],
                    )
                else:
                    self._last_agent_output = AgentOutputSummary(
                        agent_name="intake",
                        success=False,
                        message=result.message,
                        error=result.error,
                    )
            except Exception as e:
                self._simulate_agent("intake", e)
        else:
            self._simulate_intake()

    async def _do_architect(self):
        """Delegate to the Architect Agent for system design."""
        if self._current_phase not in ("specification",):
            return

        self._current_phase = "architecture"

        if self._orchestrator and self._use_real_llm:
            try:
                from forgeai.models.agent_state import AgentContext, AgentRole
                spec = self._orchestrator.state.specification
                if not spec:
                    self._last_agent_output = AgentOutputSummary(
                        agent_name="architect", success=False,
                        error="No specification available",
                    )
                    return

                context = AgentContext(role=AgentRole.ARCHITECT, specification=spec)
                result = self._orchestrator.architect_agent.execute(context)
                self._total_api_calls += result.api_calls_made
                self._tokens_this_step += result.tokens_used

                if result.success and result.architecture:
                    self._orchestrator.state.architecture = result.architecture
                    self._has_architecture = True
                    self._last_agent_output = AgentOutputSummary(
                        agent_name="architect",
                        success=True,
                        message=result.message,
                        files_produced=["architecture.json"],
                    )
                else:
                    self._last_agent_output = AgentOutputSummary(
                        agent_name="architect", success=False,
                        message=result.message, error=result.error,
                    )
            except Exception as e:
                self._simulate_agent("architect", e)
        else:
            self._simulate_architect()

    async def _do_planner(self):
        """Delegate to the Planner Agent for task decomposition."""
        if self._current_phase not in ("architecture",):
            return

        self._current_phase = "planning"

        if self._orchestrator and self._use_real_llm:
            try:
                from forgeai.models.agent_state import AgentContext, AgentRole
                spec = self._orchestrator.state.specification
                arch = self._orchestrator.state.architecture
                if not spec or not arch:
                    self._last_agent_output = AgentOutputSummary(
                        agent_name="planner", success=False,
                        error="Missing spec or architecture",
                    )
                    return

                context = AgentContext(
                    role=AgentRole.PLANNER, specification=spec, architecture=arch,
                )
                result = self._orchestrator.planner_agent.execute(context)
                self._total_api_calls += result.api_calls_made
                self._tokens_this_step += result.tokens_used

                if result.success and result.implementation_plan:
                    self._orchestrator.state.implementation_plan = result.implementation_plan
                    self._has_plan = True
                    self._tasks_total = len(result.implementation_plan.tasks)
                    self._last_agent_output = AgentOutputSummary(
                        agent_name="planner",
                        success=True,
                        message=f"Plan: {self._tasks_total} tasks",
                        files_produced=["implementation_plan.json"],
                    )
                else:
                    self._last_agent_output = AgentOutputSummary(
                        agent_name="planner", success=False,
                        message=result.message, error=result.error,
                    )
            except Exception as e:
                self._simulate_agent("planner", e)
        else:
            self._simulate_planner()

    async def _do_qa(self):
        """Delegate to QA Agent for TDD test generation."""
        if self._current_phase not in ("execution", "plan_review"):
            return

        self._current_phase = "task_qa"

        if self._orchestrator and self._use_real_llm:
            try:
                from forgeai.models.agent_state import AgentContext, AgentRole
                plan = self._orchestrator.state.implementation_plan
                if plan:
                    task = plan.get_next_task()
                    if task:
                        self._current_task_id = task.id
                        self._current_task_title = task.title
                        context = AgentContext(
                            role=AgentRole.QA,
                            specification=self._orchestrator.state.specification,
                            architecture=self._orchestrator.state.architecture,
                            current_task=task,
                            project_dir=self._project_dir,
                            existing_files=self._orchestrator.file_manager.get_all_source_files(),
                        )
                        result = self._orchestrator.qa_agent.execute(context)
                        self._total_api_calls += result.api_calls_made
                        self._tokens_this_step += result.tokens_used

                        if result.success:
                            for fp, content in result.generated_files.items():
                                self._orchestrator.file_manager.write_file(fp, content)
                                self._files_generated.append(fp)
                            self._last_agent_output = AgentOutputSummary(
                                agent_name="qa", success=True,
                                message=result.message,
                                files_produced=list(result.generated_files.keys()),
                            )
                            self._current_phase = "task_qa"
                        else:
                            self._last_agent_output = AgentOutputSummary(
                                agent_name="qa", success=False, error=result.error,
                            )
            except Exception as e:
                self._simulate_agent("qa", e)
        else:
            self._simulate_qa()

    async def _do_coder(self):
        """Delegate to Coder Agent for code generation."""
        if self._current_phase not in ("task_qa", "task_recovery"):
            return

        self._current_phase = "task_code"

        if self._orchestrator and self._use_real_llm:
            try:
                from forgeai.models.agent_state import AgentContext, AgentRole
                plan = self._orchestrator.state.implementation_plan
                if plan:
                    task = plan.get_next_task()
                    if not task:
                        # Find current in-progress task
                        from forgeai.models.task import TaskStatus
                        for t in plan.tasks:
                            if t.status in (TaskStatus.IN_PROGRESS, TaskStatus.TESTS_WRITTEN):
                                task = t
                                break

                    if task:
                        context = AgentContext(
                            role=AgentRole.CODER,
                            specification=self._orchestrator.state.specification,
                            architecture=self._orchestrator.state.architecture,
                            current_task=task,
                            project_dir=self._project_dir,
                            existing_files=self._orchestrator.file_manager.get_all_source_files(),
                            error_message=self._error_context,
                        )
                        result = self._orchestrator.coder_agent.execute(context)
                        self._total_api_calls += result.api_calls_made
                        self._tokens_this_step += result.tokens_used

                        if result.success:
                            for fp, content in result.generated_files.items():
                                self._orchestrator.file_manager.write_file(fp, content)
                                if fp not in self._files_generated:
                                    self._files_generated.append(fp)

                            # Run tests
                            test_result = self._orchestrator.test_runner.run_tests()
                            if test_result.success:
                                self._tests_passed += test_result.passed
                                self._tasks_completed += 1
                                from forgeai.models.task import TaskStatus
                                task.status = TaskStatus.PASSED
                                self._current_phase = "execution"
                                self._error_context = ""
                            else:
                                self._tests_failed += test_result.failed
                                self._error_context = test_result.output[-500:]
                                self._current_phase = "task_recovery"

                            self._last_agent_output = AgentOutputSummary(
                                agent_name="coder", success=test_result.success,
                                message=f"Tests: {test_result.passed}P/{test_result.failed}F",
                                files_produced=list(result.generated_files.keys()),
                            )
                        else:
                            self._last_agent_output = AgentOutputSummary(
                                agent_name="coder", success=False, error=result.error,
                            )
                            self._error_context = result.error
            except Exception as e:
                self._simulate_agent("coder", e)
        else:
            self._simulate_coder()

    async def _do_recovery(self):
        """Delegate to Recovery Agent for failure diagnosis."""
        if self._current_phase not in ("task_recovery", "task_code"):
            return

        self._current_phase = "task_recovery"

        if self._orchestrator and self._use_real_llm:
            try:
                from forgeai.models.agent_state import AgentContext, AgentRole
                context = AgentContext(
                    role=AgentRole.RECOVERY,
                    error_message=self._error_context,
                    existing_files=self._orchestrator.file_manager.get_all_source_files(),
                )
                result = self._orchestrator.recovery_agent.execute(context)
                self._total_api_calls += result.api_calls_made

                strategy = "RETRY_WITH_FIX"
                if result.architecture:
                    strategy = result.architecture.get("strategy", "RETRY_WITH_FIX")

                self._last_agent_output = AgentOutputSummary(
                    agent_name="recovery", success=result.success,
                    message=f"Strategy: {strategy}",
                )

                if strategy == "SKIP_TASK":
                    self._tasks_skipped += 1
                    self._current_phase = "execution"
                    self._error_context = ""
                elif strategy in ("RETRY_WITH_FIX", "MODIFY_APPROACH"):
                    self._current_phase = "task_code"
                else:
                    self._current_phase = "execution"
                    self._error_context = ""

            except Exception as e:
                self._simulate_agent("recovery", e)
        else:
            self._simulate_recovery()

    async def _do_security(self):
        """Delegate to Security Agent for code audit."""
        if self._current_phase not in ("execution", "security_audit"):
            return

        self._current_phase = "security_audit"

        if self._orchestrator and self._use_real_llm:
            try:
                from forgeai.models.agent_state import AgentContext, AgentRole
                context = AgentContext(
                    role=AgentRole.SECURITY,
                    specification=self._orchestrator.state.specification,
                    existing_files=self._orchestrator.file_manager.get_all_source_files(),
                )
                result = self._orchestrator.security_agent.execute(context)
                self._total_api_calls += result.api_calls_made

                self._last_agent_output = AgentOutputSummary(
                    agent_name="security", success=result.success,
                    message=result.message,
                    files_produced=["security_report.json"],
                )
            except Exception as e:
                self._simulate_agent("security", e)
        else:
            self._simulate_security()

    async def _do_oversight(self):
        """Delegate to Oversight Agent for quality monitoring."""
        self._current_phase_before = self._current_phase

        if self._orchestrator and self._use_real_llm:
            try:
                from forgeai.agents.oversight_agent import OversightAgent
                from forgeai.models.agent_state import AgentContext, AgentRole

                oversight = OversightAgent(self._orchestrator.llm, self._orchestrator.logger)
                context = AgentContext(
                    role=AgentRole.OVERSIGHT,
                    existing_files=self._orchestrator.file_manager.get_all_source_files(),
                    specification=self._orchestrator.state.specification,
                )
                result = oversight.execute(context)
                self._total_api_calls += result.api_calls_made

                if result.success and result.architecture:
                    report_data = result.architecture
                    self._oversight_report = OversightReport(
                        issues_found=report_data.get("issues_found", 0),
                        critical_issues=report_data.get("critical_issues", 0),
                        quality_score=report_data.get("quality_score", 0.5),
                        hallucination_flags=report_data.get("hallucination_flags", []),
                        recommendations=report_data.get("recommendations", []),
                        agent_behavior_summary=report_data.get("behavior_summary", ""),
                    )
                    self._last_agent_output = AgentOutputSummary(
                        agent_name="oversight", success=True,
                        message=f"Quality: {self._oversight_report.quality_score:.2f}, "
                                f"Issues: {self._oversight_report.issues_found}",
                    )
            except Exception as e:
                self._simulate_oversight()
        else:
            self._simulate_oversight()

    async def _do_approve_plan(self):
        """Approve the implementation plan and transition to execution."""
        if self._current_phase in ("planning",) and self._has_plan:
            self._current_phase = "execution"
            self._last_agent_output = AgentOutputSummary(
                agent_name="orchestrator", success=True,
                message="Plan approved. Starting execution.",
            )
            if self._orchestrator and self._orchestrator.file_manager:
                self._orchestrator.file_manager.initialize_project()

    async def _do_reject_plan(self):
        """Reject the plan and request replanning."""
        if self._current_phase in ("planning",):
            self._current_phase = "architecture"
            self._has_plan = False
            self._last_agent_output = AgentOutputSummary(
                agent_name="orchestrator", success=True,
                message="Plan rejected. Returning to architecture phase.",
            )

    async def _do_approve_code(self):
        """Approve generated code for the current task."""
        if self._current_phase in ("task_code",):
            self._current_phase = "execution"
            self._last_agent_output = AgentOutputSummary(
                agent_name="orchestrator", success=True,
                message="Code approved.",
            )

    async def _do_reject_code(self):
        """Reject generated code and trigger recovery."""
        if self._current_phase in ("task_code",):
            self._current_phase = "task_recovery"
            self._error_context = "Code rejected by meta-agent"
            self._last_agent_output = AgentOutputSummary(
                agent_name="orchestrator", success=True,
                message="Code rejected. Entering recovery.",
            )

    async def _do_provide_feedback(self, params: dict):
        """Provide feedback to guide the next sub-agent action."""
        feedback = params.get("feedback", "")
        self._last_agent_output = AgentOutputSummary(
            agent_name="orchestrator", success=True,
            message=f"Feedback recorded: {feedback[:100]}",
        )

    async def _do_skip_task(self):
        """Skip the current task."""
        if self._current_phase in ("task_qa", "task_code", "task_recovery"):
            self._tasks_skipped += 1
            self._current_phase = "execution"
            self._error_context = ""
            self._last_agent_output = AgentOutputSummary(
                agent_name="orchestrator", success=True,
                message="Task skipped.",
            )

    async def _do_retry_task(self):
        """Retry the current task from QA phase."""
        if self._current_phase in ("task_recovery", "task_code"):
            self._current_phase = "task_qa"
            self._error_context = ""
            self._last_agent_output = AgentOutputSummary(
                agent_name="orchestrator", success=True,
                message="Retrying task from QA phase.",
            )

    async def _do_escalate(self):
        """Escalate — end the episode as the agent cannot proceed."""
        self._is_terminal = True
        self._termination_reason = "agent_escalated"
        self._last_agent_output = AgentOutputSummary(
            agent_name="orchestrator", success=False,
            message="Agent escalated. Episode terminated.",
        )

    async def _do_finalize(self):
        """Finalize the project and terminate the episode."""
        self._is_terminal = True
        self._termination_reason = "agent_finalized"
        self._current_phase = "done"
        self._last_agent_output = AgentOutputSummary(
            agent_name="orchestrator", success=True,
            message=f"Project finalized. {self._tasks_completed}/{self._tasks_total} tasks completed.",
        )

    # ── Simulated Sub-agents (for fast training without LLM) ──────────────────

    def _simulate_agent(self, name: str, error: Exception):
        """Fallback to simulation when real agent fails."""
        self._last_agent_output = AgentOutputSummary(
            agent_name=name, success=False,
            message=f"Real agent failed, simulating: {str(error)[:100]}",
            error=str(error)[:200],
        )

    def _simulate_intake(self):
        self._has_specification = True
        self._current_phase = "specification"
        self._last_agent_output = AgentOutputSummary(
            agent_name="intake", success=True,
            message="[SIM] Specification created",
            files_produced=["structured_specification.yaml"],
        )

    def _simulate_architect(self):
        self._has_architecture = True
        self._last_agent_output = AgentOutputSummary(
            agent_name="architect", success=True,
            message="[SIM] Architecture designed",
            files_produced=["architecture.json"],
        )

    def _simulate_planner(self):
        import random
        self._has_plan = True
        self._tasks_total = self._current_spec.expected_tasks if self._current_spec else 8
        self._last_agent_output = AgentOutputSummary(
            agent_name="planner", success=True,
            message=f"[SIM] Plan: {self._tasks_total} tasks",
            files_produced=["implementation_plan.json"],
        )

    def _simulate_qa(self):
        self._current_phase = "task_qa"
        self._last_agent_output = AgentOutputSummary(
            agent_name="qa", success=True,
            message="[SIM] Tests generated",
            files_produced=["tests/test_task.py"],
        )
        self._files_generated.append("tests/test_task.py")

    def _simulate_coder(self):
        import random
        success = random.random() > 0.3  # 70% success rate
        if success:
            self._tasks_completed += 1
            self._tests_passed += random.randint(2, 5)
            self._current_phase = "execution"
            self._error_context = ""
            self._files_generated.append(f"src/module_{self._tasks_completed}.py")
        else:
            self._tests_failed += random.randint(1, 3)
            self._current_phase = "task_recovery"
            self._error_context = "[SIM] Test failures detected"

        self._last_agent_output = AgentOutputSummary(
            agent_name="coder", success=success,
            message=f"[SIM] Code {'passed' if success else 'failed'} tests",
            files_produced=[f"src/module_{self._tasks_completed}.py"] if success else [],
        )

    def _simulate_recovery(self):
        import random
        strategies = ["RETRY_WITH_FIX", "SKIP_TASK", "MODIFY_APPROACH"]
        strategy = random.choice(strategies)

        if strategy == "SKIP_TASK":
            self._tasks_skipped += 1
            self._current_phase = "execution"
            self._error_context = ""
        else:
            self._current_phase = "task_code"

        self._last_agent_output = AgentOutputSummary(
            agent_name="recovery", success=True,
            message=f"[SIM] Strategy: {strategy}",
        )

    def _simulate_security(self):
        import random
        self._last_agent_output = AgentOutputSummary(
            agent_name="security", success=True,
            message=f"[SIM] {random.randint(0, 3)} findings",
            files_produced=["security_report.json"],
        )

    def _simulate_oversight(self):
        import random
        self._oversight_report = OversightReport(
            issues_found=random.randint(0, 5),
            critical_issues=random.randint(0, 1),
            quality_score=random.uniform(0.4, 0.95),
            hallucination_flags=[],
            recommendations=["Consider error handling improvements"],
            agent_behavior_summary="Sub-agents operating within expected parameters",
        )
        self._last_agent_output = AgentOutputSummary(
            agent_name="oversight", success=True,
            message=f"[SIM] Quality: {self._oversight_report.quality_score:.2f}",
        )

    # ── Observation & State Building ──────────────────────────────────────────

    def _build_observation(self) -> ForgeObservation:
        """Build the current observation for the meta-agent."""
        test_total = self._tests_passed + self._tests_failed
        return ForgeObservation(
            current_phase=self._current_phase,
            step_count=self._step_count,
            max_steps=self._max_steps,
            episode_reward=self._total_reward,
            last_agent_output=self._last_agent_output,
            project_state=ProjectSnapshot(
                files_generated=self._files_generated.copy(),
                total_files=len(self._files_generated),
                tests_passed=self._tests_passed,
                tests_failed=self._tests_failed,
                test_pass_rate=(
                    self._tests_passed / max(test_total, 1)
                ),
                has_specification=self._has_specification,
                has_architecture=self._has_architecture,
                has_plan=self._has_plan,
                total_api_calls=self._total_api_calls,
                error_count=self._tasks_failed,
            ),
            task_progress=TaskProgressSummary(
                total_tasks=self._tasks_total,
                completed=self._tasks_completed,
                failed=self._tasks_failed,
                skipped=self._tasks_skipped,
                in_progress=1 if self._current_phase.startswith("task_") else 0,
                pending=max(0, self._tasks_total - self._tasks_completed - self._tasks_failed - self._tasks_skipped),
                percent_complete=(
                    (self._tasks_completed / max(self._tasks_total, 1)) * 100
                ),
                current_task_title=self._current_task_title,
                current_task_id=self._current_task_id,
            ),
            available_actions=self._get_available_actions(),
            oversight_report=self._oversight_report,
            reviewer_feedback=self._reviewer_feedback,
            spec_summary=self._spec_text[:300] if self._spec_text else "",
            spec_tier=self._current_spec.tier if self._current_spec else 1,
            phase_history=self._phase_history.copy(),
            error_context=self._error_context,
        )

    def _get_available_actions(self) -> list[str]:
        """Return valid actions for the current phase."""
        phase = self._current_phase

        phase_actions = {
            "idle": [ActionType.DELEGATE_INTAKE],
            "intake": [ActionType.DELEGATE_INTAKE],
            "specification": [ActionType.DELEGATE_ARCHITECT, ActionType.DELEGATE_OVERSIGHT],
            "architecture": [ActionType.DELEGATE_PLANNER, ActionType.DELEGATE_OVERSIGHT],
            "planning": [ActionType.APPROVE_PLAN, ActionType.REJECT_PLAN, ActionType.DELEGATE_OVERSIGHT],
            "plan_review": [ActionType.APPROVE_PLAN, ActionType.REJECT_PLAN],
            "execution": [
                ActionType.DELEGATE_QA,
                ActionType.DELEGATE_SECURITY,
                ActionType.DELEGATE_OVERSIGHT,
                ActionType.FINALIZE,
            ],
            "task_qa": [ActionType.DELEGATE_CODER, ActionType.SKIP_TASK],
            "task_code": [
                ActionType.APPROVE_CODE,
                ActionType.REJECT_CODE,
                ActionType.DELEGATE_CODER,
                ActionType.SKIP_TASK,
            ],
            "task_recovery": [
                ActionType.DELEGATE_RECOVERY,
                ActionType.RETRY_TASK,
                ActionType.SKIP_TASK,
                ActionType.ESCALATE,
            ],
            "security_audit": [ActionType.FINALIZE, ActionType.DELEGATE_OVERSIGHT],
            "done": [],
        }

        # Always allow these in any non-terminal state
        always_available = [ActionType.PROVIDE_FEEDBACK, ActionType.ESCALATE]

        actions = phase_actions.get(phase, [ActionType.ESCALATE])
        actions.extend(always_available)

        return list(set(a.value for a in actions))

    def _check_terminal(self) -> bool:
        """Check if the episode should terminate."""
        if self._is_terminal:
            return True

        # All tasks completed
        if (
            self._tasks_total > 0
            and self._tasks_completed + self._tasks_failed + self._tasks_skipped
            >= self._tasks_total
        ):
            self._termination_reason = "all_tasks_processed"
            return True

        # Finalized
        if self._current_phase == "done":
            self._termination_reason = "finalized"
            return True

        return False

    def _compute_quality_score(self) -> float:
        """Compute overall code quality score (0.0 - 1.0)."""
        scores = []

        # Test pass rate
        test_total = self._tests_passed + self._tests_failed
        if test_total > 0:
            scores.append(self._tests_passed / test_total)

        # Task completion rate
        if self._tasks_total > 0:
            scores.append(self._tasks_completed / self._tasks_total)

        # Oversight score
        if self._oversight_report:
            scores.append(self._oversight_report.quality_score)

        return sum(scores) / max(len(scores), 1)

    # ── Utility ───────────────────────────────────────────────────────────────

    def cleanup(self):
        """Clean up temporary project directory."""
        if self._project_dir and os.path.exists(self._project_dir):
            try:
                shutil.rmtree(self._project_dir)
            except Exception:
                pass

    def get_curriculum(self) -> AdaptiveCurriculum:
        """Access the curriculum for stats and configuration."""
        return self._curriculum

    def __repr__(self) -> str:
        return (
            f"ForgeEnvironment(episode={self._episode_id}, "
            f"step={self._step_count}, phase={self._current_phase}, "
            f"tier={self._current_spec.tier if self._current_spec else '?'}, "
            f"reward={self._total_reward:.2f})"
        )
