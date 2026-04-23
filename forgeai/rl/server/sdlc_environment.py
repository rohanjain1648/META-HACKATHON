"""SDLC RL Environment — MCPEnvironment server implementation.

Inherits from openenv's MCPEnvironment so the framework can wire it into
the standard MCP tool protocol.  The single exposed MCP tool is named
``submit_solution`` — deliberately NOT a reserved name (reset / step /
state / close are reserved and must not be used as tool names).

State machine per episode:
    reset()  → sample task from adaptive curriculum
    submit_solution(code)  → sandbox-execute code → pytest → 5-component reward
    done=True when tests pass OR max_steps exceeded OR anti-cheat triggered

Anti-cheat fires a hard -1.0 penalty — strictly worse than generating
nothing — so reward-hacking is never profitable.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Observation, State

from forgeai.rl.curriculum import CurriculumManager, DifficultyLevel, CodingTask
from forgeai.rl.models import SDLCObservation, SDLCState
from forgeai.rl.reward_functions import RewardEngine
from forgeai.rl.verifier import CodeVerifier


class SDLCEnvironment(MCPEnvironment):
    """MCP-based SDLC coding-task RL environment.

    MCPEnvironment provides:
      - self.mcp  : FastMCP instance (populated by super().__init__())
      - HTTP routing for reset / state endpoints
      - Session management

    This subclass adds:
      - Adaptive curriculum sampling (EASY → MEDIUM → HARD)
      - Sandboxed pytest execution via CodeVerifier
      - 5-component reward engine with anti-cheat hard penalty
      - ``submit_solution`` MCP tool (the agent's action interface)
    """

    def __init__(
        self,
        start_difficulty: DifficultyLevel = DifficultyLevel.EASY,
        timeout_seconds: int = 10,
    ) -> None:
        # MCPEnvironment.__init__ sets up self.mcp (FastMCP instance)
        super().__init__()

        self._curriculum = CurriculumManager(start_level=start_difficulty)
        self._verifier = CodeVerifier(timeout_seconds=timeout_seconds)
        self._reward_engine = RewardEngine(timeout_seconds=timeout_seconds)

        # Per-episode state
        self._episode_id: str = ""
        self._current_task: Optional[CodingTask] = None
        self._step_count: int = 0
        self._last_error: str = ""

        # Aggregate metrics for state property
        self._total_episodes: int = 0
        self._total_rewards: list[float] = []

        self._register_tools()

    # ── OpenEnv interface ─────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> SDLCObservation:
        """Start a new episode by sampling a task from the curriculum."""
        self._episode_id = episode_id or str(uuid.uuid4())[:8]
        self._step_count = 0
        self._last_error = ""
        self._current_task = self._curriculum.sample_task()
        self._total_episodes += 1

        return SDLCObservation(
            task_description=self._current_task.description,
            function_signature=self._current_task.function_signature,
            test_code=self._current_task.test_code,
            previous_error="",
            step=0,
            episode_id=self._episode_id,
        )

    @property
    def state(self) -> SDLCState:
        """Current environment state (property, not a method call)."""
        return SDLCState(
            episode_id=self._episode_id,
            current_task=self._current_task.task_id if self._current_task else None,
            difficulty=self._curriculum.current_level.value,
            step=self._step_count,
            total_episodes=self._total_episodes,
            mean_reward=(
                sum(self._total_rewards) / len(self._total_rewards)
                if self._total_rewards else 0.0
            ),
            recent_success_rate=self._curriculum.stats.recent_success_rate,
        )

    # ── MCP tool registration ─────────────────────────────────────────────

    def _register_tools(self) -> None:
        """Register domain tools with the FastMCP server.

        IMPORTANT: tool names must NOT be any of the reserved OpenEnv names:
        reset, step, state, close.
        """

        @self.mcp.tool()
        def submit_solution(code: str) -> dict:
            """Submit Python code for evaluation against the current task's tests.

            Args:
                code: Raw Python source — no markdown fences, no explanation.

            Returns:
                dict with keys: observation, reward, done, info.
                info.reward_breakdown breaks down all 5 reward components.
            """
            return self._process_submission(code)

    # ── Core episode logic ────────────────────────────────────────────────

    def _process_submission(self, code: str) -> dict:
        """Run code through verifier + reward engine, advance curriculum."""
        if self._current_task is None:
            raise RuntimeError("Call reset() before submitting a solution.")

        self._step_count += 1

        verification = self._verifier.verify(
            generated_code=code,
            test_code=self._current_task.test_code,
            task_signature=self._current_task.function_signature,
        )
        breakdown = self._reward_engine.compute(
            result=verification,
            generated_code=code,
            expected_signature=self._current_task.function_signature,
        )
        reward = breakdown.total
        self._total_rewards.append(reward)

        done = (
            verification.success
            or self._step_count >= 1      # single-step episodes
            or bool(verification.anti_cheat_violations)
        )

        if done:
            self._curriculum.record_result(
                success=verification.success,
                pass_rate=verification.pass_rate,
            )

        # Feed failure context back to the model on next step
        error_context = ""
        if not verification.success and not verification.anti_cheat_violations:
            error_context = (
                verification.stdout[-600:]
                if verification.stdout
                else verification.stderr[-600:]
            )
        self._last_error = error_context

        next_obs = SDLCObservation(
            task_description=self._current_task.description,
            function_signature=self._current_task.function_signature,
            test_code=self._current_task.test_code,
            previous_error=error_context,
            step=self._step_count,
            episode_id=self._episode_id,
        )

        return {
            "observation": next_obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": {
                "episode_id": self._episode_id,
                "task_id": self._current_task.task_id,
                "difficulty": self._current_task.difficulty.value,
                "reward_breakdown": breakdown.to_dict(),
                "verification": verification.to_dict(),
                "step": self._step_count,
            },
        }
