"""OpenEnv type models for the SDLC RL environment.

These are the transport types shared by client and server.
Inheriting from openenv base classes ensures compatibility with
the MCPToolClient / MCPEnvironment machinery.
"""

from __future__ import annotations

from typing import Optional

from openenv.core.env_server.types import Action, Observation, State


class SDLCAction(Action):
    """One agent action: raw Python code submitted for evaluation."""
    code: str


class SDLCObservation(Observation):
    """What the agent sees at each step of an episode."""
    task_description: str
    function_signature: str
    test_code: str
    previous_error: str = ""
    step: int = 0
    episode_id: str = ""

    def to_prompt(self) -> str:
        """Build the LLM prompt string from this observation."""
        parts = [
            "You are an expert Python developer. Implement the following function/class.",
            "",
            f"## Task\n{self.task_description}",
            "",
            f"## Required Signature\n```python\n{self.function_signature}\n```",
            "",
            "## Tests Your Implementation Must Pass",
            f"```python\n{self.test_code}\n```",
        ]
        if self.previous_error:
            parts += [
                "",
                "## Previous Attempt Failed With",
                f"```\n{self.previous_error[:800]}\n```",
                "Fix the implementation so all tests pass.",
            ]
        parts += [
            "",
            "## Your Implementation",
            "Write ONLY the Python code (no markdown, no explanation):",
            "",
        ]
        return "\n".join(parts)


class SDLCState(State):
    """Full environment state — used for monitoring and health checks."""
    episode_id: str = ""
    current_task: Optional[str] = None
    difficulty: str = "easy"
    step: int = 0
    total_episodes: int = 0
    mean_reward: float = 0.0
    recent_success_rate: float = 0.0
