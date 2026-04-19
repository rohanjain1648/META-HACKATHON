"""ForgeRL — Multi-Agent Software Engineering RL Environment.

An OpenEnv-compatible reinforcement learning environment where an LLM
meta-agent learns to orchestrate a team of specialized sub-agents
(Intake, Architect, Planner, QA, Coder, Recovery, Security, Oversight)
to autonomously build working software from natural-language specifications.

Themes covered:
  - Multi-Agent Interactions (Fleet AI + Halluminate)
  - Long-Horizon Reasoning (50-300+ step episodes)
  - Professional Tasks (real file system, pytest, Docker)
  - Self-Improvement (adaptive difficulty curriculum)
"""

__version__ = "1.0.0"
__author__ = "ForgeRL Team"

from forge_env.environment import ForgeEnvironment
from forge_env.models import ForgeAction, ForgeObservation, ForgeState
from forge_env.reward import ForgeRewardCalculator
from forge_env.curriculum import AdaptiveCurriculum

__all__ = [
    "ForgeEnvironment",
    "ForgeAction",
    "ForgeObservation",
    "ForgeState",
    "ForgeRewardCalculator",
    "AdaptiveCurriculum",
]
