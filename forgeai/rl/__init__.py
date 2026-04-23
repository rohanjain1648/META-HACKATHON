"""ForgeAI RL — Reinforcement Learning layer for the SDLC pipeline.

Stack:
    OpenEnv MCPEnvironment  →  TRL GRPOTrainer  →  Unsloth 4-bit QLoRA

Public API (client-side only — never import forgeai.rl.server here):
    SDLCEnv           — MCPToolClient for connecting to the hosted environment
    SDLCObservation   — typed observation returned by reset() / step()
    SDLCState         — typed state returned by env.state property
    CallToolAction    — action wrapper: CallToolAction("submit_solution", {"code": ...})
    ListToolsAction   — introspect available tools from the environment
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from forgeai.rl.client import SDLCEnv
from forgeai.rl.models import SDLCAction, SDLCObservation, SDLCState

# Training utilities (no MCP dependency — used for local GRPO training)
from forgeai.rl.curriculum import CurriculumManager, DifficultyLevel
from forgeai.rl.reward_functions import RewardEngine, RewardBreakdown
from forgeai.rl.verifier import CodeVerifier

__all__ = [
    # Client interface
    "SDLCEnv",
    "SDLCAction",
    "SDLCObservation",
    "SDLCState",
    "CallToolAction",
    "ListToolsAction",
    # Training utilities
    "CurriculumManager",
    "DifficultyLevel",
    "RewardEngine",
    "RewardBreakdown",
    "CodeVerifier",
]
