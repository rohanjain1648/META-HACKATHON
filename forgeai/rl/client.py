"""Client for the ForgeAI SDLC RL environment.

This module is the only thing client / training code should import.
It never imports from forgeai.rl.server — that package is server-only.

Usage (remote HuggingFace Spaces deployment)::

    from forgeai.rl import SDLCEnv, CallToolAction

    env = SDLCEnv("https://<username>-forgeai-sdlc.hf.space")
    obs = await env.reset()
    print(obs["prompt"])

    action = CallToolAction(
        tool_name="submit_solution",
        tool_input={"code": "def fibonacci(n): ..."},
    )
    obs, reward, done, info = await env.step(action)

MCPToolClient provides reset(), step(), and state as properties — the
standard Gym-style API — over HTTP to whatever server URL is given.
"""

from openenv.core.mcp_client import MCPToolClient


class SDLCEnv(MCPToolClient):
    """Remote client for the ForgeAI SDLC coding-task RL environment.

    The single MCP tool exposed by the server is ``submit_solution``:

        action = CallToolAction(
            tool_name="submit_solution",
            tool_input={"code": "<python source>"},
        )

    Inherits the full Gym-style interface from MCPToolClient:
        - reset()      → SDLCObservation
        - step(action) → (SDLCObservation, reward, done, info)
        - state        → SDLCState  (property, not a method call)
    """
