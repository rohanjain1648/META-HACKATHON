"""FastAPI application for the ForgeAI SDLC RL environment.

Uses openenv's create_app() factory so the server follows the standard
OpenEnv HTTP protocol — any OpenEnv-compatible client can connect without
knowing the environment's internal details.

The app is referenced in openenv.yaml as:
    app: forgeai.rl.server.app:app

Run locally:
    uvicorn forgeai.rl.server.app:app --host 0.0.0.0 --port 7860 --reload

HuggingFace Spaces Dockerfile CMD:
    python app.py   (which calls uvicorn on this module)
"""

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from forgeai.rl.server.sdlc_environment import SDLCEnvironment

app = create_app(
    SDLCEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="forgeai_sdlc",
)
