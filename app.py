"""HuggingFace Spaces Entry Point — ForgeAI-RL OpenEnv Environment.

This file is the entry point for the HuggingFace Space.
It serves the SDLCEnvironment as an OpenEnv-compatible REST API.

HF Spaces URL structure:
    https://huggingface.co/spaces/<your-username>/forgeai-rl

The environment can be used by TRL training scripts via:
    from openenv.client import OpenEnvClient
    env = OpenEnvClient("https://<username>-forgeai-rl.hf.space")
    obs = env.reset()
    obs, reward, done, info = env.step(generated_code)

Or via HTTP directly:
    POST /reset  → {session_id, observation}
    POST /step   → {observation, reward, done, info}
    GET  /health → {status, active_sessions}
    GET  /state  → {state}

Deployment on HF Spaces:
    1. Create a new Space (Docker SDK)
    2. Push this repo to the Space
    3. HF Spaces will build the Dockerfile and run this app
"""

import os

import uvicorn
from forgeai.rl.server.app import app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))   # HF Spaces default port
    uvicorn.run(
        app,          # single-worker: pass object directly (safe with workers=1)
        host="0.0.0.0",
        port=port,
        log_level="info",
        workers=1,    # Single worker for stateful session management
    )
