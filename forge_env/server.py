"""ForgeRL OpenEnv HTTP Server — Deploys the environment as an API.

Uses OpenEnv's create_fastapi_app pattern to expose the ForgeEnvironment
as an HTTP service with Gymnasium-style reset/step/state endpoints.

Deployment options:
  1. Local: `uvicorn forge_env.server:app --host 0.0.0.0 --port 7860`
  2. Docker: Build and run the Dockerfile
  3. HuggingFace Space: Deploy as a Gradio/FastAPI Space
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from forge_env.environment import ForgeEnvironment
from forge_env.models import (
    ForgeAction,
    ForgeObservation,
    ForgeState,
    ResetResult,
    StepResult,
)


# ── Environment Instance ──────────────────────────────────────────────────────

_env: ForgeEnvironment | None = None


def get_env() -> ForgeEnvironment:
    """Get or create the environment singleton."""
    global _env
    if _env is None:
        _env = ForgeEnvironment(
            api_key=os.environ.get("GOOGLE_API_KEY", ""),
            use_real_llm=bool(os.environ.get("GOOGLE_API_KEY")),
            max_steps=int(os.environ.get("FORGERL_MAX_STEPS", "200")),
        )
    return _env


# ── FastAPI App ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize environment on startup."""
    get_env()
    yield
    # Cleanup on shutdown
    if _env:
        _env.cleanup()


app = FastAPI(
    title="ForgeRL — Multi-Agent Software Engineering RL Environment",
    description=(
        "An OpenEnv-compatible environment where an LLM meta-agent learns "
        "to orchestrate specialized sub-agents to build software autonomously. "
        "Covers Multi-Agent Interactions, Long-Horizon Reasoning, Professional "
        "Tasks, and Self-Improvement themes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ───────────────────────────────────────────────────


class ResetRequest(BaseModel):
    spec_text: str | None = None
    tier: int | None = None


class StepRequest(BaseModel):
    action: ForgeAction


class EnvironmentInfo(BaseModel):
    name: str = "ForgeRL"
    version: str = "1.0.0"
    description: str = "Multi-Agent Software Engineering RL Environment"
    action_space: list[str] = []
    themes: list[str] = [
        "Multi-Agent Interactions",
        "Long-Horizon Reasoning",
        "Professional Tasks (World Modeling)",
        "Self-Improvement",
    ]
    sub_themes: list[str] = [
        "Fleet AI: Scalable Oversight",
        "Halluminate: Multi-Actor Environments",
        "Mercor: Token-Scaled Rewards",
        "Snorkel AI: Simulated Experts-in-the-Loop",
    ]
    max_steps: int = 300
    num_sub_agents: int = 9
    difficulty_tiers: int = 5


# ── API Endpoints (OpenEnv-compatible) ────────────────────────────────────────


@app.get("/", response_model=EnvironmentInfo)
async def root():
    """Environment metadata and description."""
    from forge_env.models import ActionType
    return EnvironmentInfo(
        action_space=[a.value for a in ActionType],
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "environment": "ForgeRL", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResult)
async def reset(request: ResetRequest = ResetRequest()):
    """Reset the environment for a new episode.

    Optionally provide a custom specification text and/or tier.
    If not provided, the adaptive curriculum selects automatically.
    """
    env = get_env()
    try:
        result = await env.reset(
            spec_text=request.spec_text,
            tier=request.tier,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResult)
async def step(request: StepRequest):
    """Execute one orchestration step.

    The meta-agent provides an action, and the environment returns
    the resulting observation, reward, and termination flags.
    """
    env = get_env()
    try:
        result = await env.step(request.action)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=ForgeState)
async def get_state():
    """Get the full internal environment state (for debugging)."""
    env = get_env()
    return env.state


@app.get("/curriculum")
async def get_curriculum():
    """Get adaptive curriculum statistics."""
    env = get_env()
    return env.get_curriculum().get_stats()


@app.get("/actions")
async def get_available_actions():
    """Get the list of valid actions in the current state."""
    env = get_env()
    obs = env._build_observation()
    return {
        "current_phase": obs.current_phase,
        "available_actions": obs.available_actions,
        "step_count": obs.step_count,
        "max_steps": obs.max_steps,
    }


# ── Gradio UI for HuggingFace Spaces ─────────────────────────────────────────


def create_gradio_ui():
    """Create a Gradio interface for interactive environment exploration."""
    try:
        import gradio as gr
    except ImportError:
        return None

    env = get_env()

    async def reset_env(spec_text, tier):
        tier_val = int(tier) if tier else None
        result = await env.reset(
            spec_text=spec_text if spec_text else None,
            tier=tier_val,
        )
        return (
            f"Episode: {result.info.get('episode_id', '?')}\n"
            f"Tier: {result.info.get('tier', '?')}\n"
            f"Spec: {result.info.get('spec_name', '?')}\n"
            f"Max Steps: {result.info.get('max_steps', '?')}",
            result.observation.model_dump_json(indent=2),
        )

    async def step_env(action_type, reasoning):
        from forge_env.models import ActionType as AT
        action = ForgeAction(
            action_type=AT(action_type),
            reasoning=reasoning,
        )
        result = await env.step(action)
        reward_info = result.info.get("reward_breakdown", {})
        return (
            f"Reward: {result.reward:.3f}\n"
            f"Phase: {result.observation.current_phase}\n"
            f"Step: {result.observation.step_count}/{result.observation.max_steps}\n"
            f"Terminal: {result.terminated}\n"
            f"Tasks: {result.observation.task_progress.completed}/{result.observation.task_progress.total_tasks}",
            result.observation.model_dump_json(indent=2),
            "\n".join(f"  {k}: {v:.3f}" for k, v in reward_info.items() if v != 0),
        )

    from forge_env.models import ActionType as AT
    action_choices = [a.value for a in AT]

    with gr.Blocks(title="ForgeRL Environment", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🔨 ForgeRL — Multi-Agent Software Engineering RL Environment\n"
            "An OpenEnv-compatible environment where an LLM learns to orchestrate "
            "sub-agents to build software autonomously."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Reset Environment")
                spec_input = gr.Textbox(
                    label="Specification (leave empty for curriculum)",
                    lines=3,
                    placeholder="Build a REST API for...",
                )
                tier_input = gr.Dropdown(
                    choices=["1", "2", "3", "4", "5"],
                    label="Tier",
                    value="1",
                )
                reset_btn = gr.Button("🔄 Reset", variant="primary")
                reset_output = gr.Textbox(label="Episode Info", lines=5)

            with gr.Column(scale=1):
                gr.Markdown("### Take Action")
                action_input = gr.Dropdown(
                    choices=action_choices,
                    label="Action Type",
                    value="delegate_intake",
                )
                reasoning_input = gr.Textbox(
                    label="Reasoning",
                    placeholder="Why are you taking this action?",
                )
                step_btn = gr.Button("▶️ Step", variant="primary")
                step_output = gr.Textbox(label="Step Result", lines=5)

        with gr.Row():
            obs_output = gr.JSON(label="Observation")
            reward_output = gr.Textbox(label="Reward Breakdown", lines=8)

        reset_btn.click(reset_env, [spec_input, tier_input], [reset_output, obs_output])
        step_btn.click(
            step_env,
            [action_input, reasoning_input],
            [step_output, obs_output, reward_output],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    # Check if we should launch Gradio (for HF Spaces)
    if os.environ.get("SPACE_ID") or os.environ.get("GRADIO_UI"):
        demo = create_gradio_ui()
        if demo:
            # Mount FastAPI inside Gradio
            import gradio as gr
            demo.launch(server_name="0.0.0.0", server_port=7860)
        else:
            uvicorn.run(app, host="0.0.0.0", port=7860)
    else:
        uvicorn.run(app, host="0.0.0.0", port=7860)
