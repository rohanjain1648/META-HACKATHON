"""ForgeAI — Main Entry Point.

Ties together the Orchestrator, CLI, Web Dashboard, and RL training pipeline.

Modes:
    Default  : Run the SDLC pipeline to generate a project from a spec
    --web    : Launch the web dashboard
    --train-rl: Train an LLM with GRPO on the SDLC environment
    --rl-server: Serve the SDLC environment as an OpenEnv-compatible API
"""

import os
import sys
import click
from pathlib import Path
from dotenv import load_dotenv

from forgeai.config.config_manager import ConfigManager
from forgeai.core.orchestrator import Orchestrator
from forgeai.ui.cli_interface import CLIInterface

# Load .env file if it exists
load_dotenv()


@click.command()
@click.option("--spec", "-s", help="Natural language project specification", required=False)
@click.option("--config", "-c", help="Path to YAML config file", default=None)
@click.option("--web", "-w", is_flag=True, help="Launch the web dashboard")
@click.option("--train-rl", is_flag=True, help="Train an LLM via GRPO on the SDLC RL environment")
@click.option("--rl-server", is_flag=True, help="Serve the SDLC RL environment as an OpenEnv API")
@click.option("--rl-model", default="Qwen/Qwen2.5-Coder-3B-Instruct",
              help="[--train-rl] Base model for GRPO training")
@click.option("--rl-steps", default=300, type=int,
              help="[--train-rl] Maximum training steps")
@click.option("--rl-difficulty", default="easy",
              type=click.Choice(["easy", "medium", "hard"]),
              help="[--train-rl] Starting curriculum difficulty level")
@click.option("--rl-port", default=8001, type=int,
              help="[--rl-server] Port for the OpenEnv API server")
def main(spec: str, config: str, web: bool, train_rl: bool, rl_server: bool,
         rl_model: str, rl_steps: int, rl_difficulty: str, rl_port: int):
    """ForgeAI: Agentic Software Development Framework with RL Training."""

    # ── RL Training Mode ──────────────────────────────────────────────────
    if train_rl:
        _run_rl_training(rl_model, rl_steps, rl_difficulty)
        return

    # ── OpenEnv Server Mode ───────────────────────────────────────────────
    if rl_server:
        _run_rl_server(rl_port)
        return

    # ── Standard SDLC Pipeline Mode ──────────────────────────────────────
    try:
        cfg_manager = ConfigManager.get_instance(config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    cli = CLIInterface()
    cli.clear()
    cli.print_banner()

    if web or cfg_manager.web_dashboard.enabled:
        cli.console.print("[yellow]Web dashboard enabled. URL: http://127.0.0.1:8000[/yellow]")

    orchestrator = Orchestrator(cfg_manager)
    orchestrator.set_callbacks(
        on_phase_change=cli.show_phase_change,
        on_checkpoint=cli.request_checkpoint,
        on_question=cli.ask_questions,
        on_task_progress=cli.show_task_progress,
        on_diff_review=cli.show_diff_review,
    )

    final_spec = spec
    if not final_spec:
        final_spec = click.prompt("\n[bold cyan]What would you like to build?[/bold cyan]")

    summary = orchestrator.run(final_spec)
    cli.show_summary(summary)


# ---------------------------------------------------------------------------
# RL sub-commands
# ---------------------------------------------------------------------------

def _run_rl_training(model_name: str, max_steps: int, difficulty: str) -> None:
    """Launch the GRPO training pipeline."""
    try:
        from forgeai.rl.trainer import TrainingConfig, run_training
    except ImportError as e:
        print(f"RL dependencies not installed: {e}")
        print("Run: pip install trl>=0.9.0 unsloth transformers datasets accelerate peft")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  ForgeAI-RL: GRPO Training with TRL + Unsloth")
    print(f"{'='*60}")
    print(f"  Base model : {model_name}")
    print(f"  Max steps  : {max_steps}")
    print(f"  Difficulty : {difficulty}")
    print(f"  Environment: SDLC Code Generation (OpenEnv-compatible)")
    print(f"{'='*60}\n")

    config = TrainingConfig(
        model_name=model_name,
        max_steps=max_steps,
        start_difficulty=difficulty,
    )

    save_path = run_training(config)
    print(f"\nTraining complete. Model saved to: {save_path}")
    print("Push to HuggingFace Hub: huggingface-cli upload <your-repo> " + save_path)


def _run_rl_server(port: int) -> None:
    """Launch the OpenEnv-compatible FastAPI server."""
    try:
        import uvicorn
        from forgeai.rl.server.app import app
    except ImportError as e:
        print(f"Server dependencies not installed: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  ForgeAI-RL: OpenEnv SDLC Environment Server")
    print(f"{'='*60}")
    print(f"  API docs   : http://0.0.0.0:{port}/docs")
    print(f"  Health     : http://0.0.0.0:{port}/health")
    print(f"  OpenEnv    : openenv.yaml → forgeai.rl.server.app:app")
    print(f"{'='*60}\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
