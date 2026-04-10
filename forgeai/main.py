"""ForgeAI — Main Entry Point.

Ties together the Orchestrator, CLI, and Web Dashboard.
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
def main(spec: str, config: str, web: bool):
    """ForgeAI: Agentic Software Development Framework."""
    
    # 1. Initialize Configuration
    try:
        cfg_manager = ConfigManager.get_instance(config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # 2. Initialize UI
    cli = CLIInterface()
    cli.clear()
    cli.print_banner()

    # 3. Launch Web Dashboard in background if requested
    if web or cfg_manager.web_dashboard.enabled:
        cli.console.print("[yellow]Web dashboard enabled. URL: http://127.0.0.1:8000[/yellow]")
        # We would normally start the FastAPI server in a thread here
        # or separate process. For simplicity in the demo, we focus on the core flow.

    # 4. Initialize Orchestrator
    orchestrator = Orchestrator(cfg_manager)

    # 5. Connect UI to Orchestrator
    orchestrator.set_callbacks(
        on_phase_change=cli.show_phase_change,
        on_checkpoint=cli.request_checkpoint,
        on_question=cli.ask_questions,
        on_task_progress=cli.show_task_progress,
        on_diff_review=cli.show_diff_review
    )

    # 6. Get specification from user if not provided in CLI
    final_spec = spec
    if not final_spec:
        final_spec = click.prompt("\n[bold cyan]What would you like to build?[/bold cyan]")

    # 7. Run Pipeline
    summary = orchestrator.run(final_spec)

    # 8. Show Results
    cli.show_summary(summary)

if __name__ == "__main__":
    main()
