"""CLI Interface — Rich terminal UI for ForgeAI.

Provides animated panels, real-time log streaming, and interactive
checkpoints for a premium developer experience.
"""

import sys
import time
from typing import Any, Dict, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.logging import RichHandler

from forgeai.models.workflow_state import WorkflowPhase
from forgeai.models.task import AtomicTask, TaskStatus

class CLIInterface:
    """Rich CLI for interacting with the ForgeAI framework."""

    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.current_phase = WorkflowPhase.IDLE
        self.log_entries = []

    def clear(self):
        self.console.clear()

    def print_banner(self):
        banner = Text("FORGE AI", style="bold cyan", justify="center")
        banner.append("\nAgentic Software Development Framework", style="italic white")
        self.console.print(Panel(banner, border_style="cyan"))

    def show_phase_change(self, phase: WorkflowPhase):
        self.current_phase = phase
        self.console.print(f"\n[bold cyan]▶ Transitioning to {phase.value.upper()}[/bold cyan]")

    def ask_questions(self, questions: List[str]) -> Dict[str, str]:
        self.console.print("\n[bold yellow]The Intake Agent needs more information:[/bold yellow]")
        answers = {}
        for q in questions:
            ans = Prompt.ask(f"[cyan]{q}[/cyan]")
            answers[q] = ans
        return answers

    def request_checkpoint(self, title: str, content: str) -> bool:
        self.console.print(f"\n[bold magenta]━━━━━━━━━ CHECKPOINT: {title} ━━━━━━━━━[/bold magenta]")
        self.console.print(content)
        return Confirm.ask("[bold yellow]Do you approve this step to proceed?[/bold yellow]", default=True)

    def show_task_progress(self, task: AtomicTask, progress_data: Dict[str, Any]):
        status_color = {
            TaskStatus.PASSED: "green",
            TaskStatus.FAILED: "red",
            TaskStatus.IN_PROGRESS: "yellow",
            TaskStatus.RETRYING: "magenta",
            TaskStatus.PENDING: "white"
        }.get(task.status, "white")
        
        table = Table(title=f"Task Progress: {progress_data['percent_complete']}%", border_style="cyan")
        table.add_column("Task ID", justify="right")
        table.add_column("Title")
        table.add_column("Status")
        
        table.add_row(
            str(task.id),
            task.title,
            f"[{status_color}]{task.status.value}[/{status_color}]"
        )
        self.console.print(table)

    def show_diff_review(self, task: AtomicTask, files: Dict[str, str]) -> bool:
        self.console.print(f"\n[bold yellow]Reviewing changes for Task #{task.id}: {task.title}[/bold yellow]")
        for filepath, content in files.items():
            self.console.print(Panel(
                Syntax(content, "python", theme="monokai", line_numbers=True),
                title=f"[cyan]{filepath}[/cyan]",
                border_style="green"
            ))
        return Confirm.ask("[bold yellow]Apply these changes?[/bold yellow]", default=True)

    def show_summary(self, summary: Dict[str, Any]):
        self.console.print("\n[bold green]━━━━━━━━━━━━━ PIPELINE SUMMARY ━━━━━━━━━━━━━[/bold green]")
        table = Table(show_header=False, box=None)
        table.add_row("Status", f"[bold green]{summary['status']}[/bold green]")
        table.add_row("Tasks Completed", str(summary['tasks_completed']))
        table.add_row("Files Generated", str(summary['files_generated']))
        table.add_row("Tests Passed", f"[green]{summary['tests_passed']}[/green]")
        table.add_row("Tests Failed", f"[red]{summary['tests_failed']}[/red]")
        table.add_row("Total API Calls", str(summary['total_api_calls']))
        table.add_row("Duration", f"{summary['duration_seconds']:.1f}s")
        
        self.console.print(Panel(table, title="Results", border_style="green"))

        if summary.get('errors'):
          self.console.print("\n[bold red]Errors encountered:[/bold red]")
          for err in summary['errors']:
            self.console.print(f"- {err}")

    def log(self, entry: Any):
        # We can pipe the ActivityLogger here if we want real-time rich logs
        pass
