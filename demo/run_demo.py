"""ForgeRL Interactive Demo — Shows the environment in action.

This script provides a compelling demo of the ForgeRL environment:
1. Resets with a Tier 1 specification
2. Walks through the full SDLC workflow with the heuristic policy
3. Shows oversight agent catching issues
4. Demonstrates recovery from test failures
5. Displays the final reward breakdown
6. Generates a visual reward chart

Usage:
    python demo/run_demo.py
    python demo/run_demo.py --tier 2 --interactive
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.live import Live
    from rich.columns import Columns
    from rich.markdown import Markdown

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from forge_env.environment import ForgeEnvironment
from forge_env.models import ForgeAction, ActionType


class ForgeRLDemo:
    """Interactive demo of the ForgeRL environment."""

    def __init__(self, tier: int = 1, interactive: bool = False, use_real_llm: bool = False):
        self.tier = tier
        self.interactive = interactive
        self.env = ForgeEnvironment(
            use_real_llm=use_real_llm,
            max_steps=100,
        )
        self.console = Console(force_terminal=True) if RICH_AVAILABLE else None
        self.step_log = []

    def print(self, text: str, style: str = ""):
        if self.console:
            self.console.print(text, style=style)
        else:
            print(text)

    def print_banner(self):
        if self.console:
            banner = Text()
            banner.append("[ForgeRL]", style="bold cyan")
            banner.append("\nMulti-Agent Software Engineering RL Environment", style="italic")
            banner.append("\n\nMeta PyTorch x HuggingFace OpenEnv Hackathon", style="dim")
            self.console.print(Panel(banner, border_style="cyan", padding=(1, 3)))
        else:
            print("=" * 50)
            print("  ForgeRL Demo")
            print("=" * 50)

    def print_observation(self, obs, step_num: int):
        if self.console:
            table = Table(title=f"Step {step_num} — Observation", border_style="cyan", show_lines=True)
            table.add_column("Field", style="bold cyan", width=20)
            table.add_column("Value", width=50)

            table.add_row("Phase", f"[bold yellow]{obs.current_phase}[/]")
            table.add_row("Step", f"{obs.step_count}/{obs.max_steps}")
            table.add_row("Reward", f"[green]{obs.episode_reward:.3f}[/]")
            table.add_row(
                "Tasks",
                f"{obs.task_progress.completed}/{obs.task_progress.total_tasks} "
                f"({obs.task_progress.percent_complete:.0f}%)",
            )
            table.add_row(
                "Tests",
                f"[green]{obs.project_state.tests_passed}[/] passed, "
                f"[red]{obs.project_state.tests_failed}[/] failed",
            )
            table.add_row("Files", str(obs.project_state.total_files))
            table.add_row("Last Agent", obs.last_agent_output.agent_name)
            table.add_row(
                "Agent Status",
                f"[green]✓[/]" if obs.last_agent_output.success else f"[red]✗[/]",
            )
            table.add_row("Message", obs.last_agent_output.message[:80])

            if obs.error_context:
                table.add_row("Error", f"[red]{obs.error_context[:80]}[/]")

            if obs.oversight_report:
                table.add_row(
                    "Oversight",
                    f"Quality: {obs.oversight_report.quality_score:.2f}, "
                    f"Issues: {obs.oversight_report.issues_found}",
                )

            self.console.print(table)
        else:
            print(f"\n--- Step {step_num} ---")
            print(f"Phase: {obs.current_phase}")
            print(f"Reward: {obs.episode_reward:.3f}")

    def print_action(self, action: ForgeAction, reward: float):
        if self.console:
            color = "green" if reward > 0 else "red" if reward < 0 else "yellow"
            self.console.print(
                f"  → Action: [bold]{action.action_type.value}[/] "
                f"| Reward: [{color}]{reward:+.3f}[/] "
                f"| Reason: {action.reasoning[:60]}"
            )

    def print_summary(self, state, rewards: list[float]):
        if self.console:
            summary = Table(title="Episode Summary", border_style="green", show_lines=True)
            summary.add_column("Metric", style="bold", width=25)
            summary.add_column("Value", width=30)

            summary.add_row("Episode ID", state.episode_id)
            summary.add_row("Tier", str(state.spec_tier))
            summary.add_row("Total Steps", str(state.step_count))
            summary.add_row("Total Reward", f"[bold green]{state.total_reward:.3f}[/]")
            summary.add_row("Test Pass Rate", f"{state.true_test_pass_rate:.1%}")
            summary.add_row("Quality Score", f"{state.true_code_quality_score:.2f}")
            summary.add_row("Files Generated", str(state.true_files_generated))
            summary.add_row("API Calls", str(state.total_api_calls))
            summary.add_row("Termination", state.termination_reason)

            self.console.print(summary)

            # Phases visited
            phases_text = " → ".join(state.phase_trace[:15])
            self.console.print(f"\n[dim]Phase Trace: {phases_text}[/dim]")

    async def run(self):
        """Run the full demo."""
        self.print_banner()

        # ── 1. Reset Environment ──
        self.print("\n[bold cyan]1. Resetting Environment[/bold cyan]", style="")
        reset_result = await self.env.reset(tier=self.tier)
        obs = reset_result.observation

        self.print(
            f"  Episode: {reset_result.info['episode_id']}\n"
            f"  Tier: {reset_result.info['tier']}\n"
            f"  Spec: {reset_result.info['spec_name']}\n"
            f"  Max Steps: {reset_result.info['max_steps']}\n"
            f"  Reviewer: {reset_result.info['reviewer']}",
        )

        # ── 2. Run Episode ──
        self.print("\n[bold cyan]2. Running Episode (Heuristic Policy)[/bold cyan]")

        rewards = []
        step = 0

        # Define the SDLC workflow
        workflow = [
            (ActionType.DELEGATE_INTAKE, "Starting requirements analysis"),
            (ActionType.DELEGATE_ARCHITECT, "Designing system architecture"),
            (ActionType.DELEGATE_PLANNER, "Decomposing into atomic tasks"),
            (ActionType.APPROVE_PLAN, "Plan looks good, proceeding to execution"),
            (ActionType.DELEGATE_QA, "Writing TDD tests for first task"),
            (ActionType.DELEGATE_CODER, "Generating code to pass tests"),
            (ActionType.DELEGATE_OVERSIGHT, "Running quality oversight check"),
            (ActionType.DELEGATE_QA, "Tests for second task"),
            (ActionType.DELEGATE_CODER, "Code for second task"),
            (ActionType.DELEGATE_QA, "Tests for third task"),
            (ActionType.DELEGATE_CODER, "Code for third task"),
            (ActionType.DELEGATE_OVERSIGHT, "Mid-project quality check"),
            (ActionType.DELEGATE_SECURITY, "Security audit on generated code"),
            (ActionType.FINALIZE, "All tasks complete, finalizing project"),
        ]

        for action_type, reasoning in workflow:
            if obs.current_phase == "done" or step >= 50:
                break

            # Check if action is valid
            available = obs.available_actions
            if action_type.value not in available:
                # Fall back to first available
                if available:
                    try:
                        action_type = ActionType(available[0])
                        reasoning = f"Fallback: {available[0]}"
                    except ValueError:
                        continue
                else:
                    continue

            action = ForgeAction(
                action_type=action_type,
                reasoning=reasoning,
            )

            result = await self.env.step(action)
            obs = result.observation
            rewards.append(result.reward)
            step += 1

            self.print_action(action, result.reward)

            if step % 4 == 0 or result.terminated:
                self.print_observation(obs, step)

            if result.terminated:
                break

        # ── 3. Summary ──
        self.print("\n[bold cyan]3. Episode Results[/bold cyan]")
        state = self.env.state
        self.print_summary(state, rewards)

        # ── 4. Reward Chart ──
        self.print("\n[bold cyan]4. Generating Reward Analysis[/bold cyan]")
        try:
            self._plot_demo_rewards(rewards)
            self.print("  ✓ Reward chart saved to demo/demo_rewards.png")
        except Exception as e:
            self.print(f"  ⚠ Could not generate chart: {e}")

        # ── 5. Theme Alignment ──
        if self.console:
            themes = Table(title="Hackathon Theme Coverage", border_style="magenta")
            themes.add_column("Theme", style="bold")
            themes.add_column("How ForgeRL Addresses It")
            themes.add_column("Evidence", style="dim")

            themes.add_row(
                "Multi-Agent Interactions",
                "Meta-agent coordinates 9 sub-agents",
                f"Phases visited: {len(state.phase_trace)}",
            )
            themes.add_row(
                "Long-Horizon Reasoning",
                f"{state.step_count} steps with sparse terminal reward",
                f"Reward: {state.total_reward:.2f}",
            )
            themes.add_row(
                "Professional Tasks",
                "Real file system, pytest, Docker",
                f"Files: {state.true_files_generated}",
            )
            themes.add_row(
                "Self-Improvement",
                "Adaptive curriculum Tier 1→5",
                f"Current tier: {state.spec_tier}",
            )
            themes.add_row(
                "Fleet AI Oversight",
                "Oversight agent monitors all agents",
                f"Quality: {state.true_code_quality_score:.2f}",
            )
            themes.add_row(
                "Snorkel Expert-in-Loop",
                "Simulated reviewer with changing prefs",
                "Preferences may shift mid-episode",
            )
            themes.add_row(
                "Mercor Token Scaling",
                "Rewards scale with useful tokens",
                f"Tokens: {state.total_tokens_used}",
            )

            self.console.print(themes)

        self.env.cleanup()
        self.print("\n[bold green]Demo complete![/bold green]")

    def _plot_demo_rewards(self, rewards: list[float]):
        """Generate a demo reward plot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        os.makedirs("demo", exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("ForgeRL Demo — Reward Analysis", fontweight="bold")

        # Per-step rewards
        colors = ["#10b981" if r >= 0 else "#ef4444" for r in rewards]
        ax1.bar(range(len(rewards)), rewards, color=colors, alpha=0.7)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Reward")
        ax1.set_title("Per-Step Rewards")
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax1.grid(True, alpha=0.3)

        # Cumulative reward
        cumulative = np.cumsum(rewards)
        ax2.plot(cumulative, color="#6366f1", linewidth=2)
        ax2.fill_between(range(len(cumulative)), cumulative, alpha=0.2, color="#6366f1")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Cumulative Reward")
        ax2.set_title("Cumulative Episode Reward")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("demo/demo_rewards.png", dpi=150)
        plt.close()


async def main():
    parser = argparse.ArgumentParser(description="ForgeRL Interactive Demo")
    parser.add_argument("--tier", type=int, default=1, help="Spec difficulty tier (1-5)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--real-llm", action="store_true", help="Use real LLM (requires API key)")
    args = parser.parse_args()

    demo = ForgeRLDemo(
        tier=args.tier,
        interactive=args.interactive,
        use_real_llm=args.real_llm,
    )
    await demo.run()


if __name__ == "__main__":
    asyncio.run(main())
