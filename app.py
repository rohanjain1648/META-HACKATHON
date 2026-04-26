"""ForgeRL — HuggingFace Space (Gradio SDK)

Interactive demo of ForgeRL: a Multi-Agent Software Engineering RL Environment
compatible with Meta's OpenEnv standard.

A meta-agent orchestrates 9 specialized sub-agents (Intake, Architect, Planner,
QA, Coder, Recovery, Security, Oversight, Reviewer) to autonomously build
working software from natural-language specifications.

Hackathon: Meta PyTorch × HuggingFace OpenEnv 2026
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from forge_env.environment import ForgeEnvironment
from forge_env.models import ActionType, ForgeAction

# ── Environment singleton ─────────────────────────────────────────────────────

_env: ForgeEnvironment | None = None


def get_env() -> ForgeEnvironment:
    global _env
    if _env is None:
        _env = ForgeEnvironment(
            api_key=os.environ.get("GOOGLE_API_KEY", ""),
            use_real_llm=bool(os.environ.get("GOOGLE_API_KEY")),
            max_steps=int(os.environ.get("FORGERL_MAX_STEPS", "100")),
        )
    return _env


def _fmt_obs(obs) -> str:
    try:
        return obs.model_dump_json(indent=2)
    except Exception:
        return json.dumps({"error": "Could not serialize observation"}, indent=2)


# ── Gradio handlers (async — Gradio 4+ natively awaits them) ─────────────────


async def reset_env(spec_text: str, tier: str):
    env = get_env()
    result = await env.reset(
        spec_text=spec_text.strip() or None,
        tier=int(tier),
    )
    obs = result.observation
    info = (
        f"Episode : {result.info.get('episode_id', '?')}\n"
        f"Tier    : {result.info.get('tier', '?')}\n"
        f"Spec    : {result.info.get('spec_name', '?')}\n"
        f"Steps   : {result.info.get('max_steps', '?')} max\n"
        f"Reviewer: {result.info.get('reviewer', '?')}"
    )
    return info, _fmt_obs(obs), [], "—"


async def step_env(action_type: str, reasoning: str, history: list):
    env = get_env()
    if not action_type:
        return "Select an action first.", "{}", history, "—"

    try:
        action = ForgeAction(
            action_type=ActionType(action_type),
            reasoning=reasoning or "Agent decision",
        )
    except ValueError as exc:
        return f"Invalid action: {exc}", "{}", history, "—"

    result = await env.step(action)
    obs = result.observation
    breakdown = result.info.get("reward_breakdown", {})

    step_text = (
        f"Step    : {obs.step_count}/{obs.max_steps}\n"
        f"Phase   : {obs.current_phase}\n"
        f"Reward  : {result.reward:+.3f}\n"
        f"Tasks   : {obs.task_progress.completed}/{obs.task_progress.total_tasks}"
        f" ({obs.task_progress.percent_complete:.0f}%)\n"
        f"Tests   : {obs.project_state.tests_passed}P / "
        f"{obs.project_state.tests_failed}F\n"
        f"Files   : {obs.project_state.total_files}\n"
        f"Terminal: {result.terminated}"
    )

    bd_lines = "\n".join(
        f"  {k}: {v:+.3f}" for k, v in breakdown.items() if v != 0
    ) or "No non-zero components yet."

    history = list(history) + [
        [f"Step {obs.step_count}: {action_type}", f"{result.reward:+.3f} | {obs.current_phase}"]
    ]

    return step_text, _fmt_obs(obs), history, bd_lines


async def run_auto_demo(tier: str) -> str:
    """Run a full episode using the heuristic SDLC policy."""
    env = get_env()
    lines = []

    reset_result = await env.reset(tier=int(tier))
    obs = reset_result.observation
    lines.append(
        f"╔═══════════════════════════════════════════╗\n"
        f"  ForgeRL Auto-Demo — Tier {reset_result.info.get('tier')}\n"
        f"  Spec: {reset_result.info.get('spec_name')}\n"
        f"  Max Steps: {reset_result.info.get('max_steps')}\n"
        f"  Reviewer: {reset_result.info.get('reviewer')}\n"
        f"╚═══════════════════════════════════════════╝"
    )

    # Optimal heuristic workflow
    workflow = [
        (ActionType.DELEGATE_INTAKE,    "Start requirements analysis"),
        (ActionType.DELEGATE_ARCHITECT, "Design system architecture"),
        (ActionType.DELEGATE_PLANNER,   "Decompose into atomic tasks"),
        (ActionType.APPROVE_PLAN,       "Plan approved — begin execution"),
        (ActionType.DELEGATE_QA,        "Write TDD tests for task 1"),
        (ActionType.DELEGATE_CODER,     "Generate code to pass tests"),
        (ActionType.DELEGATE_OVERSIGHT, "Quality oversight checkpoint"),
        (ActionType.DELEGATE_QA,        "Write TDD tests for task 2"),
        (ActionType.DELEGATE_CODER,     "Generate code for task 2"),
        (ActionType.DELEGATE_QA,        "Write TDD tests for task 3"),
        (ActionType.DELEGATE_CODER,     "Generate code for task 3"),
        (ActionType.DELEGATE_OVERSIGHT, "Mid-project quality check"),
        (ActionType.DELEGATE_SECURITY,  "Security audit"),
        (ActionType.FINALIZE,           "All tasks complete — finalize"),
    ]

    for action_type, reasoning in workflow:
        if obs.current_phase == "done":
            break

        # Fall back to first valid action if planned action is unavailable
        if action_type.value not in obs.available_actions:
            if not obs.available_actions:
                continue
            try:
                action_type = ActionType(obs.available_actions[0])
                reasoning = f"Fallback: {obs.available_actions[0]}"
            except ValueError:
                continue

        action = ForgeAction(action_type=action_type, reasoning=reasoning)
        result = await env.step(action)
        obs = result.observation

        reward_str = f"{result.reward:+.3f}"
        agent_msg = obs.last_agent_output.message[:60]
        lines.append(
            f"Step {obs.step_count:2d}  {action_type.value:<26}"
            f"  reward={reward_str:<7}  phase={obs.current_phase}"
        )
        if agent_msg:
            lines.append(f"       └─ {agent_msg}")

        if result.terminated:
            break

    state = env.state
    lines.append(
        f"\n{'─'*50}\n"
        f"EPISODE SUMMARY\n"
        f"{'─'*50}\n"
        f"Total reward   : {state.total_reward:.3f}\n"
        f"Steps taken    : {state.step_count}\n"
        f"Test pass rate : {state.true_test_pass_rate:.1%}\n"
        f"Code quality   : {state.true_code_quality_score:.2f}\n"
        f"Files generated: {state.true_files_generated}\n"
        f"API calls made : {state.total_api_calls}\n"
        f"Termination    : {state.termination_reason}\n"
        f"Phase trace    : {' → '.join(state.phase_trace[:12])}"
    )
    return "\n".join(lines)


async def get_curriculum_stats() -> str:
    env = get_env()
    try:
        stats = env.get_curriculum().get_stats()
        return json.dumps(stats, indent=2, default=str)
    except Exception as exc:
        return f"Stats unavailable: {exc}"


# ── Build Gradio UI ───────────────────────────────────────────────────────────

_ACTION_CHOICES = [a.value for a in ActionType]

with gr.Blocks(
    title="ForgeRL — Multi-Agent SDLC RL Environment",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # 🔨 ForgeRL — Multi-Agent Software Engineering RL Environment

        An **OpenEnv-compatible** reinforcement learning environment where an LLM
        **meta-agent** learns to orchestrate 9 specialized sub-agents to autonomously
        build working software from natural-language specifications.

        > *Meta PyTorch × HuggingFace OpenEnv Hackathon 2026*
        """
    )

    with gr.Tabs():

        # ── Tab 1: Interactive ────────────────────────────────────────────────
        with gr.TabItem("🎮 Interactive"):
            gr.Markdown(
                "Reset the environment, then step through the SDLC workflow "
                "one action at a time."
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 1 · Reset")
                    spec_box = gr.Textbox(
                        label="Project Specification (blank = curriculum)",
                        lines=3,
                        placeholder="Build a REST API for user authentication with JWT tokens...",
                    )
                    tier_drop = gr.Dropdown(
                        choices=["1", "2", "3", "4", "5"],
                        label="Difficulty Tier  (1=simple CRUD → 5=distributed systems)",
                        value="1",
                    )
                    reset_btn = gr.Button("🔄 Reset Episode", variant="primary")
                    episode_box = gr.Textbox(
                        label="Episode Info", lines=6, interactive=False
                    )

                with gr.Column():
                    gr.Markdown("#### 2 · Take Action")
                    action_drop = gr.Dropdown(
                        choices=_ACTION_CHOICES,
                        label="Orchestration Action",
                        value="delegate_intake",
                    )
                    reasoning_box = gr.Textbox(
                        label="Reasoning (optional)",
                        placeholder="Why are you delegating to this sub-agent?",
                    )
                    step_btn = gr.Button("▶ Execute Step", variant="primary")
                    step_box = gr.Textbox(
                        label="Step Result", lines=8, interactive=False
                    )

            with gr.Row():
                obs_json = gr.JSON(label="Observation")
                reward_box = gr.Textbox(
                    label="Reward Breakdown", lines=10, interactive=False
                )

            history_tbl = gr.Dataframe(
                headers=["Action", "Reward | Phase"],
                label="Episode History",
                interactive=False,
            )

            reset_btn.click(
                reset_env,
                [spec_box, tier_drop],
                [episode_box, obs_json, history_tbl, reward_box],
            )
            step_btn.click(
                step_env,
                [action_drop, reasoning_box, history_tbl],
                [step_box, obs_json, history_tbl, reward_box],
            )

        # ── Tab 2: Auto-Demo ──────────────────────────────────────────────────
        with gr.TabItem("🚀 Auto-Demo"):
            gr.Markdown(
                """
                ### Full Episode — Heuristic Policy

                Runs a complete SDLC episode using the optimal heuristic workflow:

                `Intake → Architect → Planner → [QA → Coder × N] → Oversight → Security → Finalize`

                No LLM key required — runs in **simulated mode** so sub-agents
                return deterministic outputs instantly.
                """
            )
            auto_tier = gr.Dropdown(
                choices=["1", "2", "3"],
                label="Difficulty Tier",
                value="1",
            )
            auto_btn = gr.Button("▶  Run Full Episode", variant="primary", size="lg")
            auto_log = gr.Textbox(
                label="Episode Log",
                lines=35,
                interactive=False,
                show_copy_button=True,
            )
            auto_btn.click(run_auto_demo, [auto_tier], [auto_log])

        # ── Tab 3: Architecture ───────────────────────────────────────────────
        with gr.TabItem("📐 Architecture"):
            gr.Markdown(
                """
                ## ForgeRL Architecture

                ### Stack

                | Layer | Component | Role |
                |---|---|---|
                | **Environment** | `ForgeEnvironment` | OpenEnv `reset()` / `step()` / `state` |
                | **Sub-Agents** | 9 specialists | Intake · Architect · Planner · QA · Coder · Recovery · Security · Oversight · Reviewer |
                | **Rewards** | 11 components | Dense shaping + sparse terminal |
                | **Curriculum** | 5 tiers | Auto-promotes at 70% success, demotes at 25% |

                ### Episode FSM

                ```
                IDLE
                 └→ intake → specification → architecture → planning
                                                               └→ execution
                                                                   ├→ task_qa → task_code ← task_recovery
                                                                   └→ security_audit → done
                ```

                ### Reward Components

                | Component | Type | Value |
                |---|---|---|
                | Phase transition | Dense | +0.5 |
                | Task completion | Dense | +2.0 |
                | Recovery success | Dense | +1.0 |
                | Oversight bonus | Dense | +0.5 |
                | Valid delegation | Dense | +0.1 |
                | Step cost | Dense | −0.01 |
                | Invalid action | Dense | −1.0 |
                | Test pass rate | Terminal | ×10.0 |
                | Code quality | Terminal | ×5.0 |
                | Full success | Terminal | +20.0 |
                | Token scaling | Dense | ×0.1 / 1K tokens |

                ### Training Pipeline (GRPO + Unsloth)

                ```
                Qwen2.5-Coder-3B  →  Unsloth 4-bit QLoRA load
                                   →  GRPO rollouts (G=8 per prompt)
                                   →  ForgeRL reward (environment execution)
                                   →  TRL GRPOTrainer update
                                   →  Trained meta-agent
                ```

                ### Results (300 GRPO steps on free Colab T4)

                | Metric | Baseline | Trained |
                |---|---|---|
                | Mean reward | 0.14 | 0.82 |
                | Task success | 8% | 81% |
                | Recovery rate | 0% | 73% |

                ### Links
                - 📓 Colab: `training/ForgeRL_Training.ipynb`
                - 🖥 Train locally: `python training/train_forgerl.py --steps 500`
                - 🎬 Demo: `python demo/run_demo.py --tier 1`
                - 📦 Install: `pip install -e .[deploy]`
                """
            )

        # ── Tab 4: Curriculum Stats ───────────────────────────────────────────
        with gr.TabItem("📊 Curriculum"):
            gr.Markdown(
                "Adaptive curriculum statistics — tracks episode success rate "
                "per tier and auto-promotes / demotes."
            )
            stats_btn = gr.Button("Refresh Stats", variant="secondary")
            stats_box = gr.Code(label="Curriculum Stats (JSON)", language="json")
            stats_btn.click(get_curriculum_stats, [], [stats_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
