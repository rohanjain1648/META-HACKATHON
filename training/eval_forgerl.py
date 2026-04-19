"""ForgeRL Evaluation Script — Before/After analysis and reward measurement.

Evaluates a trained (or untrained) model on held-out specifications,
generating metrics, reward curves, and behavioral comparisons.

Usage:
    python training/eval_forgerl.py --model ./forgerl_outputs/final_model
    python training/eval_forgerl.py --baseline  # Evaluate random policy
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from forge_env.environment import ForgeEnvironment
from forge_env.models import ForgeAction, ActionType


# ── Policies ──────────────────────────────────────────────────────────────────


def random_policy(observation: dict) -> ForgeAction:
    """Baseline: randomly pick from available actions."""
    import random
    available = observation.get("available_actions", ["escalate"])
    action_str = random.choice(available)
    try:
        action_type = ActionType(action_str)
    except ValueError:
        action_type = ActionType.ESCALATE
    return ForgeAction(action_type=action_type, reasoning="Random selection")


def heuristic_policy(observation: dict) -> ForgeAction:
    """Hand-coded heuristic policy that follows the optimal workflow."""
    phase = observation.get("current_phase", "idle")
    available = observation.get("available_actions", [])
    task_progress = observation.get("task_progress", {})
    error_context = observation.get("error_context", "")

    # Follow the natural SDLC flow
    phase_to_action = {
        "idle": "delegate_intake",
        "intake": "delegate_intake",
        "specification": "delegate_architect",
        "architecture": "delegate_planner",
        "planning": "approve_plan",
        "plan_review": "approve_plan",
        "execution": "delegate_qa",
        "task_qa": "delegate_coder",
        "task_code": "delegate_coder",
        "task_recovery": "delegate_recovery",
        "security_audit": "finalize",
        "done": "finalize",
    }

    # Special cases
    if error_context and phase == "task_recovery":
        if "delegate_recovery" in available:
            action_str = "delegate_recovery"
        elif "skip_task" in available:
            action_str = "skip_task"
        else:
            action_str = "escalate"
    elif (
        phase == "execution"
        and task_progress.get("completed", 0) >= task_progress.get("total_tasks", 1)
    ):
        action_str = "finalize"
    elif (
        phase == "execution"
        and task_progress.get("completed", 0) > 0
        and task_progress.get("completed", 0) % 3 == 0
    ):
        # Run oversight every 3 tasks
        if "delegate_oversight" in available:
            action_str = "delegate_oversight"
        else:
            action_str = "delegate_qa"
    else:
        action_str = phase_to_action.get(phase, "escalate")

    # Validate action is available
    if action_str not in available and available:
        action_str = available[0]

    try:
        action_type = ActionType(action_str)
    except ValueError:
        action_type = ActionType.ESCALATE

    return ForgeAction(
        action_type=action_type,
        reasoning=f"Heuristic: phase={phase}, action={action_str}",
    )


async def run_episode(
    env: ForgeEnvironment,
    policy,
    tier: int = 1,
    max_steps: int = 100,
    verbose: bool = False,
) -> dict:
    """Run a single episode with a given policy.

    Returns:
        Episode metrics dict.
    """
    result = await env.reset(tier=tier)
    obs = result.observation

    total_reward = 0.0
    step_rewards = []
    actions_taken = []
    start_time = time.time()

    for step in range(max_steps):
        obs_dict = obs.model_dump()
        action = policy(obs_dict)

        step_result = await env.step(action)
        obs = step_result.observation

        total_reward += step_result.reward
        step_rewards.append(step_result.reward)
        actions_taken.append(action.action_type.value)

        if verbose and step % 10 == 0:
            print(
                f"  Step {step}: phase={obs.current_phase}, "
                f"reward={step_result.reward:.3f}, "
                f"tasks={obs.task_progress.completed}/{obs.task_progress.total_tasks}"
            )

        if step_result.terminated:
            break

    duration = time.time() - start_time
    state = env.state

    return {
        "tier": tier,
        "steps": len(step_rewards),
        "total_reward": total_reward,
        "avg_step_reward": total_reward / max(len(step_rewards), 1),
        "tasks_completed": state.true_files_generated,
        "test_pass_rate": state.true_test_pass_rate,
        "quality_score": state.true_code_quality_score,
        "termination_reason": state.termination_reason,
        "duration": duration,
        "step_rewards": step_rewards,
        "actions": actions_taken,
    }


async def evaluate(args):
    """Run evaluation suite."""
    print("=" * 60)
    print("  ForgeRL Evaluation")
    print("=" * 60)

    # Select policy
    if args.baseline:
        policy = random_policy
        policy_name = "Random Baseline"
    else:
        policy = heuristic_policy
        policy_name = "Heuristic Policy"

    print(f"  Policy: {policy_name}")
    print(f"  Episodes per tier: {args.episodes}")
    print(f"  Tiers: 1-{args.max_tier}")
    print("=" * 60)

    env = ForgeEnvironment(use_real_llm=False, max_steps=args.max_steps)
    all_results = []

    for tier in range(1, args.max_tier + 1):
        print(f"\n── Tier {tier} ──")
        tier_results = []

        for ep in range(args.episodes):
            result = await run_episode(
                env, policy, tier=tier,
                max_steps=args.max_steps,
                verbose=args.verbose,
            )
            tier_results.append(result)
            print(
                f"  Episode {ep + 1}: "
                f"reward={result['total_reward']:.2f}, "
                f"steps={result['steps']}, "
                f"quality={result['quality_score']:.2f}, "
                f"reason={result['termination_reason']}"
            )

        # Tier summary
        avg_reward = sum(r["total_reward"] for r in tier_results) / len(tier_results)
        avg_steps = sum(r["steps"] for r in tier_results) / len(tier_results)
        avg_quality = sum(r["quality_score"] for r in tier_results) / len(tier_results)
        success_rate = sum(
            1 for r in tier_results if r["quality_score"] > 0.5
        ) / len(tier_results)

        print(f"  ───────────────")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Steps: {avg_steps:.1f}")
        print(f"  Avg Quality: {avg_quality:.2f}")
        print(f"  Success Rate: {success_rate:.1%}")

        all_results.extend(tier_results)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f"eval_{policy_name.lower().replace(' ', '_')}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {results_path}")

    # Plot comparison
    try:
        plot_evaluation_results(all_results, policy_name, args.output_dir)
        print(f"✓ Plots saved to {args.output_dir}/")
    except Exception as e:
        print(f"⚠ Could not generate plots: {e}")

    env.cleanup()


def plot_evaluation_results(results: list[dict], policy_name: str, output_dir: str):
    """Generate evaluation plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tiers = sorted(set(r["tier"] for r in results))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"ForgeRL Evaluation — {policy_name}", fontsize=14, fontweight="bold")

    colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

    # 1. Reward by tier
    ax = axes[0, 0]
    tier_rewards = {t: [r["total_reward"] for r in results if r["tier"] == t] for t in tiers}
    positions = range(len(tiers))
    bp = ax.boxplot(
        [tier_rewards[t] for t in tiers],
        positions=positions,
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], colors[: len(tiers)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Tier {t}" for t in tiers])
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward Distribution by Tier")
    ax.grid(True, alpha=0.3)

    # 2. Steps by tier
    ax = axes[0, 1]
    tier_steps = {t: [r["steps"] for r in results if r["tier"] == t] for t in tiers}
    for i, t in enumerate(tiers):
        ax.bar(i, np.mean(tier_steps[t]), color=colors[i % len(colors)], alpha=0.7, label=f"Tier {t}")
        ax.errorbar(i, np.mean(tier_steps[t]), yerr=np.std(tier_steps[t]), color="black", capsize=5)
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([f"Tier {t}" for t in tiers])
    ax.set_ylabel("Steps")
    ax.set_title("Average Steps per Episode")
    ax.grid(True, alpha=0.3)

    # 3. Quality score
    ax = axes[1, 0]
    tier_quality = {t: [r["quality_score"] for r in results if r["tier"] == t] for t in tiers}
    for i, t in enumerate(tiers):
        ax.bar(i, np.mean(tier_quality[t]), color=colors[i % len(colors)], alpha=0.7)
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([f"Tier {t}" for t in tiers])
    ax.set_ylabel("Quality Score (0-1)")
    ax.set_title("Code Quality by Tier")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 4. Example reward trajectory
    ax = axes[1, 1]
    for i, t in enumerate(tiers[:3]):
        tier_eps = [r for r in results if r["tier"] == t]
        if tier_eps:
            rewards = tier_eps[0]["step_rewards"]
            cumulative = np.cumsum(rewards)
            ax.plot(cumulative, color=colors[i], label=f"Tier {t}", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Example Episode Reward Trajectories")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_results.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ForgeRL Evaluation")
    parser.add_argument("--baseline", action="store_true", help="Evaluate random policy")
    parser.add_argument("--model", default=None, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per tier")
    parser.add_argument("--max-tier", type=int, default=3, help="Maximum tier to evaluate")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--output-dir", default="./forgerl_outputs", help="Output dir")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()
    asyncio.run(evaluate(args))
