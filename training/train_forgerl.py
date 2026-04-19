"""ForgeRL Training Script — GRPO with Unsloth + HuggingFace TRL.

This script trains an LLM to be a software engineering "manager" that
orchestrates sub-agents in the ForgeRL environment using Group Relative
Policy Optimization (GRPO).

Designed to run in Google Colab with a T4/A100 GPU.

Usage:
    python training/train_forgerl.py --steps 500 --tier 1

Hackathon Requirements Met:
    ✓ Uses OpenEnv (latest release) — ForgeEnvironment
    ✓ Minimal training script using HF TRL — GRPOTrainer
    ✓ Compatible with Unsloth for efficiency — FastLanguageModel
    ✓ Shows reward improvement over training
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_action_from_text(text: str) -> dict:
    """Parse an LLM-generated action from text output.

    The model outputs structured text like:
        ACTION: delegate_intake
        REASONING: Starting with requirements analysis...
        PARAMETERS: {}

    Returns:
        Dict with action_type, reasoning, parameters keys.
    """
    action_type = "escalate"
    reasoning = ""
    parameters = {}

    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.upper().startswith("ACTION:"):
            action_type = line.split(":", 1)[1].strip().lower()
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
        elif line.upper().startswith("PARAMETERS:"):
            try:
                params_str = line.split(":", 1)[1].strip()
                parameters = json.loads(params_str) if params_str else {}
            except (json.JSONDecodeError, IndexError):
                parameters = {}

    # Try to extract from JSON format
    if action_type == "escalate":
        try:
            data = json.loads(text)
            action_type = data.get("action_type", data.get("action", "escalate"))
            reasoning = data.get("reasoning", "")
            parameters = data.get("parameters", {})
        except (json.JSONDecodeError, AttributeError):
            pass

    return {
        "action_type": action_type,
        "reasoning": reasoning,
        "parameters": parameters,
    }


def build_prompt_from_observation(observation: dict, spec_text: str) -> str:
    """Convert a ForgeRL observation into a prompt for the LLM.

    Args:
        observation: ForgeObservation as a dict.
        spec_text: The project specification text.

    Returns:
        A formatted prompt string.
    """
    phase = observation.get("current_phase", "idle")
    step = observation.get("step_count", 0)
    max_steps = observation.get("max_steps", 100)
    available = observation.get("available_actions", [])
    task_progress = observation.get("task_progress", {})
    project_state = observation.get("project_state", {})
    last_output = observation.get("last_agent_output", {})
    error_context = observation.get("error_context", "")

    prompt = f"""You are a Software Engineering Manager AI operating in the ForgeRL environment.
Your goal is to orchestrate a team of AI sub-agents to build working software.

## Current Project Specification
{spec_text[:500]}

## Current State
- Phase: {phase}
- Step: {step}/{max_steps}
- Tasks: {task_progress.get('completed', 0)}/{task_progress.get('total_tasks', 0)} completed
- Tests: {project_state.get('tests_passed', 0)} passed, {project_state.get('tests_failed', 0)} failed
- Files: {project_state.get('total_files', 0)} generated

## Last Agent Output
- Agent: {last_output.get('agent_name', 'none')}
- Success: {last_output.get('success', False)}
- Message: {last_output.get('message', '')}
"""

    if error_context:
        prompt += f"\n## Error Context\n{error_context[:300]}\n"

    prompt += f"""
## Available Actions
{', '.join(available)}

Choose the best next action. Respond in this format:
ACTION: <action_type>
REASONING: <brief explanation>
PARAMETERS: {{}}
"""
    return prompt


def create_training_dataset(num_specs: int = 20) -> list[dict]:
    """Create a dataset of project specifications for training.

    Each item has a 'prompt' (the initial observation) and metadata.
    """
    from forge_env.curriculum import SPEC_BANK, AdaptiveCurriculum

    curriculum = AdaptiveCurriculum()
    dataset = []

    for i in range(num_specs):
        # Cycle through tiers
        tier = (i % 5) + 1
        curriculum.current_tier = tier
        spec, description = curriculum.sample_spec()

        dataset.append({
            "prompt": build_prompt_from_observation(
                {
                    "current_phase": "idle",
                    "step_count": 0,
                    "max_steps": spec.max_steps,
                    "available_actions": ["delegate_intake"],
                    "task_progress": {"completed": 0, "total_tasks": 0},
                    "project_state": {"tests_passed": 0, "tests_failed": 0, "total_files": 0},
                    "last_agent_output": {"agent_name": "none", "success": False, "message": "Episode start"},
                    "error_context": "",
                },
                description,
            ),
            "spec_text": description,
            "tier": tier,
            "spec_name": spec.name,
        })

    return dataset


def environment_reward_function(prompts, completions, **kwargs) -> list[float]:
    """Reward function that runs completions through the ForgeRL environment.

    For each completion, we:
    1. Parse the sequence of actions from the LLM output
    2. Execute them in the environment
    3. Return the cumulative reward

    This is the core integration between TRL's GRPOTrainer and our environment.
    """
    from forge_env.environment import ForgeEnvironment
    from forge_env.models import ForgeAction, ActionType

    rewards = []

    for prompt, completion in zip(prompts, completions):
        try:
            # Create a fresh environment (simulated mode for speed)
            env = ForgeEnvironment(use_real_llm=False, max_steps=50)

            # Extract spec text from the prompt
            spec_start = prompt.find("## Current Project Specification\n")
            spec_end = prompt.find("\n## Current State")
            spec_text = prompt[spec_start + 35:spec_end].strip() if spec_start > 0 else ""

            # Reset environment
            loop = asyncio.new_event_loop()
            try:
                reset_result = loop.run_until_complete(env.reset(spec_text=spec_text, tier=1))

                # Parse actions from completion
                action_texts = completion.strip().split("\n\n")
                total_reward = 0.0

                for action_text in action_texts[:30]:  # Cap at 30 actions
                    parsed = parse_action_from_text(action_text)

                    # Convert to ForgeAction
                    try:
                        action_type = ActionType(parsed["action_type"])
                    except ValueError:
                        action_type = ActionType.ESCALATE

                    action = ForgeAction(
                        action_type=action_type,
                        reasoning=parsed.get("reasoning", ""),
                        parameters=parsed.get("parameters", {}),
                    )

                    result = loop.run_until_complete(env.step(action))
                    total_reward += result.reward

                    if result.terminated:
                        break

                rewards.append(total_reward)
            finally:
                loop.close()
                env.cleanup()

        except Exception as e:
            print(f"Reward computation error: {e}")
            rewards.append(-5.0)  # Penalty for crashes

    return rewards


def format_reward_function(prompts, completions, **kwargs) -> list[float]:
    """Reward function for output format compliance.

    Awards points for outputs that follow the ACTION/REASONING/PARAMETERS format.
    """
    rewards = []
    for completion in completions:
        score = 0.0
        text = completion.strip().upper()

        if "ACTION:" in text:
            score += 0.3
        if "REASONING:" in text:
            score += 0.2

        # Check if action is a valid ActionType
        from forge_env.models import ActionType
        valid_actions = {a.value for a in ActionType}
        parsed = parse_action_from_text(completion)
        if parsed["action_type"] in valid_actions:
            score += 0.5

        rewards.append(score)

    return rewards


def train(args):
    """Main training function using GRPO with Unsloth + TRL."""
    print("=" * 60)
    print("  ForgeRL GRPO Training")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Generations: {args.generations}")
    print(f"  Learning rate: {args.lr}")
    print("=" * 60)

    # ── 1. Load Model with Unsloth ──
    print("\n[1/5] Loading model with Unsloth...")

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        print(f"  ✓ Model loaded: {args.model}")

    except ImportError:
        print("  ⚠ Unsloth not available. Using transformers directly.")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto",
            device_map="auto",
        )
        print(f"  ✓ Model loaded (no Unsloth): {args.model}")

    # ── 2. Create Training Dataset ──
    print("\n[2/5] Creating training dataset...")
    dataset = create_training_dataset(num_specs=args.num_specs)

    # Convert to HF format
    from datasets import Dataset
    hf_dataset = Dataset.from_list([
        {"prompt": item["prompt"], "spec_text": item["spec_text"]}
        for item in dataset
    ])
    print(f"  ✓ Dataset: {len(hf_dataset)} specifications")

    # ── 3. Configure GRPO ──
    print("\n[3/5] Configuring GRPO trainer...")
    from trl import GRPOTrainer, GRPOConfig

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.generations,
        max_steps=args.steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_completion_length=512,
        max_prompt_length=1024,
        logging_steps=10,
        save_steps=100,
        report_to="none",
    )

    # Try to enable vLLM for fast generation
    try:
        training_args.use_vllm = True
        print("  ✓ vLLM enabled for fast generation")
    except Exception:
        print("  ⚠ vLLM not available, using standard generation")

    # ── 4. Train ──
    print("\n[4/5] Starting GRPO training...")
    print(f"  Reward functions: environment_reward + format_reward")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[environment_reward_function, format_reward_function],
        args=training_args,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
    )

    # Track rewards for plotting
    reward_history = []

    class RewardCallback:
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "reward" in logs:
                reward_history.append(logs["reward"])

    trainer.add_callback(RewardCallback())

    # Train!
    start = time.time()
    train_result = trainer.train()
    duration = time.time() - start

    print(f"\n  ✓ Training complete in {duration:.1f}s")
    print(f"  Total steps: {train_result.global_step}")

    # ── 5. Save & Evaluate ──
    print("\n[5/5] Saving model and generating evaluation...")

    # Save model
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    print(f"  ✓ Model saved to {args.output_dir}/final_model")

    # Save reward history
    reward_path = os.path.join(args.output_dir, "reward_history.json")
    with open(reward_path, "w") as f:
        json.dump(reward_history, f)
    print(f"  ✓ Reward history saved to {reward_path}")

    # Plot reward curve
    try:
        plot_reward_curve(reward_history, args.output_dir)
        print(f"  ✓ Reward curve saved to {args.output_dir}/reward_curve.png")
    except Exception as e:
        print(f"  ⚠ Could not plot reward curve: {e}")

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)

    return reward_history


def plot_reward_curve(rewards: list[float], output_dir: str):
    """Plot and save the reward curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not rewards:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Raw rewards
    ax1.plot(rewards, alpha=0.3, color="#6366f1", label="Raw")
    if len(rewards) > 10:
        window = min(20, len(rewards) // 3)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(
            range(window - 1, len(rewards)),
            smoothed,
            color="#6366f1",
            linewidth=2,
            label=f"Smoothed ({window}-step)",
        )
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Reward")
    ax1.set_title("ForgeRL Training Reward Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative reward
    cumulative = np.cumsum(rewards)
    ax2.plot(cumulative, color="#10b981", linewidth=2)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Cumulative Reward")
    ax2.set_title("Cumulative Training Reward")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_curve.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ForgeRL GRPO Training")
    parser.add_argument(
        "--model", default="unsloth/Qwen3-1.7B-Base",
        help="Base model for training",
    )
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--generations", type=int, default=4, help="GRPO group size")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--num-specs", type=int, default=20, help="Number of training specs")
    parser.add_argument(
        "--output-dir", default="./forgerl_outputs",
        help="Output directory for model and logs",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)
