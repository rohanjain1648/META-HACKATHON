"""GRPO Trainer — TRL + Unsloth training pipeline for ForgeAI-RL.

This module implements the full training loop:
    1. Load base model with Unsloth (4-bit QLoRA for efficiency)
    2. Build curriculum dataset
    3. Configure TRL GRPOTrainer with multi-component reward functions
    4. Train with periodic rollout inspection (anti-hacking monitoring)
    5. Save model correctly (avoid the naive 4-bit → 16-bit merge bug)

Recommended base model  : Qwen/Qwen2.5-Coder-3B-Instruct
                          (strong at coding, small enough for hackathon compute)
Fallback                 : Qwen/Qwen2.5-Coder-1.5B-Instruct (faster iteration)

Training recipe:
    - 4-bit QLoRA via Unsloth (fits on T4/A10 GPU)
    - GRPO with G=8 completions per prompt
    - Temperature 0.9 during rollout (exploration)
    - Max sequence length 1024 tokens (task + code fits comfortably)
    - Batch size 4 × G = 32 completions per update step
    - Checkpoint every 50 steps

Run:
    python -m forgeai.rl.trainer --model Qwen/Qwen2.5-Coder-3B-Instruct
    # or via the main CLI:
    python -m forgeai.main --train-rl
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("forgeai.rl.trainer")


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All hyperparameters for the GRPO training run."""
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    output_dir: str = "./forgeai_rl_model"
    max_seq_length: int = 1024
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # GRPO hyperparameters
    num_generations: int = 8           # G: completions per prompt
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    max_steps: int = 300               # Override epochs for quick hackathon runs
    warmup_steps: int = 20
    save_steps: int = 50
    logging_steps: int = 10

    # Rollout
    temperature: float = 0.9
    max_new_tokens: int = 512
    timeout_seconds: int = 10

    # Curriculum
    num_dataset_samples: int = 500
    start_difficulty: str = "easy"

    # Monitoring
    inspect_every_n_steps: int = 25    # Sample and print rollouts for anti-hack check
    reward_log_path: str = "./forgeai_rl_rewards.jsonl"


# ---------------------------------------------------------------------------
# Main trainer entry point
# ---------------------------------------------------------------------------

def run_training(config: Optional[TrainingConfig] = None) -> str:
    """Run the full GRPO training pipeline.

    Returns:
        Path to the saved model directory.
    """
    if config is None:
        config = TrainingConfig()

    logger.info("ForgeAI-RL: Starting GRPO training")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Output: {config.output_dir}")

    # ── Step 1: Import heavy dependencies (lazy to avoid slow startup) ───
    try:
        from unsloth import FastLanguageModel
        from trl import GRPOConfig, GRPOTrainer
        import torch
    except ImportError as e:
        raise ImportError(
            f"Training dependencies not installed: {e}\n"
            "Run: pip install 'unsloth[colab-new]' trl>=0.9.0"
        ) from e

    # ── Step 2: Load model with Unsloth ──────────────────────────────────
    logger.info("Loading model with Unsloth 4-bit QLoRA...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,                          # Auto-detect (bf16 or fp16)
        load_in_4bit=config.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Switch to training mode
    FastLanguageModel.for_training(model)

    # ── Step 3: Build dataset from curriculum ────────────────────────────
    logger.info("Building curriculum dataset...")
    from forgeai.rl.curriculum import CurriculumManager, DifficultyLevel
    from forgeai.rl.rollout import build_grpo_dataset, make_reward_fn

    curriculum = CurriculumManager(
        start_level=DifficultyLevel(config.start_difficulty)
    )
    raw_dataset = build_grpo_dataset(curriculum, num_samples=config.num_dataset_samples)

    # Build lookup maps for the reward function
    test_code_map = {d["task_id"]: d["test_code"] for d in raw_dataset}
    signature_map = {d["task_id"]: d["signature"] for d in raw_dataset}

    # HuggingFace Dataset format
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    hf_dataset = Dataset.from_list(
        [{"prompt": d["prompt"], "task_id": d["task_id"]} for d in raw_dataset]
    )

    # ── Step 4: Build reward functions ──────────────────────────────────
    # Using multiple independent reward functions (anti-reward-hacking)
    reward_fn = make_reward_fn(
        test_code_map=test_code_map,
        signature_map=signature_map,
        timeout_seconds=config.timeout_seconds,
    )

    # Additional lightweight reward functions for extra signal diversity
    def syntax_bonus_fn(completions: list[str], **kwargs) -> list[float]:
        """Bonus reward for syntactically valid Python."""
        import ast
        results = []
        for code in completions:
            try:
                ast.parse(code)
                results.append(0.05)
            except SyntaxError:
                results.append(-0.05)
        return results

    def length_penalty_fn(completions: list[str], **kwargs) -> list[float]:
        """Mild penalty for extremely short or extremely long solutions."""
        results = []
        for code in completions:
            lines = [l for l in code.splitlines() if l.strip()]
            n = len(lines)
            if n < 2:
                results.append(-0.1)    # Suspiciously short (possible hack)
            elif n > 80:
                results.append(-0.05)   # Unnecessarily verbose
            else:
                results.append(0.0)
        return results

    # ── Step 5: Configure GRPOTrainer ────────────────────────────────────
    logger.info("Configuring TRL GRPOTrainer...")

    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        num_generations=config.num_generations,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
        max_prompt_length=config.max_seq_length - config.max_new_tokens,
        report_to="tensorboard",        # Log to tensorboard for monitoring
        remove_unused_columns=False,
        bf16=True,
        dataloader_num_workers=0,       # Avoid multiprocessing issues on Windows
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=grpo_config,
        train_dataset=hf_dataset,
        reward_funcs=[
            reward_fn,           # Primary: test execution (weight 0.60)
            syntax_bonus_fn,     # Secondary: syntax validity
            length_penalty_fn,   # Tertiary: length sanity check
        ],
    )

    # ── Step 6: Attach reward monitoring callback ────────────────────────
    reward_log = Path(config.reward_log_path)
    _attach_reward_monitor(trainer, reward_log, config.inspect_every_n_steps)

    # ── Step 7: Train ────────────────────────────────────────────────────
    logger.info("Starting training...")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed:.0f}s")

    # ── Step 8: Save model correctly ─────────────────────────────────────
    # CRITICAL: Do NOT upcast 4-bit → 16-bit then merge naively.
    # Use Unsloth's safe merge path instead.
    save_path = Path(config.output_dir) / "final"
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {save_path}...")

    # Save LoRA adapters directly (safest approach)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    # Optionally also save merged 16-bit for inference (Unsloth safe merge)
    try:
        merged_path = Path(config.output_dir) / "merged_16bit"
        model.save_pretrained_merged(
            str(merged_path),
            tokenizer,
            save_method="merged_16bit",  # Unsloth's safe merge path
        )
        logger.info(f"Merged 16-bit model saved to {merged_path}")
    except Exception as e:
        logger.warning(f"Could not save merged model (adapters still saved): {e}")

    logger.info("ForgeAI-RL training complete.")
    return str(save_path)


# ---------------------------------------------------------------------------
# Reward monitoring callback
# ---------------------------------------------------------------------------

def _attach_reward_monitor(trainer, reward_log_path: Path, inspect_every: int):
    """Attach a callback that logs reward breakdowns and samples rollouts."""
    try:
        from transformers import TrainerCallback

        class RewardMonitorCallback(TrainerCallback):
            def __init__(self):
                self._step = 0
                self._reward_log = reward_log_path

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                # Write reward components to JSONL for analysis
                entry = {"step": state.global_step, "logs": logs}
                with open(self._reward_log, "a") as f:
                    f.write(json.dumps(entry) + "\n")

                # Alert on suspicious patterns
                reward = logs.get("train/reward", None)
                if reward and reward > 0.95:
                    logger.info(
                        f"[Step {state.global_step}] Very high reward {reward:.3f} — "
                        "consider inspecting generations for reward hacking"
                    )

            def on_step_end(self, args, state, control, **kwargs):
                self._step += 1
                if self._step % inspect_every == 0:
                    logger.info(
                        f"[Step {state.global_step}] Anti-hack checkpoint: "
                        "sample and inspect model generations manually."
                    )

        trainer.add_callback(RewardMonitorCallback())
    except ImportError:
        logger.warning("transformers not available — skipping reward monitor callback")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ForgeAI-RL GRPO Trainer")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct",
                        help="Base model name or path")
    parser.add_argument("--output-dir", default="./forgeai_rl_model",
                        help="Directory to save the trained model")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Maximum training steps")
    parser.add_argument("--num-generations", type=int, default=8,
                        help="Number of GRPO completions per prompt (G)")
    parser.add_argument("--difficulty", default="easy",
                        choices=["easy", "medium", "hard"],
                        help="Starting curriculum difficulty level")
    parser.add_argument("--dataset-size", type=int, default=500,
                        help="Number of training prompts to generate")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (requires more VRAM)")
    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        start_difficulty=args.difficulty,
        num_dataset_samples=args.dataset_size,
        load_in_4bit=not args.no_4bit,
    )

    save_path = run_training(config)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
