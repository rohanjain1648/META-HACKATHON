"""GRPO Rollout Collection — Generates G rollouts per prompt and scores them.

This module bridges the SDLCEnvironment with TRL's GRPOTrainer.
It implements the rollout function that TRL calls during each training step.

GRPO Algorithm (Group Relative Policy Optimisation):
    For each prompt p in batch:
        1. Generate G completions: {c_1, ..., c_G}
        2. Score each: {r_1, ..., r_G} via environment
        3. Compute group relative advantage: A_i = (r_i - mean(r)) / std(r)
        4. GRPO loss: -sum(A_i * log_prob(c_i | p))

Key design choices:
    - G = 8 completions per prompt (balances diversity vs. compute)
    - Temperature = 0.9 during rollout (encourages exploration)
    - Batch evaluation: all completions executed in parallel threads
    - Full reward breakdown logged per step for monitoring
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass
from typing import Any

from forgeai.rl.curriculum import CurriculumManager
from forgeai.rl.reward_functions import RewardBreakdown, RewardEngine
from forgeai.rl.verifier import CodeVerifier, VerificationResult

logger = logging.getLogger("forgeai.rl.rollout")


# ---------------------------------------------------------------------------
# Rollout result
# ---------------------------------------------------------------------------

@dataclass
class RolloutResult:
    """Stores one completed rollout (one code completion + its reward)."""
    prompt: str
    completion: str
    reward: float
    breakdown: RewardBreakdown
    verification: VerificationResult
    task_id: str = ""
    difficulty: str = "easy"


@dataclass
class RolloutBatch:
    """A batch of rollouts ready to feed into GRPOTrainer."""
    prompts: list[str]                   # Repeated G times per original prompt
    completions: list[str]               # Model outputs
    rewards: list[float]                 # Scalar rewards
    breakdowns: list[dict]               # Per-component reward breakdowns
    mean_reward: float = 0.0
    pass_rate: float = 0.0              # Fraction of fully-passing completions


# ---------------------------------------------------------------------------
# Rollout collector
# ---------------------------------------------------------------------------

class RolloutCollector:
    """Collects G rollouts per prompt using the SDLC environment.

    Usage in TRL training loop::

        collector = RolloutCollector(G=8, timeout_seconds=10)
        batch = collector.collect(prompts, completions)
        # batch.rewards → feed to GRPOTrainer
    """

    def __init__(
        self,
        G: int = 8,
        timeout_seconds: int = 10,
        max_workers: int = 4,
    ):
        self.G = G
        self._verifier = CodeVerifier(timeout_seconds=timeout_seconds)
        self._reward_engine = RewardEngine(timeout_seconds=timeout_seconds)
        self._max_workers = max_workers

        # Running stats for monitoring
        self._total_rollouts: int = 0
        self._reward_history: list[float] = []
        self._component_history: list[dict] = []

    def score_completions(
        self,
        completions: list[str],
        test_codes: list[str],
        signatures: list[str],
    ) -> list[float]:
        """Score a list of code completions against their respective test suites.

        This is the callable passed to TRL's GRPOTrainer as `reward_funcs`.

        Args:
            completions  : List of model-generated code strings.
            test_codes   : Corresponding pytest test suite per completion.
            signatures   : Expected function signature per completion.

        Returns:
            List of scalar rewards aligned with completions.
        """
        def _score_one(args: tuple[str, str, str]) -> float:
            code, test, sig = args
            verification = self._verifier.verify(code, test, sig)
            reward = self._reward_engine.compute_scalar(verification, code, sig)
            self._reward_history.append(reward)
            self._total_rollouts += 1
            return reward

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            rewards = list(executor.map(
                _score_one,
                zip(completions, test_codes, signatures),
            ))

        return rewards

    def score_with_breakdown(
        self,
        completions: list[str],
        test_codes: list[str],
        signatures: list[str],
    ) -> tuple[list[float], list[dict]]:
        """Like score_completions but also returns per-component breakdowns."""

        def _score_one(args: tuple[str, str, str]) -> tuple[float, dict]:
            code, test, sig = args
            verification = self._verifier.verify(code, test, sig)
            breakdown = self._reward_engine.compute(verification, code, sig)
            self._reward_history.append(breakdown.total)
            self._component_history.append(breakdown.to_dict())
            self._total_rollouts += 1
            return breakdown.total, breakdown.to_dict()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            results = list(executor.map(
                _score_one,
                zip(completions, test_codes, signatures),
            ))

        rewards = [r for r, _ in results]
        breakdowns = [b for _, b in results]
        return rewards, breakdowns

    def get_monitoring_stats(self) -> dict:
        """Return aggregated stats for training monitoring dashboard."""
        if not self._reward_history:
            return {"total_rollouts": 0}

        recent = self._reward_history[-100:]
        pass_count = sum(1 for r in recent if r >= 0.9)

        stats: dict[str, Any] = {
            "total_rollouts": self._total_rollouts,
            "mean_reward_recent_100": sum(recent) / len(recent),
            "pass_rate_recent_100": pass_count / len(recent),
        }

        if self._component_history:
            last = self._component_history[-100:]
            for key in ["test_pass", "syntax", "efficiency", "format_score", "security"]:
                stats[f"mean_{key}"] = sum(d.get(key, 0) for d in last) / len(last)

        return stats


# ---------------------------------------------------------------------------
# TRL-compatible reward function factory
# ---------------------------------------------------------------------------

def make_reward_fn(
    test_code_map: dict[str, str],
    signature_map: dict[str, str],
    timeout_seconds: int = 10,
):
    """Create a TRL-compatible reward function for GRPOTrainer.

    TRL's GRPOTrainer expects a callable:
        reward_fn(completions: list[str], **kwargs) -> list[float]

    This factory creates such a callable that looks up the test code and
    signature for each prompt from pre-built maps.

    Args:
        test_code_map : {task_id: test_code}
        signature_map : {task_id: function_signature}
        timeout_seconds: Execution timeout per completion.

    Returns:
        reward_fn callable compatible with TRL GRPOTrainer.

    Example::

        reward_fn = make_reward_fn(test_map, sig_map)
        trainer = GRPOTrainer(..., reward_funcs=[reward_fn])
    """
    verifier = CodeVerifier(timeout_seconds=timeout_seconds)
    engine = RewardEngine(timeout_seconds=timeout_seconds)

    def reward_fn(
        completions: list[str],
        task_ids: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        rewards = []
        for i, code in enumerate(completions):
            # Resolve task_id from kwargs or use a default fallback
            task_id = (task_ids[i] if task_ids else None) or list(test_code_map.keys())[0]
            test_code = test_code_map.get(task_id, "")
            signature = signature_map.get(task_id, "")

            if not test_code:
                rewards.append(0.0)
                continue

            verification = verifier.verify(code, test_code, signature)
            reward = engine.compute_scalar(verification, code, signature)
            rewards.append(reward)

        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Convenience: build TRL dataset from curriculum
# ---------------------------------------------------------------------------

def build_grpo_dataset(
    curriculum: CurriculumManager,
    num_samples: int = 500,
) -> list[dict]:
    """Build a dataset of {prompt, task_id} dicts for GRPOTrainer.

    Each example is one prompt.  During training, the model generates G
    completions per prompt, all scored by the reward_fn.

    Returns:
        List of dicts with keys: prompt, task_id, test_code, signature
    """
    import random
    tasks = curriculum.get_all_tasks()
    dataset = []
    for _ in range(num_samples):
        task = random.choice(tasks)
        from forgeai.rl.models import SDLCObservation
        obs = SDLCObservation(
            task_description=task.description,
            function_signature=task.function_signature,
            test_code=task.test_code,
        )
        dataset.append({
            "prompt": obs.to_prompt(),
            "task_id": task.task_id,
            "test_code": task.test_code,
            "signature": task.function_signature,
        })
    return dataset
