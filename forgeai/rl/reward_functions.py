"""Reward Functions — Multiple independent reward verifiers.

Using a single reward signal makes it easy for a model to game the reward.
This module provides 5 independent components that are aggregated with
explicit weights.  The anti-cheat component acts as a hard penalty.

Component weights (sum to 1.0 for legitimate code):
    test_pass      : 0.60   — fraction of tests passing
    syntax         : 0.10   — syntactically valid Python
    efficiency     : 0.10   — runs within time budget
    format         : 0.10   — proper structure (imports, function def)
    security       : 0.10   — no dangerous patterns

Anti-cheat penalty    : -1.0  (overrides everything if triggered)
"""

import ast
import re
from dataclasses import dataclass
from typing import Optional

from forgeai.rl.verifier import VerificationResult


# ---------------------------------------------------------------------------
# Reward breakdown data class
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    """Per-component reward scores for a single rollout."""
    test_pass: float = 0.0      # [0, 0.60]
    syntax: float = 0.0         # 0 or 0.10
    efficiency: float = 0.0     # 0 or 0.10
    format_score: float = 0.0   # [0, 0.10]
    security: float = 0.0       # 0 or 0.10
    anti_cheat: float = 0.0     # 0 or -1.0
    total: float = 0.0

    def to_dict(self) -> dict:
        return {
            "test_pass": round(self.test_pass, 4),
            "syntax": round(self.syntax, 4),
            "efficiency": round(self.efficiency, 4),
            "format_score": round(self.format_score, 4),
            "security": round(self.security, 4),
            "anti_cheat": round(self.anti_cheat, 4),
            "total": round(self.total, 4),
        }


# ---------------------------------------------------------------------------
# Individual reward components
# ---------------------------------------------------------------------------

class TestPassReward:
    """Primary reward: fraction of tests passing, weighted 0.60."""

    WEIGHT = 0.60

    def score(self, result: VerificationResult) -> float:
        if result.anti_cheat_violations:
            return 0.0
        return self.WEIGHT * result.pass_rate


class SyntaxReward:
    """Binary reward for syntactically valid Python, weighted 0.10."""

    WEIGHT = 0.10

    def score(self, result: VerificationResult, generated_code: str) -> float:
        if not result.syntax_valid:
            return 0.0
        try:
            ast.parse(generated_code)
            return self.WEIGHT
        except SyntaxError:
            return 0.0


class EfficiencyReward:
    """Reward for completing within time budget, weighted 0.10."""

    WEIGHT = 0.10

    def __init__(self, timeout_seconds: int = 10):
        self.timeout = timeout_seconds

    def score(self, result: VerificationResult) -> float:
        if result.timed_out:
            return 0.0
        if result.anti_cheat_violations:
            return 0.0
        # Partial credit for speed: full reward if ≤50% of budget used
        ratio = result.duration_seconds / max(self.timeout, 1)
        if ratio <= 0.5:
            return self.WEIGHT
        elif ratio <= 1.0:
            return self.WEIGHT * (1.0 - ratio)
        return 0.0


class FormatReward:
    """Reward for code structure quality, weighted 0.10.

    Checks:
        - Has at least one function or class definition
        - Proper docstring or inline comment (optional, partial credit)
        - No trailing syntax errors in first 10 lines
        - Reasonable length (not a one-liner cheat)
    """

    WEIGHT = 0.10

    def score(self, generated_code: str, expected_signature: str = "") -> float:
        score = 0.0
        lines = [l for l in generated_code.splitlines() if l.strip()]

        # Must define at least one callable
        has_def = any(l.lstrip().startswith(("def ", "class ")) for l in lines)
        if not has_def:
            return 0.0

        score += 0.04  # Has function/class definition

        # Reasonable length (not suspiciously short)
        if len(lines) >= 3:
            score += 0.02

        # Matches expected signature if provided
        if expected_signature and expected_signature.split("(")[0].strip() in generated_code:
            score += 0.02

        # Has some form of documentation
        if '"""' in generated_code or "'''" in generated_code or "# " in generated_code:
            score += 0.02

        return min(score, self.WEIGHT)


class SecurityReward:
    """Reward for absence of common security anti-patterns, weighted 0.10.

    Penalised patterns (each found reduces score):
        - SQL string formatting
        - Hardcoded credentials
        - Shell injection via f-strings
        - Use of pickle
    """

    WEIGHT = 0.10

    _BAD_PATTERNS: list[tuple[str, float]] = [
        (r"pickle\.(loads?|dumps?)", 0.05),
        (r'f["\'].*SELECT.*\{', 0.05),
        (r'password\s*=\s*["\'][^"\']+["\']', 0.03),
        (r'os\.system\s*\(', 0.03),
        (r'shell\s*=\s*True', 0.03),
    ]

    def score(self, generated_code: str) -> float:
        deductions = 0.0
        for pattern, penalty in self._BAD_PATTERNS:
            if re.search(pattern, generated_code, re.IGNORECASE):
                deductions += penalty
        return max(0.0, self.WEIGHT - deductions)


class AntiCheatPenalty:
    """Hard penalty for detected reward-hacking attempts.

    Returns -1.0 if any violation is detected, overriding all positive rewards.
    This makes reward hacking strictly worse than doing nothing.
    """

    PENALTY = -1.0

    def score(self, result: VerificationResult) -> float:
        if result.anti_cheat_violations:
            return self.PENALTY
        return 0.0


# ---------------------------------------------------------------------------
# Reward engine — aggregates all components
# ---------------------------------------------------------------------------

class RewardEngine:
    """Compute composite reward from a VerificationResult.

    Usage::

        engine = RewardEngine(timeout_seconds=10)
        breakdown = engine.compute(result, generated_code, expected_sig)
        reward = breakdown.total  # scalar fed to GRPO
    """

    def __init__(self, timeout_seconds: int = 10):
        self._test = TestPassReward()
        self._syntax = SyntaxReward()
        self._efficiency = EfficiencyReward(timeout_seconds)
        self._format = FormatReward()
        self._security = SecurityReward()
        self._anti_cheat = AntiCheatPenalty()

    def compute(
        self,
        result: VerificationResult,
        generated_code: str,
        expected_signature: str = "",
    ) -> RewardBreakdown:
        """Compute all reward components and return breakdown + total."""
        breakdown = RewardBreakdown()

        anti = self._anti_cheat.score(result)
        if anti < 0:
            # Hard penalty: short-circuit everything
            breakdown.anti_cheat = anti
            breakdown.total = anti
            return breakdown

        breakdown.test_pass = self._test.score(result)
        breakdown.syntax = self._syntax.score(result, generated_code)
        breakdown.efficiency = self._efficiency.score(result)
        breakdown.format_score = self._format.score(generated_code, expected_signature)
        breakdown.security = self._security.score(generated_code)
        breakdown.anti_cheat = 0.0
        breakdown.total = (
            breakdown.test_pass
            + breakdown.syntax
            + breakdown.efficiency
            + breakdown.format_score
            + breakdown.security
        )
        return breakdown

    def compute_scalar(
        self,
        result: VerificationResult,
        generated_code: str,
        expected_signature: str = "",
    ) -> float:
        """Return just the total scalar reward (used by GRPO rollout)."""
        return self.compute(result, generated_code, expected_signature).total
