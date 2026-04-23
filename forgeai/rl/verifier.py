"""Code Verifier — Sandboxed Python code execution with anti-cheat guards.

Runs generated code in a restricted subprocess.  Detects reward-hacking
patterns before execution (timer edits, global mutations, test-file writes,
forbidden imports) and refuses to run suspicious code.
"""

import ast
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Patterns that indicate reward hacking attempts
_FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (r"\bos\._exit\b", "os._exit bypass"),
    (r"\bsys\.exit\b", "sys.exit bypass"),
    (r"__import__\s*\(\s*['\"]sys['\"]", "dynamic sys import"),
    (r"open\s*\(.*['\"]w['\"]", "writing files at runtime"),
    (r"\bsubprocess\b", "subprocess in generated code"),
    (r"\bexec\s*\(", "exec() call"),
    (r"\beval\s*\(", "eval() call"),
    (r"\bcompile\s*\(", "compile() call"),
    (r"importlib", "importlib manipulation"),
    (r"pytest\._", "pytest internals access"),
    (r"_pytest\b", "_pytest internals access"),
    (r"conftest\b", "conftest manipulation"),
    (r"mock\.patch", "test mocking"),
    (r"unittest\.mock", "unittest.mock"),
    (r"builtins\b", "builtins override"),
]

_FORBIDDEN_IMPORTS: set[str] = {
    "subprocess", "multiprocessing", "ctypes", "cffi",
    "socket", "requests", "httpx", "aiohttp", "urllib",
}


@dataclass
class VerificationResult:
    """Result of running generated code against tests."""
    passed: int = 0
    failed: int = 0
    errors: int = 0
    total: int = 0
    duration_seconds: float = 0.0
    timed_out: bool = False
    syntax_valid: bool = True
    anti_cheat_violations: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    success: bool = False

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "total": self.total,
            "pass_rate": self.pass_rate,
            "duration_seconds": self.duration_seconds,
            "timed_out": self.timed_out,
            "syntax_valid": self.syntax_valid,
            "anti_cheat_violations": self.anti_cheat_violations,
            "success": self.success,
        }


class CodeVerifier:
    """Runs generated code in a sandboxed subprocess and returns test results.

    Execution flow:
        1. Static anti-cheat scan (AST + regex)
        2. Syntax check
        3. Write code to temp file
        4. Run pytest with timeout
        5. Parse results
    """

    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds

    # ── Public API ────────────────────────────────────────────────────────

    def verify(self, generated_code: str, test_code: str,
               task_signature: str = "") -> VerificationResult:
        """Verify generated code against provided tests.

        Args:
            generated_code: The model's output code.
            test_code: Pytest test file content.
            task_signature: Expected function/class signature for format check.

        Returns:
            VerificationResult with detailed metrics.
        """
        result = VerificationResult()

        # Step 1: Anti-cheat scan
        violations = self._scan_anti_cheat(generated_code)
        result.anti_cheat_violations = violations
        if violations:
            result.syntax_valid = False
            return result

        # Step 2: Syntax check
        if not self._check_syntax(generated_code, result):
            return result

        # Step 3: Run tests in temp directory
        self._run_tests(generated_code, test_code, result)
        return result

    # ── Anti-Cheat ────────────────────────────────────────────────────────

    def _scan_anti_cheat(self, code: str) -> list[str]:
        """Return list of detected anti-cheat violations."""
        violations: list[str] = []

        for pattern, description in _FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(description)

        # AST-level import check
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = (
                        [alias.name for alias in node.names]
                        if isinstance(node, ast.Import)
                        else [node.module or ""]
                    )
                    for name in names:
                        root = name.split(".")[0]
                        if root in _FORBIDDEN_IMPORTS:
                            violations.append(f"forbidden import: {name}")
        except SyntaxError:
            pass  # Caught in syntax check phase

        return violations

    def _check_syntax(self, code: str, result: VerificationResult) -> bool:
        """Validate Python syntax via AST parse."""
        try:
            ast.parse(code)
            result.syntax_valid = True
            return True
        except SyntaxError as e:
            result.syntax_valid = False
            result.stderr = f"SyntaxError: {e}"
            return False

    # ── Execution ─────────────────────────────────────────────────────────

    def _run_tests(self, generated_code: str, test_code: str,
                   result: VerificationResult) -> None:
        """Write code + tests to temp dir and run pytest."""
        with tempfile.TemporaryDirectory(prefix="forgeai_rl_") as tmpdir:
            tmp = Path(tmpdir)

            # Write implementation
            (tmp / "solution.py").write_text(generated_code, encoding="utf-8")

            # Write test file (inject import of solution)
            full_test = f"from solution import *\n\n{test_code}"
            (tmp / "test_solution.py").write_text(full_test, encoding="utf-8")

            # Write minimal conftest to avoid import issues
            (tmp / "conftest.py").write_text("", encoding="utf-8")

            cmd = [
                sys.executable, "-m", "pytest",
                str(tmp / "test_solution.py"),
                "-v", "--tb=short", "--no-header", "-q",
                f"--timeout={self.timeout_seconds}",
            ]

            start = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds + 2,
                    cwd=tmpdir,
                    env=self._safe_env(),
                )
                result.duration_seconds = time.perf_counter() - start
                result.stdout = proc.stdout
                result.stderr = proc.stderr
                result.success = proc.returncode == 0
                self._parse_pytest_output(proc.stdout, result)

            except subprocess.TimeoutExpired:
                result.timed_out = True
                result.duration_seconds = self.timeout_seconds + 2

    @staticmethod
    def _safe_env() -> dict:
        """Minimal env for subprocess — removes credentials/tokens."""
        import os
        safe = {k: v for k, v in os.environ.items()
                if not any(s in k.upper() for s in
                           ["KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"])}
        safe["PYTHONDONTWRITEBYTECODE"] = "1"
        return safe

    @staticmethod
    def _parse_pytest_output(output: str, result: VerificationResult) -> None:
        """Extract pass/fail counts from pytest summary line."""
        for line in reversed(output.splitlines()):
            ll = line.lower()
            if "passed" in ll or "failed" in ll or "error" in ll:
                def extract(pattern: str) -> int:
                    m = re.search(pattern, ll)
                    return int(m.group(1)) if m else 0

                result.passed = extract(r"(\d+)\s+passed")
                result.failed = extract(r"(\d+)\s+failed")
                result.errors = extract(r"(\d+)\s+error")
                result.total = result.passed + result.failed + result.errors
                break
