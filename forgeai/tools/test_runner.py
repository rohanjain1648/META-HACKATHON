"""Test Runner — Executes pytest in a subprocess and parses results.

Runs the generated project's test suite and returns structured results
for the orchestrator to act on (FR-12).
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from forgeai.core.activity_logger import ActivityLogger


class TestResult:
    """Structured test execution result."""

    def __init__(self):
        self.passed: int = 0
        self.failed: int = 0
        self.errors: int = 0
        self.skipped: int = 0
        self.total: int = 0
        self.output: str = ""
        self.error_output: str = ""
        self.success: bool = False
        self.failure_details: list[dict] = []
        self.duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "total": self.total,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "failure_details": self.failure_details,
            "output": self.output[-2000:] if len(self.output) > 2000 else self.output,  # Truncate
        }


class TestRunner:
    """Runs pytest against the generated project and parses results."""

    def __init__(self, project_dir: str, timeout: int = 60,
                 logger: Optional[ActivityLogger] = None):
        self._project_dir = Path(project_dir).resolve()
        self._timeout = timeout
        self._logger = logger

    def run_tests(self, test_path: Optional[str] = None) -> TestResult:
        """Run pytest on the generated project.
        
        Args:
            test_path: Specific test file/dir to run. If None, runs all tests.
            
        Returns:
            TestResult with parsed pass/fail counts and output.
        """
        result = TestResult()

        # Build command
        cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short", "--no-header"]
        if test_path:
            cmd.append(str(self._project_dir / test_path))
        else:
            cmd.append(str(self._project_dir))

        if self._logger:
            self._logger.test_run("TestRunner", f"Running: {' '.join(cmd)}")

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=str(self._project_dir),
                env={**os.environ, "PYTHONPATH": str(self._project_dir)},
            )

            result.output = proc.stdout
            result.error_output = proc.stderr
            result.success = proc.returncode == 0

            # Parse pytest output for counts
            self._parse_results(result, proc.stdout)

            if self._logger:
                status = "PASSED" if result.success else "FAILED"
                self._logger.test_run("TestRunner", 
                    f"Tests {status}: {result.passed} passed, {result.failed} failed, "
                    f"{result.errors} errors, {result.skipped} skipped",
                    result.to_dict()
                )

        except subprocess.TimeoutExpired:
            result.success = False
            result.error_output = f"Tests timed out after {self._timeout}s"
            if self._logger:
                self._logger.error("TestRunner", result.error_output)

        except FileNotFoundError:
            result.success = False
            result.error_output = "pytest not found. Ensure it is installed."
            if self._logger:
                self._logger.error("TestRunner", result.error_output)

        return result

    def run_single_test_file(self, test_file: str) -> TestResult:
        """Run a single test file."""
        return self.run_tests(test_path=test_file)

    def _parse_results(self, result: TestResult, output: str):
        """Parse pytest output to extract pass/fail counts."""
        lines = output.strip().split("\n")

        # Look for the summary line like "5 passed, 2 failed, 1 error in 2.34s"
        for line in reversed(lines):
            line_lower = line.lower()
            if "passed" in line_lower or "failed" in line_lower or "error" in line_lower:
                import re
                passed_match = re.search(r"(\d+)\s+passed", line_lower)
                failed_match = re.search(r"(\d+)\s+failed", line_lower)
                error_match = re.search(r"(\d+)\s+error", line_lower)
                skipped_match = re.search(r"(\d+)\s+skipped", line_lower)
                duration_match = re.search(r"in\s+([\d.]+)s", line_lower)

                if passed_match:
                    result.passed = int(passed_match.group(1))
                if failed_match:
                    result.failed = int(failed_match.group(1))
                if error_match:
                    result.errors = int(error_match.group(1))
                if skipped_match:
                    result.skipped = int(skipped_match.group(1))
                if duration_match:
                    result.duration_seconds = float(duration_match.group(1))

                result.total = result.passed + result.failed + result.errors
                break

        # Extract failure details
        self._extract_failure_details(result, lines)

    def _extract_failure_details(self, result: TestResult, lines: list[str]):
        """Extract individual test failure messages."""
        in_failure = False
        current_failure: dict = {}

        for line in lines:
            if line.startswith("FAILED ") or "FAILED" in line and "::" in line:
                if current_failure:
                    result.failure_details.append(current_failure)
                current_failure = {"test": line.strip(), "details": ""}
                in_failure = True
            elif in_failure and line.startswith("    "):
                current_failure["details"] += line + "\n"
            elif in_failure and not line.strip():
                in_failure = False

        if current_failure:
            result.failure_details.append(current_failure)

    def check_syntax(self, filepath: str) -> dict:
        """Check Python file for syntax errors using py_compile."""
        full_path = self._project_dir / filepath
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "py_compile", str(full_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return {
                "valid": proc.returncode == 0,
                "error": proc.stderr if proc.returncode != 0 else "",
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}
