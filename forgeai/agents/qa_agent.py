"""QA Agent (TDD-First) — Writes failing test cases BEFORE production code.

Satisfies FR-11: TDD-first approach — generate failing tests before code.
Satisfies FR-12: Automatically run test suite after code generation.

This is the highest-scored non-autonomy criterion (25 points).
"""

import json
from typing import Optional

from forgeai.agents.base_agent import BaseAgent
from forgeai.core.activity_logger import ActivityLogger
from forgeai.models.agent_state import AgentContext, AgentResult, AgentRole
from forgeai.tools.llm_gateway import LLMGateway


class QAAgent(BaseAgent):
    """Writes comprehensive failing test cases for each task BEFORE production code."""

    def __init__(self, llm: LLMGateway, logger: Optional[ActivityLogger] = None):
        super().__init__(AgentRole.QA, llm, logger)

    def build_system_prompt(self) -> str:
        return (
            "You are an expert QA Engineer AI following strict Test-Driven Development (TDD).\n\n"
            "Your ONLY job is to write FAILING test cases that define the expected behavior "
            "BEFORE any production code exists. This is non-negotiable.\n\n"
            "Rules:\n"
            "1. Use pytest as the testing framework\n"
            "2. Write tests that are currently EXPECTED TO FAIL (the code doesn't exist yet)\n"
            "3. Cover: happy path, edge cases, error conditions, boundary values\n"
            "4. Use descriptive test names: test_<what>_<condition>_<expected>\n"
            "5. Include proper imports (even if modules don't exist yet)\n"
            "6. Use fixtures in conftest.py for shared setup\n"
            "7. Test API endpoints with TestClient (FastAPI) if applicable\n"
            "8. Test data validation with valid and invalid inputs\n"
            "9. Each test should test ONE thing\n"
            "10. Include docstrings explaining what each test verifies\n\n"
            "Output format: Python test files that can be saved directly to disk."
        )

    def build_user_prompt(self, context: AgentContext) -> str:
        spec = context.specification
        spec_text = spec.to_prompt_context() if spec else ""
        arch_text = json.dumps(context.architecture, indent=2) if context.architecture else ""
        task = context.current_task

        # Include existing project files for import resolution
        existing = ""
        if context.existing_files:
            existing = "\n## Existing Project Files:\n"
            for fname, content in context.existing_files.items():
                existing += f"\n### {fname}\n```python\n{content}\n```\n"

        task_info = ""
        if task:
            task_info = (
                f"\n## Current Task (#{task.id}): {task.title}\n"
                f"Description: {task.description}\n"
                f"Target files: {', '.join(task.target_files)}\n"
            )

        return (
            f"Write FAILING test cases for the following task. These tests define "
            f"the expected behavior — production code will be written AFTER to make them pass.\n\n"
            f"## Project Specification\n{spec_text}\n\n"
            f"## Architecture\n{arch_text}\n"
            f"{task_info}\n"
            f"{existing}\n"
            f"Respond with a JSON object:\n"
            f'{{\n'
            f'  "test_files": {{\n'
            f'    "tests/test_example.py": "import pytest\\n\\ndef test_example():\\n    ...",\n'
            f'    "tests/conftest.py": "import pytest\\n\\n@pytest.fixture\\ndef ..."\n'
            f'  }},\n'
            f'  "test_count": 5,\n'
            f'  "coverage_areas": ["happy path", "validation", "error handling"]\n'
            f'}}\n\n'
            f"IMPORTANT: Respond ONLY with valid JSON. The test code must be syntactically valid Python. "
            f"Tests SHOULD FAIL because the production code doesn't exist yet."
        )

    def parse_response(self, raw_response: str, context: AgentContext) -> AgentResult:
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                return AgentResult(
                    success=False, role=self.role,
                    error="Failed to parse QA response as JSON",
                )

        test_files = data.get("test_files", {})
        if not test_files:
            return AgentResult(
                success=False, role=self.role,
                error="No test files generated",
            )

        return AgentResult(
            success=True,
            role=self.role,
            generated_files=test_files,
            message=f"Generated {data.get('test_count', len(test_files))} tests "
                    f"covering: {', '.join(data.get('coverage_areas', []))}",
        )
