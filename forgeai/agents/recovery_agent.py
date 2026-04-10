"""Recovery Agent — Diagnoses failures and orchestrates recovery strategies.

Satisfies FR-15: Auto-retry with error context on failure.
Satisfies FR-17: Rollback to last checkpoint on abort.
"""

import json
from typing import Optional

from forgeai.agents.base_agent import BaseAgent
from forgeai.core.activity_logger import ActivityLogger
from forgeai.models.agent_state import AgentContext, AgentResult, AgentRole
from forgeai.tools.llm_gateway import LLMGateway


class RecoveryAgent(BaseAgent):
    """Diagnoses failures and decides recovery strategy: retry, modify, rollback, or escalate."""

    def __init__(self, llm: LLMGateway, logger: Optional[ActivityLogger] = None):
        super().__init__(AgentRole.RECOVERY, llm, logger)

    def build_system_prompt(self) -> str:
        return (
            "You are an expert Debugging and Recovery AI. When code generation or tests fail, "
            "you diagnose the root cause and recommend a recovery strategy.\n\n"
            "Recovery strategies (in priority order):\n"
            "1. RETRY_WITH_FIX — You understand the error and can provide specific fix guidance\n"
            "2. MODIFY_APPROACH — The approach is fundamentally wrong, suggest a different approach\n"
            "3. SKIP_TASK — The task is non-critical and can be skipped safely\n"
            "4. ESCALATE — The failure requires human intervention\n\n"
            "Your analysis must include:\n"
            "- Root cause identification\n"
            "- Specific error type (syntax, import, logic, runtime, timeout)\n"
            "- Concrete fix instructions for the Coder Agent\n"
            "- Whether the error is likely in the test or production code"
        )

    def build_user_prompt(self, context: AgentContext) -> str:
        task = context.current_task
        task_info = f"Task #{task.id}: {task.title}\n{task.description}" if task else "Unknown task"

        # Existing code context
        code_summary = ""
        for fname, content in context.existing_files.items():
            code_summary += f"\n### {fname}\n```python\n{content[:1500]}\n```\n"

        return (
            f"A task has FAILED. Diagnose the error and recommend recovery.\n\n"
            f"## Task Information\n{task_info}\n\n"
            f"## Error Message\n```\n{context.error_message}\n```\n\n"
            f"## Error Traceback\n```\n{context.error_traceback}\n```\n\n"
            f"## Attempt #{context.retry_count + 1}\n"
            f"Previous attempts: {context.previous_attempts}\n\n"
            f"## Current Code\n{code_summary}\n\n"
            f"Respond with a JSON object:\n"
            f'{{\n'
            f'  "diagnosis": {{\n'
            f'    "root_cause": "Clear description of what went wrong",\n'
            f'    "error_type": "syntax|import|logic|runtime|timeout|test_design",\n'
            f'    "error_in": "test_code|production_code|both|configuration"\n'
            f'  }},\n'
            f'  "strategy": "RETRY_WITH_FIX|MODIFY_APPROACH|SKIP_TASK|ESCALATE",\n'
            f'  "fix_instructions": "Specific instructions for the Coder Agent to fix the issue",\n'
            f'  "modified_test_code": {{}},\n'
            f'  "confidence": 0.85\n'
            f'}}\n\n'
            f"IMPORTANT: Respond ONLY with valid JSON."
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
                    error="Failed to parse recovery response",
                )

        strategy = data.get("strategy", "ESCALATE")
        diagnosis = data.get("diagnosis", {})
        fix_instructions = data.get("fix_instructions", "")
        modified_tests = data.get("modified_test_code", {})

        needs_human = strategy == "ESCALATE"

        return AgentResult(
            success=True,
            role=self.role,
            message=f"Recovery strategy: {strategy}\n"
                    f"Root cause: {diagnosis.get('root_cause', 'Unknown')}\n"
                    f"Fix: {fix_instructions}",
            requires_human_input=needs_human,
            generated_files=modified_tests if modified_tests else {},
            # Store recovery details in the architecture field for downstream use
            architecture={
                "strategy": strategy,
                "diagnosis": diagnosis,
                "fix_instructions": fix_instructions,
                "confidence": data.get("confidence", 0.0),
            },
        )
