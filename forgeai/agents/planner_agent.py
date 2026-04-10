"""Planner Agent — Decomposes architecture into ordered atomic tasks.

Satisfies FR-05: Decompose into ordered atomic implementation tasks.
Satisfies FR-06: Present plan for user review before execution.
"""

import json
from typing import Optional

from forgeai.agents.base_agent import BaseAgent
from forgeai.core.activity_logger import ActivityLogger
from forgeai.models.agent_state import AgentContext, AgentResult, AgentRole
from forgeai.models.task import AtomicTask, ImplementationPlan, RiskLevel
from forgeai.tools.llm_gateway import LLMGateway


class PlannerAgent(BaseAgent):
    """Converts specification + architecture into an ordered task list."""

    def __init__(self, llm: LLMGateway, logger: Optional[ActivityLogger] = None):
        super().__init__(AgentRole.PLANNER, llm, logger)

    def build_system_prompt(self) -> str:
        return (
            "You are an expert Software Project Planner AI. Given a project specification "
            "and architecture, you decompose the work into an ordered list of atomic "
            "implementation tasks.\n\n"
            "Rules for task decomposition:\n"
            "1. Each task must produce a single, independently verifiable change\n"
            "2. Tasks must be ordered so dependencies come first\n"
            "3. Each task should specify which files it creates/modifies\n"
            "4. Start with foundational tasks (models, config, database) before features\n"
            "5. End with integration and final cleanup tasks\n"
            "6. Mark high-risk tasks (database schema, auth, external APIs)\n"
            "7. Each task MUST be testable — include what should be tested\n"
            "8. First task should always be project setup (requirements.txt, config)\n\n"
            "The tasks will be executed in a TDD workflow:\n"
            "  - For each task, tests are written FIRST\n"
            "  - Then production code is generated to make tests pass\n"
            "  - This means each task description must be clear enough to write tests from"
        )

    def build_user_prompt(self, context: AgentContext) -> str:
        spec = context.specification
        spec_text = spec.to_prompt_context() if spec else "No spec"
        arch_text = json.dumps(context.architecture, indent=2) if context.architecture else "No architecture"

        return (
            f"Decompose this project into atomic implementation tasks:\n\n"
            f"## Specification\n{spec_text}\n\n"
            f"## Architecture\n{arch_text}\n\n"
            f"Respond with a JSON object:\n"
            f'{{\n'
            f'  "project_name": "string",\n'
            f'  "architecture_summary": "Brief summary of approach",\n'
            f'  "total_estimated_files": 15,\n'
            f'  "tasks": [\n'
            f'    {{\n'
            f'      "id": 1,\n'
            f'      "title": "Project Setup",\n'
            f'      "description": "Create requirements.txt, main.py entry point, and base configuration",\n'
            f'      "target_files": ["requirements.txt", "src/main.py", "src/config.py"],\n'
            f'      "dependencies": [],\n'
            f'      "risk_level": "low",\n'
            f'      "estimated_complexity": "simple",\n'
            f'      "test_description": "Test that the app starts and config loads correctly",\n'
            f'      "is_checkpoint": false\n'
            f'    }}\n'
            f'  ]\n'
            f'}}\n\n'
            f"IMPORTANT: Respond ONLY with valid JSON. Create {self._estimate_task_count(context)} tasks covering ALL features."
        )

    def _estimate_task_count(self, context: AgentContext) -> str:
        """Estimate reasonable task count based on project complexity."""
        if context.specification:
            tier = context.specification.tier
            return {1: "6-10", 2: "8-12", 3: "10-15", 4: "12-18", 5: "15-20"}.get(tier, "8-12")
        return "8-12"

    def parse_response(self, raw_response: str, context: AgentContext) -> AgentResult:
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                return AgentResult(success=False, role=self.role,
                                   error="Failed to parse plan as JSON")

        tasks = []
        for t in data.get("tasks", []):
            risk = t.get("risk_level", "low")
            if risk not in ("low", "medium", "high", "critical"):
                risk = "low"
            tasks.append(AtomicTask(
                id=t.get("id", 0),
                title=t.get("title", ""),
                description=t.get("description", ""),
                target_files=t.get("target_files", []),
                dependencies=t.get("dependencies", []),
                risk_level=RiskLevel(risk),
                estimated_complexity=t.get("estimated_complexity", "simple"),
                is_checkpoint=t.get("is_checkpoint", False),
            ))

        plan = ImplementationPlan(
            project_name=data.get("project_name", ""),
            tasks=tasks,
            total_estimated_files=data.get("total_estimated_files", 0),
            architecture_summary=data.get("architecture_summary", ""),
        )

        return AgentResult(
            success=True,
            role=self.role,
            implementation_plan=plan,
            message=f"Implementation plan created: {len(tasks)} tasks, "
                    f"~{plan.total_estimated_files} files",
        )
