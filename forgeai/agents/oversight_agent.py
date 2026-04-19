"""Oversight Agent — Fleet AI / Scalable Oversight for multi-agent monitoring.

Satisfies the Fleet AI sub-theme: Trains oversight agents to monitor, analyze,
and explain the behavior of other AI agents operating in complex, multi-agent settings.

This agent:
  - Monitors outputs from all other agents for quality and consistency
  - Detects hallucinations (unreferenced APIs, impossible data models)
  - Flags quality issues (missing error handling, untested edge cases)
  - Generates natural-language explanations of agent behavior
  - Produces a structured OversightReport consumed by the meta-agent
"""

import json
from typing import Optional

from forgeai.agents.base_agent import BaseAgent
from forgeai.core.activity_logger import ActivityLogger
from forgeai.models.agent_state import AgentContext, AgentResult, AgentRole
from forgeai.tools.llm_gateway import LLMGateway


class OversightAgent(BaseAgent):
    """Monitors and evaluates the quality of all sub-agent outputs.

    Acts as a 'manager' that reviews the collective work of the team,
    ensuring coherence, quality, and adherence to the specification.
    """

    def __init__(self, llm: LLMGateway, logger: Optional[ActivityLogger] = None):
        super().__init__(AgentRole.OVERSIGHT, llm, logger)

    def build_system_prompt(self) -> str:
        return (
            "You are an expert AI Oversight Agent. Your role is to monitor, analyze, "
            "and evaluate the outputs of other AI agents working on a software project.\n\n"
            "Your responsibilities:\n"
            "1. QUALITY ASSESSMENT: Evaluate code quality, architecture coherence, and test coverage\n"
            "2. HALLUCINATION DETECTION: Flag any outputs that reference non-existent APIs, "
            "impossible data models, or unsupported features\n"
            "3. CONSISTENCY CHECK: Ensure all agent outputs are consistent with the specification\n"
            "4. BEHAVIOR ANALYSIS: Explain what each agent did and whether it was appropriate\n"
            "5. RISK IDENTIFICATION: Flag potential issues before they become problems\n\n"
            "You are objective, thorough, and constructive. Your analysis helps the meta-agent "
            "make better orchestration decisions.\n\n"
            "Rate quality on a 0.0 to 1.0 scale where:\n"
            "  0.0-0.3: Critical issues, likely non-functional\n"
            "  0.3-0.5: Significant issues, needs major rework\n"
            "  0.5-0.7: Acceptable but with notable gaps\n"
            "  0.7-0.9: Good quality with minor improvements possible\n"
            "  0.9-1.0: Excellent quality, production-ready"
        )

    def build_user_prompt(self, context: AgentContext) -> str:
        spec_text = ""
        if context.specification:
            spec_text = context.specification.to_prompt_context()

        code_review = ""
        test_review = ""
        for fname, content in context.existing_files.items():
            if "test" in fname.lower() or fname.startswith("tests/"):
                test_review += f"\n### {fname}\n```python\n{content[:1500]}\n```\n"
            elif fname.endswith(".py"):
                code_review += f"\n### {fname}\n```python\n{content[:1500]}\n```\n"

        return (
            f"Perform a comprehensive oversight review of the current project state.\n\n"
            f"## Project Specification\n{spec_text}\n\n"
            f"## Production Code\n{code_review}\n\n"
            f"## Test Code\n{test_review}\n\n"
            f"Respond with a JSON object:\n"
            f'{{\n'
            f'  "quality_score": 0.75,\n'
            f'  "issues_found": 3,\n'
            f'  "critical_issues": 0,\n'
            f'  "hallucination_flags": [\n'
            f'    "Description of any hallucinated/impossible code"\n'
            f'  ],\n'
            f'  "quality_issues": [\n'
            f'    {{"severity": "medium", "description": "Missing error handling in X", '
            f'"file": "src/main.py", "recommendation": "Add try/except"}}\n'
            f'  ],\n'
            f'  "consistency_issues": [\n'
            f'    "Any inconsistencies between spec and implementation"\n'
            f'  ],\n'
            f'  "recommendations": [\n'
            f'    "Actionable recommendations for improvement"\n'
            f'  ],\n'
            f'  "behavior_summary": "Natural language summary of agent behavior and project status",\n'
            f'  "test_coverage_assessment": "Analysis of test coverage gaps",\n'
            f'  "architecture_coherence": 0.8\n'
            f'}}\n\n'
            f"IMPORTANT: Respond ONLY with valid JSON. Be thorough but constructive."
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
                    success=False,
                    role=self.role,
                    error="Failed to parse oversight response as JSON",
                )

        quality_score = data.get("quality_score", 0.5)
        issues_found = data.get("issues_found", 0)
        critical_issues = data.get("critical_issues", 0)

        return AgentResult(
            success=True,
            role=self.role,
            message=(
                f"Oversight review complete: Quality={quality_score:.2f}, "
                f"Issues={issues_found} (Critical={critical_issues})"
            ),
            # Store the full oversight report in architecture field for downstream use
            architecture={
                "quality_score": quality_score,
                "issues_found": issues_found,
                "critical_issues": critical_issues,
                "hallucination_flags": data.get("hallucination_flags", []),
                "quality_issues": data.get("quality_issues", []),
                "consistency_issues": data.get("consistency_issues", []),
                "recommendations": data.get("recommendations", []),
                "behavior_summary": data.get("behavior_summary", ""),
                "test_coverage_assessment": data.get("test_coverage_assessment", ""),
                "architecture_coherence": data.get("architecture_coherence", 0.5),
            },
        )
