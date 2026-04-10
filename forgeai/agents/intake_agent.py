"""Intake Agent — Receives raw spec, identifies ambiguity, refines requirements.

Satisfies FR-01: Accept natural-language spec + identify underspecified aspects.
Satisfies FR-02: Produce structured specification with acceptance criteria.
"""

import json
from typing import Optional

from forgeai.agents.base_agent import BaseAgent
from forgeai.core.activity_logger import ActivityLogger
from forgeai.models.agent_state import AgentContext, AgentResult, AgentRole
from forgeai.models.specification import (
    APIEndpoint,
    DataModel,
    StructuredSpecification,
)
from forgeai.tools.llm_gateway import LLMGateway


class IntakeAgent(BaseAgent):
    """Analyzes raw project specifications and produces structured output."""

    def __init__(self, llm: LLMGateway, logger: Optional[ActivityLogger] = None):
        super().__init__(AgentRole.INTAKE, llm, logger)

    def build_system_prompt(self) -> str:
        return (
            "You are an expert Requirements Analyst AI. Your job is to take a raw, "
            "natural-language project specification and:\n"
            "1. Identify underspecified or ambiguous aspects\n"
            "2. Generate targeted clarifying questions (max 5-7)\n"
            "3. After clarification, produce a complete structured specification\n\n"
            "You are meticulous about edge cases, validation rules, and acceptance criteria.\n"
            "The backend MUST be in Python. Frontend (if needed) must be React or Angular.\n"
            "Always think about: data models, API contracts, error handling, validation, "
            "and non-functional requirements."
        )

    def build_user_prompt(self, context: AgentContext) -> str:
        if context.clarification_responses:
            # Phase 2: We have clarification answers, produce the full spec
            return self._build_specification_prompt(context)
        else:
            # Phase 1: Analyze raw spec, generate questions
            return self._build_analysis_prompt(context)

    def _build_analysis_prompt(self, context: AgentContext) -> str:
        return (
            f"Analyze the following project specification and identify ambiguities.\n\n"
            f"## Raw Specification:\n{context.user_input}\n\n"
            f"Respond with a JSON object containing:\n"
            f'{{\n'
            f'  "project_name": "descriptive name",\n'
            f'  "summary": "1-2 sentence summary of what this project does",\n'
            f'  "identified_ambiguities": ["list of unclear aspects"],\n'
            f'  "clarifying_questions": ["specific questions to resolve ambiguities"],\n'
            f'  "initial_requirements": ["list of requirements you can already extract"],\n'
            f'  "suggested_tier": 1,\n'
            f'  "tech_stack": {{"backend": "Python/FastAPI", "database": "...", "frontend": "..."}}\n'
            f'}}\n\n'
            f"IMPORTANT: Respond ONLY with valid JSON."
        )

    def _build_specification_prompt(self, context: AgentContext) -> str:
        q_and_a = "\n".join([
            f"Q: {q}\nA: {a}"
            for q, a in context.clarification_responses.items()
        ])
        return (
            f"Based on the original specification and clarification answers, "
            f"produce a complete structured specification.\n\n"
            f"## Original Specification:\n{context.user_input}\n\n"
            f"## Clarifications:\n{q_and_a}\n\n"
            f"Respond with a JSON object containing:\n"
            f'{{\n'
            f'  "project_name": "string",\n'
            f'  "summary": "full description",\n'
            f'  "tier": 1,\n'
            f'  "acceptance_criteria": ["list of testable criteria"],\n'
            f'  "constraints": ["technical/business constraints"],\n'
            f'  "functional_requirements": ["detailed FR list"],\n'
            f'  "non_functional_requirements": ["NFR list"],\n'
            f'  "tech_stack": {{"backend": "...", "database": "...", "frontend": "..."}},\n'
            f'  "data_models": [{{"name": "...", "fields": {{"field": "type"}}, "description": "...", "validations": ["..."]}}],\n'
            f'  "api_endpoints": [{{"method": "GET", "path": "/...", "description": "...", "auth_required": false}}],\n'
            f'  "architecture_style": "monolith|microservice|layered"\n'
            f'}}\n\n'
            f"IMPORTANT: Respond ONLY with valid JSON. Be thorough and specific."
        )

    def parse_response(self, raw_response: str, context: AgentContext) -> AgentResult:
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                return AgentResult(
                    success=False,
                    role=self.role,
                    error="Failed to parse LLM response as JSON",
                )

        if "clarifying_questions" in data and not context.clarification_responses:
            # Phase 1: Return questions for user
            return AgentResult(
                success=True,
                role=self.role,
                clarifying_questions=data.get("clarifying_questions", []),
                requires_human_input=True,
                message=f"Project: {data.get('project_name', 'Unknown')}\n"
                        f"Summary: {data.get('summary', '')}\n"
                        f"Suggested Tier: {data.get('suggested_tier', 1)}",
                specification=StructuredSpecification(
                    project_name=data.get("project_name", ""),
                    summary=data.get("summary", ""),
                    tier=data.get("suggested_tier", 1),
                    raw_input=context.user_input,
                    clarification_questions=data.get("clarifying_questions", []),
                    functional_requirements=data.get("initial_requirements", []),
                    tech_stack=data.get("tech_stack", {}),
                ),
            )
        else:
            # Phase 2: Full specification
            data_models = []
            for dm in data.get("data_models", []):
                data_models.append(DataModel(
                    name=dm.get("name", ""),
                    fields=dm.get("fields", {}),
                    description=dm.get("description", ""),
                    validations=dm.get("validations", []),
                ))

            api_endpoints = []
            for ep in data.get("api_endpoints", []):
                api_endpoints.append(APIEndpoint(
                    method=ep.get("method", "GET"),
                    path=ep.get("path", "/"),
                    description=ep.get("description", ""),
                    auth_required=ep.get("auth_required", False),
                ))

            spec = StructuredSpecification(
                project_name=data.get("project_name", ""),
                summary=data.get("summary", ""),
                tier=data.get("tier", 1),
                acceptance_criteria=data.get("acceptance_criteria", []),
                constraints=data.get("constraints", []),
                functional_requirements=data.get("functional_requirements", []),
                non_functional_requirements=data.get("non_functional_requirements", []),
                tech_stack=data.get("tech_stack", {}),
                data_models=data_models,
                api_endpoints=api_endpoints,
                architecture_style=data.get("architecture_style", "monolith"),
                raw_input=context.user_input,
                clarification_answers=list(context.clarification_responses.values()),
            )

            return AgentResult(
                success=True,
                role=self.role,
                specification=spec,
                message=f"Structured specification created for: {spec.project_name}",
            )
