"""Architect Agent — Designs project structure, data models, and API contracts.

Satisfies FR-04: Design project structure from scratch including directory layout,
modules, data models, and API contracts before any code is written.
"""

import json
from typing import Optional

from forgeai.agents.base_agent import BaseAgent
from forgeai.core.activity_logger import ActivityLogger
from forgeai.models.agent_state import AgentContext, AgentResult, AgentRole
from forgeai.tools.llm_gateway import LLMGateway


class ArchitectAgent(BaseAgent):
    """Designs the complete project architecture from the structured specification."""

    def __init__(self, llm: LLMGateway, logger: Optional[ActivityLogger] = None):
        super().__init__(AgentRole.ARCHITECT, llm, logger)

    def build_system_prompt(self) -> str:
        return (
            "You are an expert Software Architect AI. Given a structured project specification, "
            "you design the complete project architecture from scratch.\n\n"
            "Your output must include:\n"
            "1. Directory/file layout (Python backend, with React/Angular frontend if needed)\n"
            "2. Module responsibilities and boundaries\n"
            "3. Data model schemas (Pydantic models for Python)\n"
            "4. API contract definitions (endpoints, request/response schemas)\n"
            "5. Database schema design\n"
            "6. Key dependencies and their versions\n"
            "7. Configuration approach\n\n"
            "Rules:\n"
            "- Backend MUST be Python (FastAPI recommended)\n"
            "- Frontend (if needed) MUST be React or Angular\n"
            "- Design for testability — every module should be independently testable\n"
            "- Follow clean architecture / layered design principles\n"
            "- Include proper error handling patterns\n"
            "- Include requirements.txt for Python dependencies"
        )

    def build_user_prompt(self, context: AgentContext) -> str:
        spec = context.specification
        spec_context = spec.to_prompt_context() if spec else "No specification provided"

        return (
            f"Design the complete project architecture based on this specification:\n\n"
            f"{spec_context}\n\n"
            f"Respond with a JSON object:\n"
            f'{{\n'
            f'  "project_name": "string",\n'
            f'  "architecture_style": "layered",\n'
            f'  "directory_structure": {{\n'
            f'    "src/": {{\n'
            f'      "main.py": "Application entry point",\n'
            f'      "models/": {{"__init__.py": "...", "schemas.py": "Pydantic models"}},\n'
            f'      "routes/": {{"__init__.py": "...", "api.py": "API endpoints"}},\n'
            f'      "services/": {{"__init__.py": "...", "business_logic.py": "Core logic"}},\n'
            f'      "database/": {{"__init__.py": "...", "connection.py": "DB setup"}}\n'
            f'    }},\n'
            f'    "tests/": {{"test_models.py": "...", "test_routes.py": "...", "test_services.py": "...", "conftest.py": "Fixtures"}},\n'
            f'    "requirements.txt": "Python dependencies",\n'
            f'    "README.md": "Project documentation"\n'
            f'  }},\n'
            f'  "modules": [\n'
            f'    {{"name": "models", "responsibility": "Data models and validation", "files": ["src/models/schemas.py"]}}\n'
            f'  ],\n'
            f'  "data_models": [\n'
            f'    {{"name": "ModelName", "fields": {{"id": "int", "name": "str"}}, "validations": ["name must not be empty"]}}\n'
            f'  ],\n'
            f'  "api_endpoints": [\n'
            f'    {{"method": "POST", "path": "/api/resource", "description": "Create resource", "request_schema": {{}}, "response_schema": {{}}}}\n'
            f'  ],\n'
            f'  "dependencies": {{"fastapi": ">=0.110.0", "uvicorn": ">=0.27.0", "pydantic": ">=2.0.0", "pytest": ">=8.0.0"}},\n'
            f'  "database": {{"type": "sqlite|mongodb|postgres", "schema": "..."}},\n'
            f'  "configuration": {{"approach": "env_variables", "files": [".env", "config.py"]}},\n'
            f'  "error_handling": "Description of error handling strategy"\n'
            f'}}\n\n'
            f"IMPORTANT: Respond ONLY with valid JSON. Be thorough and production-quality."
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
                    error="Failed to parse architecture response as JSON",
                )

        return AgentResult(
            success=True,
            role=self.role,
            architecture=data,
            message=f"Architecture designed: {data.get('architecture_style', 'layered')} "
                    f"with {len(data.get('modules', []))} modules, "
                    f"{len(data.get('api_endpoints', []))} endpoints",
        )
