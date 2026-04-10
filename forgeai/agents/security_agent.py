"""Security Auditor Agent — Post-module security sweep.

Satisfies FR-14 [EXTENDED]: AI-powered security audit flagging vulnerabilities.
"""

import json
from typing import Optional

from forgeai.agents.base_agent import BaseAgent
from forgeai.core.activity_logger import ActivityLogger
from forgeai.models.agent_state import AgentContext, AgentResult, AgentRole
from forgeai.tools.llm_gateway import LLMGateway


class SecurityAgent(BaseAgent):
    """Performs security audits on generated code, flagging vulnerabilities."""

    def __init__(self, llm: LLMGateway, logger: Optional[ActivityLogger] = None):
        super().__init__(AgentRole.SECURITY, llm, logger)

    def build_system_prompt(self) -> str:
        return (
            "You are an expert Security Auditor AI. You review generated source code "
            "for security vulnerabilities and best practice violations.\n\n"
            "Check for:\n"
            "1. Hardcoded secrets, API keys, passwords\n"
            "2. SQL injection vulnerabilities\n"
            "3. Command injection vulnerabilities\n"
            "4. Path traversal attacks\n"
            "5. Missing input validation\n"
            "6. Authentication/authorization bypasses\n"
            "7. Insecure deserialization\n"
            "8. Missing CORS configuration\n"
            "9. Sensitive data exposure\n"
            "10. Missing rate limiting\n\n"
            "Rate each finding by severity: CRITICAL, HIGH, MEDIUM, LOW, INFO\n"
            "Provide specific remediation for each finding."
        )

    def build_user_prompt(self, context: AgentContext) -> str:
        code_review = ""
        for fname, content in context.existing_files.items():
            if fname.endswith(".py") and "test" not in fname.lower():
                code_review += f"\n### {fname}\n```python\n{content}\n```\n"

        return (
            f"Perform a security audit on the following codebase:\n\n"
            f"{code_review}\n\n"
            f"Respond with a JSON object:\n"
            f'{{\n'
            f'  "findings": [\n'
            f'    {{\n'
            f'      "severity": "HIGH",\n'
            f'      "category": "hardcoded_secrets",\n'
            f'      "file": "src/config.py",\n'
            f'      "line": 15,\n'
            f'      "description": "Hardcoded database password found",\n'
            f'      "remediation": "Use environment variable instead"\n'
            f'    }}\n'
            f'  ],\n'
            f'  "summary": {{\n'
            f'    "total_findings": 3,\n'
            f'    "critical": 0,\n'
            f'    "high": 1,\n'
            f'    "medium": 1,\n'
            f'    "low": 1,\n'
            f'    "overall_risk": "MEDIUM"\n'
            f'  }},\n'
            f'  "recommendations": ["Top-level security recommendations"]\n'
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
                    error="Failed to parse security audit response",
                )

        summary = data.get("summary", {})
        findings = data.get("findings", [])

        return AgentResult(
            success=True,
            role=self.role,
            security_report=data,
            message=f"Security audit complete: {summary.get('total_findings', 0)} findings, "
                    f"Risk: {summary.get('overall_risk', 'UNKNOWN')}",
        )
