"""Simulated Expert Reviewer — Snorkel AI Expert-in-the-Loop theme.

Simulates interactions with real subject-matter experts who have:
  - Changing coding preferences (OOP vs functional, strict vs relaxed typing)
  - Evolving requirements mid-project ("actually, use MongoDB instead")
  - Variable approval thresholds based on reviewer "personality"
  - Constructive feedback that guides the meta-agent's decisions

This forces the RL agent to develop theory-of-mind reasoning:
  - Model the reviewer's current preferences
  - Anticipate preference changes
  - Adapt code generation strategy to satisfy shifting requirements
"""

import json
import random
from typing import Optional

from forgeai.agents.base_agent import BaseAgent
from forgeai.core.activity_logger import ActivityLogger
from forgeai.models.agent_state import AgentContext, AgentResult, AgentRole
from forgeai.tools.llm_gateway import LLMGateway


class SimulatedReviewer(BaseAgent):
    """Simulates a human expert code reviewer with changing preferences.

    Each reviewer personality has different coding style preferences,
    strictness levels, and approval thresholds. Preferences may change
    during a review session, forcing the meta-agent to adapt.
    """

    def __init__(
        self,
        llm: LLMGateway,
        logger: Optional[ActivityLogger] = None,
        reviewer_name: str = "Senior Architect",
        strictness: float = 0.7,
        preferences: Optional[dict] = None,
    ):
        super().__init__(AgentRole.REVIEWER, llm, logger)
        self.reviewer_name = reviewer_name
        self.strictness = strictness
        self.preferences = preferences or {
            "code_style": "functional",
            "typing": "strict",
            "error_handling": "comprehensive",
            "testing": "thorough",
            "documentation": "docstring",
        }

    def build_system_prompt(self) -> str:
        pref_text = "\n".join(
            [f"  - {k}: {v}" for k, v in self.preferences.items()]
        )
        return (
            f"You are {self.reviewer_name}, a demanding code reviewer with "
            f"specific preferences and high standards.\n\n"
            f"Your strictness level: {self.strictness:.1f}/1.0\n\n"
            f"Your current coding preferences:\n{pref_text}\n\n"
            f"Rules for review:\n"
            f"1. Evaluate code against YOUR specific preferences, not general best practices\n"
            f"2. Be {'very strict' if self.strictness > 0.7 else 'moderately strict' if self.strictness > 0.4 else 'lenient'} in your evaluation\n"
            f"3. Give a satisfaction score between 0.0 and 1.0\n"
            f"4. Provide constructive but personality-consistent feedback\n"
            f"5. Only approve if the code meets YOUR threshold of {self.strictness:.1f}\n\n"
            f"You are opinionated, consistent with your preferences, and provide "
            f"actionable feedback. You sometimes change your mind about requirements."
        )

    def build_user_prompt(self, context: AgentContext) -> str:
        spec_text = ""
        if context.specification:
            spec_text = context.specification.to_prompt_context()

        code_to_review = ""
        for fname, content in context.existing_files.items():
            if fname.endswith(".py") and "test" not in fname.lower():
                code_to_review += f"\n### {fname}\n```python\n{content[:2000]}\n```\n"

        return (
            f"Review the following code as {self.reviewer_name}.\n\n"
            f"## Project Specification\n{spec_text}\n\n"
            f"## Code to Review\n{code_to_review}\n\n"
            f"Respond with a JSON object:\n"
            f'{{\n'
            f'  "approved": false,\n'
            f'  "satisfaction_score": 0.65,\n'
            f'  "feedback": "Detailed feedback based on your preferences",\n'
            f'  "style_compliance": {{\n'
            f'    "code_style": {{"score": 0.7, "notes": "..."}},\n'
            f'    "typing": {{"score": 0.8, "notes": "..."}},\n'
            f'    "error_handling": {{"score": 0.6, "notes": "..."}},\n'
            f'    "testing": {{"score": 0.5, "notes": "..."}},\n'
            f'    "documentation": {{"score": 0.4, "notes": "..."}}\n'
            f'  }},\n'
            f'  "suggestions": ["Specific actionable suggestions"],\n'
            f'  "requirement_change": null\n'
            f'}}\n\n'
            f"IMPORTANT: Respond ONLY with valid JSON.\n"
            f"If you want to change a requirement, set 'requirement_change' to a string "
            f"describing the new requirement (e.g., 'Switch database from SQLite to PostgreSQL')."
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
                    error="Failed to parse reviewer response as JSON",
                )

        satisfaction = data.get("satisfaction_score", 0.5)
        approved = data.get("approved", False)
        feedback = data.get("feedback", "")
        requirement_change = data.get("requirement_change")

        return AgentResult(
            success=True,
            role=self.role,
            requires_human_input=not approved,
            message=(
                f"Reviewer {self.reviewer_name}: "
                f"{'APPROVED' if approved else 'CHANGES REQUESTED'} "
                f"(satisfaction: {satisfaction:.2f})"
            ),
            architecture={
                "approved": approved,
                "satisfaction_score": satisfaction,
                "feedback": feedback,
                "style_compliance": data.get("style_compliance", {}),
                "suggestions": data.get("suggestions", []),
                "requirement_change": requirement_change,
                "reviewer_name": self.reviewer_name,
                "preferences": self.preferences,
            },
        )

    def randomize_preference_change(self) -> Optional[tuple[str, str, str]]:
        """Randomly mutate one preference to simulate changing requirements.

        Returns:
            Tuple of (key, old_value, new_value) if changed, else None.
        """
        alternatives = {
            "code_style": ["functional", "oop", "procedural"],
            "typing": ["strict", "relaxed", "gradual"],
            "error_handling": ["comprehensive", "minimal", "fail_fast"],
            "testing": ["thorough", "minimal", "property_based"],
            "documentation": ["inline", "docstring", "none"],
        }

        key = random.choice(list(self.preferences.keys()))
        old_value = self.preferences[key]
        options = [v for v in alternatives.get(key, []) if v != old_value]

        if options:
            new_value = random.choice(options)
            self.preferences[key] = new_value

            if self.logger:
                self.logger.info(
                    "SimulatedReviewer",
                    f"{self.reviewer_name} changed preference: {key} from '{old_value}' to '{new_value}'",
                )

            return (key, old_value, new_value)
        return None
