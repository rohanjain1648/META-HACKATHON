"""Base Agent — Abstract base class for all ForgeAI agents.

Every agent follows the same contract: receive context, execute via LLM,
validate output, return structured result.
"""

import time
from abc import ABC, abstractmethod
from typing import Optional

from forgeai.core.activity_logger import ActivityLogger
from forgeai.models.agent_state import AgentContext, AgentResult, AgentRole
from forgeai.tools.llm_gateway import LLMGateway


class BaseAgent(ABC):
    """Abstract base class for all ForgeAI agents."""

    def __init__(self, role: AgentRole, llm: LLMGateway,
                 logger: Optional[ActivityLogger] = None):
        self.role = role
        self.llm = llm
        self.logger = logger

    @abstractmethod
    def build_system_prompt(self) -> str:
        """Build the system-level instruction for the LLM."""
        ...

    @abstractmethod
    def build_user_prompt(self, context: AgentContext) -> str:
        """Build the task-specific prompt from the agent context."""
        ...

    @abstractmethod
    def parse_response(self, raw_response: str, context: AgentContext) -> AgentResult:
        """Parse the LLM's raw response into a structured AgentResult."""
        ...

    def execute(self, context: AgentContext) -> AgentResult:
        """Execute the agent's task.
        
        This is the main entry point. It:
        1. Builds the system and user prompts
        2. Calls the LLM
        3. Parses the response
        4. Logs the activity
        """
        start_time = time.time()
        
        if self.logger:
            self.logger.agent(self.role.value, f"Agent starting execution")

        try:
            system_prompt = self.build_system_prompt()
            user_prompt = self.build_user_prompt(context)
            
            if self.logger:
                self.logger.agent(self.role.value, 
                    f"Calling LLM with {len(user_prompt)} char prompt")

            raw_response = self.llm.generate(
                prompt=user_prompt,
                system_instruction=system_prompt,
            )

            result = self.parse_response(raw_response, context)
            result.role = self.role
            result.duration_seconds = time.time() - start_time
            result.api_calls_made = 1

            if self.logger:
                status = "SUCCESS" if result.success else "NEEDS_INPUT" if result.requires_human_input else "FAILED"
                self.logger.agent(self.role.value, 
                    f"Execution complete: {status} ({result.duration_seconds:.1f}s)")

            return result

        except Exception as e:
            duration = time.time() - start_time
            if self.logger:
                self.logger.error(self.role.value, f"Agent failed: {str(e)}")

            return AgentResult(
                success=False,
                role=self.role,
                error=str(e),
                duration_seconds=duration,
            )

    def execute_json(self, context: AgentContext) -> AgentResult:
        """Execute the agent expecting a JSON response from the LLM."""
        start_time = time.time()

        if self.logger:
            self.logger.agent(self.role.value, f"Agent starting (JSON mode)")

        try:
            system_prompt = self.build_system_prompt()
            user_prompt = self.build_user_prompt(context)

            raw_json = self.llm.generate_json(
                prompt=user_prompt,
                system_instruction=system_prompt,
            )

            # Pass the parsed JSON as a string for parse_response
            import json
            result = self.parse_response(json.dumps(raw_json), context)
            result.role = self.role
            result.duration_seconds = time.time() - start_time
            result.api_calls_made = 1

            if self.logger:
                status = "SUCCESS" if result.success else "FAILED"
                self.logger.agent(self.role.value, f"JSON execution: {status}")

            return result

        except Exception as e:
            if self.logger:
                self.logger.error(self.role.value, f"JSON agent failed: {str(e)}")
            return AgentResult(
                success=False,
                role=self.role,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )
