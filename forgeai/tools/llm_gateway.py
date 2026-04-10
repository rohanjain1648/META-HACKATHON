"""LLM Gateway — Unified interface for calling LLM APIs.

Supports Google Gemini (primary), with extensible design for OpenAI/Anthropic.
Handles rate limiting, retries, token counting, and request/response logging.
"""

import json
import time
import traceback
from typing import Optional

import google.generativeai as genai

from forgeai.core.activity_logger import ActivityLogger


class LLMGateway:
    """Unified LLM API interface with logging and retry logic."""

    def __init__(self, provider: str, model: str, api_key: str,
                 temperature: float = 0.2, max_tokens: int = 8192,
                 logger: Optional[ActivityLogger] = None):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger

        # Counters for summary report
        self.total_api_calls = 0
        self.total_tokens_used = 0

        # Initialize provider
        if provider == "google":
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(model)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def generate(self, prompt: str, system_instruction: str = "",
                 temperature: Optional[float] = None,
                 max_retries: int = 3) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The user/task prompt.
            system_instruction: System-level instruction for the model.
            temperature: Override default temperature.
            max_retries: Number of retry attempts on failure.
            
        Returns:
            The generated text response.
        """
        temp = temperature if temperature is not None else self.temperature

        for attempt in range(max_retries):
            try:
                if self.logger:
                    self.logger.api_call("LLMGateway", f"API call #{self.total_api_calls + 1} to {self.provider}/{self.model}", {
                        "prompt_length": len(prompt),
                        "attempt": attempt + 1,
                    })

                response = self._call_provider(prompt, system_instruction, temp)
                self.total_api_calls += 1

                # Estimate tokens (rough: 1 token ≈ 4 chars)
                est_tokens = (len(prompt) + len(response)) // 4
                self.total_tokens_used += est_tokens

                if self.logger:
                    self.logger.api_call("LLMGateway", f"Response received ({len(response)} chars, ~{est_tokens} tokens)")

                return response

            except Exception as e:
                error_msg = f"LLM API error (attempt {attempt + 1}/{max_retries}): {str(e)}"
                if self.logger:
                    self.logger.error("LLMGateway", error_msg)

                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)  # Exponential backoff
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"LLM API failed after {max_retries} attempts: {str(e)}")

    def _call_provider(self, prompt: str, system_instruction: str, temperature: float) -> str:
        """Call the configured LLM provider."""
        if self.provider == "google":
            return self._call_gemini(prompt, system_instruction, temperature)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _call_gemini(self, prompt: str, system_instruction: str, temperature: float) -> str:
        """Call Google Gemini API."""
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=self.max_tokens,
        )

        # If there's a system instruction, create a model with it
        if system_instruction:
            model = genai.GenerativeModel(
                self.model,
                system_instruction=system_instruction,
            )
        else:
            model = self._gemini_model

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        return response.text

    def generate_json(self, prompt: str, system_instruction: str = "",
                      max_retries: int = 3) -> dict:
        """Generate a response and parse it as JSON.
        
        Automatically wraps the prompt to request JSON output and 
        handles extraction from markdown code blocks.
        """
        json_prompt = (
            f"{prompt}\n\n"
            "IMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation, "
            "no code blocks. Just the raw JSON object."
        )

        for attempt in range(max_retries):
            try:
                raw = self.generate(json_prompt, system_instruction, max_retries=1)
                # Clean response — strip markdown code fences if present
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    # Remove ```json and trailing ```
                    lines = cleaned.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    cleaned = "\n".join(lines)

                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                if self.logger:
                    self.logger.warn("LLMGateway", f"JSON parse failed (attempt {attempt + 1}): {str(e)}")
                if attempt >= max_retries - 1:
                    raise

    def get_stats(self) -> dict:
        """Return usage statistics."""
        return {
            "total_api_calls": self.total_api_calls,
            "total_tokens_used": self.total_tokens_used,
            "provider": self.provider,
            "model": self.model,
        }
