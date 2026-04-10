"""Structured Specification — The refined, machine-readable project spec.

Produced by the Intake Agent after requirement clarification (FR-01, FR-02).
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class APIEndpoint(BaseModel):
    """Describes an API endpoint in the project."""
    method: str = "GET"
    path: str = "/"
    description: str = ""
    request_body: Optional[dict] = None
    response_schema: Optional[dict] = None
    auth_required: bool = False


class DataModel(BaseModel):
    """Describes a data model / schema in the project."""
    name: str
    fields: dict[str, str] = Field(default_factory=dict)  # field_name -> type
    description: str = ""
    validations: list[str] = Field(default_factory=list)


class StructuredSpecification(BaseModel):
    """The complete refined specification for the project to be built.
    
    This is the primary artifact produced by the Intake Agent and consumed
    by all downstream agents. It serves as the single source of truth.
    """
    # Core identity
    project_name: str = ""
    summary: str = ""
    tier: int = 1  # Complexity tier 1-5

    # Requirements
    acceptance_criteria: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    functional_requirements: list[str] = Field(default_factory=list)
    non_functional_requirements: list[str] = Field(default_factory=list)

    # Technical design
    tech_stack: dict[str, str] = Field(default_factory=dict)
    data_models: list[DataModel] = Field(default_factory=list)
    api_endpoints: list[APIEndpoint] = Field(default_factory=list)

    # Architecture hints
    architecture_style: str = "monolith"  # monolith | microservice | layered
    directory_structure: dict = Field(default_factory=dict)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    clarification_questions: list[str] = Field(default_factory=list)
    clarification_answers: list[str] = Field(default_factory=list)
    raw_input: str = ""  # Original user specification

    def to_prompt_context(self) -> str:
        """Serialize to a compact string for LLM prompt injection."""
        lines = [
            f"# Project: {self.project_name}",
            f"## Summary\n{self.summary}",
            f"## Tier: {self.tier}",
            f"## Acceptance Criteria",
        ]
        for i, ac in enumerate(self.acceptance_criteria, 1):
            lines.append(f"  {i}. {ac}")
        lines.append("## Constraints")
        for c in self.constraints:
            lines.append(f"  - {c}")
        lines.append("## Data Models")
        for dm in self.data_models:
            lines.append(f"  - {dm.name}: {dm.fields}")
        lines.append("## API Endpoints")
        for ep in self.api_endpoints:
            lines.append(f"  - {ep.method} {ep.path}: {ep.description}")
        lines.append(f"## Tech Stack: {self.tech_stack}")
        return "\n".join(lines)
