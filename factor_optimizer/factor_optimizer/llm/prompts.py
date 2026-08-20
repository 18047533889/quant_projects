"""Versioned prompt templates for LLM proposal generation."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PromptTemplate:
    """
    A versioned prompt template with metadata.

    Attributes:
        template_id: Unique identifier for this template
        version: Template version (semantic versioning recommended)
        system_prompt: System/role prompt
        user_prompt_template: User prompt with {placeholders}
        output_schema: Expected output schema description
        model_hints: Model-specific tuning hints (temperature, etc.)
        description: Human-readable description of template purpose
    """

    template_id: str
    version: str
    system_prompt: str
    user_prompt_template: str
    output_schema: Dict[str, Any]
    model_hints: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self):
        """Validate required fields."""
        if not self.template_id:
            raise ValueError("template_id is required")
        if not self.version:
            raise ValueError("version is required")
        if not self.system_prompt:
            raise ValueError("system_prompt is required")
        if not self.user_prompt_template:
            raise ValueError("user_prompt_template is required")

    def render(self, **kwargs) -> str:
        """
        Render the user prompt with provided variables.

        Args:
            **kwargs: Variables to substitute into the template

        Returns:
            Rendered prompt string

        Raises:
            KeyError: If a required placeholder is missing
        """
        try:
            return self.user_prompt_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required placeholder: {e}")

    def get_full_prompt(self, **kwargs) -> tuple[str, str]:
        """
        Get both system and rendered user prompts.

        Returns:
            (system_prompt, rendered_user_prompt)
        """
        return self.system_prompt, self.render(**kwargs)


class PromptRegistry:
    """
    Registry for managing versioned prompt templates.

    Supports multiple versions of the same template.
    """

    def __init__(self):
        self._templates: Dict[str, Dict[str, PromptTemplate]] = {}

    def register(self, template: PromptTemplate) -> None:
        """
        Register a prompt template.

        Args:
            template: The template to register

        Raises:
            ValueError: If the exact template_id+version already exists
        """
        if template.template_id not in self._templates:
            self._templates[template.template_id] = {}

        versions = self._templates[template.template_id]
        if template.version in versions:
            raise ValueError(
                f"Template {template.template_id} version {template.version} already registered"
            )

        versions[template.version] = template

    def get(self, template_id: str, version: Optional[str] = None) -> PromptTemplate:
        """
        Retrieve a prompt template.

        Args:
            template_id: Template identifier
            version: Specific version (if None, returns latest)

        Returns:
            The requested template

        Raises:
            KeyError: If template or version not found
        """
        if template_id not in self._templates:
            raise KeyError(f"Template {template_id} not found")

        versions = self._templates[template_id]
        if not versions:
            raise KeyError(f"Template {template_id} has no versions")

        if version is None:
            # Return latest version (lexicographic sort, assumes semantic versioning)
            version = max(versions.keys())

        if version not in versions:
            raise KeyError(f"Template {template_id} version {version} not found")

        return versions[version]

    def list_templates(self) -> list[str]:
        """List all registered template IDs."""
        return sorted(self._templates.keys())

    def list_versions(self, template_id: str) -> list[str]:
        """List all versions of a template."""
        if template_id not in self._templates:
            return []
        return sorted(self._templates[template_id].keys())

    def clear(self) -> None:
        """Clear all registered templates."""
        self._templates.clear()


# Default prompt templates
DEFAULT_MUTATION_PROPOSAL_SYSTEM = """You are a quantitative factor research assistant. Your task is to propose mutations to existing factors that may improve their predictive power, risk characteristics, or computational efficiency.

You should:
- Understand the parent factor's mechanism and characteristics
- Propose mutations that are theoretically motivated
- Provide clear hypotheses about why the mutation might improve performance
- Consider trade-offs (e.g., complexity vs. signal strength)

Output your proposal as valid JSON matching the specified schema."""

DEFAULT_MUTATION_PROPOSAL_USER = """Parent Factor Analysis:
{parent_factor_info}

Current Performance Metrics:
{performance_metrics}

Search Objective:
{search_objective}

Available Mutation Types:
{available_mutations}

Propose 1-3 mutation candidates. For each, specify:
- mutation_type: Type of mutation from available list
- parameters: Mutation-specific parameters
- mechanism_hypothesis: Why this mutation might improve the factor
- expected_signatures: Expected changes in metrics or behavior

Return a JSON object with a "proposals" array."""

DEFAULT_MUTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["mutation_type", "parameters", "mechanism_hypothesis"],
                "properties": {
                    "mutation_type": {"type": "string"},
                    "parameters": {"type": "object"},
                    "mechanism_hypothesis": {"type": "string"},
                    "expected_signatures": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
    "required": ["proposals"],
}


def create_default_registry() -> PromptRegistry:
    """Create a registry with default templates."""
    registry = PromptRegistry()

    mutation_proposal_v1 = PromptTemplate(
        template_id="mutation_proposal",
        version="1.0.0",
        system_prompt=DEFAULT_MUTATION_PROPOSAL_SYSTEM,
        user_prompt_template=DEFAULT_MUTATION_PROPOSAL_USER,
        output_schema=DEFAULT_MUTATION_SCHEMA,
        model_hints={
            "temperature": 0.7,
            "max_tokens": 2000,
        },
        description="Generate mutation proposals for factor optimization",
    )

    registry.register(mutation_proposal_v1)
    return registry
