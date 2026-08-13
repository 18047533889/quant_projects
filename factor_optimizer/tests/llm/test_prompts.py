"""Tests for prompt templates and registry."""

from factor_optimizer.llm.prompts import (
    PromptTemplate,
    PromptRegistry,
    create_default_registry,
    DEFAULT_MUTATION_PROPOSAL_SYSTEM,
    DEFAULT_MUTATION_PROPOSAL_USER,
)


def test_prompt_template_creation():
    """Test basic prompt template creation."""
    template = PromptTemplate(
        template_id="test_template",
        version="1.0.0",
        system_prompt="You are a helpful assistant.",
        user_prompt_template="Analyze this: {content}",
        output_schema={"type": "object"},
        model_hints={"temperature": 0.5},
        description="Test template",
    )

    assert template.template_id == "test_template"
    assert template.version == "1.0.0"
    assert template.system_prompt == "You are a helpful assistant."
    assert template.model_hints["temperature"] == 0.5


def test_prompt_template_validation():
    """Test prompt template validation."""
    try:
        PromptTemplate(
            template_id="",  # Invalid
            version="1.0.0",
            system_prompt="test",
            user_prompt_template="test",
            output_schema={},
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "template_id is required" in str(e)


def test_prompt_template_render():
    """Test prompt rendering with variables."""
    template = PromptTemplate(
        template_id="test",
        version="1.0.0",
        system_prompt="System",
        user_prompt_template="Hello {name}, your score is {score}",
        output_schema={},
    )

    rendered = template.render(name="Alice", score=95)
    assert rendered == "Hello Alice, your score is 95"


def test_prompt_template_render_missing_variable():
    """Test that rendering fails with missing variables."""
    template = PromptTemplate(
        template_id="test",
        version="1.0.0",
        system_prompt="System",
        user_prompt_template="Hello {name}",
        output_schema={},
    )

    try:
        template.render()  # Missing 'name'
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Missing required placeholder" in str(e)


def test_prompt_template_get_full_prompt():
    """Test getting both system and rendered prompts."""
    template = PromptTemplate(
        template_id="test",
        version="1.0.0",
        system_prompt="You are helpful",
        user_prompt_template="Process {data}",
        output_schema={},
    )

    system, user = template.get_full_prompt(data="sample")
    assert system == "You are helpful"
    assert user == "Process sample"


def test_prompt_registry_register_and_get():
    """Test basic registry operations."""
    registry = PromptRegistry()

    template = PromptTemplate(
        template_id="test_template",
        version="1.0.0",
        system_prompt="System",
        user_prompt_template="User",
        output_schema={},
    )

    registry.register(template)

    # Get by ID (latest version)
    retrieved = registry.get("test_template")
    assert retrieved.template_id == "test_template"
    assert retrieved.version == "1.0.0"

    # Get by ID and version
    retrieved = registry.get("test_template", "1.0.0")
    assert retrieved.version == "1.0.0"


def test_prompt_registry_multiple_versions():
    """Test registering and retrieving multiple versions."""
    registry = PromptRegistry()

    v1 = PromptTemplate(
        template_id="evolving",
        version="1.0.0",
        system_prompt="Version 1",
        user_prompt_template="V1",
        output_schema={},
    )

    v2 = PromptTemplate(
        template_id="evolving",
        version="2.0.0",
        system_prompt="Version 2",
        user_prompt_template="V2",
        output_schema={},
    )

    registry.register(v1)
    registry.register(v2)

    # Get latest (should be v2)
    latest = registry.get("evolving")
    assert latest.version == "2.0.0"
    assert latest.system_prompt == "Version 2"

    # Get specific version
    specific = registry.get("evolving", "1.0.0")
    assert specific.version == "1.0.0"
    assert specific.system_prompt == "Version 1"


def test_prompt_registry_duplicate_version_rejection():
    """Test that duplicate template+version is rejected."""
    registry = PromptRegistry()

    template1 = PromptTemplate(
        template_id="test",
        version="1.0.0",
        system_prompt="First",
        user_prompt_template="First",
        output_schema={},
    )

    template2 = PromptTemplate(
        template_id="test",
        version="1.0.0",  # Same version
        system_prompt="Second",
        user_prompt_template="Second",
        output_schema={},
    )

    registry.register(template1)

    try:
        registry.register(template2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "already registered" in str(e)


def test_prompt_registry_list_templates():
    """Test listing all template IDs."""
    registry = PromptRegistry()

    templates = [
        PromptTemplate(
            template_id=f"template_{i}",
            version="1.0.0",
            system_prompt="System",
            user_prompt_template="User",
            output_schema={},
        )
        for i in range(3)
    ]

    for template in templates:
        registry.register(template)

    template_ids = registry.list_templates()
    assert len(template_ids) == 3
    assert "template_0" in template_ids
    assert "template_1" in template_ids
    assert "template_2" in template_ids
    assert template_ids == sorted(template_ids)  # Should be sorted


def test_prompt_registry_list_versions():
    """Test listing versions of a specific template."""
    registry = PromptRegistry()

    versions = ["1.0.0", "1.1.0", "2.0.0"]
    for version in versions:
        template = PromptTemplate(
            template_id="multi_version",
            version=version,
            system_prompt="System",
            user_prompt_template="User",
            output_schema={},
        )
        registry.register(template)

    listed_versions = registry.list_versions("multi_version")
    assert len(listed_versions) == 3
    assert "1.0.0" in listed_versions
    assert "2.0.0" in listed_versions
    assert listed_versions == sorted(listed_versions)


def test_prompt_registry_list_versions_empty():
    """Test listing versions for non-existent template."""
    registry = PromptRegistry()
    versions = registry.list_versions("nonexistent")
    assert versions == []


def test_prompt_registry_get_nonexistent_template():
    """Test getting a non-existent template raises KeyError."""
    registry = PromptRegistry()

    try:
        registry.get("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError as e:
        assert "not found" in str(e)


def test_prompt_registry_get_nonexistent_version():
    """Test getting a non-existent version raises KeyError."""
    registry = PromptRegistry()

    template = PromptTemplate(
        template_id="test",
        version="1.0.0",
        system_prompt="System",
        user_prompt_template="User",
        output_schema={},
    )
    registry.register(template)

    try:
        registry.get("test", "2.0.0")  # Version doesn't exist
        assert False, "Should have raised KeyError"
    except KeyError as e:
        assert "version" in str(e).lower()


def test_prompt_registry_clear():
    """Test clearing the registry."""
    registry = PromptRegistry()

    template = PromptTemplate(
        template_id="test",
        version="1.0.0",
        system_prompt="System",
        user_prompt_template="User",
        output_schema={},
    )
    registry.register(template)

    assert len(registry.list_templates()) == 1

    registry.clear()
    assert len(registry.list_templates()) == 0


def test_default_registry_creation():
    """Test that default registry is properly initialized."""
    registry = create_default_registry()

    # Should have mutation_proposal template
    templates = registry.list_templates()
    assert "mutation_proposal" in templates

    # Should have version 1.0.0
    versions = registry.list_versions("mutation_proposal")
    assert "1.0.0" in versions

    # Retrieve and verify
    template = registry.get("mutation_proposal", "1.0.0")
    assert template.system_prompt == DEFAULT_MUTATION_PROPOSAL_SYSTEM
    assert template.user_prompt_template == DEFAULT_MUTATION_PROPOSAL_USER
    assert "temperature" in template.model_hints


def test_default_template_placeholders():
    """Test that default template has expected placeholders."""
    registry = create_default_registry()
    template = registry.get("mutation_proposal")

    # Test rendering with required placeholders
    rendered = template.render(
        parent_factor_info="info",
        performance_metrics="metrics",
        search_objective="objective",
        available_mutations="mutations",
    )

    assert "info" in rendered
    assert "metrics" in rendered
    assert "objective" in rendered
    assert "mutations" in rendered


def test_default_template_output_schema():
    """Test that default template has proper output schema."""
    registry = create_default_registry()
    template = registry.get("mutation_proposal")

    schema = template.output_schema
    assert "properties" in schema
    assert "proposals" in schema["properties"]
    assert schema["properties"]["proposals"]["type"] == "array"
