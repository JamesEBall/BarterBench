"""Tests for model_registry.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_registry import (
    MODEL_REGISTRY, get_model_info, get_model_size, get_model_family,
    get_model_provider, get_size_tier, compute_dollar_cost,
    list_models_by_size, list_models_by_family,
)
from agent import MODEL_MAP, OPENROUTER_MODEL_MAP


def test_all_agent_models_registered():
    """Every model in MODEL_MAP and OPENROUTER_MODEL_MAP has a registry entry."""
    for model in MODEL_MAP:
        info = get_model_info(model)
        assert info["full_id"] != "unknown", f"MODEL_MAP model '{model}' not in registry"

    for model in OPENROUTER_MODEL_MAP:
        info = get_model_info(model)
        assert info["full_id"] != "unknown", f"OPENROUTER_MODEL_MAP model '{model}' not in registry"


def test_random_baseline_registered():
    """Random baseline agent has a registry entry."""
    info = get_model_info("random")
    assert info["family"] == "baseline"
    assert info["parameters_b"] == 0
    assert info["cost_tier"] == "free"


def test_model_size_ordering():
    """Anthropic models are ordered: haiku < sonnet < opus."""
    haiku = get_model_size("haiku")
    sonnet = get_model_size("sonnet")
    opus = get_model_size("opus")
    assert haiku is not None and sonnet is not None and opus is not None
    assert haiku < sonnet < opus


def test_size_tiers():
    """Size tier classification is correct."""
    assert get_size_tier("random") == "baseline"
    assert get_size_tier("haiku") == "small"
    assert get_size_tier("gemma-27b") == "medium"
    assert get_size_tier("llama-70b") == "large"
    assert get_size_tier("hunter") == "frontier"


def test_compute_dollar_cost():
    """Cost computation matches known pricing."""
    # Free model = zero cost
    assert compute_dollar_cost("random", 1000000, 1000000) == 0.0
    assert compute_dollar_cost("hunter", 1000000, 1000000) == 0.0

    # Haiku: $0.80/M input, $4.00/M output
    cost = compute_dollar_cost("haiku", 1000000, 1000000)
    assert abs(cost - 4.80) < 0.01

    # Opus: $15/M input, $75/M output
    cost = compute_dollar_cost("opus", 1000000, 1000000)
    assert abs(cost - 90.0) < 0.01


def test_unknown_model_graceful():
    """Unregistered model returns sensible defaults, not crash."""
    info = get_model_info("totally_unknown_model_xyz")
    assert info["family"] == "unknown"
    assert info["provider"] == "unknown"
    assert info["full_id"] == "totally_unknown_model_xyz"


def test_list_models_by_size():
    """list_models_by_size returns sorted list."""
    models = list_models_by_size()
    assert len(models) > 0
    # First model with known size should be smallest
    sized = [(name, size) for name, size in models if size is not None and size > 0]
    for i in range(len(sized) - 1):
        assert sized[i][1] <= sized[i + 1][1], f"{sized[i]} should be <= {sized[i+1]}"


def test_list_models_by_family():
    """list_models_by_family groups correctly."""
    families = list_models_by_family()
    assert "claude" in families
    assert "haiku" in families["claude"]
    assert "baseline" in families
    assert "random" in families["baseline"]


def test_all_required_fields():
    """Every registry entry has all required fields."""
    required = {"full_id", "family", "provider", "parameters_b",
                "context_window", "cost_tier", "cost_input_per_m", "cost_output_per_m"}
    for name, info in MODEL_REGISTRY.items():
        for field in required:
            assert field in info, f"Model '{name}' missing required field '{field}'"


def test_get_model_family():
    assert get_model_family("haiku") == "claude"
    assert get_model_family("llama-70b") == "llama"
    assert get_model_family("gemma-27b") == "gemma"


def test_get_model_provider():
    assert get_model_provider("haiku") == "anthropic"
    assert get_model_provider("llama-70b") == "meta"
    assert get_model_provider("hunter") == "openrouter"
