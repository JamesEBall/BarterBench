"""Tests for model_registry.py (local vLLM registry)."""

import sys
from pathlib import Path

from barterbench.registry import (
    MODEL_REGISTRY, BASELINES, get_model_info, get_model_size, get_model_family,
    get_model_provider, get_size_tier, compute_dollar_cost, compute_gpu_cost,
    list_models_by_size, list_models_by_family, llm_models,
)
from barterbench.agents import SCRIPTED_AGENTS, resolve_endpoint, DEFAULT_BASE_URL


def test_baselines_registered_and_scripted():
    for b in BASELINES:
        info = get_model_info(b)
        assert info["family"] == "baseline"
        assert info["parameters_b"] == 0
        assert b in SCRIPTED_AGENTS


def test_every_llm_entry_is_a_hf_repo_with_vllm_hint():
    for name in llm_models():
        info = MODEL_REGISTRY[name]
        assert "/" in info["full_id"], f"{name}: full_id should be a Hugging Face repo id"
        assert info["open_weight"] is True, f"{name}: only open-weight models can run locally"
        assert info["served_name"], f"{name}: served_name missing"
        assert isinstance(info["vllm_args"], str)
        vram = info.get("vram_gb_quantized") or info.get("vram_gb_bf16")
        assert vram is not None and vram <= 80, f"{name}: must fit a single 80GB A100 ({vram} GB)"


def test_size_tiers():
    assert get_size_tier("random") == "baseline"
    assert get_size_tier("qwen3-8b") == "small"
    assert get_size_tier("gemma-3-27b") == "medium"
    assert get_size_tier("llama-3.3-70b") == "large"
    assert get_size_tier("gpt-oss-120b") == "frontier"
    assert get_size_tier("not-a-model") == "unknown"


def test_costs():
    assert compute_dollar_cost("qwen3-8b", 1_000_000, 1_000_000) == 0.0
    assert abs(compute_gpu_cost(3600, usd_per_gpu_hour=2.0) - 2.0) < 1e-9


def test_unknown_model_treated_as_served_id():
    info = get_model_info("my-org/my-finetune")
    assert info["family"] == "unknown"
    assert info["full_id"] == "my-org/my-finetune"
    assert info["served_name"] == "my-org/my-finetune"


def test_list_models_by_size_sorted():
    models = list_models_by_size()
    sized = [s for _, s in models if s is not None and s > 0]
    assert sized == sorted(sized)


def test_list_models_by_family():
    fams = list_models_by_family()
    assert "qwen" in fams and "qwen3-8b" in fams["qwen"]
    assert "baseline" in fams and set(BASELINES) <= set(fams["baseline"])


def test_required_fields_present():
    required = {"full_id", "family", "provider", "parameters_b", "context_window",
                "cost_tier", "cost_input_per_m", "cost_output_per_m", "served_name"}
    for name, info in MODEL_REGISTRY.items():
        missing = required - set(info)
        assert not missing, f"{name} missing {missing}"


def test_family_and_provider():
    assert get_model_family("llama-3.1-8b") == "llama"
    assert get_model_provider("llama-3.1-8b") == "meta"
    assert get_model_size("qwen3-8b") == 8.2


def test_resolve_endpoint_defaults_and_overrides():
    url, served = resolve_endpoint("qwen3-8b", endpoints={})
    assert url == DEFAULT_BASE_URL
    assert served == "Qwen/Qwen3-8B"
    url, served = resolve_endpoint("qwen3-8b", endpoints={"qwen3-8b": {"base_url": "http://gpu:9000/v1", "model": "q"}})
    assert url == "http://gpu:9000/v1" and served == "q"
    # Quantised 70B is served under a friendly name
    _, served = resolve_endpoint("llama-3.3-70b", endpoints={})
    assert served == "llama-3.3-70b-instruct"
