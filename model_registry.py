"""Central model metadata registry for BarterBench.

Every supported model has an entry with size, family, provider, cost,
context window, and other measurable dimensions for benchmark analysis.
"""


MODEL_REGISTRY = {
    # ---- Anthropic API models ----
    "haiku": {
        "full_id": "claude-haiku-4-5-20251001",
        "family": "claude",
        "provider": "anthropic",
        "parameters_b": 3.5,
        "parameters_estimated": True,
        "context_window": 200_000,
        "cost_tier": "paid",
        "cost_input_per_m": 0.80,
        "cost_output_per_m": 4.00,
        "release_date": "2025-10-01",
        "open_weight": False,
        "architecture": "transformer",
    },
    "sonnet": {
        "full_id": "claude-sonnet-4-6",
        "family": "claude",
        "provider": "anthropic",
        "parameters_b": 70,
        "parameters_estimated": True,
        "context_window": 200_000,
        "cost_tier": "paid",
        "cost_input_per_m": 3.00,
        "cost_output_per_m": 15.00,
        "release_date": "2026-01-01",
        "open_weight": False,
        "architecture": "transformer",
    },
    "opus": {
        "full_id": "claude-opus-4-6",
        "family": "claude",
        "provider": "anthropic",
        "parameters_b": 176,
        "parameters_estimated": True,
        "context_window": 200_000,
        "cost_tier": "paid",
        "cost_input_per_m": 15.00,
        "cost_output_per_m": 75.00,
        "release_date": "2026-01-01",
        "open_weight": False,
        "architecture": "transformer",
    },

    # ---- OpenRouter frontier models ----
    "hunter": {
        "full_id": "openrouter/hunter-alpha",
        "family": "openrouter",
        "provider": "openrouter",
        "parameters_b": 1000,
        "parameters_estimated": True,
        "context_window": 1_000_000,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": "2026-02-01",
        "open_weight": False,
        "architecture": "moe",
    },
    "hunter-alpha": {
        "full_id": "openrouter/hunter-alpha",
        "family": "openrouter",
        "provider": "openrouter",
        "parameters_b": 1000,
        "parameters_estimated": True,
        "context_window": 1_000_000,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": "2026-02-01",
        "open_weight": False,
        "architecture": "moe",
    },
    "healer": {
        "full_id": "openrouter/healer-alpha",
        "family": "openrouter",
        "provider": "openrouter",
        "parameters_b": None,
        "parameters_estimated": True,
        "context_window": 256_000,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": "2026-02-01",
        "open_weight": False,
        "architecture": "unknown",
    },
    "healer-alpha": {
        "full_id": "openrouter/healer-alpha",
        "family": "openrouter",
        "provider": "openrouter",
        "parameters_b": None,
        "parameters_estimated": True,
        "context_window": 256_000,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": "2026-02-01",
        "open_weight": False,
        "architecture": "unknown",
    },

    # ---- Big open-weight models ----
    "nemotron-120b": {
        "full_id": "nvidia/nemotron-3-super-120b-a12b:free",
        "family": "nemotron",
        "provider": "nvidia",
        "parameters_b": 120,
        "parameters_estimated": False,
        "active_parameters_b": 12,
        "context_window": 256_000,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": "2025-12-01",
        "open_weight": True,
        "architecture": "moe",
    },
    "gpt-oss-120b": {
        "full_id": "openai/gpt-oss-120b:free",
        "family": "gpt-oss",
        "provider": "openai",
        "parameters_b": 120,
        "parameters_estimated": False,
        "context_window": 128_000,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": "2026-01-01",
        "open_weight": True,
        "architecture": "transformer",
    },
    "qwen3-coder": {
        "full_id": "qwen/qwen3-coder:free",
        "family": "qwen",
        "provider": "alibaba",
        "parameters_b": 480,
        "parameters_estimated": True,
        "context_window": 256_000,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": "2026-02-01",
        "open_weight": True,
        "architecture": "moe",
    },
    "llama-70b": {
        "full_id": "meta-llama/llama-3.3-70b-instruct:free",
        "family": "llama",
        "provider": "meta",
        "parameters_b": 70,
        "parameters_estimated": False,
        "context_window": 128_000,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": "2024-12-01",
        "open_weight": True,
        "architecture": "transformer",
    },
    "trinity-large": {
        "full_id": "arcee-ai/trinity-large-preview:free",
        "family": "trinity",
        "provider": "arcee",
        "parameters_b": None,
        "parameters_estimated": True,
        "context_window": 131_000,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": "2026-01-01",
        "open_weight": True,
        "architecture": "moe",
    },

    # ---- Mid-size models ----
    "step-flash": {
        "full_id": "stepfun/step-3.5-flash:free",
        "family": "step",
        "provider": "stepfun",
        "parameters_b": 196,
        "parameters_estimated": True,
        "context_window": 256_000,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": "2025-12-01",
        "open_weight": False,
        "architecture": "moe",
    },
    "qwen3-80b": {
        "full_id": "qwen/qwen3-next-80b-a3b-instruct:free",
        "family": "qwen",
        "provider": "alibaba",
        "parameters_b": 80,
        "parameters_estimated": False,
        "active_parameters_b": 3,
        "context_window": 256_000,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": "2026-01-01",
        "open_weight": True,
        "architecture": "moe",
    },
    "nemotron-30b": {
        "full_id": "nvidia/nemotron-3-nano-30b-a3b:free",
        "family": "nemotron",
        "provider": "nvidia",
        "parameters_b": 30,
        "parameters_estimated": False,
        "active_parameters_b": 3,
        "context_window": 256_000,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": "2025-12-01",
        "open_weight": True,
        "architecture": "moe",
    },
    "gemma-27b": {
        "full_id": "google/gemma-3-27b-it:free",
        "family": "gemma",
        "provider": "google",
        "parameters_b": 27,
        "parameters_estimated": False,
        "context_window": 128_000,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": "2025-03-01",
        "open_weight": True,
        "architecture": "transformer",
    },
    "trinity-mini": {
        "full_id": "arcee-ai/trinity-mini:free",
        "family": "trinity",
        "provider": "arcee",
        "parameters_b": 26,
        "parameters_estimated": True,
        "context_window": 128_000,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": "2026-01-01",
        "open_weight": True,
        "architecture": "moe",
    },
    "mistral-small": {
        "full_id": "mistralai/mistral-small-3.1-24b-instruct:free",
        "family": "mistral",
        "provider": "mistral",
        "parameters_b": 24,
        "parameters_estimated": False,
        "context_window": 128_000,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": "2025-06-01",
        "open_weight": True,
        "architecture": "transformer",
    },
    "gpt-oss-20b": {
        "full_id": "openai/gpt-oss-20b:free",
        "family": "gpt-oss",
        "provider": "openai",
        "parameters_b": 20,
        "parameters_estimated": False,
        "context_window": 128_000,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": "2026-01-01",
        "open_weight": True,
        "architecture": "transformer",
    },
    "glm-4.5": {
        "full_id": "z-ai/glm-4.5-air:free",
        "family": "glm",
        "provider": "zhipu",
        "parameters_b": None,
        "parameters_estimated": True,
        "context_window": 128_000,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": "2025-06-01",
        "open_weight": False,
        "architecture": "transformer",
    },

    # ---- Paid OpenRouter models ----
    "gpt4o": {
        "full_id": "openai/gpt-4o",
        "family": "gpt",
        "provider": "openai",
        "parameters_b": 200,
        "parameters_estimated": True,
        "context_window": 128_000,
        "cost_tier": "paid",
        "cost_input_per_m": 2.50,
        "cost_output_per_m": 10.00,
        "release_date": "2024-05-01",
        "open_weight": False,
        "architecture": "transformer",
    },
    "gpt4o-mini": {
        "full_id": "openai/gpt-4o-mini",
        "family": "gpt",
        "provider": "openai",
        "parameters_b": 8,
        "parameters_estimated": True,
        "context_window": 128_000,
        "cost_tier": "paid",
        "cost_input_per_m": 0.15,
        "cost_output_per_m": 0.60,
        "release_date": "2024-07-01",
        "open_weight": False,
        "architecture": "transformer",
    },
    "gemini-pro": {
        "full_id": "google/gemini-2.5-pro-preview",
        "family": "gemini",
        "provider": "google",
        "parameters_b": None,
        "parameters_estimated": True,
        "context_window": 1_000_000,
        "cost_tier": "paid",
        "cost_input_per_m": 1.25,
        "cost_output_per_m": 10.00,
        "release_date": "2026-01-01",
        "open_weight": False,
        "architecture": "moe",
    },
    "gemini-flash": {
        "full_id": "google/gemini-2.5-flash-preview",
        "family": "gemini",
        "provider": "google",
        "parameters_b": None,
        "parameters_estimated": True,
        "context_window": 1_000_000,
        "cost_tier": "paid",
        "cost_input_per_m": 0.15,
        "cost_output_per_m": 0.60,
        "release_date": "2026-01-01",
        "open_weight": False,
        "architecture": "moe",
    },
    "deepseek": {
        "full_id": "deepseek/deepseek-chat-v3-0324",
        "family": "deepseek",
        "provider": "deepseek",
        "parameters_b": 671,
        "parameters_estimated": False,
        "active_parameters_b": 37,
        "context_window": 128_000,
        "cost_tier": "paid",
        "cost_input_per_m": 0.27,
        "cost_output_per_m": 1.10,
        "release_date": "2025-03-01",
        "open_weight": True,
        "architecture": "moe",
    },

    # ---- Baseline ----
    "random": {
        "full_id": "random-baseline",
        "family": "baseline",
        "provider": "none",
        "parameters_b": 0,
        "parameters_estimated": False,
        "context_window": 0,
        "cost_tier": "free",
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "release_date": None,
        "open_weight": True,
        "architecture": "random",
    },
}

# Default fallback for unknown models
_UNKNOWN_MODEL = {
    "full_id": "unknown",
    "family": "unknown",
    "provider": "unknown",
    "parameters_b": None,
    "parameters_estimated": True,
    "context_window": None,
    "cost_tier": "unknown",
    "cost_input_per_m": 0.0,
    "cost_output_per_m": 0.0,
    "release_date": None,
    "open_weight": False,
    "architecture": "unknown",
}


def get_model_info(model_name: str) -> dict:
    """Look up model metadata. Returns defaults for unregistered models."""
    if model_name in MODEL_REGISTRY:
        return dict(MODEL_REGISTRY[model_name])
    # Try without common suffixes
    for suffix in (":free", "-alpha"):
        base = model_name.replace(suffix, "")
        if base in MODEL_REGISTRY:
            return dict(MODEL_REGISTRY[base])
    return {**_UNKNOWN_MODEL, "full_id": model_name}


def get_model_size(model_name: str) -> float | None:
    """Return parameter count in billions, or None if unknown."""
    return get_model_info(model_name).get("parameters_b")


def get_model_family(model_name: str) -> str:
    """Return model family (claude, llama, gemma, qwen, etc.)."""
    return get_model_info(model_name).get("family", "unknown")


def get_model_provider(model_name: str) -> str:
    """Return model provider (anthropic, meta, google, nvidia, etc.)."""
    return get_model_info(model_name).get("provider", "unknown")


def get_size_tier(model_name: str) -> str:
    """Classify model by size: small (<10B), medium (10-50B), large (50-100B), frontier (>100B)."""
    size = get_model_size(model_name)
    if size is None:
        return "unknown"
    if size == 0:
        return "baseline"
    if size < 10:
        return "small"
    if size < 50:
        return "medium"
    if size < 100:
        return "large"
    return "frontier"


def compute_dollar_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost for given token counts."""
    info = get_model_info(model_name)
    cost = (input_tokens / 1_000_000) * info.get("cost_input_per_m", 0)
    cost += (output_tokens / 1_000_000) * info.get("cost_output_per_m", 0)
    return round(cost, 6)


def list_models_by_size() -> list[tuple[str, float | None]]:
    """Return all registered models sorted by parameter count (ascending)."""
    models = []
    for name, info in MODEL_REGISTRY.items():
        # Skip aliases
        if name in ("hunter-alpha", "healer-alpha"):
            continue
        models.append((name, info.get("parameters_b")))
    return sorted(models, key=lambda x: (x[1] is None, x[1] or 0))


def list_models_by_family() -> dict[str, list[str]]:
    """Return models grouped by family."""
    families = {}
    for name, info in MODEL_REGISTRY.items():
        if name in ("hunter-alpha", "healer-alpha"):
            continue
        fam = info.get("family", "unknown")
        families.setdefault(fam, []).append(name)
    return families


def get_registry_summary() -> str:
    """Human-readable summary of all registered models."""
    lines = ["Model Registry:"]
    for name, size in list_models_by_size():
        info = MODEL_REGISTRY[name]
        size_str = f"{size}B" if size is not None else "?B"
        tier = get_size_tier(name)
        lines.append(
            f"  {name:20s} {size_str:>8s}  {tier:10s}  "
            f"{info['provider']:12s}  {info['family']:10s}  {info['cost_tier']}"
        )
    return "\n".join(lines)
