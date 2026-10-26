"""Central model metadata registry for BarterBench.

All LLM entries are open-weight models that fit on a single 80 GB A100 served
by vLLM. ``full_id`` is the Hugging Face repo vLLM loads; ``served_name`` is
the id used in API requests (defaults to ``full_id``). ``vllm_args`` is a
suggested launch line for ``scripts/serve_vllm.sh``.

There is no per-token price for local inference. Cost is GPU time: see
``compute_gpu_cost``.
"""

MODEL_REGISTRY = {
    # ---- Small (≤10B) ----
    "qwen3-8b": {
        "full_id": "Qwen/Qwen3-8B",
        "family": "qwen", "provider": "alibaba",
        "parameters_b": 8.2, "parameters_estimated": False,
        "context_window": 32_768, "release_date": "2025-04-29",
        "open_weight": True, "architecture": "transformer",
        "vram_gb_bf16": 17, "vllm_args": "--max-model-len 16384",
    },
    "llama-3.1-8b": {
        "full_id": "meta-llama/Llama-3.1-8B-Instruct",
        "family": "llama", "provider": "meta",
        "parameters_b": 8.0, "parameters_estimated": False,
        "context_window": 131_072, "release_date": "2024-07-23",
        "open_weight": True, "architecture": "transformer",
        "vram_gb_bf16": 16, "vllm_args": "--max-model-len 16384",
    },
    # ---- Medium (10-50B) ----
    "qwen3-14b": {
        "full_id": "Qwen/Qwen3-14B",
        "family": "qwen", "provider": "alibaba",
        "parameters_b": 14.8, "parameters_estimated": False,
        "context_window": 32_768, "release_date": "2025-04-29",
        "open_weight": True, "architecture": "transformer",
        "vram_gb_bf16": 30, "vllm_args": "--max-model-len 16384",
    },
    "phi-4": {
        "full_id": "microsoft/phi-4",
        "family": "phi", "provider": "microsoft",
        "parameters_b": 14.7, "parameters_estimated": False,
        "context_window": 16_384, "release_date": "2024-12-12",
        "open_weight": True, "architecture": "transformer",
        "vram_gb_bf16": 30, "vllm_args": "--max-model-len 16384",
    },
    "gpt-oss-20b": {
        "full_id": "openai/gpt-oss-20b",
        "family": "gpt-oss", "provider": "openai",
        "parameters_b": 21, "parameters_estimated": False, "active_parameters_b": 3.6,
        "context_window": 131_072, "release_date": "2025-08-05",
        "open_weight": True, "architecture": "moe",
        "vram_gb_bf16": 16, "vllm_args": "--max-model-len 16384",
    },
    "mistral-small-24b": {
        "full_id": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "family": "mistral", "provider": "mistral",
        "parameters_b": 24, "parameters_estimated": False,
        "context_window": 131_072, "release_date": "2025-03-17",
        "open_weight": True, "architecture": "transformer",
        "vram_gb_bf16": 48, "vllm_args": "--max-model-len 16384 --tokenizer-mode mistral --config-format mistral --load-format mistral",
    },
    "gemma-3-27b": {
        "full_id": "google/gemma-3-27b-it",
        "family": "gemma", "provider": "google",
        "parameters_b": 27, "parameters_estimated": False,
        "context_window": 131_072, "release_date": "2025-03-12",
        "open_weight": True, "architecture": "transformer",
        "vram_gb_bf16": 54, "vllm_args": "--max-model-len 16384",
    },
    "qwen3-30b-a3b": {
        "full_id": "Qwen/Qwen3-30B-A3B",
        "family": "qwen", "provider": "alibaba",
        "parameters_b": 30.5, "parameters_estimated": False, "active_parameters_b": 3.3,
        "context_window": 32_768, "release_date": "2025-04-29",
        "open_weight": True, "architecture": "moe",
        "vram_gb_bf16": 61, "vllm_args": "--max-model-len 16384",
    },
    "qwen3-32b": {
        "full_id": "Qwen/Qwen3-32B",
        "family": "qwen", "provider": "alibaba",
        "parameters_b": 32.8, "parameters_estimated": False,
        "context_window": 32_768, "release_date": "2025-04-29",
        "open_weight": True, "architecture": "transformer",
        "vram_gb_bf16": 66, "vllm_args": "--max-model-len 12288 --gpu-memory-utilization 0.95",
    },
    # ---- Large (50-100B): quantised to fit one A100 ----
    "llama-3.3-70b": {
        "full_id": "casperhansen/llama-3.3-70b-instruct-awq",
        "served_name": "llama-3.3-70b-instruct",
        "family": "llama", "provider": "meta",
        "parameters_b": 70.6, "parameters_estimated": False,
        "context_window": 131_072, "release_date": "2024-12-06",
        "open_weight": True, "architecture": "transformer",
        "quantization": "awq-int4",
        "vram_gb_bf16": 141, "vram_gb_quantized": 40,
        "vllm_args": "--quantization awq_marlin --max-model-len 16384",
    },
    # ---- Frontier-class open weights (>100B) ----
    "gpt-oss-120b": {
        "full_id": "openai/gpt-oss-120b",
        "family": "gpt-oss", "provider": "openai",
        "parameters_b": 117, "parameters_estimated": False, "active_parameters_b": 5.1,
        "context_window": 131_072, "release_date": "2025-08-05",
        "open_weight": True, "architecture": "moe",
        "quantization": "mxfp4",
        "vram_gb_bf16": 234, "vram_gb_quantized": 63,
        "vllm_args": "--max-model-len 16384 --gpu-memory-utilization 0.95",
    },

    # ---- Scripted baselines ----
    "random": {
        "full_id": "random-baseline",
        "family": "baseline", "provider": "none",
        "parameters_b": 0, "parameters_estimated": False,
        "context_window": 0, "release_date": None,
        "open_weight": True, "architecture": "random",
    },
    "heuristic": {
        "full_id": "heuristic-baseline",
        "family": "baseline", "provider": "none",
        "parameters_b": 0, "parameters_estimated": False,
        "context_window": 0, "release_date": None,
        "open_weight": True, "architecture": "greedy",
    },
}

_LOCAL_DEFAULTS = {
    "cost_tier": "local",
    "cost_input_per_m": 0.0,
    "cost_output_per_m": 0.0,
    "served_name": None,
    "base_url": None,
    "quantization": None,
    "vram_gb_bf16": None,
    "vram_gb_quantized": None,
    "vllm_args": "",
}
for _name, _info in MODEL_REGISTRY.items():
    for _k, _v in _LOCAL_DEFAULTS.items():
        _info.setdefault(_k, _v)
    if _info["served_name"] is None:
        _info["served_name"] = _info["full_id"]

_UNKNOWN_MODEL = {
    "full_id": "unknown",
    "family": "unknown",
    "provider": "unknown",
    "parameters_b": None,
    "parameters_estimated": True,
    "context_window": None,
    "release_date": None,
    "open_weight": False,
    "architecture": "unknown",
    **_LOCAL_DEFAULTS,
    "cost_tier": "unknown",
}

BASELINES = ("random", "heuristic")


def get_model_info(model_name: str) -> dict:
    """Look up model metadata. Unregistered names are treated as a served model id."""
    if model_name in MODEL_REGISTRY:
        return dict(MODEL_REGISTRY[model_name])
    return {**_UNKNOWN_MODEL, "full_id": model_name, "served_name": model_name}


def get_model_size(model_name: str):
    return get_model_info(model_name).get("parameters_b")


def get_model_family(model_name: str) -> str:
    return get_model_info(model_name).get("family", "unknown")


def get_model_provider(model_name: str) -> str:
    return get_model_info(model_name).get("provider", "unknown")


def get_size_tier(model_name: str) -> str:
    """small (<10B), medium (10-50B), large (50-100B), frontier (>100B)."""
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
    """Token-priced cost. Zero for local models; kept for result-format compatibility."""
    info = get_model_info(model_name)
    cost = (input_tokens / 1_000_000) * info.get("cost_input_per_m", 0)
    cost += (output_tokens / 1_000_000) * info.get("cost_output_per_m", 0)
    return round(cost, 6)


def compute_gpu_cost(gpu_seconds: float, usd_per_gpu_hour: float = 1.5) -> float:
    """Cost of local inference as GPU time. Default is a typical A100 rental price."""
    return round(gpu_seconds / 3600.0 * usd_per_gpu_hour, 6)


def list_models_by_size():
    models = [(name, info.get("parameters_b")) for name, info in MODEL_REGISTRY.items()]
    return sorted(models, key=lambda x: (x[1] is None, x[1] or 0))


def list_models_by_family():
    families = {}
    for name, info in MODEL_REGISTRY.items():
        families.setdefault(info.get("family", "unknown"), []).append(name)
    return families


def llm_models():
    return [m for m in MODEL_REGISTRY if m not in BASELINES]


def get_registry_summary() -> str:
    lines = ["Model Registry (local vLLM):"]
    for name, size in list_models_by_size():
        info = MODEL_REGISTRY[name]
        size_str = f"{size}B" if size is not None else "?B"
        vram = info.get("vram_gb_quantized") or info.get("vram_gb_bf16")
        vram_str = f"{vram}GB" if vram else "-"
        lines.append(
            f"  {name:20s} {size_str:>8s}  {get_size_tier(name):10s}  {info['provider']:10s}  "
            f"{vram_str:>6s}  {info['full_id']}"
        )
    return "\n".join(lines)
