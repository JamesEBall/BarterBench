#!/usr/bin/env bash
# Serve a registry model with vLLM on one GPU.
#
#   scripts/serve_vllm.sh <alias> [port] [extra vllm args...]
#   scripts/serve_vllm.sh qwen3-8b 8000
#   scripts/serve_vllm.sh llama-3.3-70b 8001 --gpu-memory-utilization 0.92
#
# Two models at once on an 80 GB A100 (for head-to-head matches) works for pairs
# whose VRAM estimates sum to < ~75 GB (see `python3 -m barterbench --registry`), e.g.
#   CUDA_VISIBLE_DEVICES=0 scripts/serve_vllm.sh qwen3-8b 8000 --gpu-memory-utilization 0.30 &
#   CUDA_VISIBLE_DEVICES=0 scripts/serve_vllm.sh llama-3.1-8b 8001 --gpu-memory-utilization 0.30 &
# then
#   python3 -m barterbench --matrix --models qwen3-8b,llama-3.1-8b \
#       --endpoint qwen3-8b=http://localhost:8000/v1 --endpoint llama-3.1-8b=http://localhost:8001/v1
#
# Otherwise serve one model at a time; BarterBench only needs the endpoint to be
# OpenAI-compatible and to support JSON-schema structured output (vLLM >= 0.8).
set -euo pipefail

ALIAS="${1:?usage: serve_vllm.sh <alias> [port] [extra vllm args]}"
PORT="${2:-8000}"
shift $(( $# >= 2 ? 2 : $# ))

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
read -r HF_ID SERVED VLLM_ARGS < <(python3 - "$ALIAS" <<'EOF'
import sys
from barterbench.registry import get_model_info, llm_models
alias = sys.argv[1]
info = get_model_info(alias)
if info["full_id"] in ("unknown",) or alias not in llm_models():
    # Allow a raw Hugging Face id too
    print(alias, alias, "")
else:
    print(info["full_id"], info["served_name"], info.get("vllm_args", ""))
EOF
)

echo "[serve_vllm] alias=$ALIAS model=$HF_ID served-name=$SERVED port=$PORT"
echo "[serve_vllm] extra: $VLLM_ARGS $*"

exec vllm serve "$HF_ID" \
  --served-model-name "$SERVED" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --dtype auto \
  --enable-prefix-caching \
  --guided-decoding-backend xgrammar \
  --max-num-seqs 64 \
  $VLLM_ARGS "$@"
