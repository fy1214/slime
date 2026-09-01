#!/bin/bash
#
# Qwen3-30B-A3B native BF16 GRPO (TP1 EP8 train + SGLang EP8 DeepGEMM rollout).
#
# Workload / rollout shape aligned with run-qwen3-30B-A3B-deterministic.sh for
# A/B comparison. No deterministic mode, logprob gate, or alignment hooks.
#
#   bash scripts/run-qwen3-30B-A3B-bf16.sh
#   NUM_ROLLOUT=1 ENABLE_EVAL=0 bash scripts/run-qwen3-30B-A3B-bf16.sh
#
# Env: HF_MODEL REF_LOAD LOAD SAVE PROMPT_DATA EVAL_DATA NUM_ROLLOUT ENABLE_EVAL
#      ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE
#      ROLLOUT_MAX_CONTEXT_LEN ROLLOUT_MAX_RESPONSE_LEN

# Shared container: do not pkill python
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
sleep 3

set -ex

ulimit -n 524288

export PYTHONUNBUFFERED=1
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="${no_proxy:-*}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${SLIME_ROOT}"

source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

HF_MODEL="${HF_MODEL:-/home/admin/mingfa/model/hf/Qwen3-30B-A3B/Qwen3-30B-A3B/}"
REF_LOAD="${REF_LOAD:-/home/admin/mingfa/model/torch_dist/Qwen3-30B-A3B_torch_dist/}"
LOAD="${LOAD:-/home/admin/mingfa/model/torch_dist/Qwen3-30B-A3B_torch_dist/}"
SAVE="${SAVE:-/root/Qwen3-30B-A3B_slime_bf16}"
PROMPT_DATA="${PROMPT_DATA:-/root/datasets/dapo-math-17k/dapo-math-17k.jsonl}"
EVAL_DATA="${EVAL_DATA:-/root/datasets/aime-2024/aime-2024.jsonl}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
NUM_ROLLOUT="${NUM_ROLLOUT:-3000}"
ENABLE_EVAL="${ENABLE_EVAL:-0}"

if [ ! -f "${PROMPT_DATA}" ] && [ -f /root/dapo-math-17k/dapo-math-17k.jsonl ]; then
    PROMPT_DATA=/root/dapo-math-17k/dapo-math-17k.jsonl
fi
if [ ! -f "${EVAL_DATA}" ] && [ -f /root/aime-2024/aime-2024.jsonl ]; then
    EVAL_DATA=/root/aime-2024/aime-2024.jsonl
fi

mkdir -p "${SAVE}"

PYTHONPATH_VALUE="${SLIME_ROOT}:${MEGATRON_ROOT}"
if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH_VALUE="${PYTHONPATH_VALUE}:${PYTHONPATH}"
fi
export PYTHONPATH="${PYTHONPATH_VALUE}"

# DeepEP low_latency prefill needs DP-attention sharding + staging env.
# Without these, first rollout prefill trips deep_ep.cpp:1105 token limit.
export SGLANG_DEEPEP_LL_PREFILL_STAGING=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-64}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_MODEL}"
   --ref-load "${REF_LOAD}"
   --load "${LOAD}"
   --save "${SAVE}"
   --save-interval 200
)

# Same production shape knobs as run-qwen3-30B-A3B-deterministic.sh
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-16384}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-20480}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-16384}"
SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-8192}"
echo "SHAPE: batch=${ROLLOUT_BATCH_SIZE} n=${N_SAMPLES_PER_PROMPT} gbs=${GLOBAL_BATCH_SIZE} resp=${ROLLOUT_MAX_RESPONSE_LEN} ctx=${ROLLOUT_MAX_CONTEXT_LEN}"

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 1
   --rollout-top-p 1.0

   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)

EVAL_ARGS=()
if [ "${ENABLE_EVAL}" != "0" ]; then
    EVAL_ARGS=(
       --eval-interval 20
       --eval-prompt-data aime "${EVAL_DATA}"
       --n-samples-per-eval-prompt 16
       --eval-max-response-len 16384
       --eval-top-p 1
    )
fi

# TP1 EP8 matches torch_dist (TP1 EP1) + enables rank-local expert weight sync.
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-deterministic
   --wandb-group qwen3-30B-A3B-bf16
   --wandb-mode offline
   --wandb-dir /home/admin/mingfa/python/exp/deterministic/wandb
   # --wandb-key ${WANDB_KEY}
)

# B200 auto-picks flashinfer_trtllm MoE (page_size=64 padding) which breaks
# Megatron->SGLang expert weight sync (w13 64 vs 2048). Force DeepGEMM+DeepEP
# like deterministic, but keep native BF16 (no FP8/deterministic/FA4 hooks).
SGLANG_ARGS=(
   --rollout-num-gpus 8
   --rollout-num-gpus-per-engine 8
   --sglang-server-concurrency 128
   --sglang-mem-fraction-static 0.7
   --sglang-enable-dp-attention
   --sglang-enable-dp-lm-head
   --sglang-ep-size 8
   --sglang-dp-size 8
   --sglang-moe-dp-size 1
   --sglang-moe-dense-tp-size 1
   --sglang-moe-a2a-backend deepep
   --sglang-deepep-mode low_latency
   --sglang-moe-runner-backend deep_gemm
   --sglang-page-size 64
   --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}"
   --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
   --sglang-max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS}"
   --sglang-disable-prefill-cuda-graph
   --sglang-cuda-graph-max-bs-decode 64
   --sglang-watchdog-timeout 7200
   --sglang-dist-timeout 1800
   --sglang-trust-remote-code
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --update-weight-mode full
   --update-weight-transport nccl
   --update-weight-buffer-size 2147483648
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export RAY_ADDRESS="${MASTER_ADDR}:6379"

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 --num-cpus 16 \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH_VALUE}\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"RAY_ADDRESS\": \"${MASTER_ADDR}:6379\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"NO_PROXY\": \"*\",
    \"no_proxy\": \"*\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_DEEPEP_LL_PREFILL_STAGING\": \"1\",
    \"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK\": \"${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
