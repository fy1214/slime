#!/bin/bash
# Qwen3-30B-A3B **NVFP4** deterministic driver, GB200 aarch64.
# Separate file on purpose: the FP8 chain is live and reads det_driver.sh at job
# start, so editing that file mid-chain would change the recipe under it.
# Mirrors scripts/run-qwen3-30B-A3B-nvfp4-deterministic.sh; keeps our ARM deltas
# (no optimizer-cpu-offload, PAOPT, --no-save-optim) from det_driver.sh.
# Qwen3-30B-A3B train/rollout logp-diff alignment on GB200 (aarch64).
# Mirrors fy1214/slime@slime-deterministic-patch scripts/run-qwen3-30B-A3B-deterministic.sh.
# MODE=det (deterministic alignment stack) | bf16 (same workload, no alignment).
#
# ===================== DELTAS vs the customer's script =======================
# 1. EP = world size = 4 (1 GB200 node) instead of 8 (their 1x8 B200 box).
#    BOTH sides move together (train EP == sglang ep_size == dp_size), which is
#    what the alignment actually requires; slime's own gate only asserts TP==1
#    and ETP==1 (alignment/deepgemm_moe_forward.py::_validate_parallelism).
#    NNODES=2 restores their exact EP8 shape (costs cross-node DeepEP).
# 2. NO --optimizer-cpu-offload / --overlap-cpu-optimizer-d2h-h2d.  On GB200/ARM
#    offload hangs at save_model (Megatron #4910, Grace-specific; their B200 is
#    fine).  --use-precision-aware-optimizer ALONE gives the memory back.
# 3. OPTIMIZER CHECKPOINTS (revised 2026-09-07).  We used to pass --no-save-optim because
#    Megatron's optimizer dist-ckpt save segfaults on ARM
#    (torch crc32, #1861).  The real fix is slime's hard-pinned dist_ckpt_save_pre_mcore_014:
#    the legacy path rebuilds the optimizer in full model space per rank (~191GB/actor -> host
#    OOM, and the serializer segfault behind it).  We now set it False (dp_reshardable: each
#    rank streams only its own DP shard), so the optimizer IS saved and resumed -- no more
#    re-warm at every segment boundary.  Set NO_SAVE_OPTIM=1 to fall back to model-only.
# 4. SAVE_INTERVAL defaults to "never" — this experiment is about a metric, not
#    a checkpoint.  Set SAVE_INTERVAL=200 for the long bf16-vs-fp8 comparison.
# 5. CUDA_LAUNCH_BLOCKING=1 dropped from --train-env-vars (debug residue in their
#    script; serializes every kernel launch, no effect on numerics).
# 6. Qwen3-30B-A3B-**Base** + our existing torch_dist, not their instruct build.
#    Alignment is a numerical property; this saves a 57GB conversion.
# 7. wandb online w/ mounted netrc (our house pattern) instead of offline.
# =============================================================================
set -euxo pipefail

# DELTA 5 RETRACTED (2026-09-06).  We had dropped the customer's CUDA_LAUNCH_BLOCKING=1
# as "debug residue, no effect on numerics".  It is load-bearing: without it the aligned
# MoE forward races and silently emits NaN on ~2/3 of steps (loss/grad_norm/abs_diff all
# NaN, no error, gradients discarded).  Measured on identical rollout data, 6-step smokes:
#   CLB=1 -> 0/6 NaN   |   CLB=0 -> 4/6 NaN   |   local torch.cuda.synchronize() -> 4-5/6
# and CLB=1's abs_diff series is bit-identical to a run serialized a different way, so
# those are the correct values.  Costs ~1.3x wall clock.  CLB=0 is for debugging only.
TRAIN_ENV_VARS='{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True","CUDA_LAUNCH_BLOCKING":"1"}'
if [ "${CLB:-1}" = "0" ]; then
  TRAIN_ENV_VARS='{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
fi

MODE="${MODE:-nvfp4}"
INFIX=/lustre/fsw/general_sa/shuazhang/python_space/infix.AI
W=/lustre/fsw/general_sa/shuazhang/python_space/verl_for_nvfp4_20251031/investigation/nvfp4_router_audit_20260911_v1/formal
EXP="${EXP:-gb200_30B_det_${MODE}_0904}"

NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
WORLD=$(( NNODES * GPUS_PER_NODE ))
# EP defaults to ONE NODE, not the whole world.  Multi-node then means N rollout
# engines of EP-size each, every engine confined to a node -- so we never touch
# cross-node DeepEP (which fails here: "socket failed to connect <node>:15002").
# Alignment only needs rollout ep_size == train EP, which still holds.
EP="${EP:-$GPUS_PER_NODE}"
ENGINE_GPUS="${ENGINE_GPUS:-$EP}"        # one rollout engine per node

# MODE=det consumes the FP8-experts checkpoint (train and rollout read the SAME
# one, which is what turns quantization error from a train/rollout *difference*
# into a shared bias).  MODE=bf16 is the unquantized control.
HF_MODEL="${HF_MODEL:-$INFIX/ckpts/Qwen3-30B-A3B-Base-NVFP4-expert}"
# Their nvfp4 script sets REF_LOAD=$HF_MODEL (the NVFP4 HF dir), not a torch_dist ckpt.
# NVFP4 uses the aligned spec, so REF_LOAD must be a torch_dist converted with it
# (input_layernorm.weight / {q,k}_norm.weight).  Feeding a stock-spec checkpoint makes DCP
# silently skip 96+96 tensors -> model rambles, reward 0, abs_diff still ~1e-7.
REF_LOAD="${REF_LOAD:-$INFIX/ckpts/Qwen3-30B-A3B-Base_torch_dist_aligned}"
_META=$(ls -d "$REF_LOAD"/*/.metadata 2>/dev/null | head -1)
if [ -n "$_META" ]; then
  _w=$(grep -ac "input_layernorm.weight" "$_META" 2>/dev/null || echo 0)
  echo "REF_LOAD_CHECK nvfp4 input_layernorm.weight=$_w <- $REF_LOAD"
  [ "$_w" = "0" ] && { echo "FATAL: $REF_LOAD was not built with --spec qwen3_moe_aligned" >&2; exit 2; }
fi
LOAD="${LOAD:-}"
SAVE="${SAVE:-$W/ckpts/$EXP}"
PROMPT_DATA="${PROMPT_DATA:-$INFIX/data/dapo-math-17k/dapo-math-17k.jsonl}"
EVAL_DATA="${EVAL_DATA:-$INFIX/data/aime-2024/aime-2024.jsonl}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"     # DELTA 4: effectively never
NUM_ROLLOUT="${NUM_ROLLOUT:-3000}"
ENABLE_EVAL="${ENABLE_EVAL:-0}"
CI_TEST="${CI_TEST:-0}"                       # keep 0 first: observe, don't assert-crash
MAX_TRAIN_ROLLOUT_DIFF="${MAX_TRAIN_ROLLOUT_DIFF:-9.999e-7}"
# Normally exported by det_prep_env.sh; defaulted here so the driver is
# self-contained under `set -u` (and so DRYRUN works off-cluster).
SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-bfloat16}"

[ -d "$HF_MODEL" ] || { echo "FATAL: HF_MODEL missing: $HF_MODEL" >&2; exit 2; }
[ -d "$REF_LOAD" ] || { echo "FATAL: REF_LOAD missing: $REF_LOAD" >&2; exit 2; }
[ "${DRYRUN:-0}" = "1" ] || mkdir -p "$SAVE"

# Ray-on-SLURM under pyxis: the driver runs in a DIFFERENT named container from
# the ray head, so it has its own /tmp and cannot see the head's raylet unix
# socket (GCS connects fine over TCP, then raylet IPC fails with
# "Failed to connect to socket at .../sockets/raylet").  Join the cluster with a
# local 0-resource raylet, then let ray.init() use that local one.
# Same fix as jobs/slime_driver_qat8.sh:58.
if [ -n "${RAY_ADDRESS:-}" ]; then
  ray start --address "$RAY_ADDRESS" --num-cpus 0 --num-gpus 0
  export ip_head="$RAY_ADDRESS"
  unset RAY_ADDRESS
fi

cd "${SLIME_ROOT:-/root/slime}"   # DRYRUN on the login node: SLIME_ROOT=$W/slime_det
DEEPGEMM_LAYERS=( $(seq 0 47) )

# ---- architecture (verbatim from the customer's MODEL_ARGS) ------------------
MODEL_ARGS=(
   --disable-bias-linear --qk-layernorm --group-query-attention
   --num-attention-heads 32 --num-query-groups 4 --kv-channels 128
   --num-layers 48 --hidden-size 2048 --ffn-hidden-size 6144
   --normalization RMSNorm --position-embedding-type rope --norm-epsilon 1e-6
   --rotary-percent 1.0 --swiglu --untie-embeddings-and-output-weights
   --vocab-size 151936 --rotary-base 1000000
   --moe-ffn-hidden-size 768 --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall --moe-router-topk 8
   --moe-layer-freq '[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]'
   --num-experts 128 --moe-grouped-gemm --moe-token-drop-policy probs
   --moe-router-dtype fp32 --moe-aux-loss-coeff 0
)
CKPT_ARGS=(
   --hf-checkpoint "$HF_MODEL" --ref-load "$REF_LOAD"
   --save "$SAVE" --save-interval "$SAVE_INTERVAL" --ckpt-format torch_dist
   ${NO_SAVE_OPTIM:+--no-save-optim}                 # DELTA 3 (see note): empty by default now
)
if [ "${RESUME:-0}" = "1" ]; then
  # A continuation MUST have a real checkpoint.  Without this guard slime treats a
  # missing --load target as iteration zero and silently restarts from scratch into
  # the same (resumed) W&B run -- the chain would look continuous and be garbage.
  RESUME_FROM="${LOAD:-$SAVE}"
  if [ ! -s "$RESUME_FROM/latest_checkpointed_iteration.txt" ]; then
    echo "FATAL: RESUME=1 but no checkpoint at $RESUME_FROM" >&2
    echo "       (the previous segment never reached --save-interval=$SAVE_INTERVAL)" >&2
    exit 2
  fi
  echo "RESUMING from $RESUME_FROM @ iter $(cat "$RESUME_FROM/latest_checkpointed_iteration.txt")"
  # --no-load-optim MUST be explicit.  It is NOT implied by --no-save-optim: the only
  # place slime derives it (ray/actor_group.py:158) sits inside save_model() and only
  # runs under _release_train_enabled(), i.e. same-process save-then-load -- never on a
  # cross-job resume.  Without it Megatron checkpointing.py:1794 does an unconditional
  # state_dict['optimizer'] and a --no-save-optim checkpoint dies with KeyError:
  # 'optimizer' (cost us chain segment 2735067).
  CKPT_ARGS+=( --load "$RESUME_FROM" --no-load-rng ${NO_SAVE_OPTIM:+--no-load-optim} )
elif [ -n "$LOAD" ]; then
  CKPT_ARGS+=( --load "$LOAD" )
fi

# ---- workload ---------------------------------------------------------------
if [ "${SLIME_E2E_SHAPE:-0}" = "1" ]; then     # smoke: seconds-per-step
  ROLLOUT_BATCH_SIZE=8; N_SAMPLES_PER_PROMPT=1; GLOBAL_BATCH_SIZE=8
  ROLLOUT_MAX_CONTEXT_LEN=4096; ROLLOUT_MAX_RESPONSE_LEN=32
  MAX_TOKENS_PER_GPU=8192; SGLANG_CHUNKED_PREFILL_SIZE=4096
  SGLANG_CONTEXT_LENGTH=8192; SGLANG_MAX_PREFILL_TOKENS=4096
else
  ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"; N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
  ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-16384}"
  ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
  MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-20480}"
  SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
  SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-16384}"
  SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-8192}"
fi
echo "SHAPE mode=$MODE world=$WORLD ep=$EP gbs=$GLOBAL_BATCH_SIZE n=$N_SAMPLES_PER_PROMPT resp=$ROLLOUT_MAX_RESPONSE_LEN"

ROLLOUT_ARGS=(
   --prompt-data "$PROMPT_DATA" --input-key prompt --label-key label
   --apply-chat-template --rollout-shuffle --rm-type "${RM_TYPE:-math}"
   --num-rollout "$NUM_ROLLOUT"
   --rollout-batch-size "$ROLLOUT_BATCH_SIZE" --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
   --rollout-max-context-len "$ROLLOUT_MAX_CONTEXT_LEN"
   --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
   --rollout-temperature 1 --rollout-top-p 1.0
   --deterministic-sampling-seed-mode sample
   --global-batch-size "$GLOBAL_BATCH_SIZE" --balance-data
)
EVAL_ARGS=()
[ "$ENABLE_EVAL" != "0" ] && EVAL_ARGS=(
   --eval-interval 20 --eval-prompt-data aime "$EVAL_DATA"
   --n-samples-per-eval-prompt 16 --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-8192}" --eval-top-p 1
)

PERF_ARGS=(                                        # DELTA 1: EP follows world
   --tensor-model-parallel-size 1 --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${SLIME_E2E_CONTEXT_PARALLEL_SIZE:-1}"
   --expert-model-parallel-size "$EP" --expert-tensor-parallel-size 1
   --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
   --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
)
# det-only: their bf16 control script carries neither knob.
PERF_ARGS+=( --data-pad-size-multiplier 512 --log-probs-chunk-size 1024 )
GRPO_ARGS=( --advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 )
# Their BF16 baselines all run with TIS on.  With alignment at ~1.5e-07 the IS ratio
# is 1 and the clip never fires, so enabling it costs nothing and removes a confound
# when comparing against those baselines.
[ "${USE_TIS:-0}" = "1" ] && GRPO_ARGS+=( --use-tis --tis-clip "${TIS_CLIP:-2.0}" )
# DELTA 8: the customer passes `--use-kl-loss --kl-loss-coef 0.00`.  With coef 0 the KL
# term contributes exactly zero to loss and gradient, so the ONLY effects are the
# train/kl_loss metric and a full reference forward every step (~2 min at production
# shape).  Dropping it is numerically identical and ~30-40% faster.  It also sidesteps
# `Route-preserving DeepEP metadata changed route probability order`, a device-side
# assert in the alignment code that fires on the step-1 REF forward (step 0 is fine).
# Their own e2e gate runs --num-rollout 1, so it never reaches a second step.
if [ "${USE_KL_LOSS:-0}" = "1" ]; then
  GRPO_ARGS+=( --use-kl-loss --kl-loss-coef "${KL_LOSS_COEF:-0.00}" --kl-loss-type low_var_kl )
fi
OPTIMIZER_ARGS=(                                   # DELTA 2: PAOPT only, no offload
   --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
   --adam-beta1 0.9 --adam-beta2 0.98
   --use-precision-aware-optimizer
)
WANDB_ARGS=(
   --use-wandb --wandb-project slime-deterministic-gb200
   --wandb-group "$EXP" --wandb-team shawnzzz
   --disable-wandb-random-suffix --wandb-dir "$W/wandb"
)
SGLANG_ARGS=(                                      # DELTA 1: ep/dp = world
   --rollout-num-gpus "$WORLD" --rollout-num-gpus-per-engine "$ENGINE_GPUS"
   --sglang-server-concurrency 128
   --sglang-mem-fraction-static "${MEM_FRACTION:-0.45}"
   --sglang-enable-dp-attention --sglang-enable-dp-lm-head
   --sglang-ep-size "$EP" --sglang-dp-size "$EP"   # per engine, intra-node
   --sglang-moe-dp-size 1 --sglang-moe-dense-tp-size 1
   --sglang-moe-a2a-backend deepep --sglang-deepep-mode low_latency
   --sglang-moe-runner-backend cutlass
   --sglang-quantization modelopt_fp4
   --sglang-page-size 64
   --sglang-chunked-prefill-size "$SGLANG_CHUNKED_PREFILL_SIZE"
   --sglang-context-length "$SGLANG_CONTEXT_LENGTH"
   --sglang-max-prefill-tokens "$SGLANG_MAX_PREFILL_TOKENS"
   --sglang-disable-prefill-cuda-graph --sglang-cuda-graph-max-bs-decode 64
   --sglang-watchdog-timeout 7200 --sglang-dist-timeout 1800 --sglang-trust-remote-code
)
# Shared core == their bf16 control script's MISC_ARGS, verbatim.
MISC_ARGS=(
   --attention-dropout 0.0 --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
   --attention-backend flash
   --update-weight-mode full --update-weight-transport nccl
   --update-weight-buffer-size 2147483648
   --train-env-vars "$TRAIN_ENV_VARS"   # DELTA 5
   --skip-eval-before-train
)

# ---- the alignment stack itself (MODE=det only) ------------------------------
if true; then   # nvfp4 driver: alignment always on
  # Megatron must use DeepEP (route-preserving normal dispatch) to align with
  # SGLang's low-latency dispatch; `flex` is the dispatcher that supports it.
  MISC_ARGS+=(
     --moe-router-topk-scaling-factor 1.0 --make-vocab-size-divisible-by 16
     --no-position-embedding --moe-token-dispatcher-type flex --moe-enable-deepep
     --no-check-for-nan-in-loss-and-grad
     --custom-megatron-before-log-prob-hook-path
       slime_plugins.router_full_audit.before_log_prob
     --custom-megatron-before-train-step-hook-path
       slime_plugins.router_full_audit.before_train
     --megatron-cutlass-nvfp4-moe-forward-layers "${DEEPGEMM_LAYERS[@]}"
     --deterministic-mode
  )
  SGLANG_ARGS+=(
     --sglang-kv-cache-dtype "$SGLANG_KV_CACHE_DTYPE"
     --sglang-attention-backend fa4 --sglang-enable-fused-qk-norm-rope
     --sglang-disable-flashinfer-autotune
     --sglang-enable-fp32-moe-router --sglang-enable-deterministic-inference )
  [ "${QWEN3_MOE_ALIGNED_SPEC:-1}" = "1" ] && MISC_ARGS+=(
     --spec slime_plugins.models.qwen3_moe_aligned get_qwen3_moe_aligned_spec )
fi

# Paired archived-batch diagnostic. W&B explicitly disabled at CLI level.
WANDB_ARGS=()
MISC_ARGS+=( --load-debug-rollout-data "/lustre/fsw/general_sa/shuazhang/python_space/verl_for_nvfp4_20251031/investigation/determinism_backward_audit_20260909/full_model_fixed_v1/batch.pt" )
CI_ARGS=()
[ "$CI_TEST" != "0" ] && CI_ARGS=(
   --ci-test --ci-disable-kl-checker
   --ci-train-rollout-logprob-abs-diff-threshold "$MAX_TRAIN_ROLLOUT_DIFF" )

if [ "${DRYRUN:-0}" = "1" ]; then
  set +x
  printf '%s\n' "python3 train.py --actor-num-nodes $NNODES --actor-num-gpus-per-node $GPUS_PER_NODE --num-gpus-per-node $GPUS_PER_NODE --colocate" \
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" "${WANDB_ARGS[@]}" "${PERF_ARGS[@]}" "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}" "${CI_ARGS[@]}"
  exit 0
fi

exec python3 train.py \
   --actor-num-nodes "$NNODES" --actor-num-gpus-per-node "$GPUS_PER_NODE" \
   --num-gpus-per-node "$GPUS_PER_NODE" --colocate \
   "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" "${WANDB_ARGS[@]}" "${PERF_ARGS[@]}" "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}" "${CI_ARGS[@]}"
