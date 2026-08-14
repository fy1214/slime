#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../models/qwen3-30B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint /mnt/scfd/mingfa/models/hf/Qwen3-30B-A3B
   --ref-load /mnt/scfd/mingfa/models/torch_dist/Qwen3-30B-A3B_torch_dist/
   --save /root/Qwen3-30B-A3B_slime/
   --save-interval 500
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/scfd/mingfa/data/rl/json/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   --global-batch-size 256
   --balance-data

   --use-rollout-routing-replay
   #--debug-train-only
   #--debug-rollout-only
   #--save-debug-rollout-data /root/saved/debug/data_{rollout_id}.pt
   #--load-debug-rollout-data /root/saved/debug/data_{rollout_id}.pt
)

EVAL_ARGS=(
   #--eval-interval 20
   --eval-prompt-data aime /mnt/scfd/mingfa/data/rl/json/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   #--use-tis
   #--tis-clip 0.2
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
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-30B-A3B-test
   # --wandb-key ${WANDB_KEY}

   #--use-wandb
   --wandb-mode offline
   --wandb-project slime-nvfp4
   --wandb-group qwen3-30B-A3B
   # --wandb-key ${WANDB_KEY}
   --wandb-dir /root/wandb/
)

TENSORBOARD_ARGS=(
   #--use-pytorch-profiler
   --profile-step-start 5
   --profile-step-end 6
   --tensorboard-dir /root/tensorboard/
   --record-memory-history
)

SGLANG_ARGS=(
   #--rollout-num-gpus 1
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)

   --sglang-moe-runner-backend triton
  #--sglang-moe-a2a-backend deepep
   #--sglang-disable-cuda-graph
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

PRECISE_ARGS=(
   --transformer-impl transformer_engine
   --bf16
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export RAY_USAGE_STATS_ENABLED=0

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8461 --port 6121 --dashboard-agent-listen-port 51019

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:/root/miniTransformer\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NVTE_NVFP4_DISABLE_RHT\": \"0\",
    \"NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING\": \"0\",
    \"NVTE_NVFP4_DISABLE_2D_QUANTIZATION\": \"1\",
    \"CUDA_VISIBLE_DEVICES\": \"0,1,2,3,4,5,6,7\",
    \"QAT_PARAMS\": \"8\"
  }
}"

ray job submit --address="http://127.0.0.1:8461" \
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
   ${TENSORBOARD_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${PRECISE_ARGS[@]}
