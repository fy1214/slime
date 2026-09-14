#!/bin/bash
set -euo pipefail
W=/lustre/fsw/general_sa/shuazhang/python_space/verl_for_nvfp4_20251031/investigation/nvfp4_router_fix_20260912_v1/formal
ACTION=${1:-dry-run}
export DQ_VARIANT=${2:?original or dequantized}
case "$DQ_VARIANT" in original|dequantized) ;; *) exit 2 ;; esac
export EXP=nvfp4_routerfixed_dq_${DQ_VARIANT}_0912_v1
export WANDB_ID=$EXP
export AUDIT_VARIANT=fixed
export NNODES=4 NUM_ROLLOUT=1 SAVE_INTERVAL=1000000 ENABLE_EVAL=0 CI_TEST=0 RESUME=0
export MAX_TOKENS_PER_GPU=20480 ROLLOUT_MAX_RESPONSE_LEN=8192 GLOBAL_BATCH_SIZE=256
export N_SAMPLES_PER_PROMPT=8 ROLLOUT_BATCH_SIZE=32 USE_TIS=1 TIS_CLIP=2.0 CLB=1 RM_TYPE=math
export ROLLOUT_MAX_CONTEXT_LEN=16384 SGLANG_CONTEXT_LENGTH=16384 SGLANG_MAX_PREFILL_TOKENS=8192
export MEM_FRACTION=0.45 WANDB_MODE=disabled USE_KL_LOSS=0 SLIME_PRE_MCORE_014=0
export HF_MODEL=/lustre/fsw/general_sa/shuazhang/python_space/infix.AI/ckpts/Qwen3-30B-A3B-Base-NVFP4-expert
export REF_LOAD=/lustre/fsw/general_sa/shuazhang/python_space/infix.AI/ckpts/Qwen3-30B-A3B-Base_torch_dist_aligned
export NO_SAVE_OPTIM= NVFP4_MAX_M=64 SGLANG_NVFP4_PERTOKEN_SCALE=1
if [ "$ACTION" = dry-run ]; then
 DRYRUN=1 SLIME_ROOT="$W/slime_nvfp4" bash "$W/jobs/nvfp4_driver.sh"
elif [ "$ACTION" = submit ]; then
 [ ! -e "$W/ckpts/$EXP" ] && [ ! -e "$W/results/$EXP" ] || { echo 'Namespace collision' >&2; exit 2; }
 [ -z "$(squeue -u shuazhang -h -o '%j' | rg -F "routerdq-$DQ_VARIANT-0912-v1")" ] || exit 2
 DEP=(); [ -z "${3:-}" ] || DEP=(--dependency="afterok:$3")
 sbatch --parsable --nodes=4 --time=00:20:00 --job-name="general_sa-infix:routerdq-$DQ_VARIANT-0912-v1" "${DEP[@]}" "$W/jobs/run_nvfp4.job" nvfp4
else
 exit 2
fi
