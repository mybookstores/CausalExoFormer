#!/bin/bash
# Parallel launcher: distribute 30 experiments (5 lakes × 6 models) across 8 GPUs.
#
# Strategy: Each GPU runs one lake dataset at a time (6 serial models per GPU).
# This avoids GPU memory conflicts while keeping all 8 GPUs busy.
#
# Usage:
#   # Run smoke test (1 epoch, all lakes, all 6 models):
#   bash run_lakes_parallel.sh --smoke
#
#   # Run full experiment (5 epochs, all lakes, all 6 models):
#   bash run_lakes_parallel.sh --full
#
#   # CausalExoFormer only on all lakes (smoke):
#   bash run_lakes_parallel.sh --causal-only
#
#   # Kill all background jobs:
#   bash run_lakes_parallel.sh --kill

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

if [ -f /volume/wzhang/miniconda3/etc/profile.d/conda.sh ]; then
    source /volume/wzhang/miniconda3/etc/profile.d/conda.sh
fi
if command -v conda >/dev/null 2>&1; then
    conda activate ssy_cu128_backup >/dev/null 2>&1 || true
fi

MODE="${1:-}"

# ── Training params ─────────────────────────────────────────────────────────
if [ "$MODE" = "--smoke" ]; then
    EPOCHS=2
    PATIENCE=5
    MODELS="causal timexer itransformer patchtst dlinear autoformer"
    echo ">>> SMOKE MODE: 2 epochs, all 6 models"
elif [ "$MODE" = "--full" ]; then
    EPOCHS=10
    PATIENCE=5
    MODELS="causal timexer itransformer patchtst dlinear autoformer"
    echo ">>> FULL MODE: 10 epochs, all 6 models"
elif [ "$MODE" = "--causal-only" ]; then
    EPOCHS=2
    PATIENCE=5
    MODELS="causal"
    echo ">>> CAUSAL-ONLY MODE: 2 epochs, CausalExoFormer only"
elif [ "$MODE" = "--kill" ]; then
    echo "Killing all background training jobs..."
    jobs -l | grep -v 'No such job' | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    pkill -f "python run.py" 2>/dev/null || true
    echo "Done."
    exit 0
else
    echo "Usage: bash run_lakes_parallel.sh [--smoke|--full|--causal-only|--kill]"
    exit 1
fi

# ── Dataset config ───────────────────────────────────────────────────────────
declare -A LAKE_DATA=(
    [erie1]="erie1.csv 10 chl_top__Chlorophyll"
    [huron1]="huron1.csv 15 chl_top__Chlorophyll"
    [huron2]="huron2.csv 15 chl_btm__Chlorophyll"
    [huron3]="huron3.csv 15 chl_top__Chlorophyll"
    [huron4]="huron4.csv 15 chl_top__Chlorophyll"
)

declare -A LAKE_GPU=(
    [erie1]=0
    [huron1]=1
    [huron2]=2
    [huron3]=3
    [huron4]=4
)

# ── Model class mapping ─────────────────────────────────────────────────────
declare -A MODEL_CLASS=(
    [causal]="CausalExoFormer"
    [timexer]="TimeXer"
    [itransformer]="iTransformer"
    [patchtst]="PatchTST"
    [dlinear]="DLinear"
    [autoformer]="Autoformer"
)

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ── Build per-dataset run script ─────────────────────────────────────────────
run_lake() {
    local lake=$1
    local csv=$2
    local enc_in=$3
    local target=$4
    local gpu=$5
    local pidfile="${LOG_DIR}/${lake}.pid"

    local ROOT_PATH="${SCRIPT_DIR}/dataset/lake"
    local DES="realtrain_${lake}"
    local GPU_ARGS="--use_gpu --gpu ${gpu}"

    echo ""
    echo "============================================"
    echo "  Starting ${lake} on GPU ${gpu} (PID will be logged)"
    echo "  Models: ${MODELS}"
    echo "============================================"

    local cmd=""
    for model_short in ${MODELS}; do
        local model_id="local_train_${model_short}_${lake}"
        local causal_args=""
        if [ "${model_short}" = "causal" ]; then
            causal_args="--revin_affine 1 --linear_residual 1 --linear_residual_init 0.1 --causal_warmup_epochs 1 --causal_rampup_epochs 3 --causal_gate_init 1.0 --lambda_sparse 0.0005 --lambda_consist 0.1"
        fi
        cmd="${cmd}
python run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ${ROOT_PATH} --data_path ${csv} \
  --data custom --features MS \
  --target ${target} \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --d_model 64 --n_heads 4 --e_layers 2 --d_layers 1 --d_ff 128 \
  --enc_in ${enc_in} --dec_in ${enc_in} --c_out 1 \
  --factor 1 --embed timeF --freq h --patch_len 16 \
  --moving_avg 25 --num_lags 14 --lag_step 1 \
  --train_epochs ${EPOCHS} --patience ${PATIENCE} --batch_size 32 --learning_rate 1e-3 \
  --num_workers 0 \
  --des ${DES} --itr 1 --save_pred 1 \
  ${GPU_ARGS} \
  --model_id ${model_id} \
  --model ${model_class} \
  ${causal_args} \
  2>&1 | tee -a ${LOG_DIR}/${lake}_${model_short}.log"
    done

    bash -lc "cd '${SCRIPT_DIR}' && source /volume/wzhang/miniconda3/etc/profile.d/conda.sh && conda activate ssy_cu128_backup >/dev/null 2>&1 && ${cmd}" &
    echo $! > "${pidfile}"
    echo "  [${lake}] launched with PID $(cat ${pidfile}) on GPU ${gpu}"
    echo "  Logs: ${LOG_DIR}/${lake}_*.log"
}

# ── Launch all 5 lakes in parallel ──────────────────────────────────────────
PIDS=""
for lake in erie1 huron1 huron2 huron3 huron4; do
    IFS=' ' read -r csv enc_in target <<< "${LAKE_DATA[$lake]}"
    gpu=${LAKE_GPU[$lake]}
    run_lake "$lake" "$csv" "$enc_in" "$target" "$gpu"
    PIDS="${PIDS} $(cat "${LOG_DIR}/${lake}.pid")"
done

echo ""
echo "============================================"
echo "  All 5 lakes launched. PIDs:${PIDS}"
echo "  Monitor with: tail -f logs/<lake>_<model>.log"
echo "  Kill all:     bash run_lakes_parallel.sh --kill"
echo "============================================"
