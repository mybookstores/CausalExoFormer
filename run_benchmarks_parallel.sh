#!/bin/bash
# Parallel launcher for non-lake forecasting benchmarks.
#
# Datasets:
#   ETTh1 ETTh2 ETTm1 ETTm2 electricity exchange_rate illness traffic weather
#
# Strategy:
#   - one dataset per worker process
#   - each worker runs 6 models serially on its assigned GPU
#   - if datasets > GPUs, datasets are queued in waves
#
# Usage:
#   bash run_benchmarks_parallel.sh --smoke
#   bash run_benchmarks_parallel.sh --full
#   bash run_benchmarks_parallel.sh --causal-only
#   bash run_benchmarks_parallel.sh --kill

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

if [ "$MODE" = "--smoke" ]; then
    EPOCHS=2
    PATIENCE=5
    MODELS="causal timexer itransformer patchtst dlinear autoformer"
    echo ">>> SMOKE MODE: 2 epochs, pred_len=96, all 6 models"
elif [ "$MODE" = "--full" ]; then
    EPOCHS=10
    PATIENCE=5
    MODELS="causal timexer itransformer patchtst dlinear autoformer"
    echo ">>> FULL MODE: 10 epochs, pred_len=96, all 6 models"
elif [ "$MODE" = "--causal-only" ]; then
    EPOCHS=10
    PATIENCE=5
    MODELS="causal"
    echo ">>> CAUSAL-ONLY MODE: 10 epochs, pred_len=96, optimized CausalExoFormer only"
elif [ "$MODE" = "--kill" ]; then
    echo "Killing all background benchmark jobs..."
    jobs -l | grep -v 'No such job' | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    pkill -f "python run.py" 2>/dev/null || true
    pkill -f "run_benchmarks_parallel.sh" 2>/dev/null || true
    echo "Done."
    exit 0
else
    echo "Usage: bash run_benchmarks_parallel.sh [--smoke|--full|--causal-only|--kill]"
    exit 1
fi

declare -A DATA_ROOT=(
    [ETTh1]="${SCRIPT_DIR}/dataset/ETT-small"
    [ETTh2]="${SCRIPT_DIR}/dataset/ETT-small"
    [ETTm1]="${SCRIPT_DIR}/dataset/ETT-small"
    [ETTm2]="${SCRIPT_DIR}/dataset/ETT-small"
    [electricity]="${SCRIPT_DIR}/dataset/electricity"
    [exchange_rate]="${SCRIPT_DIR}/dataset/exchange_rate"
    [illness]="${SCRIPT_DIR}/dataset/illness"
    [traffic]="${SCRIPT_DIR}/dataset/traffic"
    [weather]="${SCRIPT_DIR}/dataset/weather"
)

declare -A DATA_PATH=(
    [ETTh1]="ETTh1.csv"
    [ETTh2]="ETTh2.csv"
    [ETTm1]="ETTm1.csv"
    [ETTm2]="ETTm2.csv"
    [electricity]="electricity.csv"
    [exchange_rate]="exchange_rate.csv"
    [illness]="national_illness.csv"
    [traffic]="traffic.csv"
    [weather]="weather.csv"
)

declare -A DATA_NAME=(
    [ETTh1]="ETTh1"
    [ETTh2]="ETTh2"
    [ETTm1]="ETTm1"
    [ETTm2]="ETTm2"
    [electricity]="custom"
    [exchange_rate]="custom"
    [illness]="custom"
    [traffic]="custom"
    [weather]="custom"
)

declare -A DATA_FREQ=(
    [ETTh1]="h"
    [ETTh2]="h"
    [ETTm1]="t"
    [ETTm2]="t"
    [electricity]="h"
    [exchange_rate]="d"
    [illness]="w"
    [traffic]="h"
    [weather]="t"
)

declare -A DATA_ENC_IN=(
    [ETTh1]=7
    [ETTh2]=7
    [ETTm1]=7
    [ETTm2]=7
    [electricity]=321
    [exchange_rate]=8
    [illness]=7
    [traffic]=862
    [weather]=21
)

DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 electricity exchange_rate illness traffic weather)
GPUS=(0 1 2 3 4 5 6 7)

# Smaller batch sizes for high-dimensional datasets to reduce OOM risk.
declare -A DATA_BATCH_SIZE=(
    [ETTh1]=32
    [ETTh2]=32
    [ETTm1]=32
    [ETTm2]=32
    [electricity]=8
    [exchange_rate]=32
    [illness]=16
    [traffic]=4
    [weather]=16
)

declare -A MODEL_CLASS=(
    [causal]="CausalExoFormer"
    [timexer]="TimeXer"
    [itransformer]="iTransformer"
    [patchtst]="PatchTST"
    [dlinear]="DLinear"
    [autoformer]="Autoformer"
)

LOG_DIR="${SCRIPT_DIR}/logs/benchmarks"
mkdir -p "${LOG_DIR}"

# Larger model capacity for causal on low-dimensional datasets
declare -A MODEL_D_MODEL_CAUSAL=(
    [ETTh1]=128
    [ETTh2]=128
    [ETTm1]=64
    [ETTm2]=64
    [electricity]=64
    [exchange_rate]=32
    [illness]=64
    [traffic]=64
    [weather]=64
)
declare -A MODEL_E_LAYERS_CAUSAL=(
    [ETTh1]=3
    [ETTh2]=3
    [ETTm1]=2
    [ETTm2]=2
    [electricity]=2
    [exchange_rate]=1
    [illness]=1
    [traffic]=2
    [weather]=2
)
declare -A MODEL_SEQ_LEN=(
)
declare -A MODEL_D_FF_CAUSAL=(
    [ETTh1]=256
    [ETTh2]=256
    [ETTm1]=128
    [ETTm2]=128
    [electricity]=128
    [exchange_rate]=64
    [illness]=128
    [traffic]=128
    [weather]=128
)

run_dataset() {
    local dataset=$1
    local gpu=$2
    local pidfile="${LOG_DIR}/${dataset}.pid"
    local root_path="${DATA_ROOT[$dataset]}"
    local data_path="${DATA_PATH[$dataset]}"
    local data_name="${DATA_NAME[$dataset]}"
    local freq="${DATA_FREQ[$dataset]}"
    local enc_in="${DATA_ENC_IN[$dataset]}"
    local batch_size="${DATA_BATCH_SIZE[$dataset]}"
    local des="benchmark_${dataset}"
    local d_model_causal="${MODEL_D_MODEL_CAUSAL[$dataset]:-64}"
    local e_layers_causal="${MODEL_E_LAYERS_CAUSAL[$dataset]:-2}"
    local d_ff_causal="${MODEL_D_FF_CAUSAL[$dataset]:-128}"
    local seq_len_causal="${MODEL_SEQ_LEN[$dataset]:-96}"
    local gpu_args="--use_gpu --gpu ${gpu}"

    echo ""
    echo "============================================"
    echo "  Starting ${dataset} on GPU ${gpu}"
    echo "  Models: ${MODELS}"
    echo "============================================"

    local cmd=""
    for model_short in ${MODELS}; do
        local model_class="${MODEL_CLASS[$model_short]}"
        local model_id="benchmark_${model_short}_${dataset}"
        local causal_args=""
        if [ "${model_short}" = "causal" ]; then
            # ETTh1: large model + seasonal MA=25 (verified)
            if [ "${dataset}" = "ETTh1" ]; then
                causal_args="--revin_affine 1 --linear_residual 1 --linear_residual_seasonal 1 --linear_residual_seasonal_ma 25 --ablation bypass_causal --lambda_sparse 0 --lambda_consist 0"
            # ETTh2: large model + seasonal MA=48 (capture 2-day cycles in hourly oil temp data)
            elif [ "${dataset}" = "ETTh2" ]; then
                causal_args="--revin_affine 1 --linear_residual 1 --linear_residual_seasonal 1 --linear_residual_seasonal_ma 48 --ablation bypass_causal --lambda_sparse 0 --lambda_consist 0"
            # exchange_rate: small model with strong linear residual for random-walk-like dynamics
            elif [ "${dataset}" = "exchange_rate" ]; then
                causal_args="--revin_affine 0 --linear_residual 1 --linear_residual_init 1.0 --ablation bypass_causal --lambda_sparse 0 --lambda_consist 0"
            # illness: seq_len=96/d_model=64 with simple residual is more stable than seasonal MA on tiny validation split
            elif [ "${dataset}" = "illness" ]; then
                causal_args="--revin_affine 1 --linear_residual 1 --linear_residual_init 1.0 --ablation bypass_causal --lambda_sparse 0 --lambda_consist 0"
            elif [ "${dataset}" = "ETTm1" ] || [ "${dataset}" = "ETTm2" ] || [ "${dataset}" = "weather" ]; then
                causal_args="--revin_affine 1 --linear_residual 1 --linear_residual_init 1.0 --causal_warmup_epochs 1 --causal_rampup_epochs 3 --causal_gate_init 3.0 --lambda_sparse 0.00005 --lambda_consist 0.05"
            elif [ "${dataset}" = "electricity" ]; then
                causal_args="--revin_affine 1 --linear_residual 1 --linear_residual_init 1.0 --causal_warmup_epochs 1 --causal_rampup_epochs 3 --causal_gate_init 4.0 --lambda_sparse 0.0001 --lambda_consist 0.1 --causal_top_k 96"
            elif [ "${dataset}" = "traffic" ]; then
                causal_args="--revin_affine 1 --linear_residual 1 --linear_residual_init 1.0 --causal_warmup_epochs 1 --causal_rampup_epochs 3 --causal_gate_init 4.0 --lambda_sparse 0.0001 --lambda_consist 0.1 --causal_top_k 128"
            fi
        fi
        cmd="${cmd}
python run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ${root_path} --data_path ${data_path} \
  --data ${data_name} --features MS \
  --target OT \
  --seq_len ${seq_len_causal:-96} --label_len $(( ${seq_len_causal:-96} / 2 )) --pred_len 96 \
  --d_model ${d_model_causal:-64} --n_heads 4 --e_layers ${e_layers_causal:-2} --d_layers 1 --d_ff ${d_ff_causal:-128} \
  --enc_in ${enc_in} --dec_in ${enc_in} --c_out 1 \
  --factor 1 --embed timeF --freq ${freq} --patch_len 16 \
  --moving_avg 25 --num_lags 14 --lag_step 1 \
  --train_epochs ${EPOCHS} --patience ${PATIENCE} --batch_size ${batch_size} --learning_rate 1e-3 \
  --num_workers 0 \
  --des ${des} --itr 1 --save_pred 1 \
  ${gpu_args} \
  --model_id ${model_id} \
  --model ${model_class} \
  ${causal_args} \
  2>&1 | tee -a ${LOG_DIR}/${dataset}_${model_short}.log"
    done

    bash -lc "cd '${SCRIPT_DIR}' && source /volume/wzhang/miniconda3/etc/profile.d/conda.sh && conda activate ssy_cu128_backup >/dev/null 2>&1 && ${cmd}" &
    echo $! > "${pidfile}"
    echo "  [${dataset}] launched with PID $(cat ${pidfile}) on GPU ${gpu}"
}

PIDS=""
wave_start=0
while [ $wave_start -lt ${#DATASETS[@]} ]; do
    wave_pids=""
    for idx in "${!GPUS[@]}"; do
        dataset_index=$((wave_start + idx))
        if [ $dataset_index -ge ${#DATASETS[@]} ]; then
            break
        fi
        dataset="${DATASETS[$dataset_index]}"
        gpu="${GPUS[$idx]}"
        run_dataset "$dataset" "$gpu"
        wave_pids="${wave_pids} $(cat "${LOG_DIR}/${dataset}.pid")"
        PIDS="${PIDS} $(cat "${LOG_DIR}/${dataset}.pid")"
    done
    wait ${wave_pids}
    wave_start=$((wave_start + ${#GPUS[@]}))
done

echo ""
echo "============================================"
echo "  All benchmark datasets finished. PIDs:${PIDS}"
echo "  Logs: ${LOG_DIR}/*.log"
echo "  Kill all: bash run_benchmarks_parallel.sh --kill"
echo "============================================"
