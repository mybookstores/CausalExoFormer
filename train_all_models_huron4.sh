#!/bin/bash
# Train 6 models on Lake Huron4 using the current dataset layout.
set -e

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_PATH="${SCRIPT_DIR}/dataset/lake"
DATA_PATH="huron4.csv"
ENC_IN=15
DES="realtrain_huron4"
TARGET="chl_top__Chlorophyll"
GPU_ARGS="${GPU_ARGS:---use_gpu --gpu 0}"

COMMON="--task_name long_term_forecast --is_training 1 \
  --root_path ${ROOT_PATH} --data_path ${DATA_PATH} \
  --data custom --features MS \
  --target ${TARGET} \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --d_model 64 --n_heads 4 --e_layers 2 --d_layers 1 --d_ff 128 \
  --enc_in ${ENC_IN} --dec_in ${ENC_IN} --c_out 1 \
  --factor 1 --embed timeF --freq h --patch_len 16 \
  --moving_avg 25 --num_lags 14 --lag_step 1 \
  --train_epochs 2 --patience 5 --batch_size 32 --learning_rate 1e-3 \
  --num_workers 0 \
  --des ${DES} --itr 1 --save_pred 1 ${GPU_ARGS}"

run_one() {
    local model_short=$1; local model_class=$2
    local model_id="local_train_${model_short}_huron4"
    echo ""
    echo "===== TRAINING: ${model_class} @ huron4 ====="
    time python run.py ${COMMON} \
        --model_id "${model_id}" \
        --model "${model_class}" 2>&1 | tail -3
}

run_one causal       CausalExoFormer
run_one timexer      TimeXer
run_one itransformer iTransformer
run_one patchtst     PatchTST
run_one dlinear      DLinear
run_one autoformer   Autoformer

echo ""
echo "===== ALL 6 MODELS DONE @ huron4 ====="
