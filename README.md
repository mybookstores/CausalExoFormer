# CausalExoFormer

**CausalExoFormer** is a time series forecasting model that integrates causal discovery with transformer-based endogenous-aware prediction. It discovers which exogenous variables (and their temporal lags) causally influence the target variable through a learnable causal adjacency matrix embedded in cross-attention.

---

## Project Structure

The repository root mirrors the original `experiments/` directory from local development.

```text
.
├── run.py                         # Main entry point (train/evaluate)
├── run_benchmarks_parallel.sh     # Run standard benchmarks in parallel
├── run_lakes_parallel.sh          # Run lake benchmarks in parallel
├── data_provider/
│   ├── data_loader.py             # Dataset classes
│   └── data_factory.py            # Dataset factory
├── exp/
│   ├── exp_causal_exoformer.py    # CausalExoFormer training logic
│   └── ...                        # Other experiment classes
├── models/
│   ├── CausalExoFormer.py         # Main model
│   ├── TimeXer.py
│   ├── iTransformer.py
│   ├── PatchTST.py
│   ├── DLinear.py
│   └── Autoformer.py
├── layers/                        # Building blocks
├── utils/                         # Metrics, losses, scheduling, helpers
└── dataset/                       # Data directory
    ├── ETT-small/                 # ETTh1, ETTh2, ETTm1, ETTm2
    ├── electricity/
    ├── exchange_rate/
    ├── illness/
    ├── traffic/
    ├── weather/
    └── lake/                      # Great Lakes water quality data
        ├── erie1.csv
        ├── huron1.csv
        ├── huron2.csv
        ├── huron3.csv
        └── huron4.csv
```

---

## Installation

```bash
pip install torch numpy pandas scikit-learn statsmodels matplotlib
pip install datasets huggingface_hub  # optional, for auto-downloading standard datasets
```

---

## Quick Start

### Single Run

Run commands from the repository root.

```bash
# Lake water quality prediction
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model CausalExoFormer \
    --data custom \
    --root_path ./dataset/lake \
    --data_path huron1.csv \
    --features MS \
    --target chl_top__Chlorophyll \
    --seq_len 96 --label_len 48 --pred_len 96 \
    --enc_in 15 --dec_in 15 --c_out 1 \
    --d_model 64 --n_heads 4 --e_layers 2 --d_layers 1 \
    --train_epochs 10 --patience 5 --batch_size 32 \
    --num_lags 14 --lag_step 1 \
    --gpu 0
```

### Standard Benchmark

```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model CausalExoFormer \
    --data ETTh1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --features MS --target OT \
    --seq_len 96 --label_len 48 --pred_len 96 \
    --enc_in 7 --dec_in 7 --c_out 1 \
    --d_model 128 --n_heads 4 --e_layers 3 \
    --gpu 0
```

---

## Datasets

### Standard Benchmarks (8 datasets)

| Dataset | Path | Features | Frequency |
|---------|------|----------|-----------|
| ETTh1, ETTh2 | `dataset/ETT-small/*.csv` | 7 | hourly |
| ETTm1, ETTm2 | `dataset/ETT-small/*.csv` | 7 | 15-min |
| electricity | `dataset/electricity/` | 321 | hourly |
| exchange_rate | `dataset/exchange_rate/` | 8 | daily |
| illness | `dataset/illness/` | 7 | weekly |
| traffic | `dataset/traffic/` | 862 | hourly |
| weather | `dataset/weather/` | 21 | 10-min |

### Lake Water Quality Datasets (5 Great Lakes)

| Dataset | Target Variable | Variables |
|---------|----------------|-----------|
| Erie1 | chl_top__Chlorophyll | 10 |
| Huron1 | chl_top__Chlorophyll | 15 |
| Huron2 | chl_btm__Chlorophyll | 15 |
| Huron3 | chl_top__Chlorophyll | 15 |
| Huron4 | chl_top__Chlorophyll | 15 |

---

## Key Arguments

### Data
- `--data`: Dataset type (`ETTh1`, `ETTm1`, `custom`, etc.)
- `--root_path`: Root directory of data
- `--data_path`: Data file name
- `--features`: `M` (multivariate→multivariate), `S` (univariate→univariate), `MS` (multivariate→univariate)
- `--target`: Target column name (for `S` or `MS` mode)

### Model Architecture
- `--seq_len`: Input sequence length (default: 96)
- `--label_len`: Decoder start token length (default: 48)
- `--pred_len`: Prediction horizon (default: 96)
- `--d_model`: Model dimension (default: 512)
- `--n_heads`: Number of attention heads (default: 8)
- `--e_layers`: Encoder layers (default: 2)
- `--d_layers`: Decoder layers (default: 1)
- `--patch_len`: Patch length (default: 16)

### Training and Optimization
- `--loss`: `MSE`, `MAE`, `Huber`, `SmoothL1`, `MSE_MAE`, or `Tari`
- `--huber_delta`, `--smooth_l1_beta`: Loss-specific hyperparameters
- `--mse_mae_weight`, `--tari_alpha`: Blend weights for mixed losses
- `--grad_clip`: Gradient norm clipping (`0` disables clipping)
- `--use_train_loss_early_stopping`: Select the checkpoint with the best train loss instead of validation loss
- `--use_onecycle`, `--onecycle_pct_start`: Enable batch-level OneCycleLR scheduling
- `--use_swa`, `--swa_start`, `--swa_lr`: Enable stochastic weight averaging in later epochs

### CausalExoFormer Specific
- `--num_lags`: Number of causal lags for exogenous variables (default: 14)
- `--lag_step`: Step size between lags (default: 1)
- `--lambda_sparse`: Sparsity regularization weight (default: 0.01)
- `--lambda_dag`: DAG constraint penalty coefficient (default: 1.0)
- `--causal_warmup_epochs`: Warmup epochs before enabling causal losses
- `--causal_rampup_epochs`: Ramp-up epochs to full causal loss weight
- `--revin_affine`: Enable learnable affine after instance norm (0/1)
- `--linear_residual`: Enable target-linear residual branch (0/1)
- `--linear_residual_seasonal`, `--linear_residual_seasonal_ma`: Use trend/seasonal residual decomposition
- `--two_stage_residual`: Learn a delta on top of a detached residual baseline
- `--multi_scale_residual`, `--multi_scale_windows`: Use multi-window residual decomposition
- `--fft_residual`, `--fft_residual_top_k`: Use FFT-based trend/seasonal residuals
- `--adaptive_residual_gate`: Blend transformer output and residual branch with a learned gate

### Prediction Post-processing and Calibration
- `--pred_smooth_method`: `none`, `ema`, `ma`, or `endpoint_linear`
- `--pred_smooth_alpha`, `--pred_smooth_window`, `--pred_smooth_blend`: Smoothing controls
- `--endpoint_lerp_strength`: Blend predictions with the last encoder value
- `--calibration_method`: `none`, `affine`, `endpoint_lerp`, or `smooth_blend`
- `--calibration_grid_size`, `--calibration_max_strength`: Validation-time search controls
- `--calibration_smooth_method`, `--calibration_smooth_alpha`, `--calibration_smooth_window`: Calibration smoothing controls

---

## Parallel Benchmark Scripts

### Standard Benchmarks

```bash
# Smoke test (2 epochs, 96 pred_len)
bash run_benchmarks_parallel.sh --smoke

# Full run (10 epochs)
bash run_benchmarks_parallel.sh --full

# CausalExoFormer only
bash run_benchmarks_parallel.sh --causal-only

# Kill all jobs
bash run_benchmarks_parallel.sh --kill
```

The benchmark script now includes dataset-specific tuning presets for harder cases such as ETTh2 and exchange_rate, including mixed losses, gradient clipping, residual settings, and prediction smoothing.

### Lake Datasets

```bash
# Smoke test (2 epochs)
bash run_lakes_parallel.sh --smoke

# Full run (10 epochs)
bash run_lakes_parallel.sh --full

# Kill all jobs
bash run_lakes_parallel.sh --kill
```

---

## Output

- **Checkpoints**: `checkpoints/<setting>/checkpoint.pth`
- **Results**: `results/<setting>/`
  - `metrics.npy`: Evaluation metrics
  - `pred.npy`: Predictions
  - `true.npy`: Ground truth
  - `val_pred.npy`, `val_true.npy`: Saved when validation-based calibration is enabled
- **Causal Analysis** (CausalExoFormer only):
  - `causal_graph.npy`: Learned causal gate matrix
  - `lag_distribution.npy`: Lag softmax distribution
  - `causal_heatmap.png`: Visualization of causal relationships

Generated artifacts such as checkpoints, HTML, TeX intermediates, and large local datasets are intentionally ignored in Git so that repository syncs only include source changes.
---

## Metrics

The project computes: **MAE**, **MSE**, **RMSE**, **MAPE**, **MSPE**

---

## Citation

If you find this useful, please cite our work:

```bibtex
@article{causalexoformer2026,
  title={CausalExoFormer: Causal Discovery meets Endogenous-aware Time Series Forecasting},
  author={},
  year={2026}
}
```