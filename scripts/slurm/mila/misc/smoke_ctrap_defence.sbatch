#!/bin/bash
#SBATCH --job-name=ctrap-smoke
#SBATCH --output=artifacts/mila/logs/ctrap-smoke-%j.out
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================

# Setup
export CTRAP_DATA_DIR="$SCRATCH/tamper-resistance/data/ctrap"
ctrap_env="$SLURM_TMPDIR/.venv-ctrap-$SLURM_JOB_ID"
artifacts_dir="$SCRATCH/tamper-resistance/artifacts"
mkdir -p "$artifacts_dir/mila/misc/logs" \
    "$artifacts_dir/mila/misc/models" \
    "$artifacts_dir/mila/misc/evals"
UV_PROJECT_ENVIRONMENT="$ctrap_env" uv sync --project ctrap --frozen

# Inputs
base_model_name_or_path="meta-llama/Llama-2-7b-hf"
ctrap_alpha=0.1
ctrap_lambda=0.1
max_steps=1

# Set smoke model name.
smoke_model_name="llama2-7b--align-lora-ctrap-a$ctrap_alpha-l$ctrap_lambda--smoke-ms$max_steps"

# Record GPU memory during the smoke run.
(
    while true; do
        nvidia-smi --query-gpu=timestamp,memory.used,memory.total \
            --format=csv,noheader,nounits
        sleep 5
    done
) > "$artifacts_dir/mila/misc/logs/$smoke_model_name-memory.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

# Restart the smoke run from scratch.
rm -rf -- "$artifacts_dir/mila/misc/models/$smoke_model_name" \
    "$artifacts_dir/mila/misc/evals/$smoke_model_name"
mkdir -p "$artifacts_dir/mila/misc/evals/$smoke_model_name"

# One-step defence smoke test.
(
    cd "$PWD/ctrap"
    "$ctrap_env/bin/python" train.py \
        --model_name_or_path "$base_model_name_or_path" \
        --data_path PKU-Alignment/BeaverTails_safe_alignment \
        --bf16 True \
        --output_dir "$artifacts_dir/mila/misc/models/$smoke_model_name" \
        --num_train_epochs 20 \
        --per_device_train_batch_size 10 \
        --save_steps 100000 \
        --learning_rate 5e-4 \
        --weight_decay 0.1 \
        --lr_scheduler_type constant \
        --logging_steps 1 \
        --tf32 True \
        --optimizer ctrap \
        --sample_num 5000 \
        --bad_sample_num 5000 \
        --benign_dataset "$CTRAP_DATA_DIR/sst2.json" \
        --lamb "$ctrap_lambda" \
        --alpha "$ctrap_alpha" \
        --max_steps "$max_steps"
)
touch "$artifacts_dir/mila/misc/evals/$smoke_model_name/smoke-complete"
