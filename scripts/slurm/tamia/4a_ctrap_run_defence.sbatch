#!/bin/bash
#SBATCH --job-name=ctrap-defence
#SBATCH --output=artifacts/tamia/logs/ctrap-defence-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
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
artifacts_dir="$PWD/artifacts"
mkdir -p "$artifacts_dir/tamia/models" "$artifacts_dir/tamia/evals"
UV_PROJECT_ENVIRONMENT="$ctrap_env" uv sync --project ctrap --frozen

# Inputs
base_model_name_or_path="meta-llama/Llama-2-7b-hf"
ctrap_alpha=0.1
ctrap_lambda=0.1

# Set defence model name.
defence_model_name="llama2-7b--align-lora-ctrap-a$ctrap_alpha-l$ctrap_lambda"

# Resumption handling.
if [[ -f "$artifacts_dir/tamia/evals/$defence_model_name/defence-complete" ]]; then
    echo "Skipping completed defence: $defence_model_name"
    exit 0
fi

# Defence.
if [[ ! -f "$artifacts_dir/tamia/models/$defence_model_name/.trained" ]]; then
    (
        cd "$PWD/ctrap"
        "$ctrap_env/bin/torchrun" --standalone --nproc-per-node=4 train.py \
            --model_name_or_path "$base_model_name_or_path" \
            --data_path PKU-Alignment/BeaverTails_safe_alignment \
            --bf16 True \
            --output_dir "$artifacts_dir/tamia/models/$defence_model_name" \
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
            --alpha "$ctrap_alpha"
    )
    touch "$artifacts_dir/tamia/models/$defence_model_name/.trained"
else
    echo "Reusing completed defence training: $defence_model_name"
fi

# Native defence evaluation.
(
    cd "$PWD/ctrap/poison/evaluation"
    "$ctrap_env/bin/python" pred.py \
        --model_folder "$base_model_name_or_path" \
        --lora_folder "$artifacts_dir/tamia/models/$defence_model_name" \
        --instruction_path "$CTRAP_DATA_DIR/beavertails_harmful_test_1000.json" \
        --output_path "$artifacts_dir/tamia/evals/$defence_model_name/generations.json"
    "$ctrap_env/bin/python" eval_sentiment.py \
        --input_path "$artifacts_dir/tamia/evals/$defence_model_name/generations.json"
)
touch "$artifacts_dir/tamia/evals/$defence_model_name/defence-complete"
