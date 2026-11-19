#!/bin/bash
#SBATCH --job-name=gma
#SBATCH --output=artifacts/tamia/logs/run-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --array=0-11
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export AXOLOTL_DO_NOT_TRACK=1
export WANDB_MODE=offline

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================

# Setup
project_dir="$PWD"
defence_env="$SLURM_TMPDIR/.venv-defence-$SLURM_JOB_ID"
attack_env="$SLURM_TMPDIR/.venv-attack-$SLURM_JOB_ID"
UV_PROJECT_ENVIRONMENT="$defence_env" uv sync --frozen
UV_PROJECT_ENVIRONMENT="$attack_env" uv sync --frozen
export WANDB_DIR="$PWD/artifacts/tamia"
mkdir -p "$WANDB_DIR" artifacts/tamia/logs

# Inputs (both configs use response_separator: " ").
defence_config=${1:-configs/beavertails/stages/0_align/llama2-7b--align-fft-gma.yml}
attack_config=${2:-configs/beavertails/stages/1_attack/llama2-7b--attack-fft.yml}

# Inter-job sweep
DEFENCE_RETAIN_COEFFS=(0.1 1.0 5.0) # Retain loss weight (default: 1.0)
DEFENCE_GMA_COEFFS=(0.01 0.1 1.0 5.0) # Gradient cosine weight (default: 1.0)

# Intra-job sweep
attack_learning_rates=(5e-5 8e-5 1e-4 2e-4)
attack_max_steps=(5)

index=${SLURM_ARRAY_TASK_ID:?submit with sbatch}
defence_retain_coeff=${DEFENCE_RETAIN_COEFFS[index / ${#DEFENCE_GMA_COEFFS[@]}]}
defence_gma_coeff=${DEFENCE_GMA_COEFFS[index % ${#DEFENCE_GMA_COEFFS[@]}]}

# Set defence model name
defence_model_name="llama2-7b-beavertail--align-fft-gma--coeff-gma$defence_gma_coeff--coeff-retain$defence_retain_coeff"
defence_model_path="artifacts/tamia/models/$defence_model_name"

# ================================================================================
# Resumption handling. Skip iff every attack config for this defence exists
skip_defence=true
if [[ ! -f "artifacts/tamia/evals/$defence_model_name/results-1000.json" ]]; then
    skip_defence=false
fi
for max_steps in "${attack_max_steps[@]}"; do
    for attack_lr in "${attack_learning_rates[@]}"; do
        attack_model_name="$defence_model_name--relearn-fft-lr$attack_lr-ms$max_steps"
        if [[ ! -f "artifacts/tamia/evals/$attack_model_name/results-1000.json" ]]; then
            skip_defence=false
        fi
    done
done

if "$skip_defence"; then
    echo "Skipping completed defence: $defence_model_name"
    exit 0
fi

# Restart this defence and all its attacks from scratch.
rm -rf -- "artifacts/tamia/models/$defence_model_name" \
    "artifacts/tamia/evals/$defence_model_name"
for max_steps in "${attack_max_steps[@]}"; do
    for attack_lr in "${attack_learning_rates[@]}"; do
        attack_model_name="$defence_model_name--relearn-fft-lr$attack_lr-ms$max_steps"
        rm -rf -- "artifacts/tamia/models/$attack_model_name" \
            "artifacts/tamia/evals/$attack_model_name"
    done
done
# ================================================================================

# Defence.
export CUDA_VISIBLE_DEVICES=0,1,2,3
PYTHONHASHSEED=0 "$defence_env/bin/axolotl" train "$defence_config" --launcher python \
    --trainer_cls trainers.tr_gma.GMAGDTrainer \
    --output_dir "$defence_model_path" \
    --wandb_name "$defence_model_name" \
    --custom_trainer_args__coeff_retain "$defence_retain_coeff" \
    --custom_trainer_args__coeff_gma "$defence_gma_coeff" \
    --save_total_limit 1

# Defence evaluation before the attacks.
"$attack_env/bin/python" -m src.evals.evaluation \
    --model "$defence_model_path" \
    --results-path "artifacts/tamia/evals/$defence_model_name/results-1000.json" \
    --utility-samples 1000

# Attack and evaluation.
for max_steps in "${attack_max_steps[@]}"; do
    pids=()
    for gpu in 0 1 2 3; do
        attack_lr=${attack_learning_rates[gpu]}
        attack_model_name="$defence_model_name--relearn-fft-lr$attack_lr-ms$max_steps"
        (
            export CUDA_VISIBLE_DEVICES=$gpu
            "$attack_env/bin/axolotl" train "$attack_config" --launcher python \
                --base-model "$defence_model_path" \
                --output-dir "artifacts/tamia/models/$attack_model_name" \
                --learning-rate "$attack_lr" \
                --max-steps "$max_steps"
            "$attack_env/bin/python" -m src.evals.evaluation \
                --model "artifacts/tamia/models/$attack_model_name" \
                --results-path "artifacts/tamia/evals/$attack_model_name/results-1000.json" \
                --utility-samples 1000
            rm -rf -- "artifacts/tamia/models/$attack_model_name"
        ) >> "artifacts/tamia/logs/$attack_model_name.log" 2>&1 &
        pids+=("$!")
    done

    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if (( failed )); then
        exit "$failed"
    fi
done

# Delete model artifacts only after evaluation succeeds.
rm -rf -- "artifacts/tamia/models/$defence_model_name"
