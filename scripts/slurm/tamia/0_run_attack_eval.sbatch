#!/bin/bash
#SBATCH --job-name=base-attack
#SBATCH --output=artifacts/tamia/logs/run-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --array=0
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
export AXOLOTL_DO_NOT_TRACK=1
export WANDB_MODE=offline

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================

# Setup
attack_env="$SLURM_TMPDIR/.venv-attack-$SLURM_JOB_ID"
UV_PROJECT_ENVIRONMENT="$attack_env" uv sync --frozen
export WANDB_DIR="$PWD/artifacts/tamia"
mkdir -p "$WANDB_DIR" artifacts/tamia/logs

# Inputs
base_model_name_or_path=${1:?usage: 0_run_attack_eval.sbatch BASE_MODEL_NAME_OR_PATH ATTACK_CONFIG}
attack_config=${2:?missing attack config}

# Intra-job sweep
attack_learning_rates=(5e-5 8e-5 1e-4 2e-4)
attack_max_steps=(5 10 50 100)

# Get attack separator used
attack_separator_abbrv=$("$attack_env/bin/python" -m src.data.response_format \
    "$attack_config" --abbreviation)

# Set base model name
base_model_name="${base_model_name_or_path%/}"
base_model_name="${base_model_name##*/}"

# ================================================================================
# Resumption handling. Skip iff the base and every attack evaluation exist.
skip_attacks=true
if [[ ! -f "artifacts/tamia/evals/$base_model_name/results.json" ]]; then
    skip_attacks=false
fi
for max_steps in "${attack_max_steps[@]}"; do
    for attack_lr in "${attack_learning_rates[@]}"; do
        attack_model_name="$base_model_name--relearn-fft-att-$attack_separator_abbrv-lr$attack_lr-ms$max_steps"
        if [[ ! -f "artifacts/tamia/evals/$attack_model_name/results.json" ]]; then
            skip_attacks=false
        fi
    done
done

if "$skip_attacks"; then
    echo "Skipping completed attacks: $base_model_name"
    exit 0
fi

# Restart the base evaluation and all attacks from scratch.
rm -rf -- "artifacts/tamia/evals/$base_model_name"
for max_steps in "${attack_max_steps[@]}"; do
    for attack_lr in "${attack_learning_rates[@]}"; do
        attack_model_name="$base_model_name--relearn-fft-att-$attack_separator_abbrv-lr$attack_lr-ms$max_steps"
        rm -rf -- "artifacts/tamia/models/$attack_model_name" \
            "artifacts/tamia/evals/$attack_model_name"
    done
done
# ================================================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Prepare the shared attack dataset before starting parallel attacks.
"$attack_env/bin/axolotl" preprocess "$attack_config" \
    --base-model "$base_model_name_or_path"

# Base evaluation before the attacks.
"$attack_env/bin/python" -m src.evals.evaluation \
    --model "$base_model_name_or_path" \
    --results-path "artifacts/tamia/evals/$base_model_name/results.json"

# Attack and evaluation.
for max_steps in "${attack_max_steps[@]}"; do
    pids=()
    for gpu in 0 1 2 3; do
        attack_lr=${attack_learning_rates[gpu]}
        attack_model_name="$base_model_name--relearn-fft-att-$attack_separator_abbrv-lr$attack_lr-ms$max_steps"
        (
            export CUDA_VISIBLE_DEVICES=$gpu
            "$attack_env/bin/axolotl" train "$attack_config" --launcher python \
                --base-model "$base_model_name_or_path" \
                --output-dir "artifacts/tamia/models/$attack_model_name" \
                --learning-rate "$attack_lr" \
                --max-steps "$max_steps"
            "$attack_env/bin/python" -m src.evals.evaluation \
                --model "artifacts/tamia/models/$attack_model_name" \
                --results-path "artifacts/tamia/evals/$attack_model_name/results.json"
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
