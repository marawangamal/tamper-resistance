#!/bin/bash
#SBATCH --job-name=gma-coeff
#SBATCH --output=artifacts/tamia/logs/run-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --array=0-11
set -euo pipefail

# Cluster setup.
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

# Project setup.
defence_env="$SLURM_TMPDIR/.venv-defence-$SLURM_JOB_ID"
attack_env="$SLURM_TMPDIR/.venv-attack-$SLURM_JOB_ID"
UV_PROJECT_ENVIRONMENT="$defence_env" uv sync --frozen
UV_PROJECT_ENVIRONMENT="$attack_env" uv sync --frozen
export WANDB_DIR="$PWD/artifacts/tamia"
mkdir -p "$WANDB_DIR" artifacts/tamia/logs "$SLURM_TMPDIR/models"

defence_config=${1:-configs/beavertails/stages/0_align/llama2-7b--align-fft-gma.yml}
attack_config=${2:-configs/beavertails/stages/1_attack/llama2-7b--attack-fft-seam-sp-adamw-no-history.yml}

# The preceding LR sweep found capability-eligible points at both LRs, while
# changing retain weight did not improve post-attack harmfulness. Increase the
# gradient decorrelation pressure around those eligible points.
DEFENCE_LEARNING_RATES=(5e-7 1e-6)
DEFENCE_RETAIN_COEFFS=(0.1 1.0)
DEFENCE_GMA_COEFFS=(2.0 5.0 10.0)
attack_learning_rates=(5e-5 8e-5 1e-4 2e-4)
attack_max_steps=5

index=${SLURM_ARRAY_TASK_ID:?submit with sbatch}
lr_count=${#DEFENCE_LEARNING_RATES[@]}
retain_count=${#DEFENCE_RETAIN_COEFFS[@]}
defence_lr=${DEFENCE_LEARNING_RATES[index % lr_count]}
defence_retain_coeff=${DEFENCE_RETAIN_COEFFS[(index / lr_count) % retain_count]}
defence_gma_coeff=${DEFENCE_GMA_COEFFS[index / (lr_count * retain_count)]}
defence_model_name="llama2-7b-beavertail--align-fft-gma--lr$defence_lr--coeff-gma$defence_gma_coeff--coeff-retain$defence_retain_coeff"
defence_model_path="$SLURM_TMPDIR/models/$defence_model_name"
resolved_defence_config="$SLURM_TMPDIR/gma-defence-$SLURM_JOB_ID.yml"

"$defence_env/bin/python" scripts/misc/render_gma_config.py \
    "$defence_config" "$resolved_defence_config" \
    --output-dir "$defence_model_path" \
    --wandb-name "$defence_model_name" \
    --coeff-retain "$defence_retain_coeff" \
    --coeff-gma "$defence_gma_coeff" \
    --learning-rate "$defence_lr"

skip_defence=true
if [[ ! -f "artifacts/tamia/evals/$defence_model_name/results-1000.json" ]]; then
    skip_defence=false
fi
for attack_lr in "${attack_learning_rates[@]}"; do
    attack_model_name="$defence_model_name--relearn-fft-lr$attack_lr-ms$attack_max_steps"
    if [[ ! -f "artifacts/tamia/evals/$attack_model_name/results-1000.json" ]]; then
        skip_defence=false
    fi
done
if "$skip_defence"; then
    echo "Skipping completed defence: $defence_model_name"
    exit 0
fi

rm -rf -- "$defence_model_path" "artifacts/tamia/evals/$defence_model_name"
for attack_lr in "${attack_learning_rates[@]}"; do
    attack_model_name="$defence_model_name--relearn-fft-lr$attack_lr-ms$attack_max_steps"
    rm -rf -- "$SLURM_TMPDIR/models/$attack_model_name" \
        "artifacts/tamia/evals/$attack_model_name"
done

# Train and evaluate the defence.
export CUDA_VISIBLE_DEVICES=0,1,2,3
PYTHONHASHSEED=0 "$defence_env/bin/axolotl" train \
    "$resolved_defence_config" --launcher python
"$attack_env/bin/axolotl" preprocess "$attack_config" \
    --base-model "$defence_model_path"
"$attack_env/bin/python" -m src.evals.evaluation \
    --model "$defence_model_path" \
    --results-path "artifacts/tamia/evals/$defence_model_name/results-1000.json" \
    --utility-samples 1000

# Run the four comparable attacks in parallel, one per GPU.
pids=()
for gpu in 0 1 2 3; do
    attack_lr=${attack_learning_rates[gpu]}
    attack_model_name="$defence_model_name--relearn-fft-lr$attack_lr-ms$attack_max_steps"
    attack_model_path="$SLURM_TMPDIR/models/$attack_model_name"
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        "$attack_env/bin/axolotl" train "$attack_config" --launcher python \
            --base-model "$defence_model_path" \
            --output-dir "$attack_model_path" \
            --learning-rate "$attack_lr" \
            --max-steps "$attack_max_steps" \
            --wandb-name "$attack_model_name"
        "$attack_env/bin/python" -m src.evals.evaluation \
            --model "$attack_model_path" \
            --results-path "artifacts/tamia/evals/$attack_model_name/results-1000.json" \
            --utility-samples 1000
        rm -rf -- "$attack_model_path"
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

rm -rf -- "$defence_model_path"
