#!/bin/bash
#SBATCH --job-name=repnoise-defence-losses
#SBATCH --output=artifacts/tamia/logs/repnoise-defence-losses-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --array=0-1
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
export WANDB_MODE=offline

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================

# Setup
project_dir="$PWD"
defence_env="$SLURM_TMPDIR/.venv-defence-$SLURM_JOB_ID"
attack_env="$SLURM_TMPDIR/.venv-attack-$SLURM_JOB_ID"
UV_PROJECT_ENVIRONMENT="$defence_env" uv sync --project repnoise --frozen
UV_PROJECT_ENVIRONMENT="$attack_env" uv sync --frozen
export WANDB_DIR="$PWD/artifacts/tamia"
mkdir -p "$WANDB_DIR" artifacts/tamia/logs

# Authors' defence configuration, without dataset shuffling.
formats=(nl sp)
format=${formats[${SLURM_ARRAY_TASK_ID:?submit with sbatch}]}
case "$format" in
    nl)
        defence_separator=$'\n'
        attack_config="configs/beavertails/stages/1_attack/llama2-7b--attack-fft-repnoise-nl.yml"
        ;;
    sp)
        defence_separator=' '
        attack_config="configs/beavertails/stages/1_attack/llama2-7b--attack-fft-repnoise-sp.yml"
        ;;
esac
defence_model_name="llama2-7b-beavertail--align-fft-repnoise-$format-alpha1.0-beta0.001--loss-tracking"
defence_model_path="artifacts/tamia/models/$defence_model_name"
defence_results_path="artifacts/tamia/evals/$defence_model_name/results.json"
attack_model_name="$defence_model_name--attack-fft-$format-adamw-lr8e-5-ms250"
attack_model_path="artifacts/tamia/models/$attack_model_name"
attack_results_path="artifacts/tamia/evals/$attack_model_name/results.json"

if [[ -f "$attack_results_path" ]]; then
    echo "Skipping completed experiment: $attack_model_name"
    exit 0
fi

if [[ ! -f "$defence_results_path" ]]; then
    rm -rf -- "$defence_model_path" "${defence_results_path%/*}"
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    (
        cd "$project_dir/repnoise"
        PYTHONHASHSEED=0 "$defence_env/bin/python" -m representation_noising.train \
            --model meta-llama/Llama-2-7b-chat-hf \
            --output-dir "$project_dir/$defence_model_path" \
            --harmful-response-separator "$defence_separator" \
            --alpha 1.0 \
            --beta 0.001 \
            --wandb-project tamper-resistance \
            --wandb-run-name "$defence_model_name"
    )
    touch "$defence_model_path/.trained"
    "$attack_env/bin/python" -m src.evals.evaluation \
        --model "$defence_model_path" \
        --results-path "$defence_results_path"
elif [[ ! -f "$defence_model_path/.trained" ]]; then
    echo "Completed defence is missing its retained model: $defence_model_path" >&2
    exit 1
fi

"$attack_env/bin/axolotl" preprocess "$attack_config" \
    --base-model "$defence_model_path"
rm -rf -- "$attack_model_path" "${attack_results_path%/*}"
export CUDA_VISIBLE_DEVICES=0
"$attack_env/bin/axolotl" train "$attack_config" --launcher python \
    --base-model "$defence_model_path" \
    --output-dir "$attack_model_path" \
    --learning-rate 8e-5 \
    --max-steps 250 \
    --wandb-name "$attack_model_name"
"$attack_env/bin/python" -m src.evals.evaluation \
    --model "$attack_model_path" \
    --results-path "$attack_results_path"
rm -rf -- "$attack_model_path"
