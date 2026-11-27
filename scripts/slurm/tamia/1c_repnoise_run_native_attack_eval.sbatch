#!/bin/bash
#SBATCH --job-name=repnoise-native-attack
#SBATCH --output=artifacts/tamia/logs/repnoise-native-attack-%A_%a.out
#SBATCH --array=0-4
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

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================

# Setup
project_dir="$PWD"
attack_env="$SLURM_TMPDIR/.venv-attack-$SLURM_JOB_ID"
eval_env="$SLURM_TMPDIR/.venv-eval-$SLURM_JOB_ID"
mkdir -p artifacts/tamia/models artifacts/tamia/evals artifacts/tamia/logs
UV_PROJECT_ENVIRONMENT="$attack_env" uv sync --project repnoise --frozen
UV_PROJECT_ENVIRONMENT="$eval_env" uv sync --frozen

# Authors' NL/NL setup.
seeds=(1 2 3 4 5)
seed=${seeds[${SLURM_ARRAY_TASK_ID:?submit with sbatch}]}
defence_model_name="llama2-7b-beavertail--align-fft-repnoise-nl-alpha1.0-beta0.001"
attack_model_name="$defence_model_name--attack-fft-author-codebase-nl-lr8e-5-ms250-seed$seed"
defence_model_path="artifacts/tamia/models/$defence_model_name"
attack_model_path="artifacts/tamia/models/$attack_model_name"
attack_results_path="artifacts/tamia/evals/$attack_model_name/results.json"

if [[ -f "$attack_results_path" ]]; then
    echo "Skipping completed attack: $attack_model_name"
    exit 0
fi
if [[ ! -f "$defence_model_path/.trained" ]]; then
    echo "Missing defended model: $defence_model_path" >&2
    exit 1
fi

rm -rf -- "$attack_model_path" "$(dirname "$attack_results_path")"
(
    cd repnoise
    "$attack_env/bin/python" -m representation_noising.train \
        --model "$project_dir/$defence_model_path" \
        --output-dir "$project_dir/$attack_model_path" \
        --attack \
        --harmful-response-separator $'\n' \
        --size 1000 \
        --batch-size 4 \
        --learning-rate 8e-5 \
        --weight-decay 0.01 \
        --epochs 1 \
        --max-steps 250 \
        --seed "$seed" \
        --shuffle
)
"$eval_env/bin/python" -m src.evals.evaluation \
    --model "$attack_model_path" \
    --results-path "$attack_results_path"
rm -rf -- "$attack_model_path"
