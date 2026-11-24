#!/bin/bash
#SBATCH --job-name=seam-defence-sp
#SBATCH --output=artifacts/tamia/logs/run-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --array=0-2
set -euo pipefail

# Cluster setup.
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

# Project setup.
project_dir="$PWD"
defence_env="$SLURM_TMPDIR/.venv-defence-$SLURM_JOB_ID"
eval_env="$SLURM_TMPDIR/.venv-eval-$SLURM_JOB_ID"
UV_PROJECT_ENVIRONMENT="$defence_env" uv sync --project seam --frozen
UV_PROJECT_ENVIRONMENT="$eval_env" uv sync --frozen
export WANDB_DIR="$PWD/artifacts/tamia"
mkdir -p "$WANDB_DIR" artifacts/tamia/logs

# Match the existing SP defence, varying only the self-destruct coefficient.
defence_config=$(realpath -- configs/beavertails/seam/train-paper.yaml)
defence_separator=' '
defence_up_coeff=1.0
defence_sd_coeffs=(0.1 0.5 1.0)
defence_sd_coeff=${defence_sd_coeffs[SLURM_ARRAY_TASK_ID]}
defence_model_name="llama2-7b-beavertail--align-fft-seam-sp-sd$defence_sd_coeff-up$defence_up_coeff"
defence_model_dir="artifacts/tamia/models/$defence_model_name"
defence_model_path="$defence_model_dir/checkpoint-500"
defence_eval_path="artifacts/tamia/evals/$defence_model_name/results.json"

if [[ -f "$defence_eval_path" ]]; then
    if [[ ! -f "$defence_model_path/config.json" ]]; then
        echo "Completed defence is missing its retained checkpoint: $defence_model_path" >&2
        exit 1
    fi
    echo "Skipping completed defence: $defence_model_name"
    exit 0
fi

rm -rf -- "$defence_model_dir" "${defence_eval_path%/*}"

export CUDA_VISIBLE_DEVICES=0,1,2,3
(
    cd "$project_dir/seam"
    PYTHONHASHSEED=0 "$defence_env/bin/python" -m src.train \
        --config "$defence_config" \
        --output_dir "$project_dir/$defence_model_dir" \
        --wandb_run_name "$defence_model_name" \
        --harmful_response_separator "$defence_separator" \
        --alpha "$defence_up_coeff" \
        --beta "$defence_sd_coeff" \
        --save_total_limit 1
)

"$eval_env/bin/python" -m src.evals.evaluation \
    --model "$defence_model_path" \
    --results-path "$defence_eval_path"
