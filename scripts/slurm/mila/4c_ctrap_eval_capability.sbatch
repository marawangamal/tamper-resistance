#!/bin/bash
#SBATCH --job-name=ctrap-capability
#SBATCH --output=artifacts/mila/logs/ctrap-capability-%A_%a.out
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
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
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================

# Setup
export CTRAP_DATA_DIR="$SCRATCH/tamper-resistance/data/ctrap"
ctrap_env="$SLURM_TMPDIR/.venv-ctrap-$SLURM_JOB_ID"
artifacts_dir="$SCRATCH/tamper-resistance/artifacts"
sync_lock="$UV_CACHE_DIR/ctrap-sync.lock"
until mkdir "$sync_lock" 2>/dev/null; do sleep 1; done
trap 'rmdir "$sync_lock"' EXIT
UV_PROJECT_ENVIRONMENT="$ctrap_env" uv sync --project ctrap --frozen
rmdir "$sync_lock"
trap - EXIT

# Inputs
base_model_name_or_path="meta-llama/Llama-2-7b-hf"
methods=(raw ctrap)
model_names=(
    "llama2-7b--raw"
    "llama2-7b--align-lora-ctrap-a0.1-l0.1"
)
tasks=(sst2 agnews gsm8k)

# Inter-job sweep (2 models x 3 capability tasks = 6 tasks)
index=${SLURM_ARRAY_TASK_ID:?submit with sbatch}
method_index=$((index / ${#tasks[@]}))
method=${methods[method_index]}
model_name=${model_names[method_index]}
task=${tasks[index % ${#tasks[@]}]}
output_path="$artifacts_dir/mila/evals/$model_name/capability/$task.json"

# Resumption handling.
if [[ -f "$output_path" ]]; then
    echo "Skipping completed capability evaluation: $model_name/$task"
    exit 0
fi
mkdir -p "$(dirname "$output_path")"

lora_args=()
if [[ "$method" == ctrap ]]; then
    if [[ ! -f "$artifacts_dir/mila/models/$model_name/.trained" ]]; then
        echo "Missing completed CTRAP defence: $model_name" >&2
        exit 1
    fi
    lora_args=(--lora_folder "$artifacts_dir/mila/models/$model_name")
fi

# Native capability evaluation without fine-tuning.
case "$task" in
    sst2)
        (
            cd "$PWD/ctrap/sst2"
            "$ctrap_env/bin/python" pred_eval.py \
                --model_folder "$base_model_name_or_path" \
                "${lora_args[@]}" \
                --output_path "$output_path" \
                --dataset_path "$CTRAP_DATA_DIR/sst2_validation.json"
        )
        ;;
    agnews)
        (
            cd "$PWD/ctrap/agnews"
            "$ctrap_env/bin/python" pred_eval.py \
                --model_folder "$base_model_name_or_path" \
                "${lora_args[@]}" \
                --output_path "$output_path"
        )
        ;;
    gsm8k)
        (
            cd "$PWD/ctrap/gsm8k"
            "$ctrap_env/bin/python" pred_eval.py \
                --model_folder "$base_model_name_or_path" \
                "${lora_args[@]}" \
                --output_path "$output_path"
        )
        ;;
esac
