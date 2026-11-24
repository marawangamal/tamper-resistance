#!/bin/bash
#SBATCH --job-name=ctrap-defence
#SBATCH --output=artifacts/mila/logs/ctrap-defence-%j.out
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_LINK_MODE=copy
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

project_dir="$PWD"
ctrap_dir="$project_dir/ctrap"
ctrap_env="$SLURM_TMPDIR/.venv-ctrap-$SLURM_JOB_ID"
artifact_root="$project_dir/artifacts/mila"
base_model="meta-llama/Llama-2-7b-hf"
ctrap_alpha=0.1
ctrap_lambda=0.1
max_steps=${CTRAP_MAX_STEPS:--1}

if (( max_steps > 0 )); then
    run_suffix="--smoke-ms$max_steps"
else
    run_suffix=""
fi
model_name="llama2-7b--align-lora-ctrap-a$ctrap_alpha-l$ctrap_lambda$run_suffix"
model_path="$artifact_root/models/$model_name"
eval_dir="$artifact_root/evals/$model_name"

mkdir -p "$artifact_root/logs" "$artifact_root/models" "$artifact_root/evals"
UV_PROJECT_ENVIRONMENT="$ctrap_env" uv sync \
    --project "$ctrap_dir" --offline --frozen

monitor_log="$artifact_root/logs/$model_name-memory.csv"
(
    while true; do
        nvidia-smi --query-gpu=timestamp,memory.used,memory.total \
            --format=csv,noheader,nounits
        sleep 5
    done
) > "$monitor_log" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

rm -rf -- "$model_path" "$eval_dir"
mkdir -p "$eval_dir"

(
    cd "$ctrap_dir"
    "$ctrap_env/bin/python" train.py \
        --model_name_or_path "$base_model" \
        --data_path PKU-Alignment/BeaverTails_safe_alignment \
        --bf16 True \
        --output_dir "$model_path" \
        --num_train_epochs 20 \
        --per_device_train_batch_size 10 \
        --per_device_eval_batch_size 10 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 100000 \
        --save_total_limit 0 \
        --learning_rate 5e-4 \
        --weight_decay 0.1 \
        --warmup_ratio 0 \
        --lr_scheduler_type constant \
        --logging_steps 1 \
        --tf32 True \
        --cache_dir "$HF_HOME" \
        --optimizer ctrap \
        --sample_num 5000 \
        --bad_sample_num 5000 \
        --lamb "$ctrap_lambda" \
        --alpha "$ctrap_alpha" \
        --eval_steps 5000 \
        --max_steps "$max_steps"
)

if (( max_steps > 0 )); then
    touch "$eval_dir/smoke-complete"
    exit 0
fi

(
    cd "$ctrap_dir/poison/evaluation"
    "$ctrap_env/bin/python" pred.py \
        --model_folder "$base_model" \
        --lora_folder "$model_path" \
        --output_path "$eval_dir/generations.json" \
        --cache_dir "$HF_HOME"
    "$ctrap_env/bin/python" eval_sentiment.py \
        --input_path "$eval_dir/generations.json" \
        --cache_dir "$HF_HOME"
)
touch "$eval_dir/defence-complete"
