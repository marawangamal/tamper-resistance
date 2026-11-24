#!/bin/bash
#SBATCH --job-name=ctrap-attack
#SBATCH --output=artifacts/mila/logs/ctrap-attack-%A_%a.out
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SCRATCH/.cache/uv-ctrap"
export CTRAP_DATA_DIR="$SCRATCH/tamper-resistance/data/ctrap"
export UV_LINK_MODE=copy
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

project_dir="$PWD"
ctrap_dir="$project_dir/ctrap"
ctrap_env="$SLURM_TMPDIR/.venv-ctrap-$SLURM_JOB_ID"
artifact_root=${CTRAP_ARTIFACT_ROOT:-"$SCRATCH/tamper-resistance/artifacts/mila"}
base_model="meta-llama/Llama-2-7b-hf"
ctrap_alpha=0.1
ctrap_lambda=0.1
defence_name="llama2-7b--align-lora-ctrap-a$ctrap_alpha-l$ctrap_lambda"
defence_path="$artifact_root/models/$defence_name"
sample_counts=(100 200 300 400 500)

index=${SLURM_ARRAY_TASK_ID:?submit with sbatch}
if (( index < 5 )); then
    method=raw
    sample_count=${sample_counts[index]}
    attack_name="llama2-7b--raw--attack-lora-full-harmful-n$sample_count"
    lora_args=()
else
    method=ctrap
    sample_count=${sample_counts[index - 5]}
    attack_name="$defence_name--attack-lora-full-harmful-n$sample_count"
    lora_args=(--lora_folder "$defence_path")
    if [[ ! -f "$artifact_root/evals/$defence_name/defence-complete" ]]; then
        echo "Missing completed CTRAP defence: $defence_path" >&2
        exit 1
    fi
fi

attack_path="$artifact_root/models/$attack_name"
eval_dir="$artifact_root/evals/$attack_name"
if [[ -f "$eval_dir/results.json" ]]; then
    echo "Skipping completed attack: $attack_name"
    exit 0
fi

mkdir -p "$artifact_root/logs" "$artifact_root/models" "$artifact_root/evals"
UV_PROJECT_ENVIRONMENT="$ctrap_env" uv sync \
    --project "$ctrap_dir" --offline --frozen
rm -rf -- "$attack_path" "$eval_dir"
mkdir -p "$eval_dir"

(
    cd "$ctrap_dir"
    "$ctrap_env/bin/python" train.py \
        --model_name_or_path "$base_model" \
        "${lora_args[@]}" \
        --data_path PKU-Alignment/BeaverTails_dangerous \
        --bf16 True \
        --output_dir "$attack_path" \
        --num_train_epochs 20 \
        --per_device_train_batch_size 10 \
        --per_device_eval_batch_size 10 \
        --gradient_accumulation_steps 1 \
        --save_strategy steps \
        --save_steps 100000 \
        --save_total_limit 0 \
        --learning_rate 1e-5 \
        --weight_decay 0.1 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type constant \
        --logging_steps 10 \
        --tf32 True \
        --eval_steps 2000 \
        --cache_dir "$HF_HOME" \
        --optimizer normal \
        --evaluation_strategy steps \
        --sample_num "$sample_count" \
        --poison_ratio 1 \
        --label_smoothing_factor 0 \
        --benign_dataset "$CTRAP_DATA_DIR/sst2.json" \
        --lamb "$ctrap_lambda" \
        --alternating single_lora
)

(
    cd "$ctrap_dir/poison/evaluation"
    "$ctrap_env/bin/python" pred.py \
        --model_folder "$base_model" \
        --lora_folder "$attack_path" \
        --output_path "$eval_dir/generations.json" \
        --cache_dir "$HF_HOME"
    "$ctrap_env/bin/python" eval_sentiment.py \
        --input_path "$eval_dir/generations.json" \
        --cache_dir "$HF_HOME"
)

(
    cd "$ctrap_dir/sst2"
    "$ctrap_env/bin/python" pred_eval.py \
        --model_folder "$base_model" \
        --lora_folder "$attack_path" \
        --output_path "$eval_dir/sst2.json" \
        --dataset_path "$CTRAP_DATA_DIR/sst2_validation.json" \
        --cache_dir "$HF_HOME"
)

ctrap_revision=$(git -C "$ctrap_dir" rev-parse HEAD)
"$ctrap_env/bin/python" scripts/misc/parse_ctrap_results.py \
    --method "$method" \
    --sample-count "$sample_count" \
    --sentiment-eval "$eval_dir/generations.json_sentiment_eval.json" \
    --sst2-eval "$eval_dir/sst2.json" \
    --output "$eval_dir/results.json" \
    --model "$base_model" \
    --ctrap-revision "$ctrap_revision"
