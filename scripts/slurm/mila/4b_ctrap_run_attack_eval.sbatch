#!/bin/bash
#SBATCH --job-name=ctrap-attack
#SBATCH --output=artifacts/mila/logs/ctrap-attack-%A_%a.out
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-14
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
mkdir -p "$artifacts_dir/mila/models" "$artifacts_dir/mila/evals"
sync_lock="$UV_CACHE_DIR/ctrap-sync.lock"
until mkdir "$sync_lock" 2>/dev/null; do sleep 1; done
trap 'rmdir "$sync_lock"' EXIT
UV_PROJECT_ENVIRONMENT="$ctrap_env" uv sync --project ctrap --frozen
rmdir "$sync_lock"
trap - EXIT

# Inputs
base_model_name_or_path="meta-llama/Llama-2-7b-hf"
poison_ratio=1
attack_variant=${ATTACK_VARIANT:-paper}
attack_learning_rate=${ATTACK_LR:-1e-5}

case "$attack_variant" in
    paper)
        attack_variant_suffix=""
        optimizer_args=()
        ;;
    adamw-no-history)
        attack_variant_suffix="-adamw-no-history"
        optimizer_args=(--adam_beta1 0.0 --adam_beta2 0.0)
        ;;
    *)
        echo "Unknown attack variant: $attack_variant" >&2
        exit 1
        ;;
esac

# Inter-job sweep (3 methods x 5 sample counts = 15 tasks)
methods=(raw sft ctrap)
method_model_names=(
    "llama2-7b--raw"
    "llama2-7b--align-lora-sft"
    "llama2-7b--align-lora-ctrap-a0.1-l0.1"
)
sample_counts=(100 200 300 400 500)

# Set attack model name.
index=${SLURM_ARRAY_TASK_ID:?submit with sbatch}
method_index=$((index / ${#sample_counts[@]}))
method=${methods[method_index]}
method_model_name=${method_model_names[method_index]}
sample_count=${sample_counts[index % ${#sample_counts[@]}]}

attack_model_name="$method_model_name--attack-lora$attack_variant_suffix-lr$attack_learning_rate-pr$poison_ratio-n$sample_count"

# Resumption handling.
if [[ -f "$artifacts_dir/mila/evals/$attack_model_name/results.json" ]]; then
    echo "Skipping completed attack: $attack_model_name"
    exit 0
fi

lora_args=()
if [[ "$method" != raw ]]; then
    if [[ ! -f "$artifacts_dir/mila/evals/$method_model_name/defence-complete" ]]; then
        echo "Missing completed defence: $method_model_name" >&2
        exit 1
    fi
    lora_args=(--lora_folder "$artifacts_dir/mila/models/$method_model_name")
fi

# Attack.
(
    cd "$PWD/ctrap"
    "$ctrap_env/bin/python" train.py \
        --model_name_or_path "$base_model_name_or_path" \
        "${lora_args[@]}" \
        --data_path PKU-Alignment/BeaverTails_dangerous \
        --bf16 True \
        --output_dir "$artifacts_dir/mila/models/$attack_model_name" \
        --num_train_epochs 20 \
        --per_device_train_batch_size 10 \
        --per_device_eval_batch_size 10 \
        --save_steps 100000 \
        --learning_rate "$attack_learning_rate" \
        "${optimizer_args[@]}" \
        --weight_decay 0.1 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type constant \
        --logging_steps 10 \
        --tf32 True \
        --eval_steps 2000 \
        --optimizer normal \
        --evaluation_strategy steps \
        --sample_num "$sample_count" \
        --poison_ratio "$poison_ratio" \
        --benign_dataset "$CTRAP_DATA_DIR/sst2.json" \
        --lamb 0.1 \
        --alternating single_lora
)

# Native harmfulness evaluation.
(
    cd "$PWD/ctrap/poison/evaluation"
    "$ctrap_env/bin/python" pred.py \
        --model_folder "$base_model_name_or_path" \
        --lora_folder "$artifacts_dir/mila/models/$attack_model_name" \
        --output_path "$artifacts_dir/mila/evals/$attack_model_name/generations.json"
    "$ctrap_env/bin/python" eval_sentiment.py \
        --input_path "$artifacts_dir/mila/evals/$attack_model_name/generations.json"
)

# Native SST-2 evaluation.
(
    cd "$PWD/ctrap/sst2"
    "$ctrap_env/bin/python" pred_eval.py \
        --model_folder "$base_model_name_or_path" \
        --lora_folder "$artifacts_dir/mila/models/$attack_model_name" \
        --output_path "$artifacts_dir/mila/evals/$attack_model_name/sst2.json" \
        --dataset_path "$CTRAP_DATA_DIR/sst2_validation.json"
)

# Save structured metrics alongside the native outputs.
"$ctrap_env/bin/python" scripts/misc/parse_ctrap_results.py \
    --method "$method" \
    --sample-count "$sample_count" \
    --sentiment-eval "$artifacts_dir/mila/evals/$attack_model_name/generations.json_sentiment_eval.json" \
    --sst2-eval "$artifacts_dir/mila/evals/$attack_model_name/sst2.json" \
    --output "$artifacts_dir/mila/evals/$attack_model_name/results.json" \
    --model "$base_model_name_or_path" \
    --ctrap-revision "$(git -C "$PWD/ctrap" rev-parse HEAD)"
