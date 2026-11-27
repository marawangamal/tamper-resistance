#!/bin/bash
#SBATCH --job-name=tarv2-attack
#SBATCH --output=artifacts/tamia/logs/tarv2-attack-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --array=0-15
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
artifacts_dir="$PWD/artifacts"
export WANDB_DIR="$artifacts_dir/tamia"
mkdir -p "$artifacts_dir/tamia/models" "$artifacts_dir/tamia/evals"
UV_PROJECT_ENVIRONMENT="$attack_env" uv sync --frozen

# Inputs
base_model_name_or_path="lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2"
tokenizer_name_or_path="NousResearch/Meta-Llama-3-8B-Instruct"
base_model_name="llama3-8b-tar-bio-v2"
seed=1

# Inter-job sweep
attack_variants=(
    adamw
    adamw-no-history
    new-config-1-adamw
    new-config-1-adamw-no-history
)
attack_configs=(
    configs/wmdp/stages/1_relearn/llama3-8b-tar-bio-v2--attack-fft-adamw.yml
    configs/wmdp/stages/1_relearn/llama3-8b-tar-bio-v2--attack-fft-adamw-no-history.yml
    configs/wmdp/stages/1_relearn/llama3-8b-tar-bio-v2--attack-fft-new-config-1-adamw.yml
    configs/wmdp/stages/1_relearn/llama3-8b-tar-bio-v2--attack-fft-new-config-1-adamw-no-history.yml
)
attack_max_steps=(10 50 100 1000)
index=${SLURM_ARRAY_TASK_ID:?submit with sbatch}
optimizer_index=$((index / ${#attack_max_steps[@]}))
steps_index=$((index % ${#attack_max_steps[@]}))
attack_variant=${attack_variants[optimizer_index]}
attack_config=${attack_configs[optimizer_index]}
max_steps=${attack_max_steps[steps_index]}
attack_model_name="$base_model_name--attack-fft-$attack_variant-lr2e-5-ms$max_steps-s$seed"
attack_model_dir="$artifacts_dir/tamia/models/$attack_model_name"
attack_results_dir="$artifacts_dir/tamia/evals/$attack_model_name"

# Resumption handling.
if find "$attack_results_dir" -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; then
    echo "Skipping completed attack: $attack_model_name"
    exit 0
fi

# Prepare the dataset once for both array tasks.
preprocess_dir="$artifacts_dir/tamia/cache/axolotl/llama3-8b-tar-bio-v2--attack-fft-pile-bio"
mkdir -p "$preprocess_dir"
(
    flock 9
    if [[ ! -f "$preprocess_dir/complete" ]]; then
        "$attack_env/bin/axolotl" preprocess "${attack_configs[0]}"
        touch "$preprocess_dir/complete"
    fi
) 9>"$preprocess_dir/lock"

# Evaluate the defended checkpoint once before attacking it.
if (( index == 0 )); then
    base_results_dir="$artifacts_dir/tamia/evals/$base_model_name"
    if ! find "$base_results_dir" -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; then
        "$attack_env/bin/accelerate" launch \
            --multi_gpu \
            --num_processes 4 -m lm_eval run \
            --model hf \
            --model_args "pretrained=$base_model_name_or_path,tokenizer=$tokenizer_name_or_path,dtype=bfloat16,add_bos_token=False" \
            --tasks wmdp_bio \
            --batch_size 32 \
            --output_path "$base_results_dir"
    fi
fi

# Restart only this incomplete attack.
rm -rf -- "$attack_model_dir" "$attack_results_dir"
"$attack_env/bin/axolotl" train "$attack_config" --launcher python \
    --output-dir "$attack_model_dir" \
    --seed "$seed" \
    --data-seed "$seed" \
    --max-steps "$max_steps" \
    --wandb-name "$attack_model_name"

# Paper-matched WMDP-Bio evaluation: zero-shot, no chat template, no BOS.
"$attack_env/bin/accelerate" launch \
    --multi_gpu \
    --num_processes 4 -m lm_eval run \
    --model hf \
    --model_args "pretrained=$attack_model_dir,tokenizer=$tokenizer_name_or_path,dtype=bfloat16,add_bos_token=False" \
    --tasks wmdp_bio \
    --batch_size 32 \
    --output_path "$attack_results_dir"

rm -rf -- "$attack_model_dir"
