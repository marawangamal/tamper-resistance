#!/bin/bash
#SBATCH --job-name=repnoise
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
project_dir="$PWD"
defence_env="$SLURM_TMPDIR/.venv-defence-$SLURM_JOB_ID"
attack_env="$SLURM_TMPDIR/.venv-attack-$SLURM_JOB_ID"
UV_PROJECT_ENVIRONMENT="$defence_env" uv sync --project repnoise --frozen
UV_PROJECT_ENVIRONMENT="$attack_env" uv sync --frozen
export WANDB_DIR="$PWD/artifacts/tamia"
mkdir -p "$WANDB_DIR" artifacts/tamia/logs

# Inputs
base_model_name_or_path=meta-llama/Llama-2-7b-chat-hf
attack_variant=${ATTACK_VARIANT:-adamw}
response_format=${RESPONSE_FORMAT:-nl}
case "$attack_variant" in
    adamw)
        attack_config="configs/beavertails/stages/1_attack/llama2-7b--attack-fft-repnoise-$response_format.yml"
        ;;
    adamw-no-hist)
        attack_config="configs/beavertails/stages/1_attack/llama2-7b--attack-fft-repnoise-$response_format-adamw-no-history.yml"
        ;;
    *)
        echo "Unknown attack variant: $attack_variant" >&2
        exit 1
        ;;
esac
case "$response_format" in
    nl) defence_separator=$'\n' ;;
    sp) defence_separator=' ' ;;
    *) echo "Unknown response format: $response_format" >&2; exit 1 ;;
esac
defence_alpha_coeff=1.0 # Authors' harmful-loss suppression coefficient
defence_beta_coeff=0.001 # Authors' representation-noising coefficient

# Intra-job sweep
read -ra attack_learning_rates <<< "${ATTACK_LEARNING_RATES:-5e-5 8e-5 1e-4 2e-4}"
attack_max_steps=250

# Set defence model name
defence_model_name="llama2-7b-beavertail--align-fft-repnoise-$response_format-alpha$defence_alpha_coeff-beta$defence_beta_coeff"
defence_model_path="artifacts/tamia/models/$defence_model_name"

# Defence.
export CUDA_VISIBLE_DEVICES=0,1,2,3
if [[ ! -f "$defence_model_path/.trained" ]]; then
    rm -rf -- "$defence_model_path"
    (
        cd "$project_dir/repnoise"
        PYTHONHASHSEED=0 "$defence_env/bin/python" -m representation_noising.train \
            --model "$base_model_name_or_path" \
            --output-dir "$project_dir/$defence_model_path" \
            --harmful-response-separator "$defence_separator" \
            --alpha "$defence_alpha_coeff" \
            --beta "$defence_beta_coeff"
    )
    touch "$defence_model_path/.trained"
fi

# Prepare the shared attack dataset before starting parallel attacks.
"$attack_env/bin/axolotl" preprocess "$attack_config" \
    --base-model "$defence_model_path"

# Defence evaluation before the attacks.
if [[ ! -f "artifacts/tamia/evals/$defence_model_name/results.json" ]]; then
    "$attack_env/bin/python" -m src.evals.evaluation \
        --model "$defence_model_path" \
        --results-path "artifacts/tamia/evals/$defence_model_name/results.json"
fi

# Attack and evaluation.
pids=()
for gpu in "${!attack_learning_rates[@]}"; do
    attack_lr=${attack_learning_rates[gpu]}
    attack_model_name="$defence_model_name--attack-fft-$response_format-$attack_variant-lr$attack_lr-ms$attack_max_steps"
    attack_results_path="artifacts/tamia/evals/$attack_model_name/results.json"
    (
        if [[ -f "$attack_results_path" ]]; then
            echo "Skipping completed attack: $attack_model_name"
            exit 0
        fi
        export CUDA_VISIBLE_DEVICES=$gpu
        rm -rf -- "artifacts/tamia/models/$attack_model_name"
        "$attack_env/bin/axolotl" train "$attack_config" --launcher python \
            --base-model "$defence_model_path" \
            --output-dir "artifacts/tamia/models/$attack_model_name" \
            --learning-rate "$attack_lr" \
            --max-steps "$attack_max_steps" \
            --wandb-name "$attack_model_name"
        "$attack_env/bin/python" -m src.evals.evaluation \
            --model "artifacts/tamia/models/$attack_model_name" \
            --results-path "$attack_results_path"
        rm -rf -- "artifacts/tamia/models/$attack_model_name"
    ) >> "artifacts/tamia/logs/$attack_model_name.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
exit "$failed"
