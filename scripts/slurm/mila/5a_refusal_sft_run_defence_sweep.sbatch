#!/bin/bash
#SBATCH --job-name=refusal-sft-sweep
#SBATCH --output=artifacts/mila/logs/refusal-sft-sweep-%A_%a.out
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4%4
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_LINK_MODE=copy
export TOKENIZERS_PARALLELISM=false
export AXOLOTL_DO_NOT_TRACK=1
export WANDB_MODE=disabled

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================

defence_env="$SLURM_TMPDIR/.venv-defence-$SLURM_JOB_ID"
attack_env="$SLURM_TMPDIR/.venv-attack-$SLURM_JOB_ID"
artifacts_dir="$SCRATCH/tamper-resistance/artifacts"
cache_dir="$SCRATCH/tamper-resistance/cache/axolotl"
attack_prepared_path="$cache_dir/llama2-7b--attack-fft-sp-response-only/prepared"
mkdir -p "$artifacts_dir/mila/models" "$artifacts_dir/mila/evals" \
    "$cache_dir" artifacts/mila/logs
sync_lock="$UV_CACHE_DIR/refusal-sft-sync.lock"
until mkdir "$sync_lock" 2>/dev/null; do sleep 1; done
trap 'rmdir "$sync_lock"' EXIT
UV_PROJECT_ENVIRONMENT="$defence_env" uv sync --frozen
UV_PROJECT_ENVIRONMENT="$attack_env" uv sync --frozen
rmdir "$sync_lock"
trap - EXIT

base_defence_config="configs/beavertails/stages/0_align/llama2-7b--align-fft-refusal-sft-sp.yml"
base_attack_config="configs/beavertails/stages/1_attack/llama2-7b--attack-fft-seam-sp-adamw-no-history.yml"
defence_sample_counts=(100 500 1000 4000 8000)
attack_learning_rates=(5e-5 8e-5 1e-4 2e-4)
attack_max_steps=(5 10 50 100)

index=${SLURM_ARRAY_TASK_ID:?submit with sbatch}
defence_samples=${defence_sample_counts[index]}
defence_model_name="llama2-7b--align-fft-refusal-sft-sp-n$defence_samples"
defence_model_path="$artifacts_dir/mila/models/$defence_model_name"
defence_results_path="$artifacts_dir/mila/evals/$defence_model_name/results.json"
defence_config="$SLURM_TMPDIR/refusal-sft-n$defence_samples.yml"
attack_config="$SLURM_TMPDIR/refusal-sft-attack-n$defence_samples.yml"

skip_defence=true
if [[ ! -f "$defence_results_path" ]]; then
    skip_defence=false
fi
for max_steps in "${attack_max_steps[@]}"; do
    for attack_lr in "${attack_learning_rates[@]}"; do
        attack_model_name="$defence_model_name--attack-fft-sp-adamw-no-hist-lr$attack_lr-ms$max_steps"
        if [[ ! -f "$artifacts_dir/mila/evals/$attack_model_name/results.json" ]]; then
            skip_defence=false
        fi
    done
done
if "$skip_defence"; then
    echo "Skipping completed defence: $defence_model_name"
    exit 0
fi

rm -rf -- "$defence_model_path" "$artifacts_dir/mila/evals/$defence_model_name"
for max_steps in "${attack_max_steps[@]}"; do
    for attack_lr in "${attack_learning_rates[@]}"; do
        attack_model_name="$defence_model_name--attack-fft-sp-adamw-no-hist-lr$attack_lr-ms$max_steps"
        rm -rf -- "$artifacts_dir/mila/models/$attack_model_name" \
            "$artifacts_dir/mila/evals/$attack_model_name"
    done
done

"$defence_env/bin/python" - \
    "$base_defence_config" "$base_attack_config" \
    "$defence_config" "$attack_config" "$defence_samples" \
    "$defence_model_path" "$defence_model_name" "$cache_dir" \
    "$attack_prepared_path" <<'PY'
import sys

import yaml

(
    defence_source,
    attack_source,
    defence_destination,
    attack_destination,
    samples,
    output_dir,
    wandb_name,
    cache_dir,
    attack_prepared_path,
) = sys.argv[1:]
with open(defence_source) as handle:
    defence = yaml.safe_load(handle)
defence["datasets"][0]["message_property_mappings"]["max_samples"] = samples
defence["dataset_prepared_path"] = (
    f"{cache_dir}/llama2-7b--align-fft-refusal-sft-sp-n{samples}/prepared"
)
defence["output_dir"] = output_dir
defence["wandb_name"] = wandb_name
with open(defence_destination, "w") as handle:
    yaml.safe_dump(defence, handle, sort_keys=False)

with open(attack_source) as handle:
    attack = yaml.safe_load(handle)
attack["dataset_prepared_path"] = attack_prepared_path
with open(attack_destination, "w") as handle:
    yaml.safe_dump(attack, handle, sort_keys=False)
PY

attack_cache_lock="$cache_dir/refusal-sft-attack-preprocess.lock"
until mkdir "$attack_cache_lock" 2>/dev/null; do sleep 1; done
trap 'rmdir "$attack_cache_lock"' EXIT
if [[ ! -f "$attack_prepared_path/.complete" ]]; then
    "$attack_env/bin/axolotl" preprocess "$attack_config"
    touch "$attack_prepared_path/.complete"
fi
rmdir "$attack_cache_lock"
trap - EXIT

PYTHONHASHSEED=0 "$defence_env/bin/axolotl" train "$defence_config" \
    --launcher python

"$attack_env/bin/python" -m src.evals.evaluation \
    --model "$defence_model_path" \
    --results-path "$defence_results_path"

for max_steps in "${attack_max_steps[@]}"; do
    for attack_lr in "${attack_learning_rates[@]}"; do
        attack_model_name="$defence_model_name--attack-fft-sp-adamw-no-hist-lr$attack_lr-ms$max_steps"
        (
            "$attack_env/bin/axolotl" train "$attack_config" --launcher python \
                --base-model "$defence_model_path" \
                --output-dir "$artifacts_dir/mila/models/$attack_model_name" \
                --learning-rate "$attack_lr" \
                --max-steps "$max_steps" \
                --wandb-name "$attack_model_name"
            "$attack_env/bin/python" -m src.evals.evaluation \
                --model "$artifacts_dir/mila/models/$attack_model_name" \
                --results-path "$artifacts_dir/mila/evals/$attack_model_name/results.json"
            rm -rf -- "$artifacts_dir/mila/models/$attack_model_name"
        ) >> "artifacts/mila/logs/$attack_model_name.log" 2>&1
    done
done
rm -rf -- "$defence_model_path"
