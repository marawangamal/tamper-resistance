#!/bin/bash
#SBATCH --job-name=sdd-chat
#SBATCH --output=artifacts/tamia/logs/run-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
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
export WANDB_DIR="$PWD/artifacts/tamia"

train_env="$SLURM_TMPDIR/.venv-sdd-chat-$SLURM_JOB_ID"
UV_PROJECT_ENVIRONMENT="$train_env" uv sync --frozen

defence_config="configs/beavertails/stages/0_align/llama2-7b--align-fft-sdd-chat.yml"
attack_config="configs/beavertails/stages/1_attack/llama2-7b--attack-fft-sdd-chat.yml"
defence_model_name="llama2-7b-beavertail--align-fft-sdd-chat"
defence_model_dir="artifacts/tamia/models/$defence_model_name"
defence_model_path="$defence_model_dir/checkpoint-500"
defence_results="artifacts/tamia/evals/$defence_model_name/results.json"
attack_learning_rates=(5e-5 8e-5 1e-4 2e-4)
attack_max_steps=(5 10 50 100)

"$train_env/bin/python" -c 'from datasets import load_dataset; assert len(load_dataset("mremila/sdd-beavertails-alpaca", split="train")) == 8000'

if [[ "${SDD_CHAT_SMOKE:-0}" == 1 ]]; then
    smoke_dir="$SCRATCH/tamper-resistance/models/sdd-chat-smoke-$SLURM_JOB_ID"
    smoke_defence="$smoke_dir/defence"
    smoke_attack="$smoke_dir/attack"
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    "$train_env/bin/axolotl" train "$defence_config" --launcher python \
        --output-dir "$smoke_defence" \
        --max-steps 1 \
        --wandb-name "sdd-chat-defence-smoke-$SLURM_JOB_ID"
    test -f "$smoke_defence/config.json"
    export CUDA_VISIBLE_DEVICES=0
    "$train_env/bin/axolotl" train "$attack_config" --launcher python \
        --base-model "$smoke_defence" \
        --output-dir "$smoke_attack" \
        --max-steps 1 \
        --wandb-name "sdd-chat-attack-smoke-$SLURM_JOB_ID"
    test -f "$smoke_attack/config.json"
    "$train_env/bin/python" - "$smoke_attack" <<'PY'
import sys
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
roots = (
    Path("artifacts/tamia/cache/axolotl/llama2-7b--align-fft-sdd-chat/prepared"),
    Path("artifacts/tamia/cache/axolotl/llama2-7b--attack-fft-sdd-chat/prepared"),
)
for root in roots:
    datasets = [path.parent for path in root.glob("*/state.json")]
    prepared = max(datasets, key=lambda path: path.stat().st_mtime)
    row = load_from_disk(prepared)[0]
    first_label = next(i for i, label in enumerate(row["labels"]) if label != -100)
    prompt = tokenizer.decode(row["input_ids"][:first_label])
    assert "[INST]" in prompt and "[/INST]" in prompt, prompt
    assert "Question:" not in prompt and "Answer:" not in prompt, prompt
PY
    exit 0
fi

if [[ ! -f "$defence_model_path/config.json" ]]; then
    if [[ -d "$defence_model_dir" ]]; then
        echo "Incomplete defence output at $defence_model_dir; inspect it before retrying" >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    PYTHONHASHSEED=0 "$train_env/bin/axolotl" train "$defence_config" --launcher python \
        --output-dir "$defence_model_dir" \
        --wandb-name "$defence_model_name"
fi
test -f "$defence_model_path/config.json"

if [[ ! -f "$defence_results" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    "$train_env/bin/python" -m src.evals.evaluation \
        --model "$defence_model_path" \
        --results-path "$defence_results" \
        --prompt-format llama2
fi

"$train_env/bin/axolotl" preprocess "$attack_config" \
    --base-model "$defence_model_path"

for max_steps in "${attack_max_steps[@]}"; do
    pids=()
    for gpu in 0 1 2 3; do
        attack_lr=${attack_learning_rates[gpu]}
        attack_model_name="$defence_model_name--attack-fft-chat-adamw-no-hist-lr$attack_lr-ms$max_steps"
        attack_model_dir="artifacts/tamia/models/$attack_model_name"
        attack_results="artifacts/tamia/evals/$attack_model_name/results.json"
        if [[ -f "$attack_results" ]]; then
            continue
        fi
        (
            export CUDA_VISIBLE_DEVICES=$gpu
            if [[ ! -f "$attack_model_dir/config.json" ]]; then
                if [[ -d "$attack_model_dir" ]]; then
                    echo "Incomplete attack output at $attack_model_dir; inspect it before retrying" >&2
                    exit 1
                fi
                "$train_env/bin/axolotl" train "$attack_config" --launcher python \
                    --base-model "$defence_model_path" \
                    --output-dir "$attack_model_dir" \
                    --learning-rate "$attack_lr" \
                    --max-steps "$max_steps" \
                    --wandb-name "$attack_model_name"
            fi
            "$train_env/bin/python" -m src.evals.evaluation \
                --model "$attack_model_dir" \
                --results-path "$attack_results" \
                --prompt-format llama2
        ) > "artifacts/tamia/logs/$attack_model_name.log" 2>&1 &
        pids+=("$!")
    done

    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if (( failed )); then
        exit 1
    fi
done
