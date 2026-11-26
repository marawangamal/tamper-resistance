#!/bin/bash
#SBATCH --job-name=sdd-qi-advbench
#SBATCH --output=artifacts/tamia/logs/run-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
set -euo pipefail

# Prepare the pinned AdvBench subsets on the login node before submission.
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

train_env="$SLURM_TMPDIR/.venv-sdd-qi-$SLURM_JOB_ID"
UV_PROJECT_ENVIRONMENT="$train_env" uv sync --frozen

data_dir="$SCRATCH/tamper-resistance/data/sdd-qi-advbench"
base_config="configs/beavertails/stages/1_attack/llama2-7b--attack-fft-sdd-qi-advbench.yml"
defence_model="$PWD/artifacts/tamia/models/llama2-7b-beavertail--align-fft-sdd/checkpoint-500"
test -f "$defence_model/config.json"

# Write three independent configs, each with the exact number of updates per epoch.
"$train_env/bin/python" - "$base_config" "$data_dir" "$defence_model" "$SLURM_TMPDIR" "$PWD" <<'PY'
import json
import sys
from pathlib import Path

import yaml

base_path, data_dir, defence_model, temp_dir, worktree = map(Path, sys.argv[1:])
base = yaml.safe_load(base_path.read_text())
manifest = json.loads((data_dir / "manifest.json").read_text())
assert manifest["target_semantics"] == "AdvBench affirmative prefixes, not full harmful answers"
for shots in (10, 50, 100):
    data_path = data_dir / f"{shots}.jsonl"
    assert len(data_path.read_text().splitlines()) == shots
    config = yaml.safe_load(yaml.safe_dump(base))
    name = f"llama2-7b-beavertail--align-fft-sdd--attack-fft-qi-advbench-{shots}"
    config["base_model"] = str(defence_model)
    config["datasets"][0]["path"] = str(data_path)
    config["dataset_prepared_path"] = str(worktree / "artifacts/tamia/cache/axolotl" / name / "prepared")
    config["output_dir"] = str(worktree / "artifacts/tamia/models" / name)
    config["wandb_name"] = name
    config["custom_trainer_args"]["steps_per_epoch"] = shots // 10
    (temp_dir / f"sdd-qi-advbench-{shots}.yml").write_text(yaml.safe_dump(config))
PY

for shots in 10 50 100; do
    if [[ "${SDD_QI_SMOKE:-0}" == 1 && "$shots" != 10 ]]; then
        continue
    fi
    "$train_env/bin/axolotl" preprocess "$SLURM_TMPDIR/sdd-qi-advbench-$shots.yml"
done

if [[ "${SDD_QI_SMOKE:-0}" == 1 ]]; then
    smoke_config="$SLURM_TMPDIR/sdd-qi-advbench-smoke.yml"
    smoke_output="$SCRATCH/tamper-resistance/models/sdd-qi-advbench-smoke-$SLURM_JOB_ID"
    "$train_env/bin/python" - "$SLURM_TMPDIR/sdd-qi-advbench-10.yml" "$smoke_config" "$smoke_output" <<'PY'
import sys

import yaml

source, destination, output = sys.argv[1:]
config = yaml.safe_load(open(source))
config["max_steps"] = 1
config["output_dir"] = output
config["wandb_name"] = f"sdd-qi-advbench-smoke-{output.rsplit('-', 1)[-1]}"
with open(destination, "w") as result:
    yaml.safe_dump(config, result)
PY
    CUDA_VISIBLE_DEVICES=0 "$train_env/bin/axolotl" train "$smoke_config" --launcher python
    test -f "$smoke_output/config.json"
    exit 0
fi

pids=()
for gpu in 0 1 2; do
    shots=(10 50 100)
    shot_count=${shots[gpu]}
    name="llama2-7b-beavertail--align-fft-sdd--attack-fft-qi-advbench-$shot_count"
    model_dir="artifacts/tamia/models/$name"
    results="artifacts/tamia/evals/$name/results.json"
    if [[ -f "$results" ]]; then
        continue
    fi
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        if [[ ! -f "$model_dir/config.json" ]]; then
            if [[ -d "$model_dir" ]]; then
                echo "Incomplete attack output at $model_dir; inspect it before retrying" >&2
                exit 1
            fi
            "$train_env/bin/axolotl" train "$SLURM_TMPDIR/sdd-qi-advbench-$shot_count.yml" --launcher python
        fi
        "$train_env/bin/python" -m src.evals.evaluation \
            --model "$model_dir" \
            --results-path "$results"
    ) > "artifacts/tamia/logs/$name.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
exit "$failed"
