#!/bin/bash
#SBATCH --job-name=seam-preprocess
#SBATCH --output=artifacts/tamia/logs/run-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export AXOLOTL_DO_NOT_TRACK=1

attack_config="configs/beavertails/stages/1_attack/llama2-7b--attack-fft-seam-sp-adamw-no-history.yml"
attack_env="$SLURM_TMPDIR/.venv-attack-$SLURM_JOB_ID"

mkdir -p artifacts/tamia/logs
UV_PROJECT_ENVIRONMENT="$attack_env" uv sync --frozen
"$attack_env/bin/axolotl" preprocess "$attack_config"
