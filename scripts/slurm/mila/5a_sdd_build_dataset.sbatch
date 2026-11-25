#!/bin/bash
#SBATCH --job-name=sdd-data
#SBATCH --output=artifacts/mila/logs/sdd-data-%j.out
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail

# Cluster setup: keep the environment and download caches on the allocated node.
: "${SLURM_TMPDIR:?SLURM_TMPDIR must be set on the compute node}"
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/sdd-env"
export UV_CACHE_DIR="$SLURM_TMPDIR/uv-cache"
export HF_HOME="$SLURM_TMPDIR/huggingface"
export TOKENIZERS_PARALLELISM=false

# Build 8K prompt-answer pairs from BeaverTails and Alpaca.
uv sync --project sdd --frozen
"$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is unavailable"; print(torch.cuda.get_device_name(0), flush=True)'
output_dir="$SCRATCH/tamper-resistance/data/sdd"
mkdir -p "$output_dir"
"$UV_PROJECT_ENVIRONMENT/bin/python" sdd/toxity_dataset_construction.py \
    --save_dir "$output_dir"
