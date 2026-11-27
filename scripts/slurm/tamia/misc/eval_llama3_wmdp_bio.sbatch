#!/bin/bash
#SBATCH --job-name=llama3-wmdp
#SBATCH --output=artifacts/tamia/logs/llama3-wmdp-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=00:30:00
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$SCRATCH/.cache/huggingface"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

eval_env="$SLURM_TMPDIR/.venv-eval-$SLURM_JOB_ID"
results_dir="$PWD/artifacts/tamia/evals/llama3-8b-instruct"
mkdir -p "$results_dir"

if find "$results_dir" -type f -name 'results_*.json' -print -quit | grep -q .; then
    echo "Skipping completed evaluation"
    exit 0
fi

UV_PROJECT_ENVIRONMENT="$eval_env" uv sync --frozen
"$eval_env/bin/accelerate" launch \
    --multi_gpu \
    --num_processes 4 -m lm_eval run \
    --model hf \
    --model_args "pretrained=NousResearch/Meta-Llama-3-8B-Instruct,tokenizer=NousResearch/Meta-Llama-3-8B-Instruct,dtype=bfloat16,add_bos_token=False" \
    --tasks wmdp_bio \
    --batch_size 32 \
    --output_path "$results_dir"
