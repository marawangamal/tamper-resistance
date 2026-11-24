# Tamper Resistance

## Setup

```sh
git clone --recurse-submodules https://github.com/marawangamal/tamper-resistance.git
cd tamper-resistance
uv sync --frozen
```

## Quickstart

```sh
mkdir -p artifacts/tamia/logs

# [SEAM] Run Hyperparameter sweep
preprocess_job=$(sbatch --parsable \
  scripts/slurm/tamia/misc/preprocess_seam_hpo_attack.sbatch)
sbatch --dependency="afterok:$preprocess_job" \
  scripts/slurm/tamia/2a_seam_run_defence_sweep.sbatch

# [SEAM] Run best defence, followed by attacks and evals
sbatch scripts/slurm/tamia/2b_seam_run_defence_attack_eval.sbatch

# [SEAM] Run best defence, followed by attacks and evals.
# Additionally, log cosims during attack which is more expensive than
# the previous command.
sbatch scripts/slurm/tamia/2c_seam_run_defence_attack_eval_cosim.sbatch \
  configs/beavertails/stages/1_attack/llama2-7b--attack-fft-seam-sp.yml space

# [CTRAP] Cache the authors' datasets and models from a login node
HF_HOME="$SCRATCH/.cache/huggingface" \
  uv run python scripts/misc/prepare_ctrap_data.py

# [CTRAP] Verify the authors' defence fits one Mila L40S
sbatch --time=00:30:00 --export=ALL,CTRAP_MAX_STEPS=1 \
  scripts/slurm/mila/4a_ctrap_run_defence.sbatch

# [CTRAP] Run the full paper-default defence, then the raw/CTRAP attack matrix
defence_job=$(sbatch --parsable \
  scripts/slurm/mila/4a_ctrap_run_defence.sbatch)
sbatch --dependency="afterok:$defence_job" \
  scripts/slurm/mila/4b_ctrap_run_attack_eval.sbatch
```
