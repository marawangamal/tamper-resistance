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

# [SDD] After caching mremila/sdd-beavertails-alpaca on the Tamia login node,
# run SFT, the pre-attack evaluation, and the SEAM-matched attack sweep.
sbatch scripts/slurm/tamia/5a_sdd_run_defence_attack_eval.sbatch
```

Tamia compute nodes run offline. Before submitting SDD, load the dataset once on
the login node with `HF_HOME=$SCRATCH/.cache/huggingface` and the project's Python
environment. The base model, attack dataset, evaluation models, evaluation
datasets, and uv packages must also be available in their shared caches.
