# Tamper Resistance

## Setup

With Python 3.12 and uv installed:

```sh
git clone --recurse-submodules https://github.com/marawangamal/tamper-resistance.git
cd tamper-resistance
uv sync --frozen
```

For an existing checkout, initialize submodules with `git submodule update --init --recursive`.

## Run defences

Run from the repository root on Tamia:

```sh
mkdir -p artifacts/tamia/logs

# HPO for SEAM
sbatch scripts/slurm/tamia/2a_seam_run_defence_sweep.sbatch \
  configs/beavertails/stages/1_attack/llama2-7b--attack-fft.yml

# Set the best coefficients in 2b, then run defence, attacks, and evaluations
sbatch scripts/slurm/tamia/2b_seam_run_defence_attack_eval.sbatch \
  configs/beavertails/stages/1_attack/llama2-7b--attack-fft.yml
```
