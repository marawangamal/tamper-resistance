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

# SEAM with cross-source gradient-cosine tracking (sp/sp and nl/nl)
sbatch scripts/slurm/tamia/2c_seam_run_defence_attack_eval_cosim.sbatch \
  configs/beavertails/stages/1_attack/llama2-7b--attack-fft-seam-sp.yml space
sbatch scripts/slurm/tamia/2c_seam_run_defence_attack_eval_cosim.sbatch \
  configs/beavertails/stages/1_attack/llama2-7b--attack-fft-seam-nl.yml newline

# Set the best coefficients in 2b, then run defence, attacks, and evaluations
# sbatch scripts/slurm/tamia/2b_seam_run_defence_attack_eval.sbatch \
#   configs/beavertails/seam/train-paper.yml # defence config
#   configs/beavertails/stages/1_attack/llama2-7b--attack-fft.yml # attack config
#   llama2-7b--align-fft-seam-nl
#   llama2-7b--align-fft-seam-nl--attack-fft-seam-
```

2c trains on mixed microbatches of four harmful and four Alpaca examples. It logs
`avg_of_cosims`, the average gradient cosine over the 16 harmful–Alpaca pairs,
to offline W&B and the attack stdout logs. `SEAMAttackTrainer` computes
this metric without finite differences; only harmful examples contribute to the
training loss. It also logs harmful and benign losses. FP32 attacks span all four
GPUs and run sequentially. Defence and attack runs log offline to the
`tamper-resistance` W&B project. Attack names append
`--attack-fft-cosim-{sp,nl}-lr{LR}-ms{STEPS}` to the defence name. Evaluation uses the
full utility sets and saves `results.json`.

The attack selects the first 1,000 harmful examples and first 1,000 Alpaca
examples. `shuffle_before_merging_datasets: true` shuffles each selected subset
once during preprocessing, using the configured seed. The balanced sampler
preserves four harmful and four benign examples per batch.

Use `response_separator: space` or `response_separator: newline` in each BeaverTails
attack dataset's `message_property_mappings`. The optional final argument to the
SEAM and RepNoise scripts independently selects the defence format: `space`
(default) or `newline`. Attack datasets also default to `space` when the separator
is omitted. Artifact names retain `sp`/`nl` labels.
