# Tamper Resistance

Unlearning and relearning experiments for Zephyr-7B on WMDP-Bio, including GD,
NPO, SAM, and GMA. This standalone project was extracted from
[`lilpipe/examples/tamper-resistance-v2`](https://github.com/marawangamal/lilpipe)
with its code, evaluation results, figures, and project history.

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Request access to
[`cais/wmdp-bio-forget-corpus`](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus)
before downloading the data.

```bash
git clone https://github.com/marawangamal/tamper-resistance.git
cd tamper-resistance

# Install training and evaluation dependencies.
uv sync --frozen
```

Run commands from this repository root. The project can also be checked out as
the `examples/tamper-resistance-v2` submodule in lilpipe; use that directory as
the working directory in that case. No parent lilpipe checkout is required.

## Experiments

### Reference implementations

The upstream implementations are pinned as submodules using our forks:

- [`seam/`](https://github.com/marawangamal/seam): SEAM.
- [`repnoise/`](https://github.com/marawangamal/representation-noising): RepNoise.

```bash
git submodule update --init seam repnoise
```

Follow each submodule's README for its own environment and experiment commands.

### Training and evaluation

Defence configs are in `configs/wmdp/stages/0_unlearn/`, relearning configs are in
`configs/wmdp/stages/1_relearn/`, and W&B sweep configs are in `configs/wmdp/sweeps/`.
See [hyperparameter searches and results](docs/hpo.md) for settings, selection
criteria, and sweep usage.

For a fixed GD+SAM defence and relearning sweep on Tamia:

```bash
mkdir -p artifacts/tamia/logs
sbatch --partition=gpubase_bynode_b1 \
  scripts/slurm/tamia/run_unlearn_relearn_eval.sbatch \
  configs/wmdp/stages/0_unlearn/z7b-wmdp-bio--unlearn-fft-gd-sam-rho1e-2.yml \
  configs/wmdp/stages/1_relearn/z7b-wmdp-bio--relearn-20.yml \
  z7b-gd-sam
```

The Slurm scripts create separate training and evaluation environments and use
offline caches on compute nodes. Review their cluster paths, resource requests,
and cache setup before submitting on another system. W&B sweeps use the
configured HTTP proxy to reach W&B.

## Artifacts

Models, evaluation JSONs, and Slurm logs use `artifacts/tamia/{models,results,logs}/`.
Plots use `artifacts/figures/` for PDFs and `artifacts/figures-png/` for PNGs.
Historical evaluations remain in `artifacts/tamia/results-in-distribution/`,
`results-out-of-distribution/`, and `results-header-ablation/`.

Training scripts remove model weights after successful evaluations. Evaluation
results and figures are versioned; environments, weights, logs, and caches are
ignored. A cluster checkout can symlink `artifacts` to shared project storage.

## Tests

```bash
uv run --frozen --group test pytest tests
```

## Run defences

```sh
# HPO for SEAM
bash scripts/slurm/tamia/2b_seam_run_defence_attack_eval.sbatch
# Run best SEAM config then run attacks then run evals
bash scripts/slurm/tamia/2b_seam_run_defence_attack_eval.sbatch
```
