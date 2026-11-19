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

# Training and evaluation need separate environments.
UV_PROJECT_ENVIRONMENT=.venv-train uv sync --frozen --group train
UV_PROJECT_ENVIRONMENT=.venv-eval uv sync --frozen --group eval
```

Run commands from this repository root. The project can also be checked out as
the `examples/tamper-resistance-v2` submodule in lilpipe; use that directory as
the working directory in that case. No parent lilpipe checkout is required.

## Experiments

Defence configs are in `configs/stages/0_unlearn/`, relearning configs are in
`configs/stages/1_relearn/`, and W&B sweep configs are in `configs/sweeps/`.
See [hyperparameter searches and results](docs/hpo.md) for settings, selection
criteria, and sweep usage.

For a fixed GD+SAM defence and relearning sweep on Tamia:

```bash
mkdir -p artifacts/tamia/logs
sbatch --partition=gpubase_bynode_b1 \
  scripts/slurm/tamia/run_unlearn_relearn_eval.sbatch \
  configs/stages/0_unlearn/z7b-wmdp-bio--unlearn-fft-gd-sam-rho1e-2.yml \
  configs/stages/1_relearn/z7b-wmdp-bio--relearn-20.yml \
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

Compare the selected SEAM and RepNoise defenses across attack steps and LRs:

```bash
python scripts/misc/plot_seam_repnoise_trajectory.py
```

## Tests

```bash
uv run --frozen --group test pytest tests
```
