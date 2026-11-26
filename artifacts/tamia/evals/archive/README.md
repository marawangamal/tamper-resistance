# Archived evaluations

These directories contain superseded or diagnostic runs and are not used by the
active Llama plotting scripts.

- `divergent-attack-train-on-inputs`: attacks affected by the prompt-token masking mismatch.
- `pre-final-cosim-metrics`: attacks produced before the final cosine metrics were added.
- `weight-decay-mismatch`: no-history attack run with unmatched weight decay.
- `legacy-hpo-attack`: earlier SEAM HPO runs using the old attack setup.
- `in-distribution`, `out-of-distribution`, and `header-ablation`: earlier Zephyr experiments retained for their dedicated plots.

An archived run may share a model name with an active run. The active copy under
`artifacts/tamia/evals/` is the current result; the archive directory records why
the older copy was superseded.
