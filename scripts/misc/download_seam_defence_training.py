"""Download selected SEAM defence trajectories from Weights & Biases."""

import argparse
import os
import os.path as osp

import pandas as pd
import wandb


METRICS = ["loss_ul", "loss_up", "Cosine Similarity"]
RUNS = [
    {
        "id": "oane0nwa",
        "name": "llama2-7b-beavertail--align-fft-seam-sp-sd0.1-up1.0",
        "label": "SP, β=0.1",
    },
    {
        "id": "esoo6pa9",
        "name": "llama2-7b-beavertail--align-fft-seam-sp-sd1.0-up1.0",
        "label": "SP, β=1.0",
    },
    {
        "id": "932nukua",
        "name": "llama2-7b-beavertail--align-fft-seam-nl-sd0.01-up1.0",
        "label": "NL, β=0.01",
    },
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="marawan-gamal")
    parser.add_argument("--project", default="tamper-resistance")
    parser.add_argument(
        "--outpath",
        default="artifacts/tamia/seam-defence-training.csv",
    )
    args = parser.parse_args()

    api = wandb.Api(timeout=60)
    rows = []
    for cfg in RUNS:
        run = api.run(f"{args.entity}/{args.project}/{cfg['id']}")
        if run.name != cfg["name"]:
            raise ValueError(
                f"Run {cfg['id']} is named {run.name!r}, expected {cfg['name']!r}"
            )

        history = run.scan_history(keys=["_step", *METRICS], page_size=1000)
        observations = [
            observation
            for observation in history
            if all(observation.get(metric) is not None for metric in METRICS)
        ]
        if len(observations) != 500:
            raise ValueError(
                f"Run {cfg['id']} has {len(observations)} complete observations, "
                "expected 500"
            )

        for training_step, observation in enumerate(observations, start=1):
            for metric in METRICS:
                rows.append(
                    {
                        "Run": cfg["label"],
                        "Run ID": cfg["id"],
                        "Training step": training_step,
                        "W&B step": observation["_step"],
                        "Metric": metric,
                        "Value": observation[metric],
                    }
                )

    df = pd.DataFrame(rows)
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    df.to_csv(args.outpath, index=False)
    print(f"{args.outpath}: {len(df)} rows")
