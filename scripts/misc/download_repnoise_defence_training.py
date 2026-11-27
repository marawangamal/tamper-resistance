"""Download selected RepNoise defence trajectories from Weights & Biases."""

import argparse
import os
import os.path as osp

import pandas as pd
import wandb


METRICS = [
    "loss",
    "loss_harmful",
    "loss_harmful_log",
    "loss_retain",
    "loss_noise",
]
RUNS = [
    {
        "id": "0ikihktj",
        "name": (
            "llama2-7b-beavertail--align-fft-repnoise-nl-"
            "alpha1.0-beta0.001--loss-tracking"
        ),
        "label": "NL",
    },
    {
        "id": "j1bzn1cs",
        "name": (
            "llama2-7b-beavertail--align-fft-repnoise-sp-"
            "alpha1.0-beta0.001--loss-tracking"
        ),
        "label": "SP",
    },
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="marawan-gamal")
    parser.add_argument("--project", default="tamper-resistance")
    parser.add_argument(
        "--outpath",
        default="artifacts/tamia/repnoise-defence-training.csv",
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
        if len(observations) != 1250:
            raise ValueError(
                f"Run {cfg['id']} has {len(observations)} complete observations, "
                "expected 1250"
            )

        for observation in observations:
            for metric in METRICS:
                rows.append(
                    {
                        "Run": cfg["label"],
                        "Run ID": cfg["id"],
                        "Training step": observation["_step"],
                        "Metric": metric,
                        "Value": observation[metric],
                    }
                )

    df = pd.DataFrame(rows)
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    df.to_csv(args.outpath, index=False)
    print(f"{args.outpath}: {len(df)} rows")
