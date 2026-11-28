"""Download final SEAM defence objectives for Pareto plots."""

import argparse
import os
import os.path as osp

import pandas as pd
import wandb


METRICS = ["loss_ul", "loss_up", "Cosine Similarity"]
SP_RUNS = {
    (0.01, 0.01): "hyzfcr01",
    (0.01, 0.1): "yf47rgx5",
    (0.01, 1.0): "sflczlgq",
    (0.01, 5.0): "mj23ib2t",
    (0.1, 0.01): "xestbiwy",
    (0.1, 0.1): "pb7p16rx",
    (0.1, 1.0): "ss1x0mk4",
    (0.1, 5.0): "b2uwdh8n",
    (1.0, 0.01): "jc61hfjq",
    (1.0, 0.1): "8uznyd4j",
    (1.0, 1.0): "i5h6o5mi",
    (1.0, 5.0): "43atnd6l",
    (5.0, 0.01): "e3676mi3",
    (5.0, 0.1): "7jpkmg62",
    (5.0, 1.0): "nl0mpwws",
    (5.0, 5.0): "eq2bocdl",
}
NL_RUN = {"id": "932nukua", "sd": 0.01, "up": 1.0}


def expected_name(response_format, sd, up):
    if response_format == "SP":
        return (
            "llama2-7b-beavertail--align-fft-seam--def-sp"
            f"--coeff-sd{sd}--coeff-up{up}"
        )
    return f"llama2-7b-beavertail--align-fft-seam-nl-sd{sd}-up{up}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="marawan-gamal")
    parser.add_argument("--project", default="tamper-resistance")
    parser.add_argument("--tail-steps", type=int, default=10)
    parser.add_argument(
        "--outpath",
        default="artifacts/tamia/seam-defence-pareto.csv",
    )
    args = parser.parse_args()
    if not 1 <= args.tail_steps <= 500:
        parser.error("--tail-steps must be between 1 and 500")

    configs = [
        {"id": run_id, "format": "SP", "sd": sd, "up": up}
        for (sd, up), run_id in SP_RUNS.items()
    ]
    configs.append({"format": "NL", **NL_RUN})

    api = wandb.Api(timeout=60)
    rows = []
    for cfg in configs:
        run = api.run(f"{args.entity}/{args.project}/{cfg['id']}")
        name = expected_name(cfg["format"], cfg["sd"], cfg["up"])
        if run.name != name:
            raise ValueError(
                f"Run {cfg['id']} is named {run.name!r}, expected {name!r}"
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

        tail = observations[-args.tail_steps :]
        row = {
            "Run": (
                f"{cfg['format']}, sd={cfg['sd']:g}, up={cfg['up']:g}"
            ),
            "Run ID": cfg["id"],
            "Format": cfg["format"],
            "sd": cfg["sd"],
            "up": cfg["up"],
            "Tail steps": args.tail_steps,
            "Final W&B step": observations[-1]["_step"],
        }
        for metric in METRICS:
            row[metric] = sum(observation[metric] for observation in tail) / len(tail)
            row[f"Final {metric}"] = observations[-1][metric]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["Format", "sd", "up"])
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    df.to_csv(args.outpath, index=False)
    print(f"{args.outpath}: {len(df)} rows")
