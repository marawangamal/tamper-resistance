"""Compare selected SEAM and RepNoise defenses across Axolotl attack steps."""

import argparse
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootdir", default="artifacts/tamia/results")
    parser.add_argument(
        "--attack-max-steps", nargs="+", type=int, default=[5, 10, 50, 100]
    )
    parser.add_argument(
        "--outpath",
        default="artifacts/figures/llama2-7b-beavertail--seam-repnoise--attack-steps--lr-trajectory.pdf",
    )
    args = parser.parse_args()

    configs = [
        {
            "path": "llama2-7b-beavertail--align-fft-seam"
            "--def-sp--coeff-sd1.0--coeff-up5.0",
            "name": r"SEAM (def=sp, att=sp), $\alpha=5.0,\ \beta=1.0$",
        },
        {
            "path": "llama2-7b-beavertail--align-fft-repnoise"
            "--def-sp--alpha0.1--beta0.0001",
            "name": r"RepNoise (def=sp, att=sp), $\alpha=0.1,\ \beta=0.0001$",
        },
    ]
    learning_rates = ["5e-5", "8e-5", "1e-4", "2e-4"]
    task_metrics = {
        "mmlu": "acc,none",
        "truthfulqa_mc1": "acc,none",
        "arc_easy": "acc_norm,none",
        "hellaswag": "acc_norm,none",
    }

    rows = []
    for cfg in configs:
        for steps in args.attack_max_steps:
            for lr in learning_rates:
                model = cfg["path"] + f"--relearn-fft-att-sp-lr{lr}-ms{steps}"
                with open(osp.join(args.rootdir, model, "results.json")) as handle:
                    result = json.load(handle)
                capability = sum(
                    result["post_utility"][task][metric]
                    for task, metric in task_metrics.items()
                ) / len(task_metrics)
                for evaluation, score in [
                    ("Harmfulness ↓", result["harmfulness"]),
                    ("Capability ↑", capability),
                ]:
                    rows.append(
                        {
                            "Method": cfg["name"],
                            "LR": lr,
                            "Attack steps": steps,
                            "Eval": evaluation,
                            "Score (%)": 100 * score,
                        }
                    )

    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    grid = sns.catplot(
        data=df,
        x="Attack steps",
        y="Score (%)",
        hue="Method",
        col="LR",
        row="Eval",
        order=args.attack_max_steps,
        col_order=learning_rates,
        row_order=["Harmfulness ↓", "Capability ↑"],
        kind="point",
        errorbar=None,
        palette="colorblind",
        legend_out=False,
        height=3,
        aspect=1,
    )
    grid.set(ylim=(0, 100))
    grid.set_axis_labels("Attack steps", "Score (%)")
    grid.set_titles("{row_name} | LR={col_name}")
    sns.move_legend(grid, "lower center", bbox_to_anchor=(0.5, 0), frameon=False)
    grid.figure.subplots_adjust(bottom=0.30, hspace=0.4, wspace=0.3)
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    grid.figure.savefig(args.outpath, bbox_inches="tight")
