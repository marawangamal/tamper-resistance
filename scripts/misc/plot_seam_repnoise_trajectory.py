"""Compare base, SEAM, and RepNoise models across Axolotl attack steps."""

import argparse
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootdir", default="artifacts/tamia/results")
    parser.add_argument("--base-rootdir", default="artifacts/tamia/evals")
    parser.add_argument(
        "--attack-max-steps", nargs="+", type=int, default=[0, 5, 10, 50, 100]
    )
    parser.add_argument(
        "--outpath",
        default="artifacts/figures/llama2-7b-beavertail--seam-repnoise--attack-steps--lr-trajectory.pdf",
    )
    parser.add_argument("--png-outpath", help="Optional PNG copy of the figure")
    args = parser.parse_args()

    configs = [
        {
            "path": "llama2-7b-beavertail--align-fft-seam-sp-sd1.0-up0.01",
            "name": "SEAM",
            "attack": "--attack-fft-sp-adamw-no-hist-lr{lr}-ms{steps}",
        },
        {
            "path": "llama2-7b-beavertail--align-fft-repnoise"
            "--def-sp--alpha0.1--beta0.0001",
            "name": "RepNoise",
        },
        {
            "path": "Llama-2-7b-chat-hf",
            "name": "Base",
            "rootdir": args.base_rootdir,
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
                model = cfg["path"]
                if steps:
                    model += cfg.get(
                        "attack", "--relearn-fft-att-sp-lr{lr}-ms{steps}"
                    ).format(lr=lr, steps=steps)
                with open(
                    osp.join(cfg.get("rootdir", args.rootdir), model, "results.json")
                ) as handle:
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
    colors = sns.color_palette("colorblind")
    palette = {
        configs[0]["name"]: colors[0],
        configs[1]["name"]: colors[1],
        configs[2]["name"]: "black",
    }
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
        palette=palette,
        legend_out=False,
        height=3,
        aspect=1,
    )
    grid.set(ylim=(0, 100))
    grid.set_axis_labels("Attack steps", "Score (%)")
    grid.set_titles("{row_name} | LR={col_name}")
    sns.move_legend(
        grid, "lower center", bbox_to_anchor=(0.5, 0), frameon=False, title=None
    )
    grid.figure.subplots_adjust(bottom=0.30, hspace=0.4, wspace=0.3)
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    grid.figure.savefig(args.outpath, bbox_inches="tight")
    if args.png_outpath:
        os.makedirs(osp.dirname(args.png_outpath) or ".", exist_ok=True)
        grid.figure.savefig(args.png_outpath, dpi=200, bbox_inches="tight")
