"""Compare base, SEAM, and RepNoise models across Axolotl attack steps."""

import argparse
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootdir", default="artifacts/tamia/results")
    parser.add_argument("--base-rootdir", default="artifacts/tamia/evals")
    parser.add_argument("--seam-beta0-rootdir", default="artifacts/tamia/evals")
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
            "path": "llama2-7b-beavertail--align-fft-seam"
            "--def-sp--coeff-sd1.0--coeff-up5.0",
            "name": "SEAM",
            "hps": r"$\alpha=5.0,\ \beta=1.0$",
        },
        {
            "path": "llama2-7b-beavertail--align-fft-repnoise"
            "--def-sp--alpha0.1--beta0.0001",
            "name": "RepNoise",
            "hps": r"$\alpha=0.1,\ \beta=0.0001$",
        },
        {
            "path": "Llama-2-7b-chat-hf",
            "name": "Base model",
            "hps": "Default",
            "rootdir": args.base_rootdir,
        },
        {
            "path": "llama2-7b-beavertail--align-fft-seam"
            "--def-sp--coeff-sd0.0--coeff-up5.0",
            "name": "SEAM",
            "hps": r"$\alpha=5.0,\ \beta=0.0$",
            "rootdir": args.seam_beta0_rootdir,
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
                    model += f"--relearn-fft-att-sp-lr{lr}-ms{steps}"
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
                            "HPs": cfg["hps"],
                            "LR": lr,
                            "Attack steps": steps,
                            "Eval": evaluation,
                            "Score (%)": 100 * score,
                        }
                    )

    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    # Preserve the original equal spacing between sampled attack-step counts.
    df["Step index"] = df["Attack steps"].map(
        {steps: i for i, steps in enumerate(args.attack_max_steps)}
    )
    colors = sns.color_palette("colorblind")
    palette = {"SEAM": colors[0], "RepNoise": colors[1], "Base model": "black"}
    markers = {
        r"$\alpha=5.0,\ \beta=1.0$": "o",
        r"$\alpha=5.0,\ \beta=0.0$": "s",
        r"$\alpha=0.1,\ \beta=0.0001$": "^",
        "Default": "D",
    }
    dashes = {
        r"$\alpha=5.0,\ \beta=1.0$": "",
        r"$\alpha=5.0,\ \beta=0.0$": (4, 2),
        r"$\alpha=0.1,\ \beta=0.0001$": (1, 1),
        "Default": "",
    }
    grid = sns.relplot(
        data=df,
        x="Step index",
        y="Score (%)",
        hue="Method",
        style="HPs",
        markers=markers,
        dashes=dashes,
        col="LR",
        row="Eval",
        col_order=learning_rates,
        row_order=["Harmfulness ↓", "Capability ↑"],
        kind="line",
        errorbar=None,
        palette=palette,
        legend=False,
        height=3,
        aspect=1,
    )
    grid.set(
        ylim=(0, 100),
        xticks=range(len(args.attack_max_steps)),
        xticklabels=args.attack_max_steps,
    )
    grid.set_axis_labels("Attack steps", "Score (%)")
    grid.set_titles("{row_name} | LR={col_name}")
    grid.figure.legend(
        handles=[
            Line2D(
                [], [], color=color,
                marker="D" if method == "Base model" else None,
                label=method,
            )
            for method, color in palette.items()
        ],
        title="Method",
        loc="lower center",
        bbox_to_anchor=(0.28, 0),
        frameon=False,
    )
    grid.figure.legend(
        handles=[
            Line2D(
                [], [], color="0.3", marker=markers[hps],
                linestyle=(0, pattern) if pattern else "-",
                label=hps,
            )
            for hps, pattern in dashes.items()
            if hps != "Default"
        ],
        title="HPs",
        loc="lower center",
        bbox_to_anchor=(0.7, 0),
        frameon=False,
    )
    grid.figure.subplots_adjust(bottom=0.32, hspace=0.4, wspace=0.3)
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    grid.figure.savefig(args.outpath, bbox_inches="tight")
    if args.png_outpath:
        os.makedirs(osp.dirname(args.png_outpath) or ".", exist_ok=True)
        grid.figure.savefig(args.png_outpath, dpi=200, bbox_inches="tight")
