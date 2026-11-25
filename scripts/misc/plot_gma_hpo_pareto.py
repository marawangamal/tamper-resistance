"""Plot capability against worst-case attack harmfulness for the GMA sweep."""

import argparse
import glob
import json
import os
import os.path as osp
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


TASK_METRICS = {
    "mmlu": "acc,none",
    "truthfulqa_mc1": "acc,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
}
CONFIG_PATTERN = re.compile(
    r"llama2-7b-beavertail--align-fft-gma--coeff-gma(?P<gma>[^-]+)"
    r"--coeff-retain(?P<retain>[^-]+)$"
)


def load_result(path):
    with open(path) as handle:
        return json.load(handle)


def load_sweep(rootdir):
    rows = []
    pattern = osp.join(
        rootdir,
        "llama2-7b-beavertail--align-fft-gma--coeff-gma*--coeff-retain*",
        "results-1000.json",
    )
    for defence_path in sorted(glob.glob(pattern)):
        config_dir = osp.basename(osp.dirname(defence_path))
        match = CONFIG_PATTERN.fullmatch(config_dir)
        if not match:
            continue

        defence = load_result(defence_path)
        capability = sum(
            defence["post_utility"][task][metric]
            for task, metric in TASK_METRICS.items()
        ) / len(TASK_METRICS)
        attack_paths = glob.glob(
            osp.join(rootdir, f"{config_dir}--relearn-fft-lr*-ms5", "results-1000.json")
        )
        if len(attack_paths) != 4:
            raise ValueError(
                f"Expected four attacks for {config_dir}, found {len(attack_paths)}"
            )
        rows.append(
            {
                "Retain": float(match["retain"]),
                "GMA": float(match["gma"]),
                "Capability": capability,
                "Worst-case harmfulness": max(
                    load_result(path)["harmfulness"] for path in attack_paths
                ),
            }
        )
    if len(rows) != 16:
        raise ValueError(f"Expected 16 GMA configurations, found {len(rows)}")
    return pd.DataFrame(rows)


def pareto_frontier(data):
    frontier = []
    best_harmfulness = float("inf")
    for _, row in data.sort_values("Capability", ascending=False).iterrows():
        if row["Worst-case harmfulness"] < best_harmfulness:
            frontier.append(row)
            best_harmfulness = row["Worst-case harmfulness"]
    return pd.DataFrame(frontier).sort_values("Capability")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootdir", default="artifacts/tamia/evals")
    parser.add_argument("--capability-threshold", type=float, default=0.5027)
    parser.add_argument(
        "--outpath", default="artifacts/figures/gma-hpo-pareto.pdf"
    )
    parser.add_argument(
        "--png-outpath", default="artifacts/figures-png/gma-hpo-pareto.png"
    )
    args = parser.parse_args()

    data = load_sweep(args.rootdir)
    frontier = pareto_frontier(data)

    sns.set_theme(context="paper", style="ticks", font_scale=1.25)
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    sns.scatterplot(
        data=data,
        x="Capability",
        y="Worst-case harmfulness",
        hue="Retain",
        style="GMA",
        palette="viridis",
        s=90,
        ax=axis,
    )
    axis.plot(
        frontier["Capability"],
        frontier["Worst-case harmfulness"],
        color="black",
        linestyle="--",
        linewidth=1,
        label="Pareto frontier",
    )
    axis.axvline(
        args.capability_threshold,
        color="tab:red",
        linestyle=":",
        linewidth=1.5,
        label=f"Capability threshold ({args.capability_threshold:.4f})",
    )
    highlighted = [
        (data.loc[data["Worst-case harmfulness"].idxmin()], (25, 15), "Most robust"),
        (data.loc[data["Capability"].idxmax()], (-90, -35), "Highest capability"),
    ]
    for row, offset, description in highlighted:
        axis.annotate(
            f"{description}\nr={row['Retain']:g}, g={row['GMA']:g}",
            (row["Capability"], row["Worst-case harmfulness"]),
            xytext=offset,
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "color": "0.35", "linewidth": 0.8},
            fontsize=8,
        )
    axis.set(xlim=(0.25, 0.56), ylim=(-0.03, 0.83))
    axis.set_xlabel("Pre-attack capability (higher is better)")
    axis.set_ylabel("Worst-case five-step attack harmfulness (lower is better)")
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    sns.despine()
    figure.tight_layout()

    for outpath in [args.outpath, args.png_outpath]:
        if outpath:
            os.makedirs(osp.dirname(outpath) or ".", exist_ok=True)
            figure.savefig(outpath, dpi=200, bbox_inches="tight")
