"""Plot pairwise projections of the SEAM defence Pareto frontier."""

import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


METRICS = ["loss_ul", "loss_up", "Cosine Similarity"]
PANELS = [
    ("loss_ul", "loss_up"),
    ("loss_ul", "Cosine Similarity"),
    ("loss_up", "Cosine Similarity"),
]
AXIS_LABELS = {
    "loss_ul": "Unlearning loss ($L_{UL}$; ↓)",
    "loss_up": "Utility-preservation loss ($L_{UP}$; ↓)",
    "Cosine Similarity": "Gradient cosine similarity (↓)",
}


def pareto_optimal(values):
    """Return the nondominated rows when every objective is minimized."""
    optimal = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        dominates = np.all(values <= point, axis=1) & np.any(values < point, axis=1)
        optimal[index] = not dominates.any()
    return optimal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default="artifacts/tamia/seam-defence-pareto.csv",
    )
    parser.add_argument(
        "--outpath",
        default="artifacts/figures/llama2-7b--seam-defence-pareto.pdf",
    )
    args = parser.parse_args()

    configs = [
        {"format": "SP", "sd": sd, "up": up}
        for sd in [0.01, 0.1, 1.0, 5.0]
        for up in [0.01, 0.1, 1.0, 5.0]
    ]
    configs.append({"format": "NL", "sd": 0.01, "up": 1.0})

    df = pd.read_csv(args.path)
    expected = {(cfg["format"], cfg["sd"], cfg["up"]) for cfg in configs}
    observed = set(df[["Format", "sd", "up"]].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError(f"Expected configurations {expected}, found {observed}")
    if df[METRICS].isna().any().any():
        raise ValueError("Pareto metrics must not contain missing values")
    tail_steps = df["Tail steps"].unique()
    if len(tail_steps) != 1:
        raise ValueError(f"Expected one tail-window size, found {tail_steps}")

    df["Pareto status"] = np.where(
        pareto_optimal(df[METRICS].to_numpy()),
        "Pareto",
        "Dominated",
    )
    df["Response format"] = df["Format"].map({"SP": "SP/SP", "NL": "NL/NL"})

    sns.set_theme(context="paper", style="ticks", font_scale=1.0)
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
    format_order = ["SP/SP", "NL/NL"]
    palette = dict(zip(format_order, sns.color_palette("colorblind", 2)))
    markers = {"SP/SP": "o", "NL/NL": "X"}
    for axis, (x_metric, y_metric) in zip(axes, PANELS):
        sns.scatterplot(
            data=df,
            x=x_metric,
            y=y_metric,
            hue="Response format",
            hue_order=format_order,
            style="Response format",
            style_order=format_order,
            size="Pareto status",
            size_order=["Dominated", "Pareto"],
            sizes={"Dominated": 55, "Pareto": 110},
            markers=markers,
            palette=palette,
            edgecolor="black",
            linewidth=0.7,
            ax=axis,
        )
        axis.set_xlabel(AXIS_LABELS[x_metric])
        axis.set_ylabel(AXIS_LABELS[y_metric])
        if x_metric == "loss_up":
            axis.set_xscale("log")
            axis.set_xticks([0.3, 1.0, 3.0])
            axis.set_xticklabels(["0.3", "1", "3"])
            axis.xaxis.set_minor_formatter(mticker.NullFormatter())
        if y_metric == "loss_up":
            axis.set_yscale("log")
            axis.set_yticks([0.3, 1.0, 3.0, 7.0])
            axis.set_yticklabels(["0.3", "1", "3", "7"])
            axis.yaxis.set_minor_formatter(mticker.NullFormatter())
        axis.margins(0.1)

    handles, labels = axes[0].get_legend_handles_labels()
    for axis in axes:
        axis.get_legend().remove()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncols=6,
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(
        f"SEAM defence objectives (mean over final {tail_steps[0]} steps)",
        y=0.99,
    )
    sns.despine(fig=figure)
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.18, top=0.67, wspace=0.32)

    pngpath = osp.join(
        "artifacts/figures-png",
        osp.splitext(osp.basename(args.outpath))[0] + ".png",
    )
    for outpath in [args.outpath, pngpath]:
        os.makedirs(osp.dirname(outpath) or ".", exist_ok=True)
        figure.savefig(outpath, dpi=300, bbox_inches="tight")
        print(outpath)
