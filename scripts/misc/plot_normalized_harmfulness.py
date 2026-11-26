"""Plot normalized harmfulness under original and modified attacks."""

import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outpath",
        default=(
            "artifacts/figures/llama2-7b--og-vs-modified.pdf"
        ),
    )
    args = parser.parse_args()

    configs = [
        {
            "method": "RepNoise (2024)",
            "modification": "Original",
            "harmfulness": 0.0006899613,
            "baseline": 0.7614457,
        },
        {
            "method": "RepNoise (2024)",
            "modification": "Template fix",
            "harmfulness": 0.7570045636,
            "baseline": 0.7614457,
        },
        {
            "method": "SEAM (2026)",
            "modification": "Original",
            "harmfulness": 0.0018001745,
            "baseline": 0.7614457,
        },
        {
            "method": "SEAM (2026)",
            "modification": "Template fix",
            "harmfulness": 0.7804585640,
            "baseline": 0.7614457,
        },
        {
            "method": "SEAM (2026)",
            "modification": "No momentum",
            "harmfulness": 0.7836818636,
            "baseline": 0.7614457,
        },
        {
            "method": "CTRAP (2026)",
            "modification": "Original",
            "harmfulness": 0.157,
            "baseline": 0.771,
        },
        {
            "method": "CTRAP (2026)",
            "modification": "Larger LR",
            "harmfulness": 0.735,
            "baseline": 0.765,
        },
    ]
    methods = ["RepNoise (2024)", "SEAM (2026)", "CTRAP (2026)"]
    colors = dict(zip(methods, sns.color_palette("colorblind")))
    positions = [0, 1, 3, 4, 5, 7, 8]
    scores = [min(cfg["harmfulness"] / cfg["baseline"], 1) for cfg in configs]

    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    figure, axis = plt.subplots(figsize=(11, 4))
    bars = axis.bar(
        positions,
        scores,
        color=[colors[cfg["method"]] for cfg in configs],
        edgecolor="black",
        hatch=["" if cfg["modification"] == "Original" else "//" for cfg in configs],
    )
    axis.axhline(1, color="black", linestyle="--", linewidth=1)
    axis.axvline(2, color="0.9", linewidth=1)
    axis.axvline(6, color="0.9", linewidth=1)
    axis.set(
        ylabel="Normalized harmfulness\n(defended / base) ↓",
        ylim=(0, 1.12),
        xlim=(-0.8, 8.8),
        xticks=positions,
        xticklabels=[cfg["modification"] for cfg in configs],
    )
    plt.setp(
        axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )
    group_axis = axis.secondary_xaxis("bottom")
    group_axis.spines["bottom"].set_position(("outward", 100))
    group_axis.spines["bottom"].set_visible(False)
    group_axis.set_xticks([0.5, 4, 7.5], labels=methods)
    group_axis.tick_params(length=0)
    plt.setp(group_axis.get_xticklabels(), fontweight="bold")
    axis.bar_label(bars, labels=[f"{score:.2f}" for score in scores], padding=3)
    axis.legend(
        handles=[
            Patch(facecolor="white", edgecolor="black", label="Original"),
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch="//",
                label="Our modification",
            ),
        ],
        title="Attack setting",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    sns.despine()
    figure.tight_layout()

    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    figure.savefig(args.outpath, bbox_inches="tight")
    png_outpath = osp.join(
        "artifacts/figures-png", osp.splitext(osp.basename(args.outpath))[0] + ".png"
    )
    os.makedirs(osp.dirname(png_outpath), exist_ok=True)
    figure.savefig(png_outpath, dpi=200, bbox_inches="tight")
