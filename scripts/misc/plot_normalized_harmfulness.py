"""Plot normalized harmfulness under original and modified attacks."""

import json
import os

import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


def get_harmfulness(path):
    with open(path) as handle:
        result = json.load(handle)
    if isinstance(result, dict):
        return result["harmfulness"]
    return next(
        float(item.rsplit(":", 1)[1]) / 100
        for item in result
        if isinstance(item, str) and item.startswith("final  score:")
    )


if __name__ == "__main__":
    tamia = "artifacts/tamia/evals"
    mila = "artifacts/mila/evals"
    repnoise = (
        f"{tamia}/llama2-7b-beavertail--align-fft-repnoise"
        "-{format}-alpha1.0-beta0.001--attack-fft-{format}-{attack}"
        "-lr8e-5-ms250/results.json"
    )
    seam = (
        f"{tamia}/llama2-7b-beavertail--align-fft-seam-"
        "{format}-sd0.01-up1.0"
        "--attack-fft-cosim-{format}-ordered{attack}-lr1e-4-ms250/results.json"
    )
    ctrap = (
        f"{mila}/llama2-7b--align-lora-ctrap-a0.1-l0.1"
        "--attack-lora{attack}-lr{lr}-pr1-n500/"
        "generations.json_sentiment_eval.json"
    )
    sft = (
        f"{mila}/llama2-7b--align-lora-sft--attack-lora"
        "-lr{lr}-pr1-n500/generations.json_sentiment_eval.json"
    )
    configs = [
        {
            "method": "RepNoise (2024)",
            "modification": "Original",
            "path": repnoise.format(format="nl", attack="adamw"),
            "baseline": f"{tamia}/Llama-2-7b-chat-hf--attack-fft-author-codebase"
            "-lr8e-5-ms250/results.json",
        },
        {
            "method": "RepNoise (2024)",
            "modification": "No momentum",
            "path": repnoise.format(format="nl", attack="adamw-no-hist"),
            "baseline": f"{tamia}/Llama-2-7b-chat-hf--attack-fft-author-codebase"
            "-lr8e-5-ms250/results.json",
        },
        {
            "method": "RepNoise (2024)",
            "modification": "Template fix",
            "path": repnoise.format(format="sp", attack="adamw"),
            "baseline": f"{tamia}/Llama-2-7b-chat-hf--attack-fft-author-codebase"
            "-lr8e-5-ms250/results.json",
        },
        {
            "method": "SEAM (2026)",
            "modification": "Original",
            "path": seam.format(format="nl", attack=""),
            "baseline": f"{tamia}/Llama-2-7b-chat-hf--attack-fft-author-codebase"
            "-lr1e-4-ms250/results.json",
        },
        {
            "method": "SEAM (2026)",
            "modification": "No momentum",
            "path": seam.format(format="nl", attack="-adamw-no-hist"),
            "baseline": f"{tamia}/Llama-2-7b-chat-hf--attack-fft-author-codebase"
            "-lr1e-4-ms250/results.json",
        },
        {
            "method": "SEAM (2026)",
            "modification": "Template fix",
            "path": seam.format(format="sp", attack=""),
            "baseline": f"{tamia}/Llama-2-7b-chat-hf--attack-fft-author-codebase"
            "-lr1e-4-ms250/results.json",
        },
        {
            "method": "CTRAP (2026)",
            "modification": "Original",
            "path": ctrap.format(attack="", lr="1e-5"),
            "baseline": sft.format(lr="1e-5"),
        },
        {
            "method": "CTRAP (2026)",
            "modification": "No momentum",
            "path": ctrap.format(attack="-adamw-no-history", lr="1e-5"),
            "baseline": sft.format(lr="1e-5"),
        },
        {
            "method": "CTRAP (2026)",
            "modification": "Larger LR",
            "path": ctrap.format(attack="", lr="1e-4"),
            "baseline": sft.format(lr="1e-4"),
        },
    ]
    methods = ["RepNoise (2024)", "SEAM (2026)", "CTRAP (2026)"]
    rows = []
    for cfg in configs:
        rows.append(
            {
                **cfg,
                "Normalized harmfulness": min(
                    get_harmfulness(cfg["path"])
                    / get_harmfulness(cfg["baseline"]),
                    1,
                ),
            }
        )
    df = pd.DataFrame(rows)

    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    colors = dict(zip(methods, sns.color_palette("colorblind")))
    grid = sns.catplot(
        data=df,
        x="modification",
        y="Normalized harmfulness",
        hue="method",
        col="method",
        col_order=methods,
        kind="bar",
        palette=colors,
        dodge=False,
        edgecolor="black",
        legend=False,
        height=3.5,
        sharex=False,
    )
    grid.set(ylim=(0, 1.12))
    grid.set_axis_labels("", "Normalized harmfulness\n(defended / baseline) ↓")
    grid.set_titles("{col_name}")
    grid.set_xticklabels(rotation=45, ha="right")
    for axis in grid.axes.flat:
        for bar, label in zip(axis.patches, axis.get_xticklabels()):
            if label.get_text() != "Original":
                bar.set_hatch("//")
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center",
            )
        axis.axhline(1, color="black", linestyle="--", linewidth=1)
    grid.figure.legend(
        handles=[
            Patch(facecolor="white", edgecolor="black", label="Original"),
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch="//",
                label="Our modification",
            ),
        ],
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    sns.despine()

    os.makedirs("artifacts/figures", exist_ok=True)
    os.makedirs("artifacts/figures-png", exist_ok=True)
    grid.figure.savefig(
        "artifacts/figures/llama2-7b--og-vs-modified.pdf", bbox_inches="tight"
    )
    grid.figure.savefig(
        "artifacts/figures-png/llama2-7b--og-vs-modified.png",
        dpi=200,
        bbox_inches="tight",
    )
