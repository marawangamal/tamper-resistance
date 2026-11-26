"""Compare ordered SEAM and RepNoise attacks across learning rates."""

import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    rootdir = "artifacts/tamia/evals"
    outpath = "artifacts/figures/llama2-7b--og-vs-modified--lr-trajectory.pdf"
    learning_rates = ["5e-5", "8e-5", "1e-4", "2e-4"]
    base_path = (
        f"{rootdir}/Llama-2-7b-chat-hf--attack-fft-author-codebase"
        "-lr{lr}-ms250/results.json"
    )
    configs = [
        {
            "path": base_path,
            "method": "SEAM",
            "attack": "Base",
            "setup": "Original",
        },
        {
            "path": f"{rootdir}/llama2-7b-beavertail--align-fft-seam-nl-sd0.01-up1.0"
            "--attack-fft-cosim-nl-ordered-lr{lr}-ms250/results.json",
            "method": "SEAM",
            "attack": "NL/NL, AdamW",
            "setup": "Original",
        },
        {
            "path": f"{rootdir}/llama2-7b-beavertail--align-fft-seam-nl-sd0.01-up1.0"
            "--attack-fft-cosim-nl-ordered-adamw-no-hist-lr{lr}-ms250/results.json",
            "method": "SEAM",
            "attack": "NL/NL, AdamW no history",
            "setup": "Modified",
        },
        {
            "path": f"{rootdir}/llama2-7b-beavertail--align-fft-seam-sp-sd0.01-up1.0"
            "--attack-fft-cosim-sp-ordered-lr{lr}-ms250/results.json",
            "method": "SEAM",
            "attack": "SP/SP, AdamW",
            "setup": "Modified",
        },
        {
            "path": base_path,
            "method": "RepNoise",
            "attack": "Base",
            "setup": "Original",
        },
        {
            "path": f"{rootdir}/llama2-7b-beavertail--align-fft-repnoise-nl"
            "-alpha1.0-beta0.001--attack-fft-nl-adamw-lr{lr}-ms250/results.json",
            "method": "RepNoise",
            "attack": "NL/NL, AdamW",
            "setup": "Original",
        },
        {
            "path": f"{rootdir}/llama2-7b-beavertail--align-fft-repnoise-nl"
            "-alpha1.0-beta0.001--attack-fft-nl-adamw-no-hist"
            "-lr{lr}-ms250/results.json",
            "method": "RepNoise",
            "attack": "NL/NL, AdamW no history",
            "setup": "Modified",
        },
        {
            "path": f"{rootdir}/llama2-7b-beavertail--align-fft-repnoise-sp"
            "-alpha1.0-beta0.001--attack-fft-sp-adamw-lr{lr}-ms250/results.json",
            "method": "RepNoise",
            "attack": "SP/SP, AdamW",
            "setup": "Modified",
            "learning_rates": ["8e-5"],
        },
    ]

    rows = []
    for cfg in configs:
        for learning_rate in cfg.get("learning_rates", learning_rates):
            with open(cfg["path"].format(lr=learning_rate)) as handle:
                result = json.load(handle)
            rows.append(
                {
                    "Method": cfg["method"],
                    "Attack": cfg["attack"],
                    "Attack setup": cfg["setup"],
                    "Learning rate": learning_rate,
                    "Harmfulness (%)": 100 * result["harmfulness"],
                }
            )

    df = pd.DataFrame(rows)
    df["Learning rate"] = pd.Categorical(
        df["Learning rate"], categories=learning_rates, ordered=True
    )
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    colors = sns.color_palette("colorblind")
    palette = {
        "Base": "black",
        "NL/NL, AdamW": colors[0],
        "NL/NL, AdamW no history": colors[1],
        "SP/SP, AdamW": colors[2],
    }
    grid = sns.relplot(
        data=df,
        x="Learning rate",
        y="Harmfulness (%)",
        hue="Attack",
        style="Attack setup",
        col="Method",
        kind="line",
        markers=True,
        palette=palette,
        height=3.5,
    )
    grid.set(ylim=(0, 100))
    grid.set_titles("{col_name}")

    os.makedirs(osp.dirname(outpath), exist_ok=True)
    grid.figure.savefig(outpath, bbox_inches="tight")
    grid.figure.savefig(
        "artifacts/figures-png/llama2-7b--og-vs-modified--lr-trajectory.png",
        dpi=200,
        bbox_inches="tight",
    )
