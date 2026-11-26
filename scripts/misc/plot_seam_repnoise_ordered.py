"""Compare ordered SEAM and RepNoise attacks across learning rates."""

import argparse
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outpath",
        default=(
            "artifacts/figures/llama2-7b--og-vs-modified--lr-trajectory.pdf"
        ),
    )
    args = parser.parse_args()

    rootdir = "artifacts/tamia/evals"
    repnoise_root = "/home/mila/m/marawan.gamal/projects/tamper-resistance/repnoise"
    base_path = (
        f"{repnoise_root}/results/tamia/llama2-7b-beavertail"
        "--data-overlapping--relearn-fft-author-codebase-lr{lr}-ms250/results.json"
    )
    learning_rates = ["5e-5", "8e-5", "1e-4", "2e-4"]
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
            "path": f"{rootdir}/llama2-7b-beavertail--align-fft-seam-sp-sd0.01-up1.0"
            "--attack-fft-cosim-sp-ordered-lr{lr}-ms250/results.json",
            "method": "SEAM",
            "attack": "SP/SP, AdamW",
            "setup": "Modified",
        },
        {
            "path": f"{rootdir}/llama2-7b-beavertail--align-fft-seam-nl-sd0.01-up1.0"
            "--attack-fft-cosim-nl-ordered-adamw-no-hist-lr{lr}-ms250/results.json",
            "method": "SEAM",
            "attack": "NL/NL, AdamW no history",
            "setup": "Modified",
            "learning_rates": ["5e-5", "8e-5", "1e-4", "2e-4"],
        },
        {
            "path": base_path,
            "method": "RepNoise",
            "attack": "Base",
            "setup": "Original",
        },
        {
            "path": f"{repnoise_root}/results-def-nl-att-nl/tamia/"
            "llama2-7b-beavertail--align-fft-repnoise--upstream-paper-config"
            "--relearn-fft-lr{lr}-ms250/results.json",
            "method": "RepNoise",
            "attack": "NL/NL, AdamW",
            "setup": "Original",
        },
        {
            "path": f"{repnoise_root}/artifacts/tamia/evals/"
            "llama2-7b-beavertail--align-fft-repnoise--upstream-paper-config"
            "--def-sp-att-sp--relearn-fft-lr{lr}-ms250/results.json",
            "method": "RepNoise",
            "attack": "SP/SP, AdamW",
            "setup": "Modified",
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
    sns.set_theme(context="paper", style="ticks", font_scale=1.7)
    grid = sns.relplot(
        data=df,
        x="Learning rate",
        y="Harmfulness (%)",
        hue="Attack",
        style="Attack setup",
        col="Method",
        kind="line",
        markers=True,
        palette=["black", "C0", "C1", "C2"],
        height=3.25,
    )
    grid.set(ylim=(0, 100))
    grid.set_titles("{col_name}")

    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    grid.figure.savefig(args.outpath, bbox_inches="tight")
    png_outpath = osp.join(
        "artifacts/figures-png", osp.splitext(osp.basename(args.outpath))[0] + ".png"
    )
    os.makedirs(osp.dirname(png_outpath), exist_ok=True)
    grid.figure.savefig(png_outpath, dpi=200, bbox_inches="tight")
