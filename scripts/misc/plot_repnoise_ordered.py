"""Compare overlapping RepNoise attacks across learning rates."""

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
            "artifacts/figures/llama2-7b--repnoise--og-vs-modified.pdf"
        ),
    )
    args = parser.parse_args()

    rootdir = "/home/mila/m/marawan.gamal/projects/tamper-resistance/repnoise"
    learning_rates = ["5e-5", "8e-5", "1e-4", "2e-4"]
    configs = [
        {
            "path": f"{rootdir}/results/tamia/llama2-7b-beavertail"
            "--data-overlapping--relearn-fft-author-codebase-lr{lr}-ms250/"
            "results.json",
            "method": "Base",
            "setup": "Original",
        },
        {
            "path": f"{rootdir}/results-def-nl-att-nl/tamia/llama2-7b-beavertail"
            "--align-fft-repnoise--upstream-paper-config"
            "--relearn-fft-lr{lr}-ms250/results.json",
            "method": "RepNoise (NL/NL)",
            "setup": "Original",
        },
        {
            "path": f"{rootdir}/artifacts/tamia/evals/llama2-7b-beavertail"
            "--align-fft-repnoise--upstream-paper-config--def-sp-att-sp"
            "--relearn-fft-lr{lr}-ms250/results.json",
            "method": "RepNoise (SP/SP)",
            "setup": "Modified",
        },
    ]

    rows = []
    for cfg in configs:
        for learning_rate in learning_rates:
            with open(cfg["path"].format(lr=learning_rate)) as handle:
                result = json.load(handle)
            capability = sum(
                [
                    result["post_utility"]["mmlu"]["acc,none"],
                    result["post_utility"]["truthfulqa_mc1"]["acc,none"],
                    result["post_utility"]["arc_easy"]["acc_norm,none"],
                    result["post_utility"]["hellaswag"]["acc_norm,none"],
                ]
            ) / 4
            for evaluation, score in [
                ("Harmfulness ↓", result["harmfulness"]),
                ("Capability ↑", capability),
            ]:
                rows.append(
                    {
                        "Method": cfg["method"],
                        "Attack setup": cfg["setup"],
                        "Learning rate": learning_rate,
                        "Eval": evaluation,
                        "Score (%)": 100 * score,
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
        y="Score (%)",
        hue="Method",
        style="Attack setup",
        col="Eval",
        kind="line",
        markers=True,
        palette=["black", "C0", "C1"],
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
