"""Compare CTRAP attacks across learning rates."""

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
        default="artifacts/figures/llama2-7b--ctrap--og-vs-modified.pdf",
    )
    args = parser.parse_args()

    rootdir = osp.expandvars("$SCRATCH/tamper-resistance/artifacts/mila/evals")
    learning_rates = ["5e-5", "8e-5", "1e-4", "2e-4"]
    configs = [
        {
            "path": f"{rootdir}/llama2-7b--raw--attack-lora"
            "-lr{lr}-pr1-n500/generations.json_sentiment_eval.json",
            "method": "Base",
            "setup": "Original",
        },
        {
            "path": f"{rootdir}/llama2-7b--align-lora-ctrap-a0.1-l0.1"
            "--attack-lora-lr{lr}-pr1-n500/generations.json_sentiment_eval.json",
            "method": "CTRAP (AdamW)",
            "setup": "Original",
        },
        {
            "path": f"{rootdir}/llama2-7b--align-lora-ctrap-a0.1-l0.1"
            "--attack-lora-adamw-no-history-lr{lr}-pr1-n500/"
            "generations.json_sentiment_eval.json",
            "method": "CTRAP (AdamW no history)",
            "setup": "Modified",
        },
    ]

    rows = []
    for cfg in configs:
        for learning_rate in learning_rates:
            with open(cfg["path"].format(lr=learning_rate)) as handle:
                result = json.load(handle)
            scores = [
                float(item.rsplit(":", 1)[1])
                for item in result
                if isinstance(item, str) and item.startswith("final")
            ]
            for evaluation, score in zip(["HS(IO) ↓", "HS(O) ↓"], scores):
                rows.append(
                    {
                        "Method": cfg["method"],
                        "Attack setup": cfg["setup"],
                        "Learning rate": learning_rate,
                        "Eval": evaluation,
                        "Score (%)": score,
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
