"""Compare Base, SFT, and CTRAP across attack samples and learning rates."""

import json
import os

import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    rootdir = "artifacts/mila/evals"
    outpath = "artifacts/figures/llama2-7b--ctrap--og-vs-modified--trajectory.pdf"
    configs = [
        {"path": "llama2-7b--raw", "method": "Base"},
        {"path": "llama2-7b--align-lora-sft", "method": "SFT"},
        {
            "path": "llama2-7b--align-lora-ctrap-a0.1-l0.1",
            "method": "CTRAP (AdamW)",
        },
        {
            "path": "llama2-7b--align-lora-ctrap-a0.1-l0.1",
            "method": "CTRAP (AdamW no history)",
            "variant": "-adamw-no-history",
        },
    ]
    learning_rates = ["1e-5", "5e-5", "8e-5", "1e-4", "2e-4"]
    sample_counts = [100, 200, 300, 400, 500]
    evaluations = ["HS(IO) ↓", "HS(O) ↓"]

    rows = []
    for cfg in configs:
        for learning_rate in learning_rates:
            for sample_count in sample_counts:
                model = (
                    f"{cfg['path']}--attack-lora{cfg.get('variant', '')}"
                    f"-lr{learning_rate}-pr1-n{sample_count}"
                )
                path = os.path.join(
                    rootdir, model, "generations.json_sentiment_eval.json"
                )
                with open(path) as handle:
                    result = json.load(handle)
                scores = [
                    float(item.rsplit(":", 1)[1])
                    for item in result
                    if isinstance(item, str) and item.startswith("final")
                ]
                for evaluation, score in zip(evaluations, scores):
                    rows.append(
                        {
                            "Method": cfg["method"],
                            "Learning rate": learning_rate,
                            "Attack samples": sample_count,
                            "Eval": evaluation,
                            "Score (%)": score,
                        }
                    )

    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="ticks", font_scale=1.4)
    colors = sns.color_palette("colorblind")
    palette = {
        "Base": "black",
        "SFT": colors[2],
        "CTRAP (AdamW)": colors[0],
        "CTRAP (AdamW no history)": colors[1],
    }
    grid = sns.relplot(
        data=df,
        x="Attack samples",
        y="Score (%)",
        hue="Method",
        col="Learning rate",
        row="Eval",
        kind="line",
        marker="o",
        palette=palette,
        height=2.7,
    )
    grid.set(ylim=(0, 100))
    grid.set_titles("{row_name} | LR={col_name}")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    grid.figure.savefig(outpath, bbox_inches="tight")
    grid.figure.savefig(
        "artifacts/figures-png/llama2-7b--ctrap--og-vs-modified--trajectory.png",
        dpi=200,
        bbox_inches="tight",
    )
