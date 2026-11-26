"""Compare Base, SFT, and CTRAP across attack samples and learning rates."""

import argparse
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootdir", default="artifacts/mila/evals")
    parser.add_argument(
        "--outpath",
        default="artifacts/figures/llama2-7b--ctrap--og-vs-modified--trajectory.pdf",
    )
    parser.add_argument("--png-outpath", help="Optional PNG copy of the figure")
    args = parser.parse_args()

    configs = [
        {"path": "llama2-7b--raw", "name": "Base"},
        {"path": "llama2-7b--align-lora-sft", "name": "SFT"},
        {
            "path": "llama2-7b--align-lora-ctrap-a0.1-l0.1",
            "name": "CTRAP (AdamW)",
        },
        {
            "path": "llama2-7b--align-lora-ctrap-a0.1-l0.1",
            "name": "CTRAP (AdamW no history)",
            "variant": "-adamw-no-history",
        },
    ]
    learning_rates = ["5e-5", "8e-5", "1e-4", "2e-4"]
    sample_counts = [100, 200, 300, 400, 500]
    metrics = ["HS(IO) ↓", "HS(O) ↓"]

    rows = []
    for cfg in configs:
        for learning_rate in learning_rates:
            for sample_count in sample_counts:
                model = (
                    f"{cfg['path']}--attack-lora{cfg.get('variant', '')}"
                    f"-lr{learning_rate}-pr1-n{sample_count}"
                )
                with open(
                    osp.join(
                        args.rootdir,
                        model,
                        "generations.json_sentiment_eval.json",
                    )
                ) as handle:
                    result = json.load(handle)
                scores = [
                    float(item.rsplit(":", 1)[1])
                    for item in result
                    if isinstance(item, str) and item.startswith("final")
                ]
                if len(scores) != len(metrics):
                    raise ValueError(f"Missing CTRAP scores for {model}")
                for evaluation, score in zip(metrics, scores):
                    rows.append(
                        {
                            "Method": cfg["name"],
                            "LR": learning_rate,
                            "Attack samples": sample_count,
                            "Eval": evaluation,
                            "Score (%)": score,
                        }
                    )

    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    colors = sns.color_palette("colorblind")
    palette = {
        "Base": "black",
        "SFT": colors[2],
        "CTRAP (AdamW)": colors[0],
        "CTRAP (AdamW no history)": colors[1],
    }
    grid = sns.catplot(
        data=df,
        x="Attack samples",
        y="Score (%)",
        hue="Method",
        col="LR",
        row="Eval",
        order=sample_counts,
        col_order=learning_rates,
        row_order=metrics,
        kind="point",
        errorbar=None,
        palette=palette,
        legend_out=False,
        height=3,
        aspect=1,
    )
    grid.set(ylim=(0, 100))
    grid.set_axis_labels("Attack samples", "Score (%)")
    grid.set_titles("{row_name} | LR={col_name}")
    sns.move_legend(
        grid, "lower center", bbox_to_anchor=(0.5, 0), frameon=False, title=None
    )
    grid.figure.subplots_adjust(bottom=0.30, hspace=0.4, wspace=0.3)
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    grid.figure.savefig(args.outpath, bbox_inches="tight")
    if args.png_outpath:
        os.makedirs(osp.dirname(args.png_outpath) or ".", exist_ok=True)
        grid.figure.savefig(args.png_outpath, dpi=200, bbox_inches="tight")
