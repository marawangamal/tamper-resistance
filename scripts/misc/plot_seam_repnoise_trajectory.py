"""Compare base, SEAM, RepNoise, and SDD across Axolotl attack steps."""

import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    rootdir = "artifacts/tamia/evals"
    scratch = os.environ.get("SCRATCH")
    mila_rootdir = (
        osp.join(scratch, "tamper-resistance/artifacts/mila/evals")
        if scratch
        else "artifacts/mila/evals"
    )
    outpath = "artifacts/figures/llama2-7b--lr-trajectory.pdf"
    attack_max_steps = [0, 5, 10, 50, 100]
    configs = [
        {
            "path": "llama2-7b-beavertail--align-fft-seam-sp-sd1.0-up0.01",
            "name": "SEAM",
            "attack": "--attack-fft-sp-adamw-no-hist-lr{lr}-ms{steps}",
        },
        {
            "path": "llama2-7b-beavertail--align-fft-repnoise-sp"
            "-alpha0.1-beta0.0001",
            "name": "RepNoise",
        },
        {
            "path": "llama2-7b-beavertail--align-fft-sdd",
            "name": "SDD",
            "attack": "--attack-fft-sp-adamw-no-hist-lr{lr}-ms{steps}",
        },
        {
            "path": "Llama-2-7b-chat-hf",
            "name": "Base",
        },
        {
            "path": "llama2-7b--align-fft-refusal-sft-sp-n4000",
            "name": "SFT (4k)",
            "rootdir": mila_rootdir,
            "attack": "--attack-fft-sp-adamw-no-hist-lr{lr}-ms{steps}",
        },
    ]
    learning_rates = ["5e-5", "8e-5", "1e-4", "2e-4"]
    task_metrics = {
        "mmlu": "acc,none",
        "truthfulqa_mc1": "acc,none",
        "arc_easy": "acc_norm,none",
        "hellaswag": "acc_norm,none",
    }

    rows = []
    for cfg in configs:
        for steps in attack_max_steps:
            for lr in learning_rates:
                model = cfg["path"]
                if steps:
                    model += cfg.get(
                        "attack", "--attack-fft-sp-lr{lr}-ms{steps}"
                    ).format(lr=lr, steps=steps)
                with open(
                    osp.join(
                        cfg.get("rootdir", rootdir), model, "results.json"
                    )
                ) as handle:
                    result = json.load(handle)
                capability = sum(
                    result["post_utility"][task][metric]
                    for task, metric in task_metrics.items()
                ) / len(task_metrics)
                for evaluation, score in [
                    ("Harmfulness ↓", result["harmfulness"]),
                    ("Capability ↑", capability),
                ]:
                    rows.append(
                        {
                            "Method": cfg["name"],
                            "LR": lr,
                            "Attack steps": steps,
                            "Eval": evaluation,
                            "Score (%)": 100 * score,
                        }
                    )

    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    colors = sns.color_palette("colorblind")
    palette = {
        "Base": "black",
        "SEAM": colors[0],
        "RepNoise": colors[1],
        "SDD": colors[2],
        "SFT (4k)": colors[3],
    }
    grid = sns.catplot(
        data=df,
        x="Attack steps",
        y="Score (%)",
        hue="Method",
        col="LR",
        row="Eval",
        order=attack_max_steps,
        kind="point",
        errorbar=None,
        palette=palette,
        height=3,
    )
    grid.set(ylim=(0, 100))
    grid.set_axis_labels("Attack steps", "Score (%)")
    grid.set_titles("{row_name} | LR={col_name}")
    os.makedirs(osp.dirname(outpath), exist_ok=True)
    grid.figure.savefig(outpath, bbox_inches="tight")
    grid.figure.savefig(
        "artifacts/figures-png/llama2-7b--lr-trajectory.png",
        dpi=200,
        bbox_inches="tight",
    )

    attack_scores = df.pivot(
        index=["Method", "LR", "Attack steps"],
        columns="Eval",
        values="Score (%)",
    ).reset_index()
    attack_scores["Attack score"] = (
        attack_scores["Harmfulness ↓"] * attack_scores["Capability ↑"]
    )
    worst_lrs = attack_scores.loc[
        attack_scores.groupby(["Method", "Attack steps"])[
            "Attack score"
        ].idxmax(),
        ["Method", "Attack steps", "LR"],
    ]
    worst_case = df.merge(
        worst_lrs,
        on=["Method", "Attack steps", "LR"],
        validate="many_to_one",
    )
    grid = sns.catplot(
        data=worst_case,
        x="Attack steps",
        y="Score (%)",
        hue="Method",
        col="Eval",
        order=attack_max_steps,
        kind="point",
        errorbar=None,
        palette=palette,
        height=3,
        aspect=1.3,
    )
    grid.set(ylim=(0, 100))
    grid.set_axis_labels("Attack steps", "Score (%)")
    grid.set_titles("{col_name}\n(max harm. × cap. LR)")
    grid.figure.savefig(
        "artifacts/figures/llama2-7b--worst-case-lr-trajectory.pdf",
        bbox_inches="tight",
    )
    grid.figure.savefig(
        "artifacts/figures-png/llama2-7b--worst-case-lr-trajectory.png",
        dpi=200,
        bbox_inches="tight",
    )
