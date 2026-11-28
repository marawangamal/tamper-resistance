"""Compare refusal-SFT defence sizes across attack steps and learning rates."""

import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    scratch = os.environ.get("SCRATCH")
    rootdir = (
        osp.join(scratch, "tamper-resistance/artifacts/mila/evals")
        if scratch
        else "artifacts/mila/evals"
    )
    outname = "llama2-7b--refusal-sft--lr-trajectory"
    defence_samples = [100, 500, 1000, 4000, 8000]
    learning_rates = ["5e-5", "8e-5", "1e-4", "2e-4"]
    attack_steps = [5, 10, 50, 100]
    attack_step_counts = [0] + attack_steps
    task_metrics = {
        "mmlu": "acc,none",
        "truthfulqa_mc1": "acc,none",
        "arc_easy": "acc_norm,none",
        "hellaswag": "acc_norm,none",
    }

    rows = []
    for sample_count in defence_samples:
        defence = f"llama2-7b--align-fft-refusal-sft-sp-n{sample_count}"
        for learning_rate in learning_rates:
            models = [(0, defence)] + [
                (
                    steps,
                    f"{defence}--attack-fft-sp-adamw-no-hist"
                    f"-lr{learning_rate}-ms{steps}",
                )
                for steps in attack_steps
            ]
            for step_count, model in models:
                path = osp.join(rootdir, model, "results.json")
                if not osp.exists(path):
                    continue
                with open(path) as handle:
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
                            "Method": f"SFT ({sample_count})",
                            "Learning rate": learning_rate,
                            "Attack steps": step_count,
                            "Eval": evaluation,
                            "Score (%)": 100 * score,
                        }
                    )

    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="ticks", font_scale=1.4)
    grid = sns.catplot(
        data=df,
        x="Attack steps",
        y="Score (%)",
        hue="Method",
        col="Learning rate",
        row="Eval",
        kind="point",
        order=attack_step_counts,
        errorbar=None,
        palette="colorblind",
        height=2.7,
    )
    grid.set(ylim=(0, 100))
    grid.set_axis_labels("Attack steps", "Score (%)")
    grid.set_titles("{row_name} | LR={col_name}")

    for outdir, extension in [
        ("artifacts/figures", "pdf"),
        ("artifacts/figures-png", "png"),
    ]:
        os.makedirs(outdir, exist_ok=True)
        grid.figure.savefig(
            osp.join(outdir, f"{outname}.{extension}"),
            dpi=300,
            bbox_inches="tight",
        )
