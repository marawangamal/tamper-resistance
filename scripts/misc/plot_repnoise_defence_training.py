"""Plot selected RepNoise defence training trajectories."""

import argparse
import os
import os.path as osp

import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default="artifacts/tamia/repnoise-defence-training.csv",
    )
    parser.add_argument(
        "--outpath",
        default="artifacts/figures/llama2-7b--repnoise-defence-training.pdf",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=15,
        help="Centered rolling-mean window; use 1 to plot raw values",
    )
    args = parser.parse_args()
    if args.smooth_window < 1:
        parser.error("--smooth-window must be at least 1")

    configs = [{"name": "NL"}, {"name": "SP"}]
    metrics = ["loss_harmful_log", "loss_retain", "loss_noise"]
    df = pd.read_csv(args.path)
    expected_runs = {cfg["name"] for cfg in configs}
    if set(df["Run"]) != expected_runs:
        raise ValueError(f"Expected runs {expected_runs}, found {set(df['Run'])}")
    df = df[df["Metric"].isin(metrics)].copy()
    df = df.sort_values(["Run", "Metric", "Training step"])
    df["Plot value"] = df.groupby(["Run", "Metric"])["Value"].transform(
        lambda values: values.rolling(
            args.smooth_window,
            center=True,
            min_periods=1,
        ).mean()
    )

    sns.set_theme(context="paper", style="ticks", font_scale=1.25)
    grid = sns.relplot(
        data=df,
        x="Training step",
        y="Plot value",
        hue="Run",
        hue_order=[cfg["name"] for cfg in configs],
        col="Metric",
        col_order=metrics,
        kind="line",
        errorbar=None,
        facet_kws={"sharey": False},
        height=3.0,
        aspect=1.05,
    )
    ylabel = (
        "Value"
        if args.smooth_window == 1
        else f"Value ({args.smooth_window}-step rolling mean)"
    )
    grid.set_axis_labels("Training step", ylabel)
    grid.set_titles("{col_name}")
    grid.tight_layout()

    pngpath = osp.join(
        "artifacts/figures-png",
        osp.splitext(osp.basename(args.outpath))[0] + ".png",
    )
    for outpath in [args.outpath, pngpath]:
        os.makedirs(osp.dirname(outpath) or ".", exist_ok=True)
        grid.figure.savefig(outpath, dpi=300, bbox_inches="tight")
        print(outpath)
