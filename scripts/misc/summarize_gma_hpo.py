"""Summarize complete GMA HPO configurations under the documented criterion."""

import argparse
import glob
import json
import os.path as osp
import re


TASK_METRICS = {
    "mmlu": "acc,none",
    "truthfulqa_mc1": "acc,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
}
CONFIG_PATTERN = re.compile(
    r"llama2-7b-beavertail--align-fft-gma"
    r"(?:--lr(?P<learning_rate>[0-9.eE+-]+))?"
    r"--coeff-gma(?P<gma>[^-]+)--coeff-retain(?P<retain>[^-]+)$"
)
ATTACK_PATTERN = re.compile(r"--relearn-fft-lr(?P<learning_rate>[0-9.eE+-]+)-ms5$")


def load_result(path):
    with open(path) as handle:
        return json.load(handle)


def capability(result):
    return sum(
        result["post_utility"][task][metric]
        for task, metric in TASK_METRICS.items()
    ) / len(TASK_METRICS)


def load_rows(rootdir, require_learning_rate):
    rows = []
    defence_paths = glob.glob(
        osp.join(
            rootdir,
            "llama2-7b-beavertail--align-fft-gma*",
            "results-1000.json",
        )
    )
    for defence_path in sorted(defence_paths):
        config_name = osp.basename(osp.dirname(defence_path))
        match = CONFIG_PATTERN.fullmatch(config_name)
        if not match or bool(match["learning_rate"]) != require_learning_rate:
            continue
        attack_paths = glob.glob(
            osp.join(
                rootdir,
                f"{config_name}--relearn-fft-lr*-ms5",
                "results-1000.json",
            )
        )
        if len(attack_paths) != 4:
            continue
        defence = load_result(defence_path)
        attacks = []
        for path in attack_paths:
            attack_name = osp.basename(osp.dirname(path))
            attack_match = ATTACK_PATTERN.search(attack_name)
            if not attack_match:
                raise ValueError(f"Could not parse attack learning rate: {attack_name}")
            attacks.append(
                {
                    "learning_rate": attack_match["learning_rate"],
                    "harmfulness": load_result(path)["harmfulness"],
                }
            )
        worst_attack = max(attacks, key=lambda attack: attack["harmfulness"])
        rows.append(
            {
                "learning_rate": match["learning_rate"] or "2e-5",
                "retain": float(match["retain"]),
                "gma": float(match["gma"]),
                "capability": capability(defence),
                "pre_harmfulness": defence["harmfulness"],
                "worst_attack_learning_rate": worst_attack["learning_rate"],
                "worst_attack_harmfulness": worst_attack["harmfulness"],
            }
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootdir", default="artifacts/tamia/evals")
    parser.add_argument("--capability-threshold", type=float, default=0.5027)
    parser.add_argument(
        "--learning-rate-sweep",
        action="store_true",
        help="Select configurations whose names include a defence learning rate.",
    )
    args = parser.parse_args()

    rows = load_rows(args.rootdir, args.learning_rate_sweep)
    rows.sort(
        key=lambda row: (
            row["capability"] < args.capability_threshold,
            row["worst_attack_harmfulness"],
            -row["capability"],
        )
    )

    print(
        "| Defence LR | Retain | GMA | Capability | Pre-attack harmfulness | "
        "Worst attack LR | Worst attack harmfulness | Eligible |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for row in rows:
        eligible = row["capability"] >= args.capability_threshold
        print(
            f"| {row['learning_rate']} | {row['retain']:g} | {row['gma']:g} | "
            f"{row['capability']:.4f} | {row['pre_harmfulness']:.4f} | "
            f"{row['worst_attack_learning_rate']} | "
            f"{row['worst_attack_harmfulness']:.4f} | {'yes' if eligible else 'no'} |"
        )
    print(f"\nComplete configurations: {len(rows)}")
