"""Render one GMA sweep configuration with explicit per-task values."""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--coeff-retain", type=float, required=True)
    parser.add_argument("--coeff-gma", type=float, required=True)
    parser.add_argument("--coeff-forget", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    custom_args = config["custom_trainer_args"]
    custom_args["coeff_retain"] = args.coeff_retain
    custom_args["coeff_gma"] = args.coeff_gma
    if args.coeff_forget is not None:
        custom_args["coeff_forget"] = args.coeff_forget
    config["output_dir"] = args.output_dir
    config["wandb_name"] = args.wandb_name
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps

    args.output.write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
