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
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    custom_args = config["custom_trainer_args"]
    custom_args["coeff_retain"] = args.coeff_retain
    custom_args["coeff_gma"] = args.coeff_gma
    config["output_dir"] = args.output_dir
    config["wandb_name"] = args.wandb_name

    args.output.write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
