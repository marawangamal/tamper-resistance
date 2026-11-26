"""Convert CTRAP's native evaluation outputs into a structured result file."""

import argparse
import json
import re
from pathlib import Path


def trailing_score(path, pattern):
    with open(path, encoding="utf-8") as handle:
        values = json.load(handle)
    regex = re.compile(pattern)
    for value in reversed(values):
        if isinstance(value, str) and (match := regex.fullmatch(value)):
            return float(match.group(1))
    raise ValueError(f"Could not find {pattern!r} in {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["raw", "sft", "ctrap"], required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--sentiment-eval", type=Path, required=True)
    parser.add_argument("--sst2-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ctrap-revision", required=True)
    args = parser.parse_args()

    result = {
        "method": args.method,
        "model": args.model,
        "attack": "lora-full-harmful",
        "requested_harmful_samples": args.sample_count,
        "effective_harmful_samples": args.sample_count - 1,
        "hs_input_output_percent": trailing_score(
            args.sentiment_eval, r"final  score:([0-9.]+)"
        ),
        "hs_output_percent": trailing_score(
            args.sentiment_eval, r"final  score\+:([0-9.]+)"
        ),
        "sst2_accuracy_percent": trailing_score(
            args.sst2_eval, r"score=([0-9.]+)"
        ),
        "ctrap_revision": args.ctrap_revision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
