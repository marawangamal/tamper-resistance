"""Prepare nested AdvBench 10/50/100-shot goal/target subsets for the Qi proxy."""

import argparse
import csv
import hashlib
import io
import json
import random
import urllib.request
from pathlib import Path


SOURCE_COMMIT = "098262edf85f807224e70ecd87b9d83716bf6b73"
SOURCE_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    f"{SOURCE_COMMIT}/data/advbench/harmful_behaviors.csv"
)
SOURCE_SHA256 = "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source = urllib.request.urlopen(SOURCE_URL, timeout=30).read()
    digest = hashlib.sha256(source).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"AdvBench source changed: {digest}")
    rows = list(csv.DictReader(io.StringIO(source.decode("utf-8-sig"))))
    if len(rows) != 520 or set(rows[0]) != {"goal", "target"}:
        raise ValueError("unexpected AdvBench source schema or size")
    if any(not row["goal"].strip() or not row["target"].strip() for row in rows):
        raise ValueError("empty AdvBench goal or target")

    selected = random.Random(42).sample(range(len(rows)), 100)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for shots in (10, 50, 100):
        path = args.output_dir / f"{shots}.jsonl"
        with path.open("w") as output:
            for index in selected[:shots]:
                row = rows[index]
                output.write(json.dumps({"prompt": row["goal"], "completion": row["target"]}) + "\n")
    manifest = {
        "source": SOURCE_URL,
        "sha256": digest,
        "selection": "random.Random(42).sample(range(520), 100); nested prefixes",
        "indices": {str(shots): selected[:shots] for shots in (10, 50, 100)},
        "target_semantics": "AdvBench affirmative prefixes, not full harmful answers",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
