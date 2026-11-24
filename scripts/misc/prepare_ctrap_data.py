"""Prepare the exact local datasets and shared cache used by CTRAP."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download


REFUSAL_REPO = "anonymous4486/repnoise_beavertail"
REFUSAL_REVISION = "11ea63756195527164a9aa2850f7544bd251eab1"
REFUSAL_SHA256 = "c3c92961d834afe3ecbfbfa29e87ca759b90a73498aaf807162328e0ebe8e4ee"
ULTRACHAT_REPO = "HuggingFaceH4/ultrachat_200k"
ULTRACHAT_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
BEAVERDAM_REPO = "PKU-Alignment/beaver-dam-7b"
BEAVERDAM_REVISION = "3a5b70888869f04336d0d8344185ed74268ef896"
BASE_MODEL = "meta-llama/Llama-2-7b-hf"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_refusals(data_dir):
    source = hf_hub_download(
        repo_id=REFUSAL_REPO,
        repo_type="dataset",
        filename="beavertails_with_refusals_train.json",
        revision=REFUSAL_REVISION,
    )
    destination = data_dir / "beavertails_with_refusals_train.json"
    shutil.copyfile(source, destination)
    actual = sha256(destination)
    if actual != REFUSAL_SHA256:
        raise ValueError(f"Unexpected refusal dataset SHA-256: {actual}")


def prepare_ultrachat(data_dir):
    snapshot_download(
        repo_id=ULTRACHAT_REPO,
        repo_type="dataset",
        revision=ULTRACHAT_REVISION,
        local_dir=data_dir / "ultrachat_200k",
    )


def prepare_sst2(data_dir):
    dataset = load_dataset("nyu-mll/glue", "sst2")
    rows = []
    for example in dataset["train"]:
        rows.append(
            {
                "instruction": (
                    "Analyze the sentiment of the input, and respond only "
                    "positive or negative"
                ),
                "input": example["sentence"],
                "output": "negative" if example["label"] == 0 else "positive",
            }
        )
    with open(data_dir / "sst2.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=4)
    validation_rows = [
        {"sentence": example["sentence"], "label": example["label"]}
        for example in dataset["validation"]
    ]
    with open(data_dir / "sst2_validation.json", "w", encoding="utf-8") as handle:
        json.dump(validation_rows, handle, indent=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctrap-dir", type=Path, default=Path("ctrap"))
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()

    data_dir = (
        args.data_dir.resolve()
        if args.data_dir is not None
        else args.ctrap_dir.resolve() / "data"
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    prepare_refusals(data_dir)
    prepare_ultrachat(data_dir)
    prepare_sst2(data_dir)

    load_dataset("PKU-Alignment/BeaverTails")
    snapshot_download(repo_id=BASE_MODEL)
    snapshot_download(repo_id=BEAVERDAM_REPO, revision=BEAVERDAM_REVISION)

    print(f"Prepared CTRAP data in {data_dir}")
    print(f"Refusal SHA-256: {REFUSAL_SHA256}")
    print(f"UltraChat revision: {ULTRACHAT_REVISION}")
    print(f"BeaverDam revision: {BEAVERDAM_REVISION}")


if __name__ == "__main__":
    main()
