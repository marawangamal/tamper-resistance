"""Cache the model, tokenizer, and datasets needed by TAR-v2 jobs."""

from datasets import load_dataset
from huggingface_hub import snapshot_download


def main():
    snapshot_download(repo_id="lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2")
    snapshot_download(repo_id="NousResearch/Meta-Llama-3-8B-Instruct")
    load_dataset("lapisrocks/pile-bio", split="train")
    load_dataset("cais/wmdp", "wmdp-bio", split="test")


if __name__ == "__main__":
    main()
