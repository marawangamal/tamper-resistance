"""Paper-matched Pile-Bio forget-set preprocessing for TAR attacks."""

from axolotl.prompt_tokenizers import DatasetWrappingStrategy


class PileBioStrategy(DatasetWrappingStrategy):
    def __init__(self, tokenizer, sequence_len):
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len

    def wrap_dataset(self, dataset, process_count=None, **kwargs):
        dataset = dataset.filter(
            lambda row: row["label"],
            num_proc=process_count,
        )
        dataset = dataset.train_test_split(test_size=0.2, seed=42)["train"]
        return dataset.map(
            self.tokenize_row,
            remove_columns=dataset.column_names,
            num_proc=process_count,
        )

    def tokenize_row(self, row):
        input_ids = self.tokenizer(
            # Match the preprocessing in Evaluating Durable Safeguards.
            row["txt_chunk"].strip("<s>").strip(),
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"][: self.sequence_len]

        attention_mask = [1] * len(input_ids)
        labels = input_ids.copy()
        padding_length = self.sequence_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        labels += [-100] * padding_length
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load(tokenizer, cfg, ds_cfg=None):
    return PileBioStrategy(tokenizer, cfg.sequence_len)
