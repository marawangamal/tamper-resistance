"""Tokenize harmful BeaverTails responses and their paired refusals."""

from axolotl.prompt_tokenizers import DatasetWrappingStrategy

from data.response_format import response_separator as separator_character


class BeaverTailsStrategy(DatasetWrappingStrategy):
    def __init__(
        self,
        tokenizer,
        sequence_len,
        path,
        source_id,
        max_samples=None,
        train_on_inputs=True,
        response_field="response",
        response_separator="space",
    ):
        if type(source_id) is not int:
            raise ValueError("source_id must be an integer")
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len
        self.path = path
        self.source_id = source_id
        self.max_samples = max_samples
        self.train_on_inputs = train_on_inputs
        self.response_field = response_field
        self.response_separator = separator_character(response_separator)

    def wrap_dataset(self, dataset, process_count=None, **kwargs):
        dataset = dataset.filter(
            lambda row: not row["is_safe"],
            num_proc=process_count,
        )
        if self.max_samples is not None:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))
        dataset = dataset.map(
            self.tokenize_row,
            remove_columns=dataset.column_names,
            num_proc=process_count,
        )
        if not self.train_on_inputs:
            dataset = dataset.filter(
                lambda row: any(label != -100 for label in row["labels"]),
                num_proc=process_count,
            )
        return dataset

    def tokenize_row(self, row):
        response = row[self.response_field]
        prompt = f"Question: {row['prompt']}\nAnswer:{self.response_separator}"
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        response_ids = self.tokenizer(
            response,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]

        bos_ids = (
            [] if self.tokenizer.bos_token_id is None else [self.tokenizer.bos_token_id]
        )
        eos_ids = (
            [] if self.tokenizer.eos_token_id is None else [self.tokenizer.eos_token_id]
        )
        input_ids = (bos_ids + prompt_ids + response_ids + eos_ids)[: self.sequence_len]
        if self.train_on_inputs:
            labels = input_ids.copy()
        else:
            labels = (
                [-100] * (len(bos_ids) + len(prompt_ids)) + response_ids + eos_ids
            )[: self.sequence_len]

        attention_mask = [1] * len(input_ids)
        padding_length = self.sequence_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        labels += [-100] * padding_length
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source_id": self.source_id,
        }


def load(tokenizer, cfg, ds_cfg=None):
    properties = ds_cfg.message_property_mappings
    max_samples = properties.get("max_samples")
    train_on_inputs = properties.get("train_on_inputs", cfg.get("train_on_inputs", True))
    if isinstance(train_on_inputs, str):
        train_on_inputs = train_on_inputs.lower() == "true"
    return BeaverTailsStrategy(
        tokenizer,
        cfg.sequence_len,
        ds_cfg.path,
        int(properties["source_id"]),
        max_samples=int(max_samples) if max_samples is not None else None,
        train_on_inputs=train_on_inputs,
        response_field=properties.get("response_field", "response"),
        response_separator=properties.get("response_separator", "space"),
    )
