"""Tokenize harmful BeaverTails responses and their paired refusals."""

from data.document_loader import BaseDocumentStrategy


class BeaverTailsStrategy(BaseDocumentStrategy):
    def __init__(self, *args, response_field="response", response_separator=" ", **kwargs):
        super().__init__(*args, **kwargs)
        self.response_field = response_field
        self.response_separator = response_separator

    def wrap_dataset(self, dataset, process_count=None, **kwargs):
        dataset = dataset.filter(
            lambda row: not row["is_safe"],
            num_proc=process_count,
        )
        if self.max_samples is not None:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))
        return dataset.map(
            self.tokenize_row,
            remove_columns=dataset.column_names,
            num_proc=process_count,
        )

    def tokenize_row(self, row):
        response = row[self.response_field]
        text = f"Question: {row['prompt']}\nAnswer:{self.response_separator}{response}"
        return super().tokenize_row({"text": text})


def load(tokenizer, cfg, ds_cfg=None):
    properties = ds_cfg.message_property_mappings
    max_samples = properties.get("max_samples")
    return BeaverTailsStrategy(
        tokenizer,
        cfg.sequence_len,
        ds_cfg.path,
        int(properties["source_id"]),
        max_samples=int(max_samples) if max_samples is not None else None,
        response_field=properties.get("response_field", "response"),
    )
