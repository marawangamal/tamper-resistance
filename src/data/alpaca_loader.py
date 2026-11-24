"""Tokenize Alpaca instruction/response examples for benign gradient probes."""

from data.document_loader import BaseDocumentStrategy


class AlpacaStrategy(BaseDocumentStrategy):
    def wrap_dataset(self, dataset, process_count=None, **kwargs):
        if self.max_samples is not None:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))
        return dataset.map(
            self.tokenize_row,
            remove_columns=dataset.column_names,
            num_proc=process_count,
        )

    def tokenize_row(self, row):
        prompt = row["instruction"]
        if row.get("input"):
            prompt += f"\n{row['input']}"
        text = f"Question: {prompt}\nAnswer: {row['output']}"
        return super().tokenize_row({"text": text})


def load(tokenizer, cfg, ds_cfg=None):
    properties = ds_cfg.message_property_mappings
    max_samples = properties.get("max_samples")
    return AlpacaStrategy(
        tokenizer,
        cfg.sequence_len,
        ds_cfg.path,
        int(properties["source_id"]),
        max_samples=int(max_samples) if max_samples is not None else None,
    )
