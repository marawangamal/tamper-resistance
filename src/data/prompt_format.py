"""Prompt formats shared by training data loaders and harmfulness evaluation."""


PROMPT_FORMATS = ("question_answer", "llama2")


def format_prompt(prompt: str, prompt_format: str) -> str:
    if prompt_format == "question_answer":
        return f"Question: {prompt}\nAnswer:"
    if prompt_format == "llama2":
        return f"[INST] {prompt} [/INST]"
    raise ValueError(
        f"prompt_format must be one of {PROMPT_FORMATS}, got {prompt_format!r}"
    )
