import pytest

from data.prompt_format import format_prompt


def test_question_answer_prompt():
    assert format_prompt("Do the task", "question_answer") == (
        "Question: Do the task\nAnswer:"
    )


def test_llama2_prompt():
    assert format_prompt("Do the task", "llama2") == (
        "[INST] Do the task [/INST]"
    )


def test_unknown_prompt_format():
    with pytest.raises(ValueError, match="prompt_format"):
        format_prompt("Do the task", "unknown")
