import json

from scripts.misc.parse_ctrap_results import trailing_score


def test_trailing_score_reads_native_ctrap_metrics(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps([{"output": "safe"}, "final  score:2.20", "final  score+:0.70"])
    )

    assert trailing_score(path, r"final  score:([0-9.]+)") == 2.2
    assert trailing_score(path, r"final  score\+:([0-9.]+)") == 0.7
