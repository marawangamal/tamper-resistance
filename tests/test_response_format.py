import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from src.data.response_format import attack_response_format, response_separator


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("name, character", [("space", " "), ("newline", "\n")])
def test_named_separator(name, character):
    assert response_separator(name) == character


@pytest.mark.parametrize("value", [" ", "\n", "sp", "nl", "", None, []])
def test_invalid_separator(value):
    with pytest.raises(ValueError, match="space.*newline"):
        response_separator(value)


def write_config(path, *formats):
    datasets = [
        {
            "type": "data.beavertails_loader",
            "message_property_mappings": {"response_separator": name},
        }
        for name in formats
    ]
    datasets.append({"type": "data.alpaca_loader"})
    path.write_text(yaml.safe_dump({"datasets": datasets}))


@pytest.mark.parametrize("name, abbreviation", [("space", "sp"), ("newline", "nl")])
def test_config_and_cli_ignore_other_dataset_types(tmp_path, name, abbreviation):
    path = tmp_path / "attack.yml"
    write_config(path, name)
    assert attack_response_format(path) == name
    result = subprocess.run(
        [sys.executable, "-m", "src.data.response_format", str(path), "--abbreviation"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == abbreviation + "\n"


@pytest.mark.parametrize("formats", [(), (None,), ("space", "newline")])
def test_config_rejects_missing_or_conflicting_formats(tmp_path, formats):
    path = tmp_path / "attack.yml"
    write_config(path, *formats)
    with pytest.raises(ValueError):
        attack_response_format(path)


@pytest.mark.parametrize("method", ["1b_repnoise", "2b_seam"])
@pytest.mark.parametrize("defence_format", ["space", "newline"])
@pytest.mark.parametrize("attack_format", ["space", "newline"])
def test_script_forwards_independent_formats(
    tmp_path, method, defence_format, attack_format
):
    bin_dir = tmp_path / "home/.local/bin"
    bin_dir.mkdir(parents=True)
    events = tmp_path / "events.jsonl"
    stub = '''import json, os, pathlib, shutil, subprocess, sys
args = sys.argv[1:]
kind = pathlib.Path(sys.argv[0]).name
if kind == 'uv':
    target = pathlib.Path(os.environ['UV_PROJECT_ENVIRONMENT']) / 'bin'
    target.mkdir(parents=True, exist_ok=True)
    for name in ['python', 'axolotl']:
        shutil.copy2(pathlib.Path(os.environ['STUB_DIR']) / name, target / name)
    sys.exit(0)
if kind == 'python' and args[:2] == ['-m', 'src.data.response_format']:
    sys.exit(subprocess.call([os.environ['TEST_PYTHON'], *args]))
fd = os.open(os.environ['EVENTS'], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.write(fd, (json.dumps([kind, args]) + '\\n').encode())
os.close(fd)
'''
    for name in ["uv", "python", "axolotl"]:
        executable = bin_dir / name
        executable.write_text(f"#!{sys.executable}\n" + stub)
        executable.chmod(0o755)
    for directory in ["seam", "repnoise", "configs/beavertails/seam"]:
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs/beavertails/seam/train-paper.yml").touch()
    config = tmp_path / "attack.yml"
    write_config(config, attack_format)
    script = ROOT / f"scripts/slurm/tamia/{method}_run_defence_attack_eval.sbatch"
    args = [str(config), defence_format]
    if method == "1b_repnoise":
        args.insert(0, "base-model")
    env = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        STUB_DIR=str(bin_dir),
        EVENTS=str(events),
        TEST_PYTHON=sys.executable,
        PYTHONPATH=str(ROOT),
        SLURM_TMPDIR=str(tmp_path / "tmp"),
        SLURM_JOB_ID="123",
        SLURM_ARRAY_TASK_ID="0",
        SCRATCH=str(tmp_path / "scratch"),
    )
    subprocess.run(["bash", str(script), *args], cwd=tmp_path, env=env, check=True)
    calls = [json.loads(line) for line in events.read_text().splitlines()]
    module = "representation_noising.train" if method == "1b_repnoise" else "src.train"
    defence = next(args for kind, args in calls if args[:2] == ["-m", module])
    flag = (
        "--harmful-response-separator"
        if method == "1b_repnoise"
        else "--harmful_response_separator"
    )
    assert defence[defence.index(flag) + 1] == response_separator(defence_format)
    output_flag = "--output-dir" if method == "1b_repnoise" else "--output_dir"
    defence_output = defence[defence.index(output_flag) + 1]
    assert f"--def-{'sp' if defence_format == 'space' else 'nl'}--" in defence_output
    attacks = [args for kind, args in calls if kind == "axolotl" and args[0] == "train"]
    assert len(attacks) == 16
    for attack in attacks:
        output = attack[attack.index("--output-dir") + 1]
        assert f"--relearn-fft-att-{'sp' if attack_format == 'space' else 'nl'}-" in output


def test_config_defaults_to_space_when_separator_is_omitted(tmp_path):
    path = tmp_path / "attack.yml"
    path.write_text(yaml.safe_dump({"datasets": [{"type": "data.beavertails_loader"}]}))
    assert attack_response_format(path) == "space"
