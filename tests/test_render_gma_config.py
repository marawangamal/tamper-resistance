import subprocess
import sys

import yaml


def test_renders_per_task_gma_values(tmp_path):
    source = tmp_path / "source.yml"
    output = tmp_path / "rendered.yml"
    source.write_text(
        yaml.safe_dump(
            {
                "custom_trainer_args": {
                    "coeff_retain": 1.0,
                    "coeff_gma": 1.0,
                },
                "output_dir": "old-output",
                "wandb_name": "old-name",
            }
        )
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/misc/render_gma_config.py",
            source,
            output,
            "--output-dir",
            "new-output",
            "--wandb-name",
            "new-name",
            "--coeff-retain",
            "0.01",
            "--coeff-gma",
            "5.0",
        ],
        check=True,
    )

    config = yaml.safe_load(output.read_text())
    assert config["output_dir"] == "new-output"
    assert config["wandb_name"] == "new-name"
    assert config["custom_trainer_args"]["coeff_retain"] == 0.01
    assert config["custom_trainer_args"]["coeff_gma"] == 5.0
