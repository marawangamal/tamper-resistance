"""Named response separators shared by dataset loaders and Slurm scripts."""

import argparse

import yaml


SEPARATORS = {"space": " ", "newline": "\n"}
ABBREVIATIONS = {"space": "sp", "newline": "nl"}


def response_separator(name):
    if name not in ("space", "newline"):
        raise ValueError(f"response_separator must be 'space' or 'newline', got {name!r}")
    return SEPARATORS[name]


def attack_response_format(config_path):
    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    formats = set()
    for dataset in config.get("datasets", []):
        if dataset.get("type") != "data.beavertails_loader":
            continue
        properties = dataset.get("message_property_mappings", {})
        name = properties.get("response_separator", "space")
        response_separator(name)
        formats.add(name)
    if len(formats) != 1:
        raise ValueError(
            "Attack config must contain BeaverTails datasets with one shared "
            "response_separator: space or newline"
        )
    return formats.pop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Axolotl attack YAML")
    parser.add_argument("--abbreviation", action="store_true")
    args = parser.parse_args()
    try:
        name = attack_response_format(args.config)
    except ValueError as error:
        parser.error(str(error))
    print(ABBREVIATIONS[name] if args.abbreviation else name)
