"""Validate the service's config.yaml with its production loader."""

import argparse
from pathlib import Path

from grassmoth_server import config as config_mod
from grassmoth_server.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config_mod._find_config_path())
    args = parser.parse_args()
    print(f"Validating: {args.config}")
    try:
        load_config(args.config)
    except ValueError as exc:
        print(f"Schema errors: {exc}")
        return 1
    print("Schema: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
