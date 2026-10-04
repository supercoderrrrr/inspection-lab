"""Command-line entry point."""

import argparse
import json
from pathlib import Path

from inspection.data import validate_category


def main(argv=None):
    parser = argparse.ArgumentParser(description="Normal-reference visual inspection")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-data", help="Validate a MVTec-style category")
    check.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = validate_category(args.root)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(result, indent=2))
