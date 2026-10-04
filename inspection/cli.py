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
    fitting = commands.add_parser("fit", help="Fit a model on normal references only")
    fitting.add_argument("--root", type=Path, required=True)
    fitting.add_argument("--output", type=Path, required=True)
    fitting.add_argument("--method", choices=["baseline"], default="baseline")
    fitting.add_argument("--limit", type=int, default=16)
    fitting.add_argument("--seed", type=int, default=42)
    fitting.add_argument("--size", type=int, default=224)
    prediction = commands.add_parser("predict", help="Score one image using a saved model")
    prediction.add_argument("--model", type=Path, required=True)
    prediction.add_argument("--image", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "check-data":
            result = validate_category(args.root)
        elif args.command == "fit":
            from inspection.workflow import fit

            result = fit(args.root, args.output, args.limit, args.seed, args.size, args.method)
        else:
            from inspection.workflow import predict

            result = predict(args.model, args.image, args.output)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(result, indent=2))
