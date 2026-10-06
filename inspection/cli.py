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
    fitting.add_argument("--method", choices=["baseline", "patchcore"], default="baseline")
    fitting.add_argument("--limit", type=int, default=16)
    fitting.add_argument("--seed", type=int, default=42)
    fitting.add_argument("--size", type=int, default=224)
    fitting.add_argument("--calibration-count", type=int, default=30)
    fitting.add_argument("--alpha", type=float, default=0.05, help="Normal false-alarm target for calibration")
    prediction = commands.add_parser("predict", help="Score one image using a saved model")
    prediction.add_argument("--model", type=Path, required=True)
    prediction.add_argument("--image", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("evaluate", help="Evaluate a saved calibrated model on the test split")
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--root", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    verification = commands.add_parser("verify-results", help="Verify evidence hashes and recompute image metrics")
    verification.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "check-data":
            result = validate_category(args.root)
        elif args.command == "fit":
            from inspection.workflow import fit

            result = fit(args.root, args.output, args.limit, args.seed, args.size, args.method,
                         args.calibration_count, args.alpha)
        elif args.command == "predict":
            from inspection.workflow import predict

            result = predict(args.model, args.image, args.output)
        elif args.command == "evaluate":
            from inspection.evaluation import evaluate

            result = evaluate(args.model, args.root, args.output)
        else:
            from inspection.evaluation import verify_evaluation

            result = verify_evaluation(args.results)
    except ImportError:
        parser.exit(2, 'Error: Install PatchCore dependencies with pip install -e ".[patchcore]".\n')
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(result, indent=2))
