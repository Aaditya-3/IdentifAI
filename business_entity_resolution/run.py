"""CLI for validating and producing entity-resolution submissions."""
from __future__ import annotations

import argparse
from pathlib import Path

try:  # Supports both ``python run.py`` from this directory and module execution.
    from .src.pipeline import predict, validate
except ImportError:
    from src.pipeline import predict, validate


def main() -> None:
    parser = argparse.ArgumentParser(description="Business entity resolution pipeline")
    parser.add_argument("--mode", choices=("validate", "predict"), required=True)
    parser.add_argument("--data_dir", default="dataset/train", help="Directory with train_source*.tsv and ground truth")
    parser.add_argument("--train_dir", default=None, help="Training directory for predict mode (defaults to --data_dir)")
    parser.add_argument("--test_dir", default="dataset/test")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mode == "validate":
        score, threshold = validate(args.data_dir, args.top_k, args.seed)
        print(f"Validation macro F0.5: {score:.6f} (threshold={threshold:.2f})")
    else:
        threshold = predict(args.test_dir, args.output_dir, args.train_dir or args.data_dir, args.top_k, args.seed)
        print(f"Wrote submission files to {Path(args.output_dir).resolve()} (threshold={threshold:.2f})")


if __name__ == "__main__":
    main()
