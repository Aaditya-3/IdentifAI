"""CLI for validating and producing entity-resolution submissions."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:  # Supports both ``python run.py`` from this directory and module execution.
    from .src.pipeline import predict, validate
except ImportError:
    from src.pipeline import predict, validate


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Business entity resolution pipeline")
    parser.add_argument("--mode", choices=("validate", "predict"), required=True)
    parser.add_argument("--data_dir", default="dataset/train", help="Directory with train_source*.tsv and ground truth")
    parser.add_argument("--train_dir", default=None, help="Training directory for predict mode (defaults to --data_dir)")
    parser.add_argument("--test_dir", default="dataset/test")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--top_k", type=int, default=30, help="Final bounded candidates retained per Source-1 entity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scratch_dir", default="scratch", help="Disk workspace for SQLite indexes and memory-mapped features")
    args = parser.parse_args()
    if args.mode == "validate":
        score, threshold = validate(args.data_dir, args.top_k, args.seed, args.scratch_dir)
        print(f"Validation macro F0.5: {score:.6f} (threshold={threshold:.2f})")
    else:
        threshold = predict(args.test_dir, args.output_dir, args.train_dir or args.data_dir, args.top_k, args.seed, args.scratch_dir)
        print(f"Wrote submission files to {Path(args.output_dir).resolve()} (threshold={threshold:.2f})")


if __name__ == "__main__":
    main()
