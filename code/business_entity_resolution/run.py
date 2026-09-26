"""CLI for validating and producing entity-resolution submissions."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:  # Supports both ``python run.py`` from this directory and module execution.
    from .src.pipeline import DEFAULT_TOP_K, predict, validate
except ImportError:
    from src.pipeline import DEFAULT_TOP_K, predict, validate


# Repository root:
# <repo>/
#   run.py
#   code/
#     business_entity_resolution/
#       run.py  <-- this file
#
# Therefore parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAIN_DIR = REPO_ROOT / "student_resource" / "dataset" / "train"
DEFAULT_TEST_DIR = REPO_ROOT / "student_resource" / "dataset" / "test"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"
DEFAULT_SCRATCH_DIR = REPO_ROOT / "scratch"


def _resolve_dir(
    value: str | None,
    *,
    default: Path,
    description: str,
    create: bool = False,
) -> Path:
    """Resolve a directory and optionally create it.

    Data directories must already exist; output/work directories are created
    automatically so a clean checkout is runnable without manual setup.
    Relative paths are interpreted from the repository root for reproducibility.
    """
    raw = Path(value).expanduser() if value else default
    path = raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()

    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"{description} path is not a directory:\n{path}")
        return path

    if create:
        path.mkdir(parents=True, exist_ok=True)
        return path

    source = "Explicit" if value else "Default"
    raise FileNotFoundError(
        f"{source} {description} directory does not exist:\n{path}\n"
        "Pass the correct path explicitly."
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Business entity resolution pipeline"
    )

    parser.add_argument(
        "--mode",
        choices=("validate", "predict"),
        required=True,
    )

    parser.add_argument(
        "--data_dir",
        default=None,
        help=(
            "Training dataset directory. "
            "Defaults to student_resource/dataset/train."
        ),
    )

    parser.add_argument(
        "--train_dir",
        default=None,
        help=(
            "Training dataset directory for predict mode. "
            "Defaults to --data_dir or student_resource/dataset/train."
        ),
    )

    parser.add_argument(
        "--test_dir",
        default=None,
        help=(
            "Test dataset directory. "
            "Defaults to student_resource/dataset/test."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to repository-root/output.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Maximum candidates retained per Source-1 entity (default: {DEFAULT_TOP_K}).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--scratch_dir",
        default=None,
        help="Disk workspace. Defaults to repository-root/scratch.",
    )

    parser.add_argument(
        "--check-submission",
        action="store_true",
        help=(
            "After predict, also run the official stdlib submission validator "
            "with --check-ids. This can use substantial memory on the full test set."
        ),
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top_k must be at least 1")

    # ---------------------------------------------------------
    # Resolve canonical paths
    # ---------------------------------------------------------

    data_dir = _resolve_dir(
        args.data_dir,
        default=DEFAULT_TRAIN_DIR,
        description="training dataset",
    )

    train_dir = _resolve_dir(
        args.train_dir,
        default=data_dir,
        description="training dataset",
    )

    test_dir = _resolve_dir(
        args.test_dir,
        default=DEFAULT_TEST_DIR,
        description="test dataset",
    )

    output_dir = _resolve_dir(
        args.output_dir,
        default=DEFAULT_OUTPUT_DIR,
        description="output",
        create=True,
    )

    scratch_dir = _resolve_dir(
        args.scratch_dir,
        default=DEFAULT_SCRATCH_DIR,
        description="scratch",
        create=True,
    )

    # ---------------------------------------------------------
    # Print resolved paths so there is zero ambiguity
    # ---------------------------------------------------------

    logging.info("Repository root: %s", REPO_ROOT)
    logging.info("Training data:   %s", train_dir)
    logging.info("Test data:       %s", test_dir)
    logging.info("Output directory:%s", output_dir)
    logging.info("Scratch directory:%s", scratch_dir)
    logging.info("Top-K:            %d", args.top_k)

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    if args.mode == "validate":
        score, threshold = validate(
            str(train_dir),
            args.top_k,
            args.seed,
            str(scratch_dir),
        )

        print(
            f"Validation macro F0.5: "
            f"{score:.6f} "
            f"(threshold={threshold:.6f})"
        )
        return

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)

    # Never reuse stale final submission files.
    for filename in (
        "matching_results.tsv",
        "candidate_pairs.tsv",
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()

    threshold = predict(
        str(test_dir),
        str(output_dir),
        str(train_dir),
        args.top_k,
        args.seed,
        str(scratch_dir),
    )

    print(
        f"Wrote submission files to {output_dir} "
        f"(threshold={threshold:.6f})"
    )

    if args.check_submission:
        import subprocess
        validator = REPO_ROOT / "student_resource" / "utils" / "validate_submission.py"
        if not validator.exists():
            raise FileNotFoundError(f"Official submission validator not found: {validator}")
        command = [
            "python", str(validator),
            "--matching", str(output_dir / "matching_results.tsv"),
            "--candidate", str(output_dir / "candidate_pairs.tsv"),
            "--test-dir", str(test_dir),
            "--check-ids",
        ]
        logging.info("Running official submission validator with --check-ids")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()