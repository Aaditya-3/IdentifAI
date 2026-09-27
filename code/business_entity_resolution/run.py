"""CLI for validating and producing entity-resolution submissions."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:  # Supports both ``python run.py`` from this directory and module execution.
    from .src.pipeline import DEFAULT_TOP_K, predict, validate
    from .src.tuning import tune
except ImportError:
    from src.pipeline import DEFAULT_TOP_K, predict, validate
    from src.tuning import tune


# Repository root:
# <repo>/
#   run.py
#   code/
#     business_entity_resolution/
#       run.py  <-- this file
#
# Therefore parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]


def _discover_dataset_dir(split: str) -> Path:
    """Find a conventional materialized dataset directory without assuming one layout."""
    required = [f"{split}_source{i}.tsv" for i in (1, 2, 3)]
    candidates = (
        REPO_ROOT / "dataset" / split,
        REPO_ROOT / "data" / split,
        REPO_ROOT / "student_resource" / "dataset" / split,
        REPO_ROOT / split,
        REPO_ROOT,
    )
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / name).is_file() for name in required):
            return candidate
    # Keep the historical challenge path as the final diagnostic target when no
    # dataset is present yet; _resolve_dir then raises a precise error.
    return REPO_ROOT / "student_resource" / "dataset" / split


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
        choices=("validate", "tune", "predict"),
        required=True,
    )

    parser.add_argument(
        "--data_dir",
        default=None,
        help=(
            "Training dataset directory. "
            "Defaults to the first conventional materialized train dataset found in the repository."
        ),
    )

    parser.add_argument(
        "--train_dir",
        default=None,
        help=(
            "Training dataset directory for predict mode. "
            "Defaults to --data_dir or the discovered train dataset."
        ),
    )

    parser.add_argument(
        "--test_dir",
        default=None,
        help=(
            "Test dataset directory. "
            "Defaults to the first conventional materialized test dataset found in the repository."
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
            "After predict, run the official stdlib submission-format validator. "
            "The default check is memory-light and does not load all S2/S3 IDs."
        ),
    )

    parser.add_argument(
        "--check-submission-ids",
        action="store_true",
        help=(
            "Also ask the official validator to verify that every target ID exists. "
            "This is diagnostic only and can use substantial memory on the full test set."
        ),
    )

    parser.add_argument(
        "--allow-baseline-predict",
        action="store_true",
        help=(
            "Debug-only override allowing predict before a tuned_policy.json exists. "
            "Do not use this for the final submission."
        ),
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top_k must be at least 1")

    # ---------------------------------------------------------
    # Resolve canonical paths
    # ---------------------------------------------------------

    if args.mode == "predict":
        # Predict mode can be run with an explicit --train_dir even when the
        # repository itself contains no dataset. Do not resolve --data_dir first.
        train_default = Path(args.data_dir).expanduser() if args.data_dir else _discover_dataset_dir("train")
        train_dir = _resolve_dir(
            args.train_dir,
            default=train_default,
            description="training dataset",
        )
        data_dir = train_dir
    else:
        data_dir = _resolve_dir(
            args.data_dir,
            default=_discover_dataset_dir("train"),
            description="training dataset",
        )
        train_dir = data_dir

    test_dir = None
    if args.mode == "predict":
        test_dir = _resolve_dir(
            args.test_dir,
            default=_discover_dataset_dir("test"),
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
    if test_dir is not None:
        logging.info("Test data:       %s", test_dir)
    logging.info("Output directory:%s", output_dir)
    logging.info("Scratch directory:%s", scratch_dir)
    logging.info("Top-K:            %d", args.top_k)

    # ---------------------------------------------------------
    # Validate baseline
    # ---------------------------------------------------------

    if args.mode == "validate":
        score, threshold = validate(
            str(train_dir),
            args.top_k,
            args.seed,
            str(scratch_dir),
        )
        print(
            f"Validation baseline macro F0.5: "
            f"{score:.6f} "
            f"(threshold={threshold:.6f})"
        )
        return

    # ---------------------------------------------------------
    # Post-baseline tuning
    # ---------------------------------------------------------

    if args.mode == "tune":
        policy = tune(
            str(train_dir),
            args.top_k,
            args.seed,
            str(scratch_dir),
        )
        print(
            f"Tuning complete: selected={policy['selected_model']} "
            f"robust_macro_f0_5="
            f"{policy['results']['selection_table'][0]['robust_macro_f0_5']:.6f}"
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
        allow_baseline=args.allow_baseline_predict,
    )

    print(
        f"Wrote submission files to {output_dir} "
        f"(threshold={threshold:.6f})"
    )

    if args.check_submission or args.check_submission_ids:
        import subprocess

        validator_candidates = (
            Path(__file__).resolve().parent / "utils" / "validate_submission.py",
            Path(__file__).resolve().parent / "tests" / "validate_submission.py",
            REPO_ROOT / "student_resource" / "utils" / "validate_submission.py",
        )
        validator = next((path for path in validator_candidates if path.exists()), None)
        if validator is None:
            raise FileNotFoundError(
                "Official submission validator not found. Checked: "
                + ", ".join(str(path) for path in validator_candidates)
            )

        command = [
            sys.executable,
            str(validator),
            "--matching", str(output_dir / "matching_results.tsv"),
            "--candidate", str(output_dir / "candidate_pairs.tsv"),
            "--test-dir", str(test_dir),
        ]
        if args.check_submission_ids:
            command.append("--check-ids")

        logging.info(
            "Running official submission validator%s",
            " with --check-ids" if args.check_submission_ids else "",
        )
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()