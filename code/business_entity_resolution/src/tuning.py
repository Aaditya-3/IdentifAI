"""Post-baseline model and decision tuning.

Tuning is deliberately gated on a completed real baseline report.  It evaluates
only changes that are measurable on the provided training data: class weighting,
training-derived lexical-variation features, a structurally different linear
model, probability ensembling, and exact grouped macro-F0.5 thresholds.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .decision import optimize_grouped_threshold, choose_hysteresis_policy
from .model import LinearPairModel, PairModel, ProbabilityEnsemble
from .semantic_retrieval import SemanticConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    class_weight: str | None = None
    scale_pos_weight: float | None = None
    ensemble_weight: float = 0.75
    use_variation: bool = False


def _matrix_paths(scratch: Path, name: str, labels: bool, feature_count: int):
    x_path = scratch / f"{name}.features.f32"
    pairs_path = scratch / f"{name}.pairs.tsv"
    y_path = scratch / f"{name}.labels.i8" if labels else None
    return x_path, pairs_path, y_path


def _existing_matrix(pipeline_module, store, scratch: Path, name: str, labels: bool):
    MatrixFiles = pipeline_module.MatrixFiles
    feature_count = len(pipeline_module.FEATURE_NAMES)
    x_path, pairs_path, y_path = _matrix_paths(scratch, name, labels, feature_count)
    if not x_path.exists() or not pairs_path.exists():
        return None
    meta_path = scratch / f"{name}.meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    current_signature_row = store.connection.execute(
        "SELECT value FROM pipeline_meta WHERE key='signature'"
    ).fetchone()
    expected_store_signature = current_signature_row[0] if current_signature_row else None
    expected_semantic = SemanticConfig.from_environment(top_k=store.top_k).signature()
    expected_modules = pipeline_module._module_signature(
        "features.py", "variation.py", "semantic_retrieval.py"
    )
    expected_sampling = pipeline_module._training_sampling_config()
    if (
        meta.get("cache_schema") != pipeline_module.CACHE_SCHEMA_VERSION
        or meta.get("feature_count") != feature_count
        or meta.get("store_signature") != expected_store_signature
        or meta.get("semantic_config") != expected_semantic
        or meta.get("sampling_environment") != expected_sampling
        or meta.get("module_signatures") != expected_modules
    ):
        return None
    width = feature_count * np.dtype(np.float32).itemsize
    if width <= 0 or x_path.stat().st_size % width:
        return None
    rows = x_path.stat().st_size // width
    if labels and (y_path is None or not y_path.exists() or y_path.stat().st_size != rows):
        return None
    return MatrixFiles(x_path, y_path, pairs_path, rows)


def _load_matrix(
    pipeline_module,
    store,
    scratch: Path,
    name: str,
    where: str,
    params,
    labels: bool,
    subsample: bool,
    variation_model=None,
):
    existing = _existing_matrix(pipeline_module, store, scratch, name, labels)
    # A variation-enabled matrix cannot safely reuse a baseline matrix with the
    # same name.  Tune names are therefore unique and explicit.
    if existing is not None and variation_model is None:
        LOGGER.info("Reusing materialized matrix %s (%d rows)", name, existing.rows)
        return existing
    return pipeline_module._materialize(
        store,
        scratch,
        name,
        where,
        params,
        labels=labels,
        subsample=subsample,
        variation_model=variation_model,
    )


def _source_ids(matrix) -> list[str]:
    with matrix.pairs_path.open("r", encoding="utf-8") as handle:
        return [line.split("\t", 1)[0] for line in handle]


def _truth_counts(store, where: str, params: tuple) -> dict[str, int]:
    return {
        sid: int(count)
        for sid, count in store.connection.execute(
            f"""SELECT tr.source1_id, COUNT(*)
                 FROM truth tr JOIN source1 s ON s.entity_id=tr.source1_id
                 WHERE {where} GROUP BY tr.source1_id""",
            params,
        )
    }


def _evaluate(pipeline_module, store, matrix, model, where: str, params: tuple) -> dict[str, Any]:
    labels = matrix.y()
    if labels is None:
        raise RuntimeError("Evaluation matrix has no labels")
    probabilities = pipeline_module._batched_predict_proba(model, matrix)
    source_ids = _source_ids(matrix)
    truth_counts = _truth_counts(store, where, params)
    threshold, macro = optimize_grouped_threshold(source_ids, labels, probabilities, truth_counts)
    precision, recall = pipeline_module._pair_precision_recall(
        store, matrix, probabilities, threshold, where, params
    )
    _, by_country = pipeline_module._grouped_f05(
        store, matrix, probabilities, threshold, where, params
    )
    return {
        "model": model.describe(),
        "threshold": float(threshold),
        "macro_f0_5": float(macro),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f0_5_by_country": by_country,
        "probabilities": probabilities,
    }


def _fit_spec(spec: ModelSpec, X: np.ndarray, y: np.ndarray, seed: int):
    if spec.kind == "lgbm":
        return PairModel(
            seed,
            class_weight=spec.class_weight,
            scale_pos_weight=spec.scale_pos_weight,
        ).fit(X, y)
    if spec.kind == "linear":
        return LinearPairModel(seed, class_weight=spec.class_weight).fit(X, y)
    raise ValueError(f"Unknown model kind: {spec.kind}")


def _fit_ensemble(spec: ModelSpec, X: np.ndarray, y: np.ndarray, seed: int):
    primary = PairModel(
        seed,
        class_weight=spec.class_weight,
        scale_pos_weight=spec.scale_pos_weight,
    ).fit(X, y)
    secondary = LinearPairModel(seed, class_weight=spec.class_weight).fit(X, y)
    return ProbabilityEnsemble(primary, secondary, primary_weight=spec.ensemble_weight)


def _spec_score(result: dict[str, Any]) -> float:
    return float(result["macro_f0_5"])


def tune(
    data_dir: str | Path,
    top_k: int = 64,
    seed: int = 42,
    scratch_dir: str | Path = "scratch",
) -> dict[str, Any]:
    """Run measured post-baseline tuning."""
    import importlib

    pipeline = importlib.import_module(".pipeline", __package__)
    from .variation import VariationModel

    scratch = Path(scratch_dir)
    report_path = scratch / "validation_report.json"
    if not report_path.exists():
        raise RuntimeError("Run the real baseline first: python run.py --mode validate")
    baseline = json.loads(report_path.read_text(encoding="utf-8"))
    if baseline.get("status") != "completed" or not baseline.get("baseline_only"):
        raise RuntimeError("validation_report.json is not a completed untuned baseline")

    t0 = time.perf_counter()
    store = pipeline._build_store(data_dir, "train", scratch / "train.sqlite", top_k, with_truth=True)
    try:
        train_country, validation_country = pipeline._countries(store)

        # Reuse baseline matrices for the no-variation models.
        id_fit = _load_matrix(pipeline, store, scratch, "baseline_id_fit", "s.split<>0", (), True, True)
        id_valid = _load_matrix(pipeline, store, scratch, "baseline_id_valid", "s.split=0", (), True, False)

        fit_y = id_fit.y()
        if fit_y is None:
            raise RuntimeError("ID fit matrix has no labels")
        positives = max(1, int(np.count_nonzero(fit_y)))
        imbalance = max(1.0, (len(fit_y) - positives) / positives)
        moderate_weight = min(imbalance, max(1.25, float(np.sqrt(imbalance))))

        base_specs = [
            ModelSpec("lgbm_unweighted", "lgbm"),
            ModelSpec("lgbm_balanced", "lgbm", class_weight="balanced"),
            ModelSpec(f"lgbm_spw_{moderate_weight:.3f}", "lgbm", scale_pos_weight=moderate_weight),
            ModelSpec("ensemble_unweighted", "ensemble", ensemble_weight=0.75),
        ]

        results: dict[str, Any] = {}
        ranked: list[tuple[float, ModelSpec]] = []
        for spec in base_specs:
            t_model = time.perf_counter()
            model = _fit_ensemble(spec, id_fit.x(), fit_y, seed) if spec.kind == "ensemble" else _fit_spec(spec, id_fit.x(), fit_y, seed)
            result = _evaluate(pipeline, store, id_valid, model, "s.split=0", ())
            result["fit_seconds"] = time.perf_counter() - t_model
            result.pop("probabilities", None)
            result["training_features"] = "baseline"
            results[spec.name] = result
            ranked.append((_spec_score(result), spec))
            LOGGER.info("Tune ID %s: F0.5=%.6f threshold=%.6f", spec.name, result["macro_f0_5"], result["threshold"])

        # Training-derived token variation is learned only from s.split<>0, so
        # the ID validation rows never contribute labels to the variation maps.
        variation_model = pipeline._build_variation_model(
            store, "s.split<>0", (), max_positive_pairs=None
        )
        var_fit = _load_matrix(
            pipeline, store, scratch, "tuned_variation_id_fit",
            "s.split<>0", (), True, True, variation_model=variation_model
        )
        var_valid = _load_matrix(
            pipeline, store, scratch, "tuned_variation_id_valid",
            "s.split=0", (), True, False, variation_model=variation_model
        )
        var_y = var_fit.y()
        if var_y is None:
            raise RuntimeError("Variation ID fit matrix has no labels")

        variation_specs = [
            ModelSpec("lgbm_variation_unweighted", "lgbm", use_variation=True),
            ModelSpec("lgbm_variation_balanced", "lgbm", class_weight="balanced", use_variation=True),
            ModelSpec("ensemble_variation", "ensemble", ensemble_weight=0.75, use_variation=True),
        ]
        for spec in variation_specs:
            t_model = time.perf_counter()
            model = _fit_ensemble(spec, var_fit.x(), var_y, seed) if spec.kind == "ensemble" else _fit_spec(spec, var_fit.x(), var_y, seed)
            result = _evaluate(pipeline, store, var_valid, model, "s.split=0", ())
            result["fit_seconds"] = time.perf_counter() - t_model
            result.pop("probabilities", None)
            result["training_features"] = "variation"
            results[spec.name] = result
            ranked.append((_spec_score(result), spec))
            LOGGER.info("Tune ID %s: F0.5=%.6f threshold=%.6f", spec.name, result["macro_f0_5"], result["threshold"])

        ranked.sort(key=lambda item: (-item[0], item[1].name))
        top_specs = [spec for _, spec in ranked[:3]]

        cross_results: dict[str, Any] = {}
        for spec in top_specs:
            t_model = time.perf_counter()
            if spec.use_variation:
                country_variation = pipeline._build_variation_model(store, "s.country=?", (train_country,), None)
                reverse_variation = pipeline._build_variation_model(store, "s.country=?", (validation_country,), None)
                country_fit = _load_matrix(pipeline, store, scratch, "tuned_variation_country_fit", "s.country=?", (train_country,), True, True, variation_model=country_variation)
                country_valid = _load_matrix(pipeline, store, scratch, "tuned_variation_country_valid", "s.country=?", (validation_country,), True, False, variation_model=country_variation)
                reverse_fit = _load_matrix(pipeline, store, scratch, "tuned_variation_reverse_fit", "s.country=?", (validation_country,), True, True, variation_model=reverse_variation)
                reverse_valid = _load_matrix(pipeline, store, scratch, "tuned_variation_reverse_valid", "s.country=?", (train_country,), True, False, variation_model=reverse_variation)
            else:
                country_fit = _load_matrix(pipeline, store, scratch, "baseline_country_fit", "s.country=?", (train_country,), True, True)
                country_valid = _load_matrix(pipeline, store, scratch, "baseline_country_valid", "s.country=?", (validation_country,), True, False)
                reverse_fit = _load_matrix(pipeline, store, scratch, "baseline_reverse_fit", "s.country=?", (validation_country,), True, True)
                reverse_valid = _load_matrix(pipeline, store, scratch, "baseline_reverse_valid", "s.country=?", (train_country,), True, False)

            if spec.kind == "ensemble":
                country_model = _fit_ensemble(spec, country_fit.x(), country_fit.y(), seed)
                reverse_model = _fit_ensemble(spec, reverse_fit.x(), reverse_fit.y(), seed)
            else:
                country_model = _fit_spec(spec, country_fit.x(), country_fit.y(), seed)
                reverse_model = _fit_spec(spec, reverse_fit.x(), reverse_fit.y(), seed)

            country_eval = _evaluate(pipeline, store, country_valid, country_model, "s.country=?", (validation_country,))
            reverse_eval = _evaluate(pipeline, store, reverse_valid, reverse_model, "s.country=?", (train_country,))
            cross_results[spec.name] = {
                "country_holdout": {k: v for k, v in country_eval.items() if k != "probabilities"},
                "reverse_country_holdout": {k: v for k, v in reverse_eval.items() if k != "probabilities"},
                "seconds": time.perf_counter() - t_model,
            }

        selection_rows: list[tuple[float, str]] = []
        for spec_name, cross in cross_results.items():
            id_score = float(results[spec_name]["macro_f0_5"])
            country_score = float(cross["country_holdout"]["macro_f0_5"])
            reverse_score = float(cross["reverse_country_holdout"]["macro_f0_5"])
            robust_score = float(np.mean([id_score, country_score, reverse_score]))
            selection_rows.append((robust_score, spec_name))
        selection_rows.sort(key=lambda item: (-item[0], item[1]))
        selected_name = selection_rows[0][1]
        selected_spec = next(spec for spec in base_specs + variation_specs if spec.name == selected_name)

        selected_country = float(cross_results[selected_name]["country_holdout"]["threshold"])
        selected_reverse = float(cross_results[selected_name]["reverse_country_holdout"]["threshold"])
        selected_id = float(results[selected_name]["threshold"])

        # Keep inference exactly aligned with the threshold that was optimized
        # on grouped macro-F0.5.  Entity-level filtering remains available for
        # max-match and evidence-aware safeguards, but no unmeasured margin is
        # introduced after tuning.
        country_policy = choose_hysteresis_policy(
            selected_country,
            ambiguous_margin=0.0,
            second_match_delta=0.0,
            max_matches=top_k,
            min_absolute_score=0.0,
        )
        reverse_policy = choose_hysteresis_policy(
            selected_reverse,
            ambiguous_margin=0.0,
            second_match_delta=0.0,
            max_matches=top_k,
            min_absolute_score=0.0,
        )
        id_policy = choose_hysteresis_policy(
            selected_id,
            ambiguous_margin=0.0,
            second_match_delta=0.0,
            max_matches=top_k,
            min_absolute_score=0.0,
        )

        policy = {
            "version": 3,
            "status": "tuned_after_real_baseline",
            "selected_model": selected_name,
            "selected_model_spec": {
                "name": selected_spec.name,
                "kind": selected_spec.kind,
                "class_weight": selected_spec.class_weight,
                "scale_pos_weight": selected_spec.scale_pos_weight,
                "ensemble_weight": selected_spec.ensemble_weight,
                "use_variation": selected_spec.use_variation,
            },
            "country_mapping": {
                "train_country": train_country,
                "validation_country": validation_country,
            },
            "thresholds": {
                train_country: selected_reverse,
                validation_country: selected_country,
            },
            "id_threshold": selected_id,
            "unseen_country_threshold": selected_reverse,
            "decision_policies": {
                train_country: reverse_policy.to_dict(),
                validation_country: country_policy.to_dict(),
            },
            "unseen_decision_policy": reverse_policy.to_dict(),
            "selection_metric": "mean of ID, country-holdout and reverse-country macro-F0.5",
            "results": {
                "models": results,
                "cross_country": cross_results,
                "selection_table": [
                    {"model": name, "robust_macro_f0_5": score}
                    for score, name in selection_rows
                ],
            },
            "runtime_seconds": time.perf_counter() - t0,
        }
        (scratch / "tuned_policy.json").write_text(
            json.dumps(policy, indent=2),
            encoding="utf-8",
        )
        (scratch / "decision_policy.json").write_text(
            json.dumps(
                {
                    "version": policy["version"],
                    "status": "tuned",
                    "default": id_policy.to_dict(),
                    "country_policies": policy["decision_policies"],
                    "unseen_country": policy["unseen_decision_policy"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        LOGGER.info("Selected %s; robust F0.5=%.6f", selected_name, selection_rows[0][0])
        return policy
    finally:
        store.close()
