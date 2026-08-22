"""Train-only sample-weight TAR sensitivity on the 19 affected outer repeats.

Reruns TAR only, with training-row min/max sample-weight normalization, on the
exact repeats where full-dataset min/max leaked the held-out global maximum.

Does not:
- regenerate ODE simulations or labels
- modify production source or official manuscript results
- retrain unaffected repeats
- retrain RF / other comparators (existing comparator aggregates are reused)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tree_srl_benchmark import (  # noqa: E402
    REPEATED_METRIC_COLS,
    TARGET_COLS_TTHR,
    TAR_MODEL,
    _configure_parallel_worker,
    _configure_quiet_runtime,
    build_expert_factories,
    build_stack_bundles,
    collect_model_benchmark_rows,
    fit_expert_full,
    generate_expert_oof,
    inner_cv_splits,
    inverse_target_transform,
    load_benchmark_data,
    maybe_include_poisson,
    predict_expert_models,
    predict_stack,
    repeat_metric_ci,
    set_global_seed,
)

AUDIT_BY_REPEAT = ROOT / "results" / "audit_sample_weight_normalization_by_repeat.csv"
MANIFEST_PATH = ROOT / "results" / "tree_srl_benchmark" / "model_compare_manifest.json"
REPEATED_METRICS_PATH = ROOT / "results" / "tree_srl_benchmark" / "repeated_parameter_metrics.csv"
SUMMARY_PATH = ROOT / "results" / "tree_srl_benchmark" / "model_compare_summary.csv"
PREDICTIONS_DIR = ROOT / "results" / "tree_srl_benchmark" / "repeats"
OUTDIR = ROOT / "results" / "sample_weight_sensitivity"

WEIGHT_EPS = 0.05
SPAN_EPS = 1e-12
AFFECTED_ABS_EPS = 1e-12
IDENTICAL_PRED_EPS = 1e-12
SMALL_ABS_R2 = 0.001
SMALL_REL_RMSE = 0.005
GLOBAL_MAX_BIO_ID = 440
EXPECTED_N_AFFECTED = 19
EXPECTED_N_REPEATS = 100

PRODUCTION_SOURCE_FILES = [
    ROOT / "microbio_dataset.py",
    ROOT / "tree_srl_benchmark.py",
    ROOT / "multi_pathogen_simulator.py",
]
OFFICIAL_RESULT_FILES = [
    ROOT / "results" / "tree_srl_benchmark" / "model_compare_summary.csv",
    ROOT / "results" / "tree_srl_benchmark" / "model_compare_manifest.json",
    ROOT / "results" / "tree_srl_benchmark" / "repeated_parameter_metrics.csv",
    ROOT / "results" / "tree_srl_benchmark" / "predictions_manifest.json",
    ROOT / "data" / "microbio_formal_dataset" / "y_targets.csv",
    ROOT / "data" / "microbio_formal_dataset" / "sample_weights.csv",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_native(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_native(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_native(obj.tolist())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _assert_under_outdir(path: Path) -> Path:
    resolved = path.resolve()
    outdir = OUTDIR.resolve()
    if resolved != outdir and outdir not in resolved.parents:
        raise RuntimeError(f"Refusing to write outside {outdir}: {resolved}")
    return resolved


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path = _assert_under_outdir(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_text(path: Path, text: str) -> None:
    path = _assert_under_outdir(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolve_project_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def trainonly_weights(u_train: np.ndarray) -> Tuple[np.ndarray, float, float]:
    u_train = np.asarray(u_train, dtype=np.float64)
    train_u_min = float(np.min(u_train))
    train_u_max = float(np.max(u_train))
    span = max(train_u_max - train_u_min, SPAN_EPS)
    unc_norm = (u_train - train_u_min) / span
    weights = 1.0 / (unc_norm + WEIGHT_EPS)
    return weights, train_u_min, train_u_max


def _finite_pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2:
        return None
    if np.allclose(a, a[0], atol=IDENTICAL_PRED_EPS, rtol=0.0) and np.allclose(
        b, b[0], atol=IDENTICAL_PRED_EPS, rtol=0.0
    ):
        return 1.0 if np.allclose(a, b, atol=IDENTICAL_PRED_EPS, rtol=0.0) else None
    value = float(pearsonr(a, b).statistic)
    return None if not np.isfinite(value) else value


def load_executed_config() -> Dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    run_config = manifest["run_config"]
    return {
        "x_csv": str(_resolve_project_path(run_config["x_csv"])),
        "y_csv": str(_resolve_project_path(run_config["y_csv"])),
        "metadata_csv": str(_resolve_project_path(run_config["metadata_csv"])),
        "sample_weight_csv": str(_resolve_project_path(run_config["sample_weight_csv"])),
        "candidate_score_csv": str(_resolve_project_path(run_config["candidate_score_csv"]))
        if run_config.get("candidate_score_csv")
        else None,
        "split_mode": str(manifest.get("split_mode") or run_config["split_mode"]),
        "group_col": str(manifest.get("group_col") or run_config["group_col"]),
        "test_size": float(manifest.get("test_size") if manifest.get("test_size") is not None else run_config["test_size"]),
        "target_transform": str(run_config["target_transform"]),
        "expanded_tree_bank": bool(run_config["expanded_tree_bank"]),
        "max_tree_experts": int(run_config["max_tree_experts"]),
        "use_physics_features": bool(run_config.get("use_physics_features", False)),
        "max_rows": int(run_config.get("max_rows", 0)),
        "repeat_ci_method": str(run_config.get("repeat_ci_method", "t_interval")),
        "n_jobs_formal": int(run_config.get("n_jobs", 10)),
        "lgbm_device": str(run_config.get("lgbm_device", "gpu")),
    }


def identify_affected_repeats(audit_df: pd.DataFrame) -> pd.DataFrame:
    affected = audit_df.loc[audit_df["max_abs_weight_difference"] > AFFECTED_ABS_EPS].copy()
    affected = affected.sort_values("repeat_id").reset_index(drop=True)
    if len(affected) != EXPECTED_N_AFFECTED:
        raise RuntimeError(
            f"Expected {EXPECTED_N_AFFECTED} affected repeats with "
            f"max_abs_weight_difference > {AFFECTED_ABS_EPS}, found {len(affected)}."
        )
    if not bool((affected["global_max_group_location"] == "validation").all()):
        bad = affected.loc[affected["global_max_group_location"] != "validation", "repeat_id"].tolist()
        raise RuntimeError(f"Affected repeats are not exactly those holding out bio_id=440: {bad}")
    return affected


def saved_val_contains_bio(repeat_id: int, bio_id: int) -> bool:
    path = PREDICTIONS_DIR / f"repeat_{repeat_id:03d}" / "predictions.csv"
    pred = pd.read_csv(path, usecols=["bio_id"])
    return bool((pred["bio_id"] == bio_id).any())


def load_original_tar_predictions(repeat_id: int) -> pd.DataFrame:
    path = PREDICTIONS_DIR / f"repeat_{repeat_id:03d}" / "predictions.csv"
    cols = ["validation_original_row_index", "bio_id"] + [f"pred_TAR_{t}" for t in TARGET_COLS_TTHR]
    return pd.read_csv(path, usecols=cols)


def load_original_repeat_metadata(repeat_id: int) -> dict:
    path = PREDICTIONS_DIR / f"repeat_{repeat_id:03d}" / "repeat_metadata.csv"
    return pd.read_csv(path).iloc[0].to_dict()


def apply_trainonly_weights_inplace(data) -> Dict[str, float]:
    if data.target_uncertainty_train is None:
        raise RuntimeError("BenchmarkData lacks target_uncertainty_train; cannot recompute weights.")
    original_w = np.asarray(data.sample_weight_train, dtype=np.float64)
    corrected, train_u_min, train_u_max = trainonly_weights(data.target_uncertainty_train)
    data.sample_weight_train = corrected
    abs_diff = np.abs(corrected - original_w)
    return {
        "train_u_min": train_u_min,
        "train_u_max": train_u_max,
        "n_train_rows": int(len(corrected)),
        "max_abs_weight_difference_vs_loaded_global": float(np.max(abs_diff)),
        "mean_abs_weight_difference_vs_loaded_global": float(np.mean(abs_diff)),
    }


def fit_tar_only(
    data,
    *,
    seed: int,
    repeat_id: int,
    target_transform: str,
    expanded_tree_bank: bool,
    max_tree_experts: int,
) -> Dict[str, Any]:
    include_poisson = maybe_include_poisson(data.y_train, seed)
    expert_factories = build_expert_factories(
        seed, include_poisson=include_poisson, expanded_tree_bank=expanded_tree_bank
    )
    expert_names = list(expert_factories.keys())
    inner_splits = inner_cv_splits(data.groups_train, seed=seed)

    expert_oof: Dict[str, np.ndarray] = {}
    for expert_name, factory in expert_factories.items():
        expert_oof[expert_name] = generate_expert_oof(factory, data, inner_splits)

    if max_tree_experts > 0:
        from tree_srl_benchmark import select_top_tree_experts_by_oof

        expert_names = select_top_tree_experts_by_oof(
            expert_oof, expert_names, data.y_train_fit, max_tree_experts
        )
        expert_oof = {name: expert_oof[name] for name in expert_names}
        expert_factories = {name: expert_factories[name] for name in expert_names}

    expert_val_preds: Dict[str, np.ndarray] = {}
    for expert_name, factory in expert_factories.items():
        expert_models = fit_expert_full(factory, data)
        expert_val_preds[expert_name] = predict_expert_models(expert_models, data.X_val)

    ridge_bundle, convex_bundle, chosen_type = build_stack_bundles(
        expert_oof, data.y_train_fit, expert_names
    )
    chosen_bundle = ridge_bundle if chosen_type == "ridge" else convex_bundle
    chosen_val_fit = predict_stack(chosen_bundle, expert_val_preds, expert_names)
    tar_pred = inverse_target_transform(chosen_val_fit.copy(), target_transform)

    expert_scores = []
    for expert_name in expert_names:
        y_pred = inverse_target_transform(expert_val_preds[expert_name], target_transform)
        _, expert_summary = collect_model_benchmark_rows(
            expert_name, y_pred, data, target_transform, repeat_id, seed, bootstrap=0
        )
        expert_scores.append((expert_name, float(expert_summary["mean_R2_original"])))
    best_single_tree_name = max(expert_scores, key=lambda item: item[1])[0]

    _, tar_summary = collect_model_benchmark_rows(
        TAR_MODEL, tar_pred, data, target_transform, repeat_id, seed, bootstrap=0
    )
    return {
        "tar_pred": tar_pred,
        "summary": tar_summary,
        "best_single_tree_name": best_single_tree_name,
        "chosen_stacker_type": chosen_type,
        "expert_names_used": list(expert_names),
        "n_experts_used": int(len(expert_names)),
    }


def prediction_difference_stats(
    original: np.ndarray,
    corrected: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    diff = np.abs(np.asarray(corrected, dtype=float) - np.asarray(original, dtype=float))
    sq = np.square(np.asarray(corrected, dtype=float) - np.asarray(original, dtype=float))
    key = lambda name: f"{prefix}{name}" if prefix else name
    return {
        key("max_abs_prediction_difference"): float(np.max(diff)),
        key("mean_abs_prediction_difference"): float(np.mean(diff)),
        key("median_abs_prediction_difference"): float(np.median(diff)),
        key("rmse_between_predictions"): float(np.sqrt(np.mean(sq))),
        key("pearson_correlation_predictions"): _finite_pearson(original, corrected),
        key("n_predictions"): int(diff.size),
        key("n_predictions_absdiff_gt_1e-12"): int(np.sum(diff > 1e-12)),
    }


def run_one_affected_repeat(job: Dict[str, Any]) -> Dict[str, Any]:
    _configure_parallel_worker(lgbm_device=job["lgbm_device"], verbose=job["verbose"])
    repeat_id = int(job["repeat_id"])
    seed = int(job["seed"])
    set_global_seed(seed)
    print(f"[sensitivity] repeat {repeat_id} seed={seed}: loading split + fitting TAR", flush=True)

    data = load_benchmark_data(
        x_csv=job["x_csv"],
        y_csv=job["y_csv"],
        metadata_csv=job["metadata_csv"],
        sample_weight_csv=job["sample_weight_csv"],
        split_mode=job["split_mode"],
        group_col=job["group_col"],
        test_size=job["test_size"],
        seed=seed,
        target_transform=job["target_transform"],
        max_rows=job["max_rows"],
        use_physics_features=job["use_physics_features"],
        candidate_score_csv=job["candidate_score_csv"],
    )

    orig_pred_df = load_original_tar_predictions(repeat_id)
    saved_val = orig_pred_df["validation_original_row_index"].to_numpy(dtype=int)
    if set(saved_val.tolist()) != set(np.asarray(data.val_indices, dtype=int).tolist()):
        raise RuntimeError(f"Repeat {repeat_id}: reconstructed validation indices do not match saved TAR split.")
    if not bool((orig_pred_df["bio_id"] == GLOBAL_MAX_BIO_ID).any()):
        raise RuntimeError(f"Repeat {repeat_id}: saved validation partition does not contain bio_id={GLOBAL_MAX_BIO_ID}.")

    weight_meta = apply_trainonly_weights_inplace(data)
    if abs(weight_meta["train_u_max"] - float(job["audit_train_u_max"])) > 1e-6:
        raise RuntimeError(
            f"Repeat {repeat_id}: train_u_max {weight_meta['train_u_max']} != audit {job['audit_train_u_max']}"
        )
    if weight_meta["max_abs_weight_difference_vs_loaded_global"] <= AFFECTED_ABS_EPS:
        raise RuntimeError(
            f"Repeat {repeat_id}: corrected train-only weights are identical to loaded global weights."
        )

    tar_fit = fit_tar_only(
        data,
        seed=seed,
        repeat_id=repeat_id,
        target_transform=job["target_transform"],
        expanded_tree_bank=job["expanded_tree_bank"],
        max_tree_experts=job["max_tree_experts"],
    )
    corrected_pred = np.asarray(tar_fit["tar_pred"], dtype=float)
    orig_aligned = (
        orig_pred_df.set_index("validation_original_row_index")
        .loc[np.asarray(data.val_indices, dtype=int), [f"pred_TAR_{t}" for t in TARGET_COLS_TTHR]]
        .to_numpy(dtype=float)
    )
    orig_meta = load_original_repeat_metadata(repeat_id)
    orig_metrics = job["original_metrics"]

    pred_stats = prediction_difference_stats(orig_aligned, corrected_pred)
    per_target_pred_stats: Dict[str, Any] = {}
    for j, target in enumerate(TARGET_COLS_TTHR):
        per_target_pred_stats.update(
            prediction_difference_stats(orig_aligned[:, j], corrected_pred[:, j], prefix=f"{target}_")
        )

    pred_out = pd.DataFrame(
        {
            "repeat_id": repeat_id,
            "seed": seed,
            "validation_original_row_index": np.asarray(data.val_indices, dtype=int),
            "bio_id": data.val_metadata["bio_id"].to_numpy() if data.val_metadata is not None else np.nan,
        }
    )
    for j, target in enumerate(TARGET_COLS_TTHR):
        pred_out[f"true_{target}"] = data.y_val[:, j]
        pred_out[f"pred_TAR_original_{target}"] = orig_aligned[:, j]
        pred_out[f"pred_TAR_corrected_{target}"] = corrected_pred[:, j]
        pred_out[f"abs_diff_{target}"] = np.abs(corrected_pred[:, j] - orig_aligned[:, j])

    return {
        "repeat_id": repeat_id,
        "seed": seed,
        "weight_meta": weight_meta,
        "summary": tar_fit["summary"],
        "original_metrics": orig_metrics,
        "best_single_tree_name_original": orig_meta.get("best_single_tree_name"),
        "best_single_tree_name_corrected": tar_fit["best_single_tree_name"],
        "chosen_stacker_type_original": orig_meta.get("chosen_stacker_type"),
        "chosen_stacker_type_corrected": tar_fit["chosen_stacker_type"],
        "expert_names_used_corrected": tar_fit["expert_names_used"],
        "tree_config_changed": bool(
            str(orig_meta.get("best_single_tree_name")) != str(tar_fit["best_single_tree_name"])
            or str(orig_meta.get("chosen_stacker_type")) != str(tar_fit["chosen_stacker_type"])
        ),
        "pred_stats": pred_stats,
        "per_target_pred_stats": per_target_pred_stats,
        "pred_frame": pred_out,
        "val_row_count": int(len(data.val_indices)),
    }


def reconstruct_corrected_metrics(
    original_long: pd.DataFrame,
    corrected_rows: List[dict],
    ci_method: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tar_orig = original_long.loc[original_long["model"] == TAR_MODEL].copy()
    if tar_orig["repeat_id"].nunique() != EXPECTED_N_REPEATS:
        raise RuntimeError("Official repeated TAR metrics do not contain 100 repeats.")
    corrected_by_id = {int(r["repeat_id"]): r for r in corrected_rows}
    hybrid_rows = []
    for _, row in tar_orig.sort_values("repeat_id").iterrows():
        rid = int(row["repeat_id"])
        out = {col: row[col] for col in ["repeat_id", "seed", "model"] + REPEATED_METRIC_COLS}
        out["weight_source"] = "reused_unaffected_global"
        if rid in corrected_by_id:
            summary = corrected_by_id[rid]["summary"]
            for col in REPEATED_METRIC_COLS:
                out[col] = summary[col]
            out["weight_source"] = "corrected_trainonly"
        hybrid_rows.append(out)
    hybrid = pd.DataFrame(hybrid_rows)

    def _aggregate(df: pd.DataFrame, model_name: str) -> dict:
        row = {"model": model_name}
        for col in REPEATED_METRIC_COLS:
            mean, lo, hi = repeat_metric_ci(df[col].to_numpy(dtype=float), method=ci_method)
            row[col] = mean
            row[f"{col}_ci_low"] = lo
            row[f"{col}_ci_high"] = hi
            row[f"{col}_sd"] = float(np.std(df[col].to_numpy(dtype=float), ddof=1))
        return row

    original_agg = _aggregate(tar_orig, "TAR_original_global_weights")
    corrected_agg = _aggregate(hybrid, "TAR_corrected_trainonly_weights")
    return hybrid, pd.DataFrame([original_agg]), pd.DataFrame([corrected_agg])


def classify_sensitivity(
    max_pred_diff: float,
    delta_mean_r2: float,
    relative_delta_rmse: float,
    ranking_changed: bool,
) -> str:
    if max_pred_diff <= IDENTICAL_PRED_EPS:
        return "MODEL IDENTICAL"
    if (
        abs(delta_mean_r2) < SMALL_ABS_R2
        and abs(relative_delta_rmse) < SMALL_REL_RMSE
        and not ranking_changed
    ):
        return "MODEL CHANGED, AGGREGATE EFFECT VERY SMALL"
    return "MODEL CHANGED, NONTRIVIAL EFFECT"


def ranking_table(summary_df: pd.DataFrame, tar_mean_r2: float, tar_mean_rmse: float) -> Dict[str, Any]:
    models = {}
    for _, row in summary_df.iterrows():
        models[str(row["model"])] = {
            "mean_R2_original": float(row["mean_R2_original"]),
            "mean_RMSE_original": float(row["mean_RMSE_original"]),
        }
    r2_order_original = sorted(models, key=lambda m: models[m]["mean_R2_original"], reverse=True)
    rmse_order_original = sorted(models, key=lambda m: models[m]["mean_RMSE_original"])
    corrected_models = dict(models)
    corrected_models[TAR_MODEL] = {
        "mean_R2_original": tar_mean_r2,
        "mean_RMSE_original": tar_mean_rmse,
    }
    r2_order_corrected = sorted(
        corrected_models, key=lambda m: corrected_models[m]["mean_R2_original"], reverse=True
    )
    rmse_order_corrected = sorted(
        corrected_models, key=lambda m: corrected_models[m]["mean_RMSE_original"]
    )
    return {
        "original_r2_rank_order": r2_order_original,
        "corrected_r2_rank_order": r2_order_corrected,
        "original_rmse_rank_order": rmse_order_original,
        "corrected_rmse_rank_order": rmse_order_corrected,
        "tar_r2_rank_original": r2_order_original.index(TAR_MODEL) + 1,
        "tar_r2_rank_corrected": r2_order_corrected.index(TAR_MODEL) + 1,
        "ranking_changed": r2_order_original != r2_order_corrected or rmse_order_original != rmse_order_corrected,
        "comparator_means_reused_not_retrained": True,
        "models": models,
    }


def build_report(
    *,
    affected: pd.DataFrame,
    cfg: Dict[str, Any],
    comparison_df: pd.DataFrame,
    pred_diff_df: pd.DataFrame,
    original_agg: pd.Series,
    corrected_agg: pd.Series,
    ranking: Dict[str, Any],
    classification: str,
    integrity: Dict[str, Any],
) -> str:
    delta_r2 = float(corrected_agg["mean_R2_original"] - original_agg["mean_R2_original"])
    rel_r2 = delta_r2 / abs(float(original_agg["mean_R2_original"]))
    delta_rmse = float(corrected_agg["mean_RMSE_original"] - original_agg["mean_RMSE_original"])
    rel_rmse = delta_rmse / abs(float(original_agg["mean_RMSE_original"]))
    lines = [
        "# Train-only sample-weight TAR sensitivity",
        "",
        "Sensitivity audit only. This does **not** replace the official 100-repeat benchmark.",
        "The 81 unaffected repeats reuse official TAR metrics. The 19 affected repeats were",
        "refit with training-row min/max sample-weight normalization. Comparators were **not** retrained.",
        "",
        "## 1. Affected repeats",
        "",
        f"- Selected by `max_abs_weight_difference > {AFFECTED_ABS_EPS}`: **{len(affected)}**",
        f"- All hold out global-max `bio_id={GLOBAL_MAX_BIO_ID}` in validation: **yes**",
        f"- Seeds: {affected['seed'].tolist()}",
        f"- repeat_id: {affected['repeat_id'].tolist()}",
        "",
        "## 2. Method",
        "",
        "- Split helper: `tree_srl_benchmark.split_train_validation_indices` / `load_benchmark_data`",
        f"- split_mode=`{cfg['split_mode']}`, group_col=`{cfg['group_col']}`, test_size=`{cfg['test_size']}`",
        f"- expanded_tree_bank=`{cfg['expanded_tree_bank']}`, max_tree_experts=`{cfg['max_tree_experts']}`",
        f"- target_transform=`{cfg['target_transform']}`",
        "- TAR workflow: same expert bank, inner GroupKFold OOF, ridge/convex stack selection",
        "- Sample weights: `1 / ((u_train - train_min) / max(train_max-train_min, 1e-12) + 0.05)`",
        "- Validation rows do not enter `train_u_min` / `train_u_max`",
        "- RF and other comparators were not retrained",
        "- No ODE resimulation; original TAR predictions loaded from saved repeats",
        "",
        "## 3. Affected-repeat TAR performance",
        "",
        (
            f"- Original mean target-wise R2: `{comparison_df['mean_R2_original_original'].mean():.12g}`"
        ),
        (
            f"- Corrected mean target-wise R2: `{comparison_df['mean_R2_original_corrected'].mean():.12g}`"
        ),
        (
            f"- Mean delta R2 (corrected-original): `{(comparison_df['delta_R2'].mean()):.12g}`"
        ),
        (
            f"- Original mean RMSE: `{comparison_df['mean_RMSE_original_original'].mean():.12g}`"
        ),
        (
            f"- Corrected mean RMSE: `{comparison_df['mean_RMSE_original_corrected'].mean():.12g}`"
        ),
        (
            f"- Mean delta RMSE: `{comparison_df['delta_RMSE'].mean():.12g}`"
        ),
        "",
        "## 4. Prediction-level changes on affected validation rows",
        "",
        f"- Maximum abs prediction difference: `{pred_diff_df['max_abs_prediction_difference'].max():.12g}`",
        f"- Mean of per-repeat mean abs differences: `{pred_diff_df['mean_abs_prediction_difference'].mean():.12g}`",
        f"- Maximum RMSE between old/new predictions: `{pred_diff_df['rmse_between_predictions'].max():.12g}`",
        "",
        "## 5. Corrected 100-repeat TAR aggregate",
        "",
        "Official summary method reused: mean across repeats with t-interval 95% CI",
        f"(`repeat_metric_ci`, method=`{cfg['repeat_ci_method']}`).",
        "",
        f"- Original mean R2: `{original_agg['mean_R2_original']:.12g}` "
        f"(95% CI {original_agg['mean_R2_original_ci_low']:.12g} to {original_agg['mean_R2_original_ci_high']:.12g})",
        f"- Corrected mean R2: `{corrected_agg['mean_R2_original']:.12g}` "
        f"(95% CI {corrected_agg['mean_R2_original_ci_low']:.12g} to {corrected_agg['mean_R2_original_ci_high']:.12g})",
        f"- delta_mean_R2: `{delta_r2:.12g}`",
        f"- relative_delta_mean_R2: `{rel_r2:.12g}`",
        "",
        f"- Original mean RMSE: `{original_agg['mean_RMSE_original']:.12g}`",
        f"- Corrected mean RMSE: `{corrected_agg['mean_RMSE_original']:.12g}`",
        f"- delta_mean_RMSE: `{delta_rmse:.12g}`",
        f"- relative_delta_mean_RMSE: `{rel_rmse:.12g}`",
        "",
        "## 6. Ranking versus existing comparators",
        "",
        "Comparator aggregates were **not** retrained; official RF / Best tree / UniformTreeMean means were reused.",
        "",
        f"- Original R2 rank order: `{ranking['original_r2_rank_order']}`",
        f"- Corrected R2 rank order: `{ranking['corrected_r2_rank_order']}`",
        f"- TAR ranking changed: **{'YES' if ranking['ranking_changed'] else 'NO'}**",
        "",
        "## 7. Classification",
        "",
        f"**{classification}**",
        "",
        "Rules (factual only; not a scientific pass/fail):",
        "",
        "- `MODEL IDENTICAL`: all corrected predictions identical within 1e-12",
        "- `MODEL CHANGED, AGGREGATE EFFECT VERY SMALL`: predictions change, but",
        "  |delta mean R2| < 0.001 and |relative delta RMSE| < 0.5% and ranking unchanged",
        "- `MODEL CHANGED, NONTRIVIAL EFFECT`: otherwise",
        "",
        "## 8. Integrity",
        "",
        f"- raw ODE simulations rerun: **{integrity['ode_simulation_rerun']}**",
        f"- labels regenerated: **{integrity['labels_regenerated']}**",
        f"- production files modified: **{integrity['production_source_modified']}**",
        f"- official result files overwritten: **{integrity['official_results_overwritten']}**",
        f"- comparators retrained: **{integrity['comparators_retrained']}**",
        "",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-only sample-weight TAR sensitivity (19 affected repeats).")
    parser.add_argument("--n_jobs", type=int, default=None, help="Parallel repeats (default: formal-run n_jobs).")
    parser.add_argument("--lgbm_device", default=None, help="Unused for sklearn TAR trees; kept for worker parity.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_quiet_runtime(verbose=args.verbose)
    source_hash_before = {p.as_posix(): _sha256_file(p) for p in PRODUCTION_SOURCE_FILES if p.is_file()}
    official_hash_before = {p.as_posix(): _sha256_file(p) for p in OFFICIAL_RESULT_FILES if p.is_file()}

    if not AUDIT_BY_REPEAT.is_file():
        raise FileNotFoundError(f"Missing audit file: {AUDIT_BY_REPEAT}")
    audit_df = pd.read_csv(AUDIT_BY_REPEAT)
    affected = identify_affected_repeats(audit_df)
    cfg = load_executed_config()
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else int(cfg["n_jobs_formal"])
    lgbm_device = str(args.lgbm_device or cfg["lgbm_device"])

    print("=" * 60)
    print("STEP 1 — Affected repeats")
    print("=" * 60)
    print(f"n_affected = {len(affected)} (expected {EXPECTED_N_AFFECTED})")
    print("repeat_id  seed  train_u_max  max_relative_weight_difference  bio440_in_saved_val")
    saved_val_flags = []
    for _, row in affected.iterrows():
        rid = int(row["repeat_id"])
        in_val = saved_val_contains_bio(rid, GLOBAL_MAX_BIO_ID)
        saved_val_flags.append(in_val)
        print(
            f"{rid:9d}  {int(row['seed']):4d}  {float(row['train_u_max']):.8g}  "
            f"{float(row['max_relative_weight_difference']):.8g}  {in_val}"
        )
    if not all(saved_val_flags):
        raise RuntimeError("Saved prediction files do not confirm bio_id=440 in validation for every affected repeat.")
    print("Verified: these 19 repeats are exactly those holding out bio_id=440.")
    print()

    original_long = pd.read_csv(REPEATED_METRICS_PATH)
    tar_long = original_long.loc[original_long["model"] == TAR_MODEL]
    original_metrics_by_id = {
        int(rec["repeat_id"]): rec for rec in tar_long.to_dict(orient="records")
    }
    official_summary = pd.read_csv(SUMMARY_PATH)

    jobs = []
    for _, row in affected.iterrows():
        rid = int(row["repeat_id"])
        jobs.append(
            {
                "repeat_id": rid,
                "seed": int(row["seed"]),
                "audit_train_u_max": float(row["train_u_max"]),
                "original_metrics": original_metrics_by_id[rid],
                "x_csv": cfg["x_csv"],
                "y_csv": cfg["y_csv"],
                "metadata_csv": cfg["metadata_csv"],
                "sample_weight_csv": cfg["sample_weight_csv"],
                "candidate_score_csv": cfg["candidate_score_csv"],
                "split_mode": cfg["split_mode"],
                "group_col": cfg["group_col"],
                "test_size": cfg["test_size"],
                "target_transform": cfg["target_transform"],
                "expanded_tree_bank": cfg["expanded_tree_bank"],
                "max_tree_experts": cfg["max_tree_experts"],
                "max_rows": cfg["max_rows"],
                "use_physics_features": cfg["use_physics_features"],
                "lgbm_device": lgbm_device,
                "verbose": bool(args.verbose),
            }
        )

    print("=" * 60)
    print("STEP 3 — Refitting TAR on affected repeats only")
    print("=" * 60)
    print(f"n_jobs={n_jobs}; comparators will not be retrained; uncertainty disabled.")
    if n_jobs <= 1 or len(jobs) == 1:
        results = [run_one_affected_repeat(job) for job in jobs]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(run_one_affected_repeat)(job) for job in jobs)
    results = sorted(results, key=lambda r: int(r["repeat_id"]))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    pred_diff_rows = []
    for item in results:
        orig = item["original_metrics"]
        corr = item["summary"]
        comparison_rows.append(
            {
                "repeat_id": item["repeat_id"],
                "seed": item["seed"],
                "train_u_min": item["weight_meta"]["train_u_min"],
                "train_u_max": item["weight_meta"]["train_u_max"],
                "mean_R2_original_original": orig["mean_R2_original"],
                "mean_R2_original_corrected": corr["mean_R2_original"],
                "delta_R2": float(corr["mean_R2_original"] - orig["mean_R2_original"]),
                "mean_R2_log_original": orig["mean_R2_log"],
                "mean_R2_log_corrected": corr["mean_R2_log"],
                "delta_R2_log": float(corr["mean_R2_log"] - orig["mean_R2_log"]),
                "mean_RMSE_original_original": orig["mean_RMSE_original"],
                "mean_RMSE_original_corrected": corr["mean_RMSE_original"],
                "delta_RMSE": float(corr["mean_RMSE_original"] - orig["mean_RMSE_original"]),
                "mean_MAE_original_original": orig["mean_MAE_original"],
                "mean_MAE_original_corrected": corr["mean_MAE_original"],
                "mean_NRMSE_original_original": orig["mean_NRMSE_original"],
                "mean_NRMSE_original_corrected": corr["mean_NRMSE_original"],
                "best_single_tree_name_original": item["best_single_tree_name_original"],
                "best_single_tree_name_corrected": item["best_single_tree_name_corrected"],
                "chosen_stacker_type_original": item["chosen_stacker_type_original"],
                "chosen_stacker_type_corrected": item["chosen_stacker_type_corrected"],
                "tree_config_changed": item["tree_config_changed"],
                **item["pred_stats"],
            }
        )
        pred_diff_rows.append(
            {
                "repeat_id": item["repeat_id"],
                "seed": item["seed"],
                **item["pred_stats"],
                **item["per_target_pred_stats"],
            }
        )
        _write_csv(
            OUTDIR / "repeats" / f"repeat_{int(item['repeat_id']):03d}_tar_predictions.csv",
            item["pred_frame"],
        )

    comparison_df = pd.DataFrame(comparison_rows)
    pred_diff_df = pd.DataFrame(pred_diff_rows)
    hybrid, original_agg_df, corrected_agg_df = reconstruct_corrected_metrics(
        original_long, results, ci_method=cfg["repeat_ci_method"]
    )
    original_agg = original_agg_df.iloc[0]
    corrected_agg = corrected_agg_df.iloc[0]

    # Sanity: reconstructed original aggregate should match official summary.
    official_tar = official_summary.loc[official_summary["model"] == TAR_MODEL].iloc[0]
    if abs(float(original_agg["mean_R2_original"]) - float(official_tar["mean_R2_original"])) > 1e-12:
        raise RuntimeError("Reconstructed original 100-repeat TAR mean R2 does not match official summary.")

    delta_mean_r2 = float(corrected_agg["mean_R2_original"] - original_agg["mean_R2_original"])
    relative_delta_mean_r2 = delta_mean_r2 / abs(float(original_agg["mean_R2_original"]))
    delta_mean_rmse = float(corrected_agg["mean_RMSE_original"] - original_agg["mean_RMSE_original"])
    relative_delta_mean_rmse = delta_mean_rmse / abs(float(original_agg["mean_RMSE_original"]))
    ranking = ranking_table(
        official_summary,
        tar_mean_r2=float(corrected_agg["mean_R2_original"]),
        tar_mean_rmse=float(corrected_agg["mean_RMSE_original"]),
    )
    max_pred_diff = float(pred_diff_df["max_abs_prediction_difference"].max())
    classification = classify_sensitivity(
        max_pred_diff=max_pred_diff,
        delta_mean_r2=delta_mean_r2,
        relative_delta_rmse=relative_delta_mean_rmse,
        ranking_changed=bool(ranking["ranking_changed"]),
    )

    aggregate_out = pd.concat(
        [
            original_agg_df.assign(source="official_100repeat_TAR_global_weights"),
            corrected_agg_df.assign(source="hybrid_100repeat_TAR_trainonly_on_19_affected"),
        ],
        ignore_index=True,
    )
    aggregate_out["delta_mean_R2"] = [np.nan, delta_mean_r2]
    aggregate_out["relative_delta_mean_R2"] = [np.nan, relative_delta_mean_r2]
    aggregate_out["delta_mean_RMSE"] = [np.nan, delta_mean_rmse]
    aggregate_out["relative_delta_mean_RMSE"] = [np.nan, relative_delta_mean_rmse]

    _write_csv(OUTDIR / "affected_repeat_comparison.csv", comparison_df)
    _write_csv(OUTDIR / "prediction_difference_summary.csv", pred_diff_df)
    _write_csv(OUTDIR / "corrected_100repeat_summary.csv", aggregate_out)
    _write_csv(OUTDIR / "corrected_100repeat_tar_by_repeat.csv", hybrid)

    source_hash_after = {p.as_posix(): _sha256_file(p) for p in PRODUCTION_SOURCE_FILES if p.is_file()}
    official_hash_after = {p.as_posix(): _sha256_file(p) for p in OFFICIAL_RESULT_FILES if p.is_file()}
    source_changed = [k for k in source_hash_before if source_hash_before[k] != source_hash_after.get(k)]
    official_changed = [k for k in official_hash_before if official_hash_before[k] != official_hash_after.get(k)]
    integrity = {
        "ode_simulation_rerun": False,
        "labels_regenerated": False,
        "production_source_modified": bool(source_changed),
        "official_results_overwritten": bool(official_changed),
        "comparators_retrained": False,
        "unaffected_repeats_retrained": False,
        "source_hash_mismatches": source_changed,
        "official_hash_mismatches": official_changed,
    }
    if source_changed or official_changed:
        raise RuntimeError(f"Integrity failure: source={source_changed}, official={official_changed}")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sensitivity_only_not_official_replacement": True,
        "n_affected_repeats": int(len(affected)),
        "n_unaffected_repeats_reused": EXPECTED_N_REPEATS - int(len(affected)),
        "affected_repeat_ids": affected["repeat_id"].astype(int).tolist(),
        "affected_seeds": affected["seed"].astype(int).tolist(),
        "global_max_bio_id": GLOBAL_MAX_BIO_ID,
        "split_config": {
            "split_mode": cfg["split_mode"],
            "group_col": cfg["group_col"],
            "test_size": cfg["test_size"],
            "target_transform": cfg["target_transform"],
            "expanded_tree_bank": cfg["expanded_tree_bank"],
            "max_tree_experts": cfg["max_tree_experts"],
            "repeat_ci_method": cfg["repeat_ci_method"],
        },
        "affected_mean_R2_original": float(comparison_df["mean_R2_original_original"].mean()),
        "affected_mean_R2_corrected": float(comparison_df["mean_R2_original_corrected"].mean()),
        "affected_mean_delta_R2": float(comparison_df["delta_R2"].mean()),
        "full100_mean_R2_original": float(original_agg["mean_R2_original"]),
        "full100_mean_R2_corrected": float(corrected_agg["mean_R2_original"]),
        "delta_mean_R2": delta_mean_r2,
        "relative_delta_mean_R2": relative_delta_mean_r2,
        "full100_mean_RMSE_original": float(original_agg["mean_RMSE_original"]),
        "full100_mean_RMSE_corrected": float(corrected_agg["mean_RMSE_original"]),
        "delta_mean_RMSE": delta_mean_rmse,
        "relative_delta_mean_RMSE": relative_delta_mean_rmse,
        "full100_mean_R2_original_ci": [
            float(original_agg["mean_R2_original_ci_low"]),
            float(original_agg["mean_R2_original_ci_high"]),
        ],
        "full100_mean_R2_corrected_ci": [
            float(corrected_agg["mean_R2_original_ci_low"]),
            float(corrected_agg["mean_R2_original_ci_high"]),
        ],
        "max_validation_prediction_difference": max_pred_diff,
        "mean_validation_prediction_difference": float(pred_diff_df["mean_abs_prediction_difference"].mean()),
        "n_repeats_tree_config_changed": int(comparison_df["tree_config_changed"].sum()),
        "ranking": ranking,
        "sensitivity_classification": classification,
        "integrity": integrity,
    }
    _write_text(
        OUTDIR / "sample_weight_model_sensitivity_summary.json",
        json.dumps(_json_native(summary), indent=2, ensure_ascii=False) + "\n",
    )
    report = build_report(
        affected=affected,
        cfg=cfg,
        comparison_df=comparison_df,
        pred_diff_df=pred_diff_df,
        original_agg=original_agg,
        corrected_agg=corrected_agg,
        ranking=ranking,
        classification=classification,
        integrity=integrity,
    )
    _write_text(OUTDIR / "sample_weight_model_sensitivity_report.md", report)

    print()
    print("=" * 60)
    print("TRAIN-ONLY WEIGHT MODEL SENSITIVITY")
    print("=" * 60)
    print(f"Affected repeats rerun: {len(affected)}/100")
    print(f"Unaffected repeats reused: {EXPECTED_N_REPEATS - len(affected)}/100")
    print()
    print("Affected-repeat R2:")
    print(f"  Original mean: {comparison_df['mean_R2_original_original'].mean():.12g}")
    print(f"  Corrected mean: {comparison_df['mean_R2_original_corrected'].mean():.12g}")
    print(f"  Difference: {comparison_df['delta_R2'].mean():.12g}")
    print()
    print("Corrected full 100-repeat TAR:")
    print(f"  Original mean R2: {original_agg['mean_R2_original']:.12g}")
    print(f"  Corrected mean R2: {corrected_agg['mean_R2_original']:.12g}")
    print(f"  Delta: {delta_mean_r2:.12g}")
    print()
    print(f"  Original RMSE: {original_agg['mean_RMSE_original']:.12g}")
    print(f"  Corrected RMSE: {corrected_agg['mean_RMSE_original']:.12g}")
    print(f"  Delta: {delta_mean_rmse:.12g}")
    print()
    print(f"Maximum validation prediction difference: {max_pred_diff:.12g}")
    print(f"Mean validation prediction difference: {pred_diff_df['mean_abs_prediction_difference'].mean():.12g}")
    print()
    print(f"TAR ranking versus final comparators changed: {'YES' if ranking['ranking_changed'] else 'NO'}")
    print()
    print("SENSITIVITY CLASSIFICATION:")
    print(classification)
    print("=" * 60)
    print()
    print("Integrity:")
    print("  raw ODE simulations rerun: NO")
    print("  labels regenerated: NO")
    print("  production files modified: NO")
    print("  official result files overwritten: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
