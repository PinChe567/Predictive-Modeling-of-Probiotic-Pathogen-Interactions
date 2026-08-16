"""ODE-back functional validation: predicted Tthr -> ODE outcomes vs reference."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error

from closed_loop_eval import clip_tthr
from multi_pathogen_simulator import N_STRAINS, PAPER_FIGURE_PROFILE
from simulate_case_metrics_fast import (
    metrics_vector_to_dict,
    simulate_case_metrics_and_trajectory_fast,
    simulate_case_metrics_fast,
    validate_fast_backend,
    effective_backend,
)
from tree_srl_benchmark import (
    BEST_SINGLE_TREE,
    FORMAL_SIGNIFICANCE_CONTROLS,
    LEGACY_MODEL_NAME_MAP,
    RANDOM_FOREST,
    TAR_MODEL,
    UNIFORM_TREE_MEAN,
    build_paired_repeated_significance,
    load_repeat_split_ratios,
    normalize_model_name,
    print_holm_caption_stats,
    repeat_metric_ci,
    safe_r2_score,
)

ODE_BACK_REPEATED_METRICS_CSV = "ode_back_repeated_metrics.csv"
ODE_BACK_PAIRWISE_SIGNIFICANCE_CSV = "ode_back_pairwise_significance.csv"
ODE_BACK_BAR_METRIC = "mean_outcome_R2"
ODE_BACK_TRAJECTORY_R2_METRIC = "trajectory_R2"
ODE_BACK_TRAJECTORY_REPEATED_CSV = "ode_back_trajectory_repeated_metrics.csv"
ODE_BACK_TRAJECTORY_PAIRWISE_CSV = "ode_back_trajectory_pairwise_significance.csv"

DIRECT_RULE_UNCLIPPED = "DirectRuleUnclipped"
DIRECT_RULE_CLIPPED = "DirectRuleClipped"
DIRECT_RULE_MODELS = (DIRECT_RULE_UNCLIPPED, DIRECT_RULE_CLIPPED)
DIRECT_DISPLAY_NAMES = {
    TAR_MODEL: "TAR",
    DIRECT_RULE_UNCLIPPED: "Direct",
    DIRECT_RULE_CLIPPED: "Direct\n(clipped)",
}
DIRECT_MODEL_PLOT_ORDER = (TAR_MODEL, DIRECT_RULE_UNCLIPPED, DIRECT_RULE_CLIPPED)
DIRECT_CASE_RESULTS_CSV = "direct_threshold_case_results.csv"
DIRECT_SUMMARY_CSV = "direct_threshold_summary.csv"
DIRECT_SUPPORT_AUDIT_CSV = "direct_threshold_support_audit.csv"
DIRECT_COMPARISON_PNG = "direct_threshold_comparison.png"
DIRECT_TOTAL_DOSAGE_UNIT = "ug/mL"
DIRECT_BOOTSTRAP_SEED = 42
DIRECT_BOOTSTRAP_N = 2000
# Formal raw-library / Sobol threshold grid endpoints (CFU/mL).
DIRECT_TTHR_CLIP_LO = 8.0e6
DIRECT_TTHR_CLIP_HI = 6.4036102786783e7
DIRECT_PAUC_FEASIBILITY_FRACTION = 0.90
DIRECT_PATHOGEN_CEILING = 4.0e7
DIRECT_CONSTRAINT_TOL = 1e-6
DIRECT_LR_TOLERANCE = 0.0

LR_TRACKING_ERROR_DEFINITION = (
    "LR_tracking_error = (1/5) * sum_{i=1..5} |LR_achieved_i - LR_target_i| "
    "(per case; arithmetic mean of absolute strain-wise LR deviations; "
    "not RMSE; equals mean absolute error over the five strains only)."
)

OUTCOME_COLS_REFERENCE = [
    "P_AUC",
    "LR1",
    "LR2",
    "LR3",
    "LR4",
    "LR5",
    "mean_LR",
    "total_dosage",
    "dose_count",
    "final_total_pathogen",
    "terminal_total_pathogen",
    "log10_terminal_total_pathogen",
    "log10_final_total_pathogen",
]

OUTCOME_COLS_METRICS = [
    "P_AUC",
    "LR1",
    "LR2",
    "LR3",
    "LR4",
    "LR5",
    "mean_LR",
    "total_dosage",
    "dose_count",
    "log10_terminal_total_pathogen",
    "log10_final_total_pathogen",
]

MEAN_OUTCOME_R2_COMPONENTS = ["P_AUC", "LR1", "LR2", "LR3", "LR4", "LR5", "log10_terminal_total_pathogen"]
LR_OUTCOME_COLS = [f"LR{i}" for i in range(1, 6)]

PRED_COL_PATTERN = re.compile(r"^pred_(?P<model>.+)_Tthr_(?P<idx>[1-5])$", re.IGNORECASE)


def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.std(y_true))
    if not np.isfinite(denom) or denom <= 1e-30:
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)) / denom)


def resolve_gamma_cols(columns: Sequence[str]) -> List[str]:
    if all(f"g_{i}" in columns for i in range(1, 6)):
        return [f"g_{i}" for i in range(1, 6)]
    if all(f"gamma_{i}" in columns for i in range(1, 6)):
        return [f"gamma_{i}" for i in range(1, 6)]
    raise ValueError("X_features must contain g_1..g_5 or gamma_1..gamma_5")


def resolve_desired_lr_cols(columns: Sequence[str]) -> List[str]:
    for pattern in ([f"desired_LR{i}" for i in range(1, 6)], [f"desired_LR_{i}" for i in range(1, 6)]):
        if all(c in columns for c in pattern):
            return pattern
    return []


def resolve_desired_pauc_col(columns: Sequence[str]) -> Optional[str]:
    for name in ("desired_P_AUC", "desired_pauc", "desired_p_auc"):
        if name in columns:
            return name
    return None


def load_bundle_tables(
    x_csv: str,
    y_csv: str,
    metadata_csv: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    x_df = pd.read_csv(x_csv)
    y_df = pd.read_csv(y_csv)
    meta_df = pd.read_csv(metadata_csv) if metadata_csv and os.path.isfile(metadata_csv) else None
    return x_df, y_df, meta_df


def discover_prediction_files(predictions_dir: str, predictions_manifest: Optional[str]) -> List[str]:
    paths: List[str] = []
    if predictions_manifest and os.path.isfile(predictions_manifest):
        with open(predictions_manifest, encoding="utf-8") as fh:
            manifest = json.load(fh)
        base = os.path.dirname(os.path.abspath(predictions_manifest))
        for rel in manifest.get("prediction_files", []):
            path = os.path.join(base, rel.replace("/", os.sep))
            if os.path.isfile(path) and path.endswith("predictions.csv"):
                paths.append(path)
    if not paths:
        paths = sorted(glob.glob(os.path.join(predictions_dir, "repeat_*", "predictions.csv")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(predictions_dir, "predictions.csv")))
    return paths


def parse_prediction_models(df: pd.DataFrame, requested_models: Sequence[str]) -> Dict[str, List[str]]:
    """Map canonical model name -> list of 5 prediction column names."""
    found: Dict[str, Dict[int, str]] = {}
    for col in df.columns:
        match = PRED_COL_PATTERN.match(col)
        if not match:
            continue
        raw_model = match.group("model")
        canonical = normalize_model_name(raw_model)
        idx = int(match.group("idx"))
        found.setdefault(canonical, {})[idx] = col

    resolved: Dict[str, List[str]] = {}
    missing_models: List[str] = []
    for model in requested_models:
        canonical = normalize_model_name(model)
        if canonical not in found or len(found[canonical]) < 5:
            missing_models.append(model)
            continue
        cols = [found[canonical][i] for i in range(1, 6)]
        resolved[canonical] = cols

    if missing_models:
        warnings.warn(
            f"Missing prediction columns for models: {missing_models}. "
            f"Expected pred_{{model}}_Tthr_1..5 (legacy TAR-SRL maps to TAR).",
            stacklevel=2,
        )
    return resolved


def bio_vector_from_row(x_df: pd.DataFrame, row_index: int, gamma_cols: List[str]) -> Tuple[np.ndarray, ...]:
    row = x_df.iloc[int(row_index)]
    b0 = row[[f"B0_{i}" for i in range(1, 6)]].to_numpy(dtype=float)
    k_arr = row[[f"k_{i}" for i in range(1, 6)]].to_numpy(dtype=float)
    gamma_arr = row[gamma_cols].to_numpy(dtype=float)
    rho_arr = row[[f"rho_{i}" for i in range(1, 6)]].to_numpy(dtype=float)
    mu_arr = row[[f"mu_{i}" for i in range(1, 6)]].to_numpy(dtype=float)
    return b0, k_arr, gamma_arr, rho_arr, mu_arr


def reference_tthr_from_row(y_df: pd.DataFrame, row_index: int) -> np.ndarray:
    row = y_df.iloc[int(row_index)]
    cols = [f"Tthr_{i}" for i in range(1, 6)]
    missing = [c for c in cols if c not in row.index]
    if missing:
        raise ValueError(f"y_targets missing columns: {missing}")
    return row[cols].to_numpy(dtype=float)


def resolve_training_soft_umax_median(
    metadata_df: Optional[pd.DataFrame],
    bundle_dir_hint: Optional[str],
) -> Tuple[float, str]:
    if bundle_dir_hint:
        for rel in ("bundle_manifest.json", "relabel_manifest.json"):
            path = os.path.join(bundle_dir_hint, rel)
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    manifest = json.load(fh)
                for key in ("training_soft_u_max_median", "soft_u_max_median"):
                    if key in manifest and np.isfinite(manifest[key]):
                        return float(manifest[key]), f"{rel}:{key}"

    if metadata_df is not None and "soft_u_max" in metadata_df.columns:
        vals = metadata_df["soft_u_max"].dropna().to_numpy(dtype=float)
        if vals.size:
            return float(np.median(vals)), "metadata_soft_u_max_median_all_rows"

    raise ValueError(
        "training_median requires soft_u_max in metadata or training_soft_u_max_median in bundle manifest."
    )


def resolve_functional_umax(
    source: str,
    constant_value: float,
    row_index: int,
    metadata_df: Optional[pd.DataFrame],
    bundle_dir_hint: Optional[str],
) -> Tuple[float, str]:
    if source == "constant":
        return float(constant_value), "constant"
    if source == "metadata_soft_umax":
        if metadata_df is None or "soft_u_max" not in metadata_df.columns:
            raise ValueError("metadata_soft_umax requires soft_u_max in sample_metadata.csv")
        val = metadata_df.iloc[int(row_index)].get("soft_u_max")
        if pd.isna(val):
            raise ValueError(f"soft_u_max missing for row_index={row_index}")
        return float(val), "metadata_soft_umax"
    if source == "training_median":
        median, note = resolve_training_soft_umax_median(metadata_df, bundle_dir_hint)
        return median, f"training_median:{note}"
    raise ValueError(f"Unknown functional_umax_source: {source}")


def simulate_outcomes(
    bio: Tuple[np.ndarray, ...],
    tthr: np.ndarray,
    u_max: float,
    backend: str,
    *,
    return_trajectory_features: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], np.ndarray]:
    b0, k_arr, gamma_arr, rho_arr, mu_arr = bio
    tthr = clip_tthr(tthr)
    if return_trajectory_features:
        metrics, traj = simulate_case_metrics_and_trajectory_fast(
            b0, k_arr, gamma_arr, rho_arr, mu_arr, u_max, tthr, backend=backend
        )
        return metrics_vector_to_dict(metrics), traj
    metrics = simulate_case_metrics_fast(
        b0, k_arr, gamma_arr, rho_arr, mu_arr, u_max, tthr, backend=backend
    )
    return metrics_vector_to_dict(metrics)


def _relative_error(pred: float, ref: float) -> float:
    denom = max(abs(ref), 1e-12)
    return float((pred - ref) / denom)


def build_case_row(
    *,
    repeat_id: int,
    validation_original_row_index: int,
    case_index: int,
    model: str,
    functional_umax: float,
    umax_source_note: str,
    reference_mode: str,
    ref_tthr: np.ndarray,
    pred_tthr: np.ndarray,
    ref_outcomes: Dict[str, float],
    pred_outcomes: Dict[str, float],
    trajectory_r2_case: Optional[float] = None,
) -> dict:
    row = {
        "repeat_id": int(repeat_id),
        "validation_original_row_index": int(validation_original_row_index),
        "case_index": int(case_index),
        "model": model,
        "functional_umax": float(functional_umax),
        "functional_umax_source_note": umax_source_note,
        "reference_mode": reference_mode,
    }
    for i in range(1, 6):
        row[f"ref_Tthr_{i}"] = float(ref_tthr[i - 1])
        row[f"pred_Tthr_{i}"] = float(pred_tthr[i - 1])
    for outcome in OUTCOME_COLS_REFERENCE:
        ref_val = ref_outcomes.get(outcome, np.nan)
        pred_val = pred_outcomes.get(outcome, np.nan)
        row[f"ref_{outcome}"] = ref_val
        row[f"pred_{outcome}"] = pred_val
        row[f"abs_error_{outcome}"] = float(abs(pred_val - ref_val)) if np.isfinite(pred_val) and np.isfinite(ref_val) else np.nan
        row[f"relative_error_{outcome}"] = (
            _relative_error(pred_val, ref_val)
            if np.isfinite(pred_val) and np.isfinite(ref_val)
            else np.nan
        )
    if trajectory_r2_case is not None:
        row["trajectory_R2_case"] = float(trajectory_r2_case)
    return row


def process_prediction_file(
    pred_path: str,
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame],
    model_cols: Dict[str, List[str]],
    reference_mode: str,
    umax_source: str,
    umax_constant: float,
    bundle_dir_hint: Optional[str],
    backend: str,
    desired_pauc_col: Optional[str],
    desired_lr_cols: List[str],
    *,
    compute_trajectory_r2: bool = True,
) -> Tuple[List[dict], List[dict]]:
    pred_df = pd.read_csv(pred_path)
    if "repeat_id" in pred_df.columns and pred_df["repeat_id"].notna().any():
        repeat_id = int(pred_df["repeat_id"].iloc[0])
    else:
        base = os.path.basename(os.path.dirname(pred_path))
        m = re.search(r"repeat_(\d+)", base)
        repeat_id = int(m.group(1)) if m else 0

    gamma_cols = resolve_gamma_cols(x_df.columns)
    rows: List[dict] = []
    traj_pred_by_model: Dict[str, List[np.ndarray]] = {model: [] for model in model_cols}
    traj_ref_all: List[np.ndarray] = []

    for case_index, pred_row in pred_df.iterrows():
        row_index = int(pred_row["validation_original_row_index"])
        bio = bio_vector_from_row(x_df, row_index, gamma_cols)
        ref_tthr = reference_tthr_from_row(y_df, row_index)

        u_max, umax_note = resolve_functional_umax(
            umax_source, umax_constant, row_index, metadata_df, bundle_dir_hint
        )

        if compute_trajectory_r2 and reference_mode == "reference_tthr":
            ref_outcomes, ref_traj = simulate_outcomes(
                bio, ref_tthr, u_max, backend, return_trajectory_features=True
            )
            y_ref_by_outcome = ref_outcomes
            traj_ref_all.append(ref_traj)
        elif reference_mode == "reference_tthr":
            ref_outcomes = simulate_outcomes(bio, ref_tthr, u_max, backend)
            y_ref_by_outcome = ref_outcomes
            ref_traj = None
        else:
            ref_outcomes = {}
            y_ref_by_outcome = {}
            ref_traj = None
            if desired_pauc_col and metadata_df is not None:
                y_ref_by_outcome["P_AUC"] = float(metadata_df.iloc[row_index][desired_pauc_col])
            for i, col in enumerate(desired_lr_cols, start=1):
                if metadata_df is not None:
                    y_ref_by_outcome[f"LR{i}"] = float(metadata_df.iloc[row_index][col])
            if y_ref_by_outcome:
                lr_vals = [y_ref_by_outcome.get(f"LR{i}", np.nan) for i in range(1, 6)]
                finite_lr = [v for v in lr_vals if np.isfinite(v)]
                if finite_lr:
                    y_ref_by_outcome["mean_LR"] = float(np.mean(finite_lr))

        for model_name, cols in model_cols.items():
            pred_tthr = pred_row[cols].to_numpy(dtype=float)
            traj_r2_case = None
            if compute_trajectory_r2 and reference_mode == "reference_tthr" and ref_traj is not None:
                pred_outcomes, pred_traj = simulate_outcomes(
                    bio, pred_tthr, u_max, backend, return_trajectory_features=True
                )
                traj_pred_by_model[model_name].append(pred_traj)
                traj_r2_case = float(safe_r2_score(ref_traj, pred_traj))
            else:
                pred_outcomes = simulate_outcomes(bio, pred_tthr, u_max, backend)

            if reference_mode == "reference_tthr":
                row = build_case_row(
                    repeat_id=repeat_id,
                    validation_original_row_index=row_index,
                    case_index=case_index,
                    model=model_name,
                    functional_umax=u_max,
                    umax_source_note=umax_note,
                    reference_mode=reference_mode,
                    ref_tthr=ref_tthr,
                    pred_tthr=pred_tthr,
                    ref_outcomes=ref_outcomes,
                    pred_outcomes=pred_outcomes,
                    trajectory_r2_case=traj_r2_case,
                )
            else:
                ref_for_row = {k: y_ref_by_outcome.get(k, np.nan) for k in OUTCOME_COLS_REFERENCE}
                for k, v in pred_outcomes.items():
                    ref_for_row.setdefault(k, y_ref_by_outcome.get(k, np.nan))
                row = build_case_row(
                    repeat_id=repeat_id,
                    validation_original_row_index=row_index,
                    case_index=case_index,
                    model=model_name,
                    functional_umax=u_max,
                    umax_source_note=umax_note,
                    reference_mode=reference_mode,
                    ref_tthr=ref_tthr,
                    pred_tthr=pred_tthr,
                    ref_outcomes=ref_for_row,
                    pred_outcomes=pred_outcomes,
                    trajectory_r2_case=traj_r2_case,
                )
            rows.append(row)

    repeat_traj_rows: List[dict] = []
    if compute_trajectory_r2 and reference_mode == "reference_tthr" and traj_ref_all:
        ref_concat = np.concatenate(traj_ref_all)
        for model_name, pred_list in traj_pred_by_model.items():
            if not pred_list:
                continue
            pred_concat = np.concatenate(pred_list)
            repeat_traj_rows.append(
                {
                    "repeat_id": int(repeat_id),
                    "model": model_name,
                    ODE_BACK_TRAJECTORY_R2_METRIC: float(safe_r2_score(ref_concat, pred_concat)),
                    "n_cases": int(len(pred_list)),
                    "n_trajectory_points_per_case": int(len(pred_list[0])),
                }
            )

    return rows, repeat_traj_rows


def compute_outcome_metrics(
    case_df: pd.DataFrame,
    outcome: str,
    reference_mode: str,
    has_desired_pathogen: bool,
) -> pd.DataFrame:
    if outcome.startswith("log10_terminal") and reference_mode == "desired_targets" and not has_desired_pathogen:
        return pd.DataFrame()

    ref_col = f"ref_{outcome}"
    pred_col = f"pred_{outcome}"
    if ref_col not in case_df.columns or pred_col not in case_df.columns:
        return pd.DataFrame()

    rows: List[dict] = []
    for model_name, group in case_df.groupby("model"):
        y_true = group[ref_col].to_numpy(dtype=float)
        y_pred = group[pred_col].to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if int(mask.sum()) < 2:
            rows.append(
                {
                    "model": model_name,
                    "outcome": outcome,
                    "R2": np.nan,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "NRMSE": np.nan,
                    "Spearman_rho": np.nan,
                    "Pearson_r": np.nan,
                    "n_points": int(mask.sum()),
                }
            )
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        spear = float(spearmanr(yt, yp).correlation) if len(yt) > 1 else np.nan
        pear = float(pearsonr(yt, yp).statistic) if len(yt) > 1 else np.nan
        rows.append(
            {
                "model": model_name,
                "outcome": outcome,
                "R2": safe_r2_score(yt, yp),
                "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
                "MAE": float(mean_absolute_error(yt, yp)),
                "NRMSE": normalized_rmse(yt, yp),
                "Spearman_rho": spear,
                "Pearson_r": pear,
                "n_points": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def aggregate_summary_by_model(
    per_outcome_df: pd.DataFrame,
    case_df: pd.DataFrame,
    reference_mode: str,
    has_desired_pathogen: bool,
) -> pd.DataFrame:
    rows: List[dict] = []
    for model_name, group in case_df.groupby("model"):
        sub = per_outcome_df[per_outcome_df["model"] == model_name]
        r2_map = {row["outcome"]: row["R2"] for _, row in sub.iterrows()}

        mean_outcome_parts = [r2_map.get(o, np.nan) for o in MEAN_OUTCOME_R2_COMPONENTS]
        if reference_mode == "desired_targets" and not has_desired_pathogen:
            mean_outcome_parts = [r2_map.get(o, np.nan) for o in ("P_AUC", *LR_OUTCOME_COLS)]

        mean_outcome_r2 = float(np.nanmean(mean_outcome_parts)) if np.any(np.isfinite(mean_outcome_parts)) else np.nan
        mean_lr_r2 = float(np.nanmean([r2_map.get(o, np.nan) for o in LR_OUTCOME_COLS]))

        abs_cols = [c for c in group.columns if c.startswith("abs_error_")]
        rel_cols = [c for c in group.columns if c.startswith("relative_error_")]
        mean_abs = float(np.nanmean(group[abs_cols].to_numpy(dtype=float))) if abs_cols else np.nan
        mean_rel = float(np.nanmean(np.abs(group[rel_cols].to_numpy(dtype=float)))) if rel_cols else np.nan

        rmse_vals = sub["RMSE"].to_numpy(dtype=float) if not sub.empty else np.array([])
        mae_vals = sub["MAE"].to_numpy(dtype=float) if not sub.empty else np.array([])

        rows.append(
            {
                "model": model_name,
                "mean_outcome_R2": mean_outcome_r2,
                "mean_LR_R2": mean_lr_r2,
                "P_AUC_R2": r2_map.get("P_AUC", np.nan),
                "log10_terminal_pathogen_R2": r2_map.get("log10_terminal_total_pathogen", np.nan),
                "total_dosage_R2": r2_map.get("total_dosage", np.nan),
                "mean_RMSE": float(np.nanmean(rmse_vals)) if rmse_vals.size else np.nan,
                "mean_MAE": float(np.nanmean(mae_vals)) if mae_vals.size else np.nan,
                "mean_abs_functional_error": mean_abs,
                "mean_relative_functional_error": mean_rel,
                "n_cases": int(group["case_index"].nunique()),
                "n_repeats": int(group["repeat_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_ode_back_repeat_metrics(case_df: pd.DataFrame) -> pd.DataFrame:
    """Per-repeat functional mean_outcome_R2 (one row per repeat × model)."""
    rows: List[dict] = []
    for (repeat_id, model), group in case_df.groupby(["repeat_id", "model"]):
        r2_vals: List[float] = []
        for outcome in MEAN_OUTCOME_R2_COMPONENTS:
            ref_col = f"ref_{outcome}"
            pred_col = f"pred_{outcome}"
            if ref_col not in group.columns:
                continue
            yt = group[ref_col].to_numpy(dtype=float)
            yp = group[pred_col].to_numpy(dtype=float)
            r2_vals.append(safe_r2_score(yt, yp))
        rows.append(
            {
                "repeat_id": int(repeat_id),
                "model": model,
                ODE_BACK_BAR_METRIC: float(np.nanmean(r2_vals)) if r2_vals else np.nan,
            }
        )
    return pd.DataFrame(rows)


def repeat_level_summary(
    repeat_metrics_df: pd.DataFrame,
    models: Sequence[str],
    metric_col: str = ODE_BACK_BAR_METRIC,
) -> pd.DataFrame:
    """Aggregate repeat-level metric with 95% CI for bar plot."""
    if repeat_metrics_df.empty:
        return pd.DataFrame(
            columns=["model", metric_col, f"{metric_col}_ci_low", f"{metric_col}_ci_high"]
        )
    summary_rows: List[dict] = []
    for model in models:
        sub = repeat_metrics_df.loc[repeat_metrics_df["model"] == model, metric_col].to_numpy(dtype=float)
        mean, lo, hi = repeat_metric_ci(sub)
        summary_rows.append(
            {
                "model": model,
                metric_col: mean,
                f"{metric_col}_ci_low": lo,
                f"{metric_col}_ci_high": hi,
            }
        )
    return pd.DataFrame(summary_rows)


def resolve_benchmark_outdir_for_splits(outdir: str, prediction_files: Optional[Sequence[str]] = None) -> str:
    """Locate the tree_srl_benchmark outdir that owns repeat_metadata.csv for NB ratios."""
    candidates: List[str] = []
    if prediction_files:
        for pred in prediction_files:
            abs_pred = pred if os.path.isabs(pred) else os.path.normpath(os.path.join(outdir, pred))
            # .../tree_srl_benchmark/repeats/repeat_XXX/predictions.csv
            cur = abs_pred
            for _ in range(4):
                cur = os.path.dirname(cur)
                candidates.append(cur)
    sibling = os.path.normpath(os.path.join(outdir, "..", "tree_srl_benchmark"))
    candidates.append(sibling)
    candidates.append(os.path.abspath(os.path.join("results", "tree_srl_benchmark")))
    seen = set()
    for cand in candidates:
        cand = os.path.abspath(cand)
        if cand in seen:
            continue
        seen.add(cand)
        meta_glob = os.path.join(cand, "repeats", "repeat_*", "repeat_metadata.csv")
        if glob.glob(meta_glob):
            return cand
    raise FileNotFoundError(
        "Could not locate tree_srl_benchmark repeat_metadata.csv for train/test bio-group "
        f"ratios (ode_back outdir={outdir})."
    )


def build_ode_back_pairwise_significance(
    repeat_metrics_df: pd.DataFrame,
    *,
    srl_model: str = TAR_MODEL,
    control_models: Sequence[str],
    metric_col: str = ODE_BACK_BAR_METRIC,
    n_perm: int = 10_000,
    seed: int = 42,
    bidirectional_plot_stars: bool = True,
    split_df: Optional[pd.DataFrame] = None,
    benchmark_outdir: Optional[str] = None,
    outdir: Optional[str] = None,
    prediction_files: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Paired repeat-level significance via Nadeau–Bengio + Holm (Wilcoxon/perm sensitivity only)."""
    long_df = repeat_metrics_df.copy()
    if "model" in long_df.columns:
        long_df["model"] = long_df["model"].map(normalize_model_name)
    if split_df is None:
        bench = benchmark_outdir or resolve_benchmark_outdir_for_splits(
            outdir or ".", prediction_files=prediction_files
        )
        split_df = load_repeat_split_ratios(bench)
    formal = [m for m in FORMAL_SIGNIFICANCE_CONTROLS if m in set(map(normalize_model_name, control_models))]
    return build_paired_repeated_significance(
        srl_model=normalize_model_name(srl_model),
        control_models=[normalize_model_name(m) for m in control_models],
        n_perm=n_perm,
        seed=seed,
        single_split_exploratory=int(long_df["repeat_id"].nunique()) < 2,
        long_df=long_df,
        split_df=split_df,
        formal_controls=formal,
        metric_direction={metric_col: "higher_is_better"},
        bidirectional=bidirectional_plot_stars,
    )


def recompute_ode_back_pairwise_significance_from_saved(
    outdir: str,
    *,
    n_perm: int = 10_000,
    seed: int = 42,
    benchmark_outdir: Optional[str] = None,
) -> pd.DataFrame:
    """Rebuild ode_back_pairwise_significance.csv from saved repeat-level metrics."""
    metrics_path = os.path.join(outdir, ODE_BACK_REPEATED_METRICS_CSV)
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Missing {metrics_path}")
    repeat_metrics_df = pd.read_csv(metrics_path)
    pred_files = None
    manifest_path = os.path.join(outdir, "ode_back_validation_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        pred_files = manifest.get("prediction_files")
    control_models = [
        m
        for m in dict.fromkeys(repeat_metrics_df["model"].map(normalize_model_name))
        if m != TAR_MODEL
    ]
    pairwise_df = build_ode_back_pairwise_significance(
        repeat_metrics_df,
        srl_model=TAR_MODEL,
        control_models=control_models,
        n_perm=n_perm,
        seed=seed,
        outdir=outdir,
        benchmark_outdir=benchmark_outdir,
        prediction_files=pred_files,
    )
    pairwise_df.to_csv(os.path.join(outdir, ODE_BACK_PAIRWISE_SIGNIFICANCE_CSV), index=False)
    return pairwise_df


def direct_tthr_from_b0_and_lr(b0: np.ndarray, lr_target: np.ndarray) -> np.ndarray:
    """Tthr_i = B0_i * 10**(-LR_i); LR is base-10 (do not use exp)."""
    b0 = np.asarray(b0, dtype=float).reshape(N_STRAINS)
    lr_target = np.asarray(lr_target, dtype=float).reshape(N_STRAINS)
    return b0 * np.power(10.0, -lr_target)


def clip_direct_tthr_to_sampled_range(tthr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(tthr, dtype=float), DIRECT_TTHR_CLIP_LO, DIRECT_TTHR_CLIP_HI)


def lr_targets_for_row(
    metadata_df: Optional[pd.DataFrame],
    row_index: int,
    desired_lr_cols: Sequence[str],
    x_df: pd.DataFrame,
) -> np.ndarray:
    if metadata_df is not None and desired_lr_cols and all(c in metadata_df.columns for c in desired_lr_cols):
        return metadata_df.iloc[int(row_index)][list(desired_lr_cols)].to_numpy(dtype=float)
    lr_cols = [f"LR{i}" for i in range(1, 6)]
    if all(c in x_df.columns for c in lr_cols):
        return x_df.iloc[int(row_index)][lr_cols].to_numpy(dtype=float)
    raise ValueError("Need desired_LR1..5 in metadata or LR1..5 in X_features for direct-threshold baseline.")


def desired_pauc_for_row(
    metadata_df: Optional[pd.DataFrame],
    row_index: int,
    desired_pauc_col: Optional[str],
    x_df: pd.DataFrame,
) -> float:
    if metadata_df is not None and desired_pauc_col and desired_pauc_col in metadata_df.columns:
        return float(metadata_df.iloc[int(row_index)][desired_pauc_col])
    if "P_AUC" in x_df.columns:
        return float(x_df.iloc[int(row_index)]["P_AUC"])
    raise ValueError("Need desired_P_AUC in metadata or P_AUC in X_features for constraint checks.")


def direct_constraint_flags(
    *,
    achieved_lr: np.ndarray,
    p_auc: float,
    bterminal: float,
    target_lr: np.ndarray,
    target_pauc: float,
) -> Dict[str, bool]:
    min_pauc = DIRECT_PAUC_FEASIBILITY_FRACTION * float(target_pauc)
    lr_ok = bool(
        np.all(np.asarray(achieved_lr, dtype=float) >= np.asarray(target_lr, dtype=float) - DIRECT_LR_TOLERANCE)
    )
    pauc_ok = bool(float(p_auc) >= min_pauc - DIRECT_CONSTRAINT_TOL)
    pathogen_ok = bool(float(bterminal) <= DIRECT_PATHOGEN_CEILING + DIRECT_CONSTRAINT_TOL)
    return {
        "LR_constraint_satisfied": lr_ok,
        "P_AUC_constraint_satisfied": pauc_ok,
        "pathogen_constraint_satisfied": pathogen_ok,
        "all_constraints_satisfied": bool(lr_ok and pauc_ok and pathogen_ok),
    }


def _mean_lr_tracking_error(achieved_lr: np.ndarray, target_lr: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(achieved_lr, dtype=float) - np.asarray(target_lr, dtype=float))))


def print_lr_tracking_error_definition() -> None:
    print("LR tracking error definition:")
    print(f"  {LR_TRACKING_ERROR_DEFINITION}")


def build_direct_threshold_support_audit(unclipped_tthr: np.ndarray) -> dict:
    flat = np.asarray(unclipped_tthr, dtype=float).reshape(-1)
    n_total = int(flat.size)
    n_below = int(np.sum(flat < DIRECT_TTHR_CLIP_LO))
    n_above = int(np.sum(flat > DIRECT_TTHR_CLIP_HI))
    n_outside = int(n_below + n_above)
    return {
        "n_threshold_values": n_total,
        "n_outside_sampled_range": n_outside,
        "pct_outside_sampled_range": float(100.0 * n_outside / n_total) if n_total else float("nan"),
        "n_below_lo": n_below,
        "n_above_hi": n_above,
        "pct_below_lo": float(100.0 * n_below / n_total) if n_total else float("nan"),
        "pct_above_hi": float(100.0 * n_above / n_total) if n_total else float("nan"),
        "sampled_range_lo_CFU_per_mL": DIRECT_TTHR_CLIP_LO,
        "sampled_range_hi_CFU_per_mL": DIRECT_TTHR_CLIP_HI,
    }


def _attach_bio_id_to_direct_cases(
    case_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    work = case_df.copy()
    if "bio_id" in work.columns and work["bio_id"].notna().any():
        work["bio_id"] = work["bio_id"].astype(np.int64)
        return work
    if metadata_df is None:
        work["bio_id"] = np.nan
        return work
    meta = metadata_df.copy()
    if "row_id" in meta.columns:
        lookup = meta.set_index("row_id")["bio_id"]
        work["bio_id"] = work["validation_original_row_index"].map(lookup)
    elif len(meta) > int(work["validation_original_row_index"].max()):
        work["bio_id"] = meta.iloc[work["validation_original_row_index"].to_numpy(dtype=int)]["bio_id"].to_numpy()
    else:
        work["bio_id"] = np.nan
    return work


def audit_direct_threshold_case_matching(case_df: pd.DataFrame) -> None:
    """Fail loudly if TAR / Direct / Direct(clipped) are not matched on shared case inputs."""
    required_models = set(DIRECT_MODEL_PLOT_ORDER)
    present = set(case_df["model"].astype(str))
    missing = required_models - present
    if missing:
        raise ValueError(f"direct_threshold_case_results.csv missing models: {sorted(missing)}")

    keys = ["repeat_id", "validation_original_row_index", "case_index"]
    for col in keys:
        if col not in case_df.columns:
            raise ValueError(f"direct_threshold_case_results.csv missing key column: {col}")

    id_sets = {
        model: set(map(tuple, case_df.loc[case_df["model"] == model, keys].to_numpy()))
        for model in DIRECT_MODEL_PLOT_ORDER
    }
    ref_ids = id_sets[TAR_MODEL]
    for model in DIRECT_MODEL_PLOT_ORDER[1:]:
        if id_sets[model] != ref_ids:
            only_tar = len(ref_ids - id_sets[model])
            only_model = len(id_sets[model] - ref_ids)
            raise ValueError(
                f"Held-out case identifiers are not identical for TAR vs {model} "
                f"(only_TAR={only_tar}, only_{model}={only_model})."
            )

    match_cols = (
        ["functional_umax", "functional_umax_source_note"]
        + [f"B0_{i}" for i in range(1, 6)]
        + [f"LR_target_{i}" for i in range(1, 6)]
        + [f"Tthr_direct_unclipped_{i}" for i in range(1, 6)]
    )
    for col in match_cols:
        if col not in case_df.columns:
            raise ValueError(f"Cannot audit matched inputs; missing column: {col}")

    pivoted = case_df.pivot_table(index=keys, columns="model", values=match_cols, aggfunc="first")
    for col in match_cols:
        block = pivoted[col][list(DIRECT_MODEL_PLOT_ORDER)]
        if col == "functional_umax_source_note":
            ok = block.apply(lambda r: r.nunique(dropna=False) == 1, axis=1)
        else:
            arr = block.to_numpy(dtype=float)
            ok = np.all(np.isclose(arr, arr[:, :1], rtol=1e-12, atol=1e-6, equal_nan=True), axis=1)
        if not bool(np.all(ok)):
            n_bad = int(np.size(ok) - np.sum(ok))
            raise ValueError(
                f"Matched-input audit failed for '{col}': {n_bad} case keys differ across "
                "TAR / DirectRuleUnclipped / DirectRuleClipped "
                "(biological parameters, Umax, and/or controller-linked targets)."
            )

    source_notes = sorted(case_df["functional_umax_source_note"].astype(str).unique().tolist())
    print("Direct-threshold matched-input audit: PASS")
    print(f"  identical held-out keys across models: {len(ref_ids)}")
    print(f"  functional_umax_source_note values: {source_notes}")
    print("  verified identical across models: B0_1..5, LR_target_1..5, Tthr_direct_unclipped_1..5, functional_umax")


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int = DIRECT_BOOTSTRAP_SEED,
    n_boot: int = DIRECT_BOOTSTRAP_N,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True)
    boot_means = draws.mean(axis=1)
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    return mean, lo, hi


def aggregate_direct_threshold_by_bio_group(case_df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Aggregate repeated held-out appearances within bio_id × policy before summarizing."""
    has_bio = "bio_id" in case_df.columns and case_df["bio_id"].notna().all()
    metric_cols = [
        "LR_tracking_error",
        "PAUC",
        "Bterminal",
        "dose_count",
        "total_dosage",
        "all_constraints_satisfied",
        "LR_constraint_satisfied",
        "P_AUC_constraint_satisfied",
        "pathogen_constraint_satisfied",
    ]
    if not has_bio:
        # Fall back: unique held-out row × model (still collapse repeats).
        group_keys = ["validation_original_row_index", "model"]
        grouped = (
            case_df.groupby(group_keys, as_index=False)[metric_cols]
            .mean(numeric_only=True)
        )
        grouped = grouped.rename(columns={"validation_original_row_index": "base_group_id"})
        return grouped, False

    grouped = (
        case_df.groupby(["bio_id", "model"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .rename(columns={"bio_id": "base_group_id"})
    )
    return grouped, True


def summarize_direct_threshold_groups(
    group_df: pd.DataFrame,
    *,
    has_bio_groups: bool,
    n_unique_cases: int,
    n_case_policy_pairs: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    for model in DIRECT_MODEL_PLOT_ORDER:
        sub = group_df[group_df["model"] == model]
        if sub.empty:
            continue
        n_groups = int(len(sub))
        satisfied_rates = sub["all_constraints_satisfied"].to_numpy(dtype=float)
        # After within-base-group aggregation, each group contributes its satisfaction rate.
        # Plotted joint rate = sum(group_rates) / n_groups (equal weight per base group).
        numerator = float(np.sum(satisfied_rates))
        denominator = float(n_groups)
        rate = float(numerator / denominator) if denominator else float("nan")
        n_fully_satisfied_groups = int(np.sum(np.isclose(satisfied_rates, 1.0)))

        row = {
            "model": model,
            "display_name": DIRECT_DISPLAY_NAMES.get(model, model),
            "n_unique_biological_base_groups": int(n_groups),
            "bio_group_aggregation": bool(has_bio_groups),
            "n_unique_heldout_cases": int(n_unique_cases),
            "n_evaluated_case_policy_pairs": int(n_case_policy_pairs),
            "mean_LR_tracking_error": float(sub["LR_tracking_error"].mean()),
            "mean_PAUC": float(sub["PAUC"].mean()),
            "mean_Bterminal": float(sub["Bterminal"].mean()),
            "mean_dose_count": float(sub["dose_count"].mean()),
            "mean_total_dosage": float(sub["total_dosage"].mean()),
            "total_dosage_unit": DIRECT_TOTAL_DOSAGE_UNIT,
            "constraint_satisfaction_rate": rate,
            "joint_constraint_numerator": numerator,
            "joint_constraint_denominator": denominator,
            "joint_constraint_rate": rate,
            "n_fully_satisfied_base_groups": n_fully_satisfied_groups,
            "LR_constraint_rate": float(sub["LR_constraint_satisfied"].mean()),
            "P_AUC_constraint_rate": float(sub["P_AUC_constraint_satisfied"].mean()),
            "pathogen_constraint_rate": float(sub["pathogen_constraint_satisfied"].mean()),
        }
        if has_bio_groups:
            for col, key in (
                ("LR_tracking_error", "mean_LR_tracking_error"),
                ("PAUC", "mean_PAUC"),
                ("Bterminal", "mean_Bterminal"),
                ("dose_count", "mean_dose_count"),
                ("total_dosage", "mean_total_dosage"),
            ):
                mean, lo, hi = _bootstrap_mean_ci(sub[col].to_numpy(dtype=float))
                row[f"{key}_boot_mean"] = mean
                row[f"{key}_boot_ci_low"] = lo
                row[f"{key}_boot_ci_high"] = hi
            _, lo, hi = _bootstrap_mean_ci(satisfied_rates)
            row["constraint_satisfaction_rate_boot_ci_low"] = lo
            row["constraint_satisfaction_rate_boot_ci_high"] = hi
            row["bootstrap_note"] = (
                f"Paired descriptive bootstrap over base groups "
                f"(seed={DIRECT_BOOTSTRAP_SEED}, n_boot={DIRECT_BOOTSTRAP_N})."
            )
        else:
            row["bootstrap_note"] = (
                "No bio_id / base-group identifier available; "
                "bootstrap CIs omitted (not fabricated)."
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_direct_threshold_comparison(summary_df: pd.DataFrame, outpath: str) -> Tuple[str, str]:
    from figure_audit import (
        PALETTE_BLUE_LIGHT,
        PALETTE_BLUE_MID,
        PALETTE_RED_MID,
        apply_matplotlib_style,
        save_figure,
    )
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    apply_matplotlib_style()
    # (column, title, y-axis label, integer_y, scale)
    metrics = [
        (
            "mean_LR_tracking_error",
            "LR tracking error",
            r"mean $|\Delta\mathrm{LR}|$",
            False,
            None,
        ),
        (
            "mean_PAUC",
            "P_AUC",
            r"$P_{\mathrm{AUC}}$",
            False,
            None,
        ),
        (
            "mean_Bterminal",
            "Terminal burden",
            "CFU/mL",
            False,
            None,
        ),
        (
            "mean_dose_count",
            "Dose count",
            "doses",
            True,
            None,
        ),
        (
            "mean_total_dosage",
            "Total dosage",
            DIRECT_TOTAL_DOSAGE_UNIT,
            False,
            None,
        ),
        (
            "constraint_satisfaction_rate",
            "Joint constraint satisfaction",
            "% satisfied",
            False,
            "percent",
        ),
    ]
    models = [m for m in DIRECT_MODEL_PLOT_ORDER if m in set(summary_df["model"])]
    labels = [DIRECT_DISPLAY_NAMES.get(m, m) for m in models]
    color_map = {
        TAR_MODEL: PALETTE_RED_MID,
        DIRECT_RULE_UNCLIPPED: PALETTE_BLUE_MID,
        DIRECT_RULE_CLIPPED: PALETTE_BLUE_LIGHT,
    }
    has_boot = all(
        f"{col}_boot_ci_low" in summary_df.columns
        for col in (
            "mean_LR_tracking_error",
            "mean_PAUC",
            "mean_Bterminal",
            "mean_dose_count",
            "mean_total_dosage",
        )
    ) and "constraint_satisfaction_rate_boot_ci_low" in summary_df.columns

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.5), constrained_layout=True)
    x = np.arange(len(models))
    for ax, (col, title, ylabel, integer_y, scale) in zip(axes.ravel(), metrics):
        vals = []
        yerr = None
        if has_boot:
            lo_list = []
            hi_list = []
        for model in models:
            sub = summary_df.loc[summary_df["model"] == model]
            v = float(sub[col].iloc[0]) if len(sub) else float("nan")
            if scale == "percent" and np.isfinite(v):
                v = 100.0 * v
            vals.append(v)
            if has_boot and len(sub):
                if col == "constraint_satisfaction_rate":
                    lo = float(sub["constraint_satisfaction_rate_boot_ci_low"].iloc[0])
                    hi = float(sub["constraint_satisfaction_rate_boot_ci_high"].iloc[0])
                    if scale == "percent":
                        lo, hi = 100.0 * lo, 100.0 * hi
                else:
                    lo = float(sub[f"{col}_boot_ci_low"].iloc[0])
                    hi = float(sub[f"{col}_boot_ci_high"].iloc[0])
                lo_list.append(max(0.0, v - lo) if np.isfinite(v) and np.isfinite(lo) else 0.0)
                hi_list.append(max(0.0, hi - v) if np.isfinite(v) and np.isfinite(hi) else 0.0)
        colors = [color_map.get(m, PALETTE_BLUE_MID) for m in models]
        if has_boot:
            yerr = np.vstack([lo_list, hi_list])
            ax.bar(x, vals, color=colors, edgecolor="white", width=0.72, yerr=yerr, capsize=3, error_kw={"linewidth": 0.8})
        else:
            ax.bar(x, vals, color=colors, edgecolor="white", width=0.72)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
        ax.set_ylim(bottom=0.0)
        if integer_y:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="y", labelsize=8)

    paths = save_figure(fig, outpath, dpi=600)
    plt.close(fig)
    return paths


def print_direct_threshold_manuscript_values(summary_df: pd.DataFrame, audit: dict) -> None:
    print("Direct-threshold manuscript values")
    print(f"  unique biological base groups: {int(summary_df['n_unique_biological_base_groups'].iloc[0])}")
    print(f"  evaluated case-policy pairs: {int(summary_df['n_evaluated_case_policy_pairs'].iloc[0])}")
    print(
        "  unmodified direct Tthr support: "
        f"below={audit.get('pct_below_lo'):.6g}% (n={audit.get('n_below_lo')}), "
        f"above={audit.get('pct_above_hi'):.6g}% (n={audit.get('n_above_hi')}), "
        f"total_components={audit.get('n_threshold_values')}"
    )
    for _, row in summary_df.iterrows():
        name = row.get("display_name", row["model"])
        print(
            f"  {name}: "
            f"LR_tracking_error={row['mean_LR_tracking_error']:.6g}; "
            f"P_AUC={row['mean_PAUC']:.6g}; "
            f"Bterminal={row['mean_Bterminal']:.6g} CFU/mL; "
            f"dose_count={row['mean_dose_count']:.6g}; "
            f"total_dosage={row['mean_total_dosage']:.6g} {DIRECT_TOTAL_DOSAGE_UNIT}; "
            f"joint_constraint={float(row['joint_constraint_numerator']):.6g}/"
            f"{float(row['joint_constraint_denominator']):.6g} "
            f"({100.0 * float(row['joint_constraint_rate']):.6g}%)"
        )
    note = str(summary_df["bootstrap_note"].iloc[0]) if "bootstrap_note" in summary_df.columns else ""
    if note:
        print(f"  bootstrap: {note}")


def finalize_direct_threshold_artifacts(
    outdir: str,
    *,
    metadata_csv: Optional[str] = None,
    metadata_df: Optional[pd.DataFrame] = None,
    case_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Audit/aggregate/plot from saved case-level direct-baseline results (no ODE rerun)."""
    case_path = os.path.join(outdir, DIRECT_CASE_RESULTS_CSV)
    if case_df is None:
        if not os.path.isfile(case_path):
            raise FileNotFoundError(
                f"Missing {case_path}. Run ODE-back with --include_direct_threshold_baseline once "
                "before --direct_threshold_plot_only."
            )
        case_df = pd.read_csv(case_path)
    else:
        # Preserve the original case-level result file on disk.
        if not os.path.isfile(case_path):
            case_df.to_csv(case_path, index=False)

    print_lr_tracking_error_definition()
    audit_direct_threshold_case_matching(case_df)

    if metadata_df is None and metadata_csv and os.path.isfile(metadata_csv):
        metadata_df = pd.read_csv(metadata_csv)
    case_df = _attach_bio_id_to_direct_cases(case_df, metadata_df)

    n_unique_cases = int(case_df["validation_original_row_index"].nunique())
    n_policies = int(case_df["model"].nunique())
    n_case_policy_pairs = int(n_unique_cases * n_policies)

    # Support audit on unique held-out cases (unmodified direct thresholds only).
    uniq = case_df.drop_duplicates(subset=["validation_original_row_index"])
    tthr_cols = [f"Tthr_direct_unclipped_{i}" for i in range(1, 6)]
    unclipped = uniq[tthr_cols].to_numpy(dtype=float)
    audit = build_direct_threshold_support_audit(unclipped)
    audit["n_unique_heldout_cases"] = n_unique_cases
    audit["n_evaluated_case_policy_pairs"] = n_case_policy_pairs

    group_df, has_bio = aggregate_direct_threshold_by_bio_group(case_df)
    if has_bio:
        audit["n_unique_biological_base_groups"] = int(group_df["base_group_id"].nunique())
    else:
        audit["n_unique_biological_base_groups"] = int(group_df["base_group_id"].nunique())
        print(
            "WARNING: bio_id unavailable or incomplete; aggregated on held-out row ids instead "
            "and omitted fabricated bootstrap CIs."
        )

    summary_df = summarize_direct_threshold_groups(
        group_df,
        has_bio_groups=has_bio,
        n_unique_cases=n_unique_cases,
        n_case_policy_pairs=n_case_policy_pairs,
    )
    # Ensure count column reflects unique bio groups consistently.
    if has_bio:
        summary_df["n_unique_biological_base_groups"] = int(group_df["base_group_id"].nunique())

    summary_path = os.path.join(outdir, DIRECT_SUMMARY_CSV)
    audit_path = os.path.join(outdir, DIRECT_SUPPORT_AUDIT_CSV)
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame([audit]).to_csv(audit_path, index=False)

    plot_path = os.path.join(outdir, DIRECT_COMPARISON_PNG)
    png_path, svg_path = plot_direct_threshold_comparison(summary_df, plot_path)

    print(f"Wrote {summary_path}")
    print(f"Wrote {audit_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")
    print(f"Preserved case-level file: {case_path}")
    print_direct_threshold_manuscript_values(summary_df, audit)
    return {
        "summary_df": summary_df,
        "audit": audit,
        "png_path": png_path,
        "svg_path": svg_path,
        "summary_path": summary_path,
        "audit_path": audit_path,
        "case_path": case_path,
    }


def evaluate_direct_threshold_baselines(
    *,
    case_df: pd.DataFrame,
    x_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame],
    reference_mode: str,
    umax_source: str,
    umax_constant: float,
    bundle_dir_hint: Optional[str],
    backend: str,
    desired_pauc_col: Optional[str],
    desired_lr_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Build DirectRuleUnclipped / DirectRuleClipped case rows for the same held-out TAR cases."""
    if case_df.empty or TAR_MODEL not in set(case_df["model"]):
        raise ValueError("Direct-threshold baseline requires TAR rows in ode-back case results.")

    tar_df = case_df[case_df["model"] == TAR_MODEL].copy()
    gamma_cols = resolve_gamma_cols(x_df.columns)
    unique_rows = sorted(int(v) for v in tar_df["validation_original_row_index"].unique())

    cache: Dict[int, dict] = {}
    unclipped_stack: List[np.ndarray] = []

    for row_index in unique_rows:
        bio = bio_vector_from_row(x_df, row_index, gamma_cols)
        b0 = bio[0]
        lr_target = lr_targets_for_row(metadata_df, row_index, desired_lr_cols, x_df)
        target_pauc = desired_pauc_for_row(metadata_df, row_index, desired_pauc_col, x_df)
        u_max, umax_note = resolve_functional_umax(
            umax_source, umax_constant, row_index, metadata_df, bundle_dir_hint
        )
        tthr_raw = direct_tthr_from_b0_and_lr(b0, lr_target)
        tthr_clipped = clip_direct_tthr_to_sampled_range(tthr_raw)
        unclipped_stack.append(tthr_raw)

        out_unclipped = simulate_outcomes(bio, tthr_raw, u_max, backend)
        out_clipped = simulate_outcomes(bio, tthr_clipped, u_max, backend)
        cache[row_index] = {
            "bio_b0": b0,
            "lr_target": lr_target,
            "target_pauc": target_pauc,
            "u_max": u_max,
            "umax_note": umax_note,
            "tthr_raw": tthr_raw,
            "tthr_clipped": tthr_clipped,
            "out_unclipped": out_unclipped,
            "out_clipped": out_clipped,
        }

    # Reference outcomes / thresholds from existing TAR rows (same held-out cases; no retrain).
    ode_back_rows: List[dict] = []
    direct_case_rows: List[dict] = []

    for _, tar_row in tar_df.iterrows():
        repeat_id = int(tar_row["repeat_id"])
        case_index = int(tar_row["case_index"])
        row_index = int(tar_row["validation_original_row_index"])
        packed = cache[row_index]
        ref_tthr = np.array([float(tar_row[f"ref_Tthr_{i}"]) for i in range(1, 6)], dtype=float)
        ref_outcomes = {outcome: float(tar_row.get(f"ref_{outcome}", np.nan)) for outcome in OUTCOME_COLS_REFERENCE}

        for model_name, tthr_key, out_key in (
            (DIRECT_RULE_UNCLIPPED, "tthr_raw", "out_unclipped"),
            (DIRECT_RULE_CLIPPED, "tthr_clipped", "out_clipped"),
        ):
            pred_tthr = packed[tthr_key]
            pred_outcomes = packed[out_key]
            ode_back_rows.append(
                build_case_row(
                    repeat_id=repeat_id,
                    validation_original_row_index=row_index,
                    case_index=case_index,
                    model=model_name,
                    functional_umax=float(packed["u_max"]),
                    umax_source_note=str(packed["umax_note"]),
                    reference_mode=reference_mode,
                    ref_tthr=ref_tthr,
                    pred_tthr=pred_tthr,
                    ref_outcomes=ref_outcomes,
                    pred_outcomes=pred_outcomes,
                    trajectory_r2_case=None,
                )
            )

            achieved_lr = np.array([float(pred_outcomes[f"LR{i}"]) for i in range(1, 6)], dtype=float)
            bterminal = float(pred_outcomes.get("terminal_total_pathogen", np.nan))
            flags = direct_constraint_flags(
                achieved_lr=achieved_lr,
                p_auc=float(pred_outcomes["P_AUC"]),
                bterminal=bterminal,
                target_lr=packed["lr_target"],
                target_pauc=float(packed["target_pauc"]),
            )
            drow = {
                "repeat_id": repeat_id,
                "validation_original_row_index": row_index,
                "case_index": case_index,
                "model": model_name,
                "functional_umax": float(packed["u_max"]),
                "functional_umax_source_note": str(packed["umax_note"]),
                "LR_tracking_error": _mean_lr_tracking_error(achieved_lr, packed["lr_target"]),
                "PAUC": float(pred_outcomes["P_AUC"]),
                "Bterminal": bterminal,
                "dose_count": float(pred_outcomes["dose_count"]),
                "total_dosage": float(pred_outcomes["total_dosage"]),
                **flags,
            }
            for i in range(1, 6):
                drow[f"B0_{i}"] = float(packed["bio_b0"][i - 1])
                drow[f"LR_target_{i}"] = float(packed["lr_target"][i - 1])
                drow[f"Tthr_direct_unclipped_{i}"] = float(packed["tthr_raw"][i - 1])
                drow[f"Tthr_used_{i}"] = float(pred_tthr[i - 1])
                drow[f"LR_achieved_{i}"] = float(achieved_lr[i - 1])
            direct_case_rows.append(drow)

    # TAR comparison rows on the same held-out cases (desired-target tracking / constraints).
    for _, tar_row in tar_df.iterrows():
        row_index = int(tar_row["validation_original_row_index"])
        packed = cache[row_index]
        achieved_lr = np.array([float(tar_row[f"pred_LR{i}"]) for i in range(1, 6)], dtype=float)
        bterminal = float(tar_row.get("pred_terminal_total_pathogen", np.nan))
        flags = direct_constraint_flags(
            achieved_lr=achieved_lr,
            p_auc=float(tar_row["pred_P_AUC"]),
            bterminal=bterminal,
            target_lr=packed["lr_target"],
            target_pauc=float(packed["target_pauc"]),
        )
        drow = {
            "repeat_id": int(tar_row["repeat_id"]),
            "validation_original_row_index": row_index,
            "case_index": int(tar_row["case_index"]),
            "model": TAR_MODEL,
            "functional_umax": float(tar_row["functional_umax"]),
            "functional_umax_source_note": str(tar_row["functional_umax_source_note"]),
            "LR_tracking_error": _mean_lr_tracking_error(achieved_lr, packed["lr_target"]),
            "PAUC": float(tar_row["pred_P_AUC"]),
            "Bterminal": bterminal,
            "dose_count": float(tar_row["pred_dose_count"]),
            "total_dosage": float(tar_row["pred_total_dosage"]),
            **flags,
        }
        for i in range(1, 6):
            drow[f"B0_{i}"] = float(packed["bio_b0"][i - 1])
            drow[f"LR_target_{i}"] = float(packed["lr_target"][i - 1])
            drow[f"Tthr_direct_unclipped_{i}"] = float(packed["tthr_raw"][i - 1])
            drow[f"Tthr_used_{i}"] = float(tar_row[f"pred_Tthr_{i}"])
            drow[f"LR_achieved_{i}"] = float(achieved_lr[i - 1])
        direct_case_rows.append(drow)

    direct_case_df = pd.DataFrame(direct_case_rows)
    ode_back_direct_df = pd.DataFrame(ode_back_rows)
    # Summary / support audit / figure are produced by finalize_direct_threshold_artifacts
    # (bio-group aggregation; no invented CIs). Placeholder returns for callers.
    summary_df = pd.DataFrame()
    audit_df = pd.DataFrame()
    audit: dict = {}
    return ode_back_direct_df, direct_case_df, summary_df, audit_df, audit


def run_ode_back_validation(
    *,
    predictions_dir: str,
    predictions_manifest: Optional[str],
    x_csv: str,
    y_csv: str,
    metadata_csv: Optional[str],
    outdir: str,
    reference_mode: str,
    functional_umax_source: str,
    functional_umax_value: float,
    models: Sequence[str],
    backend: str,
    n_jobs: int,
    validate_fast_backend_flag: bool = False,
    bundle_dir_hint: Optional[str] = None,
    generate_plots: bool = True,
    compute_trajectory_r2: bool = True,
    include_direct_threshold_baseline: bool = False,
) -> dict:
    os.makedirs(outdir, exist_ok=True)
    figure_dir = os.path.join(outdir, "figure")
    os.makedirs(figure_dir, exist_ok=True)

    run_warnings: List[str] = []
    x_df, y_df, meta_df = load_bundle_tables(x_csv, y_csv, metadata_csv)
    if bundle_dir_hint is None:
        bundle_dir_hint = os.path.dirname(os.path.abspath(x_csv))

    pred_paths = discover_prediction_files(predictions_dir, predictions_manifest)
    if not pred_paths:
        raise FileNotFoundError(f"No predictions.csv found under {predictions_dir}")

    requested_models = [normalize_model_name(m) for m in models]
    sample_df = pd.read_csv(pred_paths[0])
    model_cols = parse_prediction_models(sample_df, requested_models)
    if not model_cols:
        raise ValueError("No model prediction columns resolved.")

    desired_lr_cols = resolve_desired_lr_cols(meta_df.columns) if meta_df is not None else []
    desired_pauc_col = resolve_desired_pauc_col(meta_df.columns) if meta_df is not None else None
    has_desired_pathogen = False  # no standard desired terminal pathogen column in bundle

    fast_ok = True
    if validate_fast_backend_flag:
        fast_ok, val_warnings = validate_fast_backend()
        run_warnings.extend(val_warnings)
        if not fast_ok:
            run_warnings.append("Fast numba backend failed validation; falling back to python backend.")
    effective = effective_backend(backend, fast_ok)
    if backend == "numba" and effective != "numba":
        run_warnings.append("Requested numba backend unavailable or failed validation; using python.")

    if n_jobs == 0:
        n_jobs = 1
    parallel_jobs = n_jobs

    def _process(path: str) -> Tuple[List[dict], List[dict]]:
        return process_prediction_file(
            path,
            x_df,
            y_df,
            meta_df,
            model_cols,
            reference_mode,
            functional_umax_source,
            functional_umax_value,
            bundle_dir_hint,
            effective,
            desired_pauc_col,
            desired_lr_cols,
            compute_trajectory_r2=compute_trajectory_r2,
        )

    if parallel_jobs == 1:
        all_rows: List[dict] = []
        all_traj_rows: List[dict] = []
        for path in pred_paths:
            rows, traj_rows = _process(path)
            all_rows.extend(rows)
            all_traj_rows.extend(traj_rows)
    else:
        chunks = Parallel(n_jobs=parallel_jobs, prefer="processes")(
            delayed(_process)(path) for path in pred_paths
        )
        all_rows = [row for rows, _ in chunks for row in rows]
        all_traj_rows = [row for _, traj_rows in chunks for row in traj_rows]

    case_df = pd.DataFrame(all_rows)
    direct_artifacts: Dict[str, str] = {}
    direct_audit: dict = {}
    if include_direct_threshold_baseline:
        (
            ode_back_direct_df,
            direct_case_df,
            _direct_summary_unused,
            _direct_audit_df_unused,
            _direct_audit_unused,
        ) = evaluate_direct_threshold_baselines(
            case_df=case_df,
            x_df=x_df,
            metadata_df=meta_df,
            reference_mode=reference_mode,
            umax_source=functional_umax_source,
            umax_constant=functional_umax_value,
            bundle_dir_hint=bundle_dir_hint,
            backend=effective,
            desired_pauc_col=desired_pauc_col,
            desired_lr_cols=desired_lr_cols,
        )
        # Append direct baselines into ODE-back case table without touching stored TAR predictions.
        case_df = pd.concat([case_df, ode_back_direct_df], ignore_index=True)

        direct_case_path = os.path.join(outdir, DIRECT_CASE_RESULTS_CSV)
        direct_case_df.to_csv(direct_case_path, index=False)
        finalized = finalize_direct_threshold_artifacts(
            outdir,
            metadata_df=meta_df,
            case_df=direct_case_df,
        )
        direct_audit = finalized["audit"]
        direct_artifacts = {
            DIRECT_CASE_RESULTS_CSV: finalized["case_path"],
            DIRECT_SUMMARY_CSV: finalized["summary_path"],
            DIRECT_SUPPORT_AUDIT_CSV: finalized["audit_path"],
            DIRECT_COMPARISON_PNG: finalized["png_path"],
            "direct_threshold_comparison.svg": finalized["svg_path"],
        }

    case_path = os.path.join(outdir, "ode_back_case_results.csv")
    case_df.to_csv(case_path, index=False)

    outcome_list = list(OUTCOME_COLS_METRICS)
    if reference_mode == "desired_targets" and not has_desired_pathogen:
        outcome_list = [o for o in outcome_list if not o.startswith("log10_terminal")]

    per_outcome_parts = [
        compute_outcome_metrics(case_df, outcome, reference_mode, has_desired_pathogen)
        for outcome in outcome_list
    ]
    per_outcome_df = pd.concat([df for df in per_outcome_parts if not df.empty], ignore_index=True)
    per_outcome_path = os.path.join(outdir, "ode_back_per_outcome_metrics.csv")
    per_outcome_df.to_csv(per_outcome_path, index=False)

    summary_df = aggregate_summary_by_model(
        per_outcome_df, case_df, reference_mode, has_desired_pathogen
    )
    repeat_metrics_df = build_ode_back_repeat_metrics(case_df)
    repeat_metrics_path = os.path.join(outdir, ODE_BACK_REPEATED_METRICS_CSV)
    repeat_metrics_df.to_csv(repeat_metrics_path, index=False)

    repeat_summary = repeat_level_summary(repeat_metrics_df, list(dict.fromkeys(case_df["model"].tolist())))
    control_models = [m for m in dict.fromkeys(case_df["model"].tolist()) if m != TAR_MODEL]
    pairwise_df = build_ode_back_pairwise_significance(
        repeat_metrics_df,
        srl_model=TAR_MODEL,
        control_models=control_models,
        outdir=outdir,
        prediction_files=pred_paths,
    )
    pairwise_path = os.path.join(outdir, ODE_BACK_PAIRWISE_SIGNIFICANCE_CSV)
    pairwise_df.to_csv(pairwise_path, index=False)

    traj_repeat_df = pd.DataFrame(all_traj_rows)
    traj_repeat_path = os.path.join(outdir, ODE_BACK_TRAJECTORY_REPEATED_CSV)
    traj_summary_df = pd.DataFrame()
    traj_pairwise_df = pd.DataFrame()
    traj_models = list(model_cols.keys())
    if not traj_repeat_df.empty:
        traj_repeat_df.to_csv(traj_repeat_path, index=False)
        traj_summary_df = repeat_level_summary(
            traj_repeat_df, traj_models, metric_col=ODE_BACK_TRAJECTORY_R2_METRIC
        )
        traj_pairwise_df = build_ode_back_pairwise_significance(
            traj_repeat_df,
            srl_model=TAR_MODEL,
            control_models=[m for m in traj_models if m != TAR_MODEL],
            metric_col=ODE_BACK_TRAJECTORY_R2_METRIC,
            bidirectional_plot_stars=True,
            outdir=outdir,
            prediction_files=pred_paths,
        )
        traj_pairwise_df.to_csv(os.path.join(outdir, ODE_BACK_TRAJECTORY_PAIRWISE_CSV), index=False)

    if not repeat_summary.empty:
        pooled_r2 = summary_df[["model", "mean_outcome_R2"]].rename(
            columns={"mean_outcome_R2": "mean_outcome_R2_pooled"}
        )
        drop_cols = [c for c in summary_df.columns if c.startswith("mean_outcome_R2")]
        summary_df = summary_df.drop(columns=drop_cols, errors="ignore").merge(
            repeat_summary, on="model", how="left"
        )
        summary_df = summary_df.merge(pooled_r2, on="model", how="left")
    if not traj_summary_df.empty:
        summary_df = summary_df.merge(traj_summary_df, on="model", how="left")
    summary_path = os.path.join(outdir, "ode_back_summary_by_model.csv")
    summary_df.to_csv(summary_path, index=False)

    training_median_value = None
    if functional_umax_source == "training_median":
        training_median_value, _ = resolve_training_soft_umax_median(meta_df, bundle_dir_hint)

    manifest = {
        "analysis": "ode_back_functional_validation",
        "reference_mode": reference_mode,
        "reference_mode_notes": {
            "reference_tthr": "Compare predicted ODE outcomes to reference ODE from soft/reference Tthr.",
            "desired_targets": (
                "Outcome tracking vs desired_P_AUC/desired_LR; not reference-controller fidelity."
            ),
        },
        "functional_umax_source": functional_umax_source,
        "functional_umax_value": float(functional_umax_value) if functional_umax_source == "constant" else None,
        "training_soft_u_max_median": training_median_value,
        "umax_optimization_used": False,
        "ode_backend_requested": backend,
        "ode_backend_effective": effective,
        "n_jobs": int(n_jobs),
        "n_repeats": int(case_df["repeat_id"].nunique()),
        "n_cases": int(case_df.groupby("repeat_id")["case_index"].nunique().median()),
        "models": list(dict.fromkeys(case_df["model"].tolist())),
        "prediction_models": list(model_cols.keys()),
        "include_direct_threshold_baseline": bool(include_direct_threshold_baseline),
        "direct_threshold_baseline": {
            "models": list(DIRECT_RULE_MODELS),
            "formula": "Tthr_i = B0_i * 10**(-LR_i_target)",
            "clip_range_CFU_per_mL": [DIRECT_TTHR_CLIP_LO, DIRECT_TTHR_CLIP_HI],
            "support_audit": direct_audit if include_direct_threshold_baseline else None,
        },
        "legacy_model_name_map": LEGACY_MODEL_NAME_MAP,
        "simulator": "simulate_case_metrics_fast / simulate_paper_case",
        "profile": PAPER_FIGURE_PROFILE.name,
        "profile_constants": {
            "u_max_rep": float(PAPER_FIGURE_PROFILE.u_max_rep),
            "t_end": float(PAPER_FIGURE_PROFILE.t_end),
            "dt": float(PAPER_FIGURE_PROFILE.dt),
            "dt_detect": float(PAPER_FIGURE_PROFILE.dt_detect),
            "dt_detect_note": (
                "Legacy nominal field (1/6 h). Implemented trajectories evaluate the "
                "threshold condition once per 0.4-h forward-Euler integration step; "
                "dt_detect is not separately resolved and is not a 10-min sensing trace."
            ),
        },
        "prediction_files": [os.path.relpath(p, outdir) for p in pred_paths],
        "fig4_panel_notes": {
            "Fig3B": "Direct Tthr prediction R2 (tree_srl_benchmark).",
            "Fig3C": "Direct Tthr prediction error heatmap.",
            "Fig4A": "ODE-back functional outcome R2 (ode_back_r2_barplot.png); fixed non-optimized Umax.",
        },
        "has_desired_terminal_pathogen": has_desired_pathogen,
        "trajectory_r2_enabled": bool(compute_trajectory_r2),
        "trajectory_r2_definition": (
            "Pooled R² between reference-Tthr and predicted-Tthr ODE trajectories "
            "(flattened C, log10 P_total, log10 total pathogen, log10 B1..B5 over all time steps), "
            "aggregated per repeat split."
        ),
        "warnings": run_warnings,
        "outputs": {
            "ode_back_case_results.csv": case_path,
            "ode_back_summary_by_model.csv": summary_path,
            "ode_back_per_outcome_metrics.csv": per_outcome_path,
            ODE_BACK_REPEATED_METRICS_CSV: repeat_metrics_path,
            ODE_BACK_PAIRWISE_SIGNIFICANCE_CSV: pairwise_path,
        },
    }
    if direct_artifacts:
        manifest["outputs"].update(direct_artifacts)
    if not traj_repeat_df.empty:
        manifest["outputs"][ODE_BACK_TRAJECTORY_REPEATED_CSV] = traj_repeat_path
        manifest["outputs"][ODE_BACK_TRAJECTORY_PAIRWISE_CSV] = os.path.join(
            outdir, ODE_BACK_TRAJECTORY_PAIRWISE_CSV
        )

    if generate_plots:
        from figure_audit import generate_ode_back_plots

        plot_outputs = generate_ode_back_plots(outdir)
        manifest["outputs"].update(plot_outputs)

    manifest_path = os.path.join(outdir, "ode_back_validation_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Wrote {case_path} ({len(case_df)} rows)")
    print(f"Wrote {summary_path}")
    print(f"Wrote {per_outcome_path}")
    print(f"Wrote {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="ODE-back functional validation (no retraining).")
    parser.add_argument("--predictions_dir", default=None)
    parser.add_argument("--predictions_manifest", default=None)
    parser.add_argument("--x_csv", default=None)
    parser.add_argument("--y_csv", default=None)
    parser.add_argument("--metadata_csv", default=None)
    parser.add_argument("--outdir", default="results/ode_back_validation")
    parser.add_argument(
        "--reference_mode",
        choices=["reference_tthr", "desired_targets"],
        default="reference_tthr",
    )
    parser.add_argument(
        "--functional_umax_source",
        choices=["metadata_soft_umax", "constant", "training_median"],
        default="metadata_soft_umax",
    )
    parser.add_argument("--functional_umax_value", type=float, default=18.0)
    parser.add_argument(
        "--models",
        default="TAR,RandomForest,BestSingleTree,UniformTreeMean",
        help="Comma-separated model names.",
    )
    parser.add_argument("--backend", choices=["numba", "python"], default="numba")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--validate_fast_backend", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument(
        "--no_trajectory_r2",
        action="store_true",
        help="Skip trajectory-level pred-vs-ref R² (same ODE runs; only skips flatten/R²).",
    )
    parser.add_argument(
        "--include_direct_threshold_baseline",
        action="store_true",
        help=(
            "Add DirectRuleUnclipped / DirectRuleClipped baselines "
            "(Tthr_i = B0_i * 10**(-LR_i_target); no TAR retrain)."
        ),
    )
    parser.add_argument(
        "--direct_threshold_plot_only",
        action="store_true",
        help=(
            "Rebuild direct-threshold summary/audit/figure from an existing "
            "direct_threshold_case_results.csv (no ODE rerun, no TAR retrain)."
        ),
    )
    args = parser.parse_args()

    if args.direct_threshold_plot_only:
        finalize_direct_threshold_artifacts(
            args.outdir,
            metadata_csv=args.metadata_csv or os.path.join("data", "microbio_formal_dataset", "sample_metadata.csv"),
        )
        return

    missing = [name for name in ("predictions_dir", "x_csv", "y_csv") if not getattr(args, name)]
    if missing:
        parser.error(
            "the following arguments are required unless --direct_threshold_plot_only: "
            + ", ".join(f"--{m}" for m in missing)
        )

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    n_jobs = os.cpu_count() or 1 if args.n_jobs < 0 else args.n_jobs

    run_ode_back_validation(
        predictions_dir=args.predictions_dir,
        predictions_manifest=args.predictions_manifest,
        x_csv=args.x_csv,
        y_csv=args.y_csv,
        metadata_csv=args.metadata_csv,
        outdir=args.outdir,
        reference_mode=args.reference_mode,
        functional_umax_source=args.functional_umax_source,
        functional_umax_value=args.functional_umax_value,
        models=models,
        backend=args.backend,
        n_jobs=n_jobs,
        validate_fast_backend_flag=args.validate_fast_backend,
        generate_plots=not args.no_plots,
        compute_trajectory_r2=not args.no_trajectory_r2,
        include_direct_threshold_baseline=args.include_direct_threshold_baseline,
    )


if __name__ == "__main__":
    main()
