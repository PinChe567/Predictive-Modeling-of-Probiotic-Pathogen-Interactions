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
    LEGACY_MODEL_NAME_MAP,
    TAR_MODEL,
    comparison_result_label,
    evaluate_significance_label,
    permutation_p_value,
    repeat_metric_ci,
    safe_r2_score,
    normalize_model_name,
)

ODE_BACK_REPEATED_METRICS_CSV = "ode_back_repeated_metrics.csv"
ODE_BACK_PAIRWISE_SIGNIFICANCE_CSV = "ode_back_pairwise_significance.csv"
ODE_BACK_BAR_METRIC = "mean_outcome_R2"
ODE_BACK_TRAJECTORY_R2_METRIC = "trajectory_R2"
ODE_BACK_TRAJECTORY_REPEATED_CSV = "ode_back_trajectory_repeated_metrics.csv"
ODE_BACK_TRAJECTORY_PAIRWISE_CSV = "ode_back_trajectory_pairwise_significance.csv"

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


def build_ode_back_pairwise_significance(
    repeat_metrics_df: pd.DataFrame,
    *,
    srl_model: str = TAR_MODEL,
    control_models: Sequence[str],
    metric_col: str = ODE_BACK_BAR_METRIC,
    n_perm: int = 10_000,
    seed: int = 42,
    bidirectional_plot_stars: bool = False,
) -> pd.DataFrame:
    """Paired repeat-level significance (TAR vs controls)."""
    metric = metric_col
    n_repeats = int(repeat_metrics_df["repeat_id"].nunique()) if not repeat_metrics_df.empty else 0
    rows: List[dict] = []
    for control_model in control_models:
        if control_model == srl_model:
            continue
        srl_vals = (
            repeat_metrics_df[repeat_metrics_df["model"] == srl_model]
            .sort_values("repeat_id")[metric]
            .to_numpy(dtype=float)
        )
        ctrl_vals = (
            repeat_metrics_df[repeat_metrics_df["model"] == control_model]
            .sort_values("repeat_id")[metric]
            .to_numpy(dtype=float)
        )
        n = min(len(srl_vals), len(ctrl_vals))
        if n == 0:
            continue
        srl_vals = srl_vals[:n]
        ctrl_vals = ctrl_vals[:n]
        diffs = srl_vals - ctrl_vals
        mean_srl = float(np.mean(srl_vals))
        mean_control = float(np.mean(ctrl_vals))
        mean_diff = float(np.mean(diffs))
        ci_low = float(np.percentile(diffs, 2.5)) if n > 1 else float("nan")
        ci_high = float(np.percentile(diffs, 97.5)) if n > 1 else float("nan")
        srl_better = mean_diff > 0
        if n > 1 and np.allclose(diffs, 0.0):
            wilcoxon_p = 1.0
        elif n > 1:
            try:
                wilcoxon_p = float(wilcoxon(diffs).pvalue)
            except Exception:
                wilcoxon_p = float("nan")
        else:
            wilcoxon_p = float("nan")
        perm_p = (
            permutation_p_value(diffs, n_perm=n_perm, seed=seed + abs(hash(metric)) % 10000)
            if n > 1
            else float("nan")
        )
        p_candidates = [p for p in (wilcoxon_p, perm_p) if np.isfinite(p)]
        p_for_label = float(min(p_candidates)) if p_candidates else float("nan")
        star_label, significance_tier = evaluate_significance_label(
            p_for_label,
            ci_low,
            ci_high,
            n_repeats=n,
            srl_better=srl_better,
            bidirectional=bidirectional_plot_stars,
        )
        comp_result = comparison_result_label(star_label, significance_tier, srl_better, n)
        rows.append(
            {
                "metric": metric,
                "srl_model": srl_model,
                "control_model": control_model,
                "direction": "higher_is_better",
                "n_repeats": n,
                "mean_srl": mean_srl,
                "mean_control": mean_control,
                "mean_diff": mean_diff,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "wilcoxon_p": wilcoxon_p,
                "permutation_p": perm_p,
                "significance_label": star_label,
                "significance_tier": significance_tier,
                "comparison_result": comp_result,
                "exploratory": bool(n < 10),
            }
        )
    return pd.DataFrame(rows)


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

    repeat_summary = repeat_level_summary(repeat_metrics_df, list(model_cols.keys()))
    control_models = [m for m in model_cols.keys() if m != TAR_MODEL]
    pairwise_df = build_ode_back_pairwise_significance(
        repeat_metrics_df,
        srl_model=TAR_MODEL,
        control_models=control_models,
    )
    pairwise_path = os.path.join(outdir, ODE_BACK_PAIRWISE_SIGNIFICANCE_CSV)
    pairwise_df.to_csv(pairwise_path, index=False)

    traj_repeat_df = pd.DataFrame(all_traj_rows)
    traj_repeat_path = os.path.join(outdir, ODE_BACK_TRAJECTORY_REPEATED_CSV)
    traj_summary_df = pd.DataFrame()
    traj_pairwise_df = pd.DataFrame()
    if not traj_repeat_df.empty:
        traj_repeat_df.to_csv(traj_repeat_path, index=False)
        traj_summary_df = repeat_level_summary(
            traj_repeat_df, list(model_cols.keys()), metric_col=ODE_BACK_TRAJECTORY_R2_METRIC
        )
        traj_pairwise_df = build_ode_back_pairwise_significance(
            traj_repeat_df,
            srl_model=TAR_MODEL,
            control_models=control_models,
            metric_col=ODE_BACK_TRAJECTORY_R2_METRIC,
            bidirectional_plot_stars=True,
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
        "models": list(model_cols.keys()),
        "legacy_model_name_map": LEGACY_MODEL_NAME_MAP,
        "simulator": "simulate_case_metrics_fast / simulate_paper_case",
        "profile": PAPER_FIGURE_PROFILE.name,
        "profile_constants": {
            "u_max_rep": float(PAPER_FIGURE_PROFILE.u_max_rep),
            "t_end": float(PAPER_FIGURE_PROFILE.t_end),
            "dt": float(PAPER_FIGURE_PROFILE.dt),
            "dt_detect": float(PAPER_FIGURE_PROFILE.dt_detect),
        },
        "prediction_files": [os.path.relpath(p, outdir) for p in pred_paths],
        "fig3_panel_notes": {
            "Fig3B": "Direct Tthr prediction R2 (tree_srl_benchmark).",
            "Fig3C": "Direct Tthr prediction error heatmap.",
            "Fig3D": "ODE-back functional outcome R2 / heatmap; fixed non-optimized Umax.",
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
    parser.add_argument("--predictions_dir", required=True)
    parser.add_argument("--predictions_manifest", default=None)
    parser.add_argument("--x_csv", required=True)
    parser.add_argument("--y_csv", required=True)
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
    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()
