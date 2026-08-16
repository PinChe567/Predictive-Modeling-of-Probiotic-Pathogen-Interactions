"""Derive training-only dose reference scales and fixed Umax policies for the Umax optimizer."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from closed_loop_eval import (
    FIXED_REPRESENTATIVE_UMAX_DEFAULT,
    FIXED_UMAX_POLICY_BY_REPEAT_CSV,
    FIXED_UMAX_POLICY_MANIFEST_JSON,
    FIXED_UMAX_POLICY_SUMMARY_CSV,
    OPTIMIZER_REFERENCE_BY_REPEAT_CSV,
    OPTIMIZER_REFERENCE_MANIFEST_JSON,
    OPTIMIZER_REFERENCE_SUMMARY_CSV,
    ClosedLoopConfig,
    U_CANDIDATE_LEGACY_RENAMES,
    U_LANDSCAPE_METRICS,
    U_OBJECTIVE_CATEGORY_LABELS,
    U_OBJECTIVE_PREFERRED_COLUMNS,
    U_OPTIMA_COLUMNS,
    U_OPTIMA_LEGACY_RENAMES,
    _ensure_dir,
    _ensure_parent_dir,
    _save_csv,
    _to_json_native,
    bio_params_from_row,
    clip_tthr,
    compute_composite_penalty,
    compute_optimizer_shortfalls,
    is_feasible_candidate,
    parse_predictions_wide_df,
    parse_u_grid,
    FORMAL_U_GRID_SPEC,
    U_GRID_ARANGE_HELP,
    resolve_dose_reference_scale_for_repeat,
    resolve_pauc_feasibility_fraction,
    resolve_prediction_csv_jobs,
    resolve_study_ode_backend,
    resolve_target_terminal_pathogen,
    subsample_prediction_jobs,
    TRAINING_TUNED_GLOBAL_SELECTION_RULE,
)

_PAUC_FEASIBILITY_EPS = 1e-12
from multi_pathogen_simulator import N_STRAINS, PAPER_FIGURE_PROFILE
from simulate_case_metrics_fast import (
    M_LR1,
    M_P_AUC,
    M_TERMINAL_TOTAL_PATHOGEN,
    M_TOTAL_DOSAGE,
    simulate_case_metrics_fast,
)


def reference_tthr_from_row(y_df: pd.DataFrame, row_index: int) -> np.ndarray:
    row = y_df.iloc[int(row_index)]
    cols = [f"Tthr_{i}" for i in range(1, 6)]
    missing = [c for c in cols if c not in row.index]
    if missing:
        raise ValueError(f"y_targets missing columns: {missing}")
    return row[cols].to_numpy(dtype=float)


def bio_vector_from_row(x_df: pd.DataFrame, row_index: int) -> Tuple[np.ndarray, ...]:
    row = x_df.iloc[int(row_index)]
    gamma_cols = [f"g_{i}" for i in range(1, 6)] if "g_1" in row.index else [f"gamma_{i}" for i in range(1, 6)]
    b0 = row[[f"B0_{i}" for i in range(1, 6)]].to_numpy(dtype=float)
    k_arr = row[[f"k_{i}" for i in range(1, 6)]].to_numpy(dtype=float)
    gamma_arr = row[gamma_cols].to_numpy(dtype=float)
    rho_arr = row[[f"rho_{i}" for i in range(1, 6)]].to_numpy(dtype=float)
    mu_arr = row[[f"mu_{i}" for i in range(1, 6)]].to_numpy(dtype=float)
    return b0, k_arr, gamma_arr, rho_arr, mu_arr


def resolve_reference_umax(
    row_index: int,
    metadata_df: Optional[pd.DataFrame],
    fixed_umax: float,
) -> Tuple[float, str]:
    if metadata_df is not None and "soft_u_max" in metadata_df.columns:
        val = metadata_df.iloc[int(row_index)].get("soft_u_max")
        if pd.notna(val):
            return float(val), "metadata_soft_umax"
    return float(fixed_umax), "fixed_representative_umax"


def simulate_reference_dosage(
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame],
    row_index: int,
    fixed_umax: float,
    backend: str,
) -> Tuple[float, float, str]:
    bio = bio_vector_from_row(x_df, row_index)
    tthr = clip_tthr(reference_tthr_from_row(y_df, row_index))
    u_max, u_src = resolve_reference_umax(row_index, metadata_df, fixed_umax)
    metrics = simulate_case_metrics_fast(*bio, u_max, tthr, backend=backend)
    return float(metrics[M_TOTAL_DOSAGE]), float(u_max), u_src


def _quantile_rows(dosages: np.ndarray) -> dict:
    dosages = dosages[np.isfinite(dosages)]
    if dosages.size == 0:
        nan = float("nan")
        return {
            "dose_reference_q50": nan,
            "dose_reference_q75": nan,
            "dose_reference_q90": nan,
            "dose_reference_q95": nan,
        }
    return {
        "dose_reference_q50": float(np.quantile(dosages, 0.50)),
        "dose_reference_q75": float(np.quantile(dosages, 0.75)),
        "dose_reference_q90": float(np.quantile(dosages, 0.90)),
        "dose_reference_q95": float(np.quantile(dosages, 0.95)),
    }


def _derive_one_repeat(
    repeat_id: int,
    pred_path: str,
    *,
    n_rows: int,
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame],
    fixed_representative_umax: float,
    dose_reference_quantile: float,
    backend: str,
    row_n_jobs: int,
) -> dict:
    pred_df = pd.read_csv(pred_path)
    _, val_idx, _, seed = parse_predictions_wide_df(pred_df)
    val_set = set(int(i) for i in np.asarray(val_idx, dtype=int).tolist())
    training_indices = [i for i in range(n_rows) if i not in val_set]
    if not training_indices:
        raise ValueError(f"repeat {repeat_id}: no training rows after excluding validation indices.")

    def _one_training_row(row_index: int) -> float:
        dosage, _, _ = simulate_reference_dosage(
            x_df, y_df, metadata_df, row_index, fixed_representative_umax, backend
        )
        return dosage

    if row_n_jobs <= 1:
        dosages = np.asarray([_one_training_row(i) for i in training_indices], dtype=float)
    else:
        dosages = np.asarray(
            Parallel(n_jobs=row_n_jobs if row_n_jobs > 0 else -1)(
                delayed(_one_training_row)(i) for i in training_indices
            ),
            dtype=float,
        )

    qcols = _quantile_rows(dosages)
    qcol_name = f"dose_reference_q{int(round(dose_reference_quantile * 100))}"
    dose_scale = float(qcols.get(qcol_name, qcols["dose_reference_q90"]))
    return {
        "repeat_id": int(repeat_id),
        "seed": int(seed) if np.isfinite(seed) else repeat_id,
        "n_training_rows": int(len(training_indices)),
        "n_validation_rows": int(len(val_set)),
        "held_out_excluded_from_reference_derivation": True,
        "reference_tthr_source": "y_targets_reference",
        "reference_umax_priority": "metadata_soft_umax_then_fixed_representative",
        "fixed_representative_umax": float(fixed_representative_umax),
        "dose_reference_source": "training_q90_reference_dosage",
        "dose_reference_quantile": float(dose_reference_quantile),
        **qcols,
        "dose_reference_scale": dose_scale,
    }


def derive_optimizer_references(
    *,
    x_csv: str,
    y_csv: str,
    metadata_csv: Optional[str],
    outdir: str,
    predictions_dir: Optional[str] = None,
    predictions_manifest: Optional[str] = None,
    predictions_csv: Optional[str] = None,
    fixed_representative_umax: float = FIXED_REPRESENTATIVE_UMAX_DEFAULT,
    dose_reference_quantile: float = 0.90,
    backend: str = "auto",
    n_jobs: int = 1,
    max_repeats: Optional[int] = None,
    repeat_subsample: str = "even",
) -> pd.DataFrame:
    """Per-repeat training-only reference-controller dosage quantiles for dose_reference_scale."""
    outdir = _ensure_dir(outdir)
    x_df = pd.read_csv(x_csv)
    y_df = pd.read_csv(y_csv)
    metadata_df = pd.read_csv(metadata_csv) if metadata_csv and os.path.isfile(metadata_csv) else None
    n_rows = len(x_df)
    if len(y_df) != n_rows:
        raise ValueError(f"x_csv rows ({n_rows}) != y_csv rows ({len(y_df)})")

    backend = resolve_study_ode_backend(backend)

    prediction_jobs = resolve_prediction_csv_jobs(
        predictions_dir=predictions_dir,
        predictions_manifest=predictions_manifest,
        predictions_csv=predictions_csv,
    )
    if not prediction_jobs:
        raise ValueError("No prediction CSV jobs found for reference derivation.")
    prediction_jobs = subsample_prediction_jobs(
        prediction_jobs, max_repeats, strategy=repeat_subsample
    )

    repeat_parallel = n_jobs != 1 and len(prediction_jobs) > 1
    row_n_jobs = 1 if repeat_parallel else (n_jobs if n_jobs != 0 else -1)
    jobs = sorted(prediction_jobs, key=lambda item: item[0])

    if repeat_parallel:
        by_repeat_rows = Parallel(n_jobs=n_jobs if n_jobs > 0 else -1)(
            delayed(_derive_one_repeat)(
                repeat_id,
                pred_path,
                n_rows=n_rows,
                x_df=x_df,
                y_df=y_df,
                metadata_df=metadata_df,
                fixed_representative_umax=fixed_representative_umax,
                dose_reference_quantile=dose_reference_quantile,
                backend=backend,
                row_n_jobs=1,
            )
            for repeat_id, pred_path, _source in jobs
        )
    else:
        by_repeat_rows = [
            _derive_one_repeat(
                repeat_id,
                pred_path,
                n_rows=n_rows,
                x_df=x_df,
                y_df=y_df,
                metadata_df=metadata_df,
                fixed_representative_umax=fixed_representative_umax,
                dose_reference_quantile=dose_reference_quantile,
                backend=backend,
                row_n_jobs=row_n_jobs,
            )
            for repeat_id, pred_path, _source in jobs
        ]

    by_repeat_df = pd.DataFrame(by_repeat_rows)
    summary_rows = []
    for col in ("dose_reference_q50", "dose_reference_q75", "dose_reference_q90", "dose_reference_q95", "dose_reference_scale"):
        if col not in by_repeat_df.columns:
            continue
        vals = by_repeat_df[col].to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": col,
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
                "median": float(np.nanmedian(vals)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    by_repeat_path = os.path.join(outdir, OPTIMIZER_REFERENCE_BY_REPEAT_CSV)
    summary_path = os.path.join(outdir, OPTIMIZER_REFERENCE_SUMMARY_CSV)
    manifest_path = os.path.join(outdir, OPTIMIZER_REFERENCE_MANIFEST_JSON)
    _save_csv(by_repeat_df, by_repeat_path)
    _save_csv(summary_df, summary_path)

    manifest = {
        "dose_reference_source": "training_q90_reference_dosage",
        "dose_reference_quantile": float(dose_reference_quantile),
        "held_out_samples_excluded": True,
        "validation_rows_excluded_from_policy_derivation": True,
        "reference_tthr_source": "y_targets reference Tthr (not model predictions)",
        "reference_umax_priority": [
            "sample_metadata.soft_u_max when present",
            f"fixed representative Umax ({fixed_representative_umax}) otherwise",
        ],
        "fixed_representative_umax": float(fixed_representative_umax),
        "n_repeats": int(len(by_repeat_df)),
        "n_rows_total": int(n_rows),
        "backend": backend,
        "n_jobs": int(n_jobs),
        "parallel_level": "repeats" if repeat_parallel else "training_rows",
        "outputs": {
            "optimizer_reference_by_repeat_csv": OPTIMIZER_REFERENCE_BY_REPEAT_CSV,
            "optimizer_reference_summary_csv": OPTIMIZER_REFERENCE_SUMMARY_CSV,
        },
        "default_dose_reference_scale_column": f"dose_reference_q{int(round(dose_reference_quantile * 100))}",
        "quantiles_recorded": ["q50", "q75", "q90", "q95"],
        "note": (
            "Dose reference scales are derived from training rows only; validation held-out rows "
            "are excluded. Reference Tthr comes from y_targets, not model predictions."
        ),
    }
    _ensure_parent_dir(manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(_to_json_native(manifest), fh, indent=2)

    return by_repeat_df


def _training_median_soft_umax(
    training_indices: Sequence[int],
    metadata_df: Optional[pd.DataFrame],
    fallback: float,
) -> float:
    if metadata_df is None or "soft_u_max" not in metadata_df.columns:
        return float(fallback)
    vals = metadata_df.iloc[list(training_indices)]["soft_u_max"].dropna().astype(float)
    if vals.empty:
        return float(fallback)
    return float(np.median(vals.to_numpy(dtype=float)))


def _eval_training_row_metrics(
    row_index: int,
    u_max: float,
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame],
    config: ClosedLoopConfig,
    dose_reference_scale: float,
    backend: str,
) -> Tuple[float, float, bool]:
    meta_row = metadata_df.iloc[int(row_index)] if metadata_df is not None else None
    bio = bio_params_from_row(x_df.iloc[int(row_index)], meta_row)
    tthr = clip_tthr(reference_tthr_from_row(y_df, row_index))
    metrics = simulate_case_metrics_fast(
        bio.B0,
        bio.k_arr,
        bio.gamma_arr,
        bio.rho_arr,
        bio.mu_arr,
        float(u_max),
        tthr,
        backend=backend,
    )
    lr = np.array([metrics[M_LR1 + i] for i in range(N_STRAINS)], dtype=float)
    terminal = float(metrics[M_TERMINAL_TOTAL_PATHOGEN])
    total_dosage = float(metrics[M_TOTAL_DOSAGE])
    pauc = float(metrics[M_P_AUC])
    lr_short, pauc_short, path_short, dose_burden = compute_optimizer_shortfalls(
        total_dosage=total_dosage,
        P_AUC=pauc,
        LR_terminal_median=lr,
        terminal_total_pathogen=terminal,
        target_pauc=bio.desired_pauc,
        target_lr=bio.desired_lr,
        dose_reference_scale=dose_reference_scale,
        config=config,
    )
    composite = compute_composite_penalty(lr_short, pauc_short, path_short, dose_burden, config)
    feasible = is_feasible_candidate(
        LR_terminal_median=lr,
        P_AUC=pauc,
        terminal_total_pathogen=terminal,
        target_lr=bio.desired_lr,
        target_pauc=bio.desired_pauc,
        config=config,
    )
    return composite, total_dosage, feasible


def _select_training_tuned_global_umax(
    aggregates: List[dict],
    *,
    tolerance_frac: float = 0.01,
) -> dict:
    if not aggregates:
        raise ValueError("No training Umax aggregates to select from.")
    min_penalty = min(float(row["mean_composite_penalty"]) for row in aggregates)
    threshold = min_penalty * (1.0 + tolerance_frac)
    near_min = [row for row in aggregates if float(row["mean_composite_penalty"]) <= threshold]
    if not near_min:
        near_min = aggregates
    return min(
        near_min,
        key=lambda row: (
            float(row["mean_total_dosage"]),
            float(row["candidate_u_max"]),
        ),
    )


def _aggregate_fixed_policy_at_u(
    candidate_u: float,
    training_indices: Sequence[int],
    *,
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame],
    config: ClosedLoopConfig,
    dose_scale: float,
    backend: str,
    row_n_jobs: int,
) -> dict:
    def _row_metrics(row_index: int, u_val: float) -> Tuple[float, float, bool]:
        return _eval_training_row_metrics(
            row_index, u_val, x_df, y_df, metadata_df, config, dose_scale, backend
        )

    if row_n_jobs <= 1:
        stats = [_row_metrics(i, float(candidate_u)) for i in training_indices]
    else:
        stats = Parallel(n_jobs=row_n_jobs if row_n_jobs > 0 else -1)(
            delayed(_row_metrics)(i, float(candidate_u)) for i in training_indices
        )
    composites = np.asarray([s[0] for s in stats], dtype=float)
    dosages = np.asarray([s[1] for s in stats], dtype=float)
    feasible = np.asarray([s[2] for s in stats], dtype=bool)
    return {
        "candidate_u_max": float(candidate_u),
        "mean_composite_penalty": float(np.nanmean(composites)),
        "feasible_fraction": float(np.mean(feasible)),
        "mean_total_dosage": float(np.nanmean(dosages)),
    }


def _derive_one_fixed_umax_policy_repeat(
    repeat_id: int,
    pred_path: str,
    *,
    n_rows: int,
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame],
    u_grid: np.ndarray,
    config: ClosedLoopConfig,
    reference_by_repeat_df: Optional[pd.DataFrame],
    rep_umax: float,
    u_min: float,
    u_max_grid: float,
    u_step: float,
    backend: str,
    row_n_jobs: int,
) -> dict:
    pred_df = pd.read_csv(pred_path)
    _, val_idx, _, seed = parse_predictions_wide_df(pred_df)
    val_set = set(int(i) for i in np.asarray(val_idx, dtype=int).tolist())
    training_indices = [i for i in range(n_rows) if i not in val_set]
    if not training_indices:
        raise ValueError(f"repeat {repeat_id}: no training rows after excluding validation indices.")

    dose_scale, _ = resolve_dose_reference_scale_for_repeat(config, repeat_id, reference_by_repeat_df)
    median_soft = _training_median_soft_umax(training_indices, metadata_df, rep_umax)

    u_grid_list = [float(u) for u in u_grid]
    if row_n_jobs <= 1 and len(u_grid_list) > 1:
        u_workers = max(1, min(len(u_grid_list), os.cpu_count() or 4))
        aggregates = Parallel(n_jobs=u_workers)(
            delayed(_aggregate_fixed_policy_at_u)(
                candidate_u,
                training_indices,
                x_df=x_df,
                y_df=y_df,
                metadata_df=metadata_df,
                config=config,
                dose_scale=dose_scale,
                backend=backend,
                row_n_jobs=1,
            )
            for candidate_u in u_grid_list
        )
    else:
        aggregates = [
            _aggregate_fixed_policy_at_u(
                candidate_u,
                training_indices,
                x_df=x_df,
                y_df=y_df,
                metadata_df=metadata_df,
                config=config,
                dose_scale=dose_scale,
                backend=backend,
                row_n_jobs=row_n_jobs,
            )
            for candidate_u in u_grid_list
        ]
    best = _select_training_tuned_global_umax(aggregates)
    return {
        "repeat_id": int(repeat_id),
        "seed": int(seed) if np.isfinite(seed) else repeat_id,
        "n_training_rows": int(len(training_indices)),
        "n_validation_rows": int(len(val_set)),
        "held_out_excluded_from_policy_derivation": True,
        "representative_fixed_u_max": float(rep_umax),
        "representative_fixed_source": "pre_specified_representative",
        "training_median_soft_u_max": float(median_soft),
        "training_median_source": "training_median_soft_umax",
        "training_tuned_global_u_max": float(best["candidate_u_max"]),
        "training_tuned_global_mean_composite_penalty": float(best["mean_composite_penalty"]),
        "training_tuned_global_score": float(best["mean_composite_penalty"]),
        "training_tuned_global_feasible_fraction": float(best["feasible_fraction"]),
        "training_tuned_global_mean_total_dosage": float(best["mean_total_dosage"]),
        "training_tuned_global_selection_rule": TRAINING_TUNED_GLOBAL_SELECTION_RULE,
        "training_tuned_global_source": "training_tuned_global_umax",
        "u_grid_min": u_min,
        "u_grid_max": u_max_grid,
        "u_grid_step": u_step,
    }


def derive_fixed_umax_policies(
    *,
    x_csv: str,
    y_csv: str,
    metadata_csv: Optional[str],
    outdir: str,
    u_grid: np.ndarray,
    config: ClosedLoopConfig,
    predictions_dir: Optional[str] = None,
    predictions_manifest: Optional[str] = None,
    predictions_csv: Optional[str] = None,
    reference_by_repeat_df: Optional[pd.DataFrame] = None,
    fixed_representative_umax: Optional[float] = None,
    backend: str = "auto",
    n_jobs: int = 1,
    max_repeats: Optional[int] = None,
    repeat_subsample: str = "even",
) -> pd.DataFrame:
    """Per-repeat training-only fixed Umax policies (representative, median, tuned global)."""
    outdir = _ensure_dir(outdir)
    x_df = pd.read_csv(x_csv)
    y_df = pd.read_csv(y_csv)
    metadata_df = pd.read_csv(metadata_csv) if metadata_csv and os.path.isfile(metadata_csv) else None
    n_rows = len(x_df)
    if len(y_df) != n_rows:
        raise ValueError(f"x_csv rows ({n_rows}) != y_csv rows ({len(y_df)})")

    rep_umax = float(
        fixed_representative_umax
        if fixed_representative_umax is not None
        else getattr(config, "fixed_representative_umax", None) or PAPER_FIGURE_PROFILE.u_max_rep
    )
    u_grid = np.asarray(u_grid, dtype=float)
    u_min = float(u_grid.min())
    u_max_grid = float(u_grid.max())
    u_step = float(u_grid[1] - u_grid[0]) if len(u_grid) > 1 else 1.0
    backend = resolve_study_ode_backend(backend)

    prediction_jobs = resolve_prediction_csv_jobs(
        predictions_dir=predictions_dir,
        predictions_manifest=predictions_manifest,
        predictions_csv=predictions_csv,
    )
    if not prediction_jobs:
        raise ValueError("No prediction CSV jobs found for fixed Umax policy derivation.")
    prediction_jobs = subsample_prediction_jobs(
        prediction_jobs, max_repeats, strategy=repeat_subsample
    )

    repeat_parallel = n_jobs != 1 and len(prediction_jobs) > 1
    row_n_jobs = 1 if repeat_parallel else (n_jobs if n_jobs != 0 else -1)
    jobs = sorted(prediction_jobs, key=lambda item: item[0])

    if repeat_parallel:
        by_repeat_rows = Parallel(n_jobs=n_jobs if n_jobs > 0 else -1)(
            delayed(_derive_one_fixed_umax_policy_repeat)(
                repeat_id,
                pred_path,
                n_rows=n_rows,
                x_df=x_df,
                y_df=y_df,
                metadata_df=metadata_df,
                u_grid=u_grid,
                config=config,
                reference_by_repeat_df=reference_by_repeat_df,
                rep_umax=rep_umax,
                u_min=u_min,
                u_max_grid=u_max_grid,
                u_step=u_step,
                backend=backend,
                row_n_jobs=1,
            )
            for repeat_id, pred_path, _source in jobs
        )
    else:
        by_repeat_rows = [
            _derive_one_fixed_umax_policy_repeat(
                repeat_id,
                pred_path,
                n_rows=n_rows,
                x_df=x_df,
                y_df=y_df,
                metadata_df=metadata_df,
                u_grid=u_grid,
                config=config,
                reference_by_repeat_df=reference_by_repeat_df,
                rep_umax=rep_umax,
                u_min=u_min,
                u_max_grid=u_max_grid,
                u_step=u_step,
                backend=backend,
                row_n_jobs=row_n_jobs,
            )
            for repeat_id, pred_path, _source in jobs
        ]

    by_repeat_df = pd.DataFrame(by_repeat_rows)
    summary_rows = []
    for col in (
        "representative_fixed_u_max",
        "training_median_soft_u_max",
        "training_tuned_global_u_max",
        "training_tuned_global_score",
        "training_tuned_global_feasible_fraction",
        "training_tuned_global_mean_total_dosage",
    ):
        if col not in by_repeat_df.columns:
            continue
        vals = by_repeat_df[col].to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": col,
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
                "median": float(np.nanmedian(vals)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    by_repeat_path = os.path.join(outdir, FIXED_UMAX_POLICY_BY_REPEAT_CSV)
    summary_path = os.path.join(outdir, FIXED_UMAX_POLICY_SUMMARY_CSV)
    manifest_path = os.path.join(outdir, FIXED_UMAX_POLICY_MANIFEST_JSON)
    _save_csv(by_repeat_df, by_repeat_path)
    _save_csv(summary_df, summary_path)

    manifest = {
        "fixed_umax_policies_training_only": True,
        "validation_rows_excluded_from_policy_derivation": True,
        "held_out_samples_excluded": True,
        "reference_tthr_source": "y_targets reference Tthr (not model predictions)",
        "training_tuned_global_umax_note": (
            "training_tuned_global_u_max is a training-only global fixed-Umax baseline; "
            "validation outcomes are never used to tune global Umax."
        ),
        "training_tuned_global_selection_rule": TRAINING_TUNED_GLOBAL_SELECTION_RULE,
        "representative_fixed_umax": float(rep_umax),
        "u_grid": u_grid.tolist(),
        "n_repeats": int(len(by_repeat_df)),
        "n_rows_total": int(n_rows),
        "backend": backend,
        "outputs": {
            "fixed_umax_policy_by_repeat_csv": FIXED_UMAX_POLICY_BY_REPEAT_CSV,
            "fixed_umax_policy_summary_csv": FIXED_UMAX_POLICY_SUMMARY_CSV,
        },
        "training_median_soft_umax_by_repeat": {
            str(int(row["repeat_id"])): float(row["training_median_soft_u_max"])
            for _, row in by_repeat_df.iterrows()
        },
        "training_tuned_global_umax_by_repeat": {
            str(int(row["repeat_id"])): float(row["training_tuned_global_u_max"])
            for _, row in by_repeat_df.iterrows()
        },
    }
    _ensure_parent_dir(manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(_to_json_native(manifest), fh, indent=2)

    return by_repeat_df


def _sorted_landscape_curve(curve: pd.DataFrame) -> pd.DataFrame:
    return curve.sort_values("candidate_u_max").reset_index(drop=True)


def _candidate_metric_series(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric in df.columns:
        return df[metric]
    for old, new in U_CANDIDATE_LEGACY_RENAMES.items():
        if new == metric and old in df.columns:
            return df[old]
    raise KeyError(f"Metric '{metric}' not found in candidate dataframe.")


def preferred_u_dose_reference_limit(curve: pd.DataFrame) -> float:
    work = _sorted_landscape_curve(curve)
    u = work["candidate_u_max"].to_numpy(dtype=float)
    dose = _candidate_metric_series(work, "total_dosage").to_numpy(dtype=float)
    if "aspiration_total_dosage" in work.columns:
        threshold = float(work["aspiration_total_dosage"].iloc[0])
    elif "dose_reference_scale" in work.columns:
        threshold = float(work["dose_reference_scale"].iloc[0])
    else:
        return float("nan")
    mask = np.isfinite(u) & np.isfinite(dose) & np.isfinite(threshold)
    u, dose = u[mask], dose[mask]
    if len(u) == 0:
        return float("nan")
    last_ok = float("nan")
    for ui, di in zip(u, dose):
        if float(di) <= float(threshold):
            last_ok = float(ui)
    return last_ok


def preferred_u_pauc_constraint_limit(
    curve: pd.DataFrame,
    *,
    pauc_feasibility_fraction: float = 0.90,
) -> Tuple[float, bool]:
    work = _sorted_landscape_curve(curve)
    u = work["candidate_u_max"].to_numpy(dtype=float)
    pauc = work["P_AUC"].to_numpy(dtype=float)
    if "aspiration_P_AUC" in work.columns:
        threshold = float(work["aspiration_P_AUC"].iloc[0])
        mask = np.isfinite(u) & np.isfinite(pauc) & np.isfinite(threshold)
        u, pauc = u[mask], pauc[mask]
        if len(u) == 0:
            return float("nan"), False
        last_ok = float("nan")
        feasible_any = False
        for ui, pi in zip(u, pauc):
            if pi >= threshold:
                last_ok = float(ui)
                feasible_any = True
        return last_ok, feasible_any
    if "target_P_AUC" in work.columns:
        target = float(work["target_P_AUC"].iloc[0])
    elif "minimum_acceptable_P_AUC" in work.columns:
        target = float(work["minimum_acceptable_P_AUC"].iloc[0]) / max(
            pauc_feasibility_fraction, _PAUC_FEASIBILITY_EPS
        )
    else:
        return float("nan"), False
    mask = np.isfinite(u) & np.isfinite(pauc) & np.isfinite(target)
    u, pauc = u[mask], pauc[mask]
    if len(u) == 0 or not np.isfinite(target):
        return float("nan"), False
    threshold = pauc_feasibility_fraction * target
    last_ok = float("nan")
    feasible_any = False
    for ui, pi in zip(u, pauc):
        if pi >= threshold:
            last_ok = float(ui)
            feasible_any = True
    return last_ok, feasible_any


def preferred_u_lr_feasibility(curve: pd.DataFrame, *, lr_tolerance: float = 0.0) -> float:
    work = _sorted_landscape_curve(curve)
    for _, row in work.iterrows():
        ok = True
        for i in range(1, N_STRAINS + 1):
            lr_col = f"LR{i}"
            tgt_col = f"aspiration_LR{i}" if f"aspiration_LR{i}" in row.index else f"target_LR{i}"
            if lr_col not in row.index or tgt_col not in row.index:
                ok = False
                break
            if float(row[lr_col]) < float(row[tgt_col]) - lr_tolerance:
                ok = False
                break
        if ok:
            return float(row["candidate_u_max"])
    return float("nan")


def preferred_u_pathogen_feasibility(curve: pd.DataFrame, *, pathogen_ceiling: float) -> float:
    work = _sorted_landscape_curve(curve)
    if "aspiration_terminal_pathogen" in work.columns:
        ceiling = float(work["aspiration_terminal_pathogen"].iloc[0])
    path = _candidate_metric_series(work, "terminal_total_pathogen").to_numpy(dtype=float)
    u = work["candidate_u_max"].to_numpy(dtype=float)
    for ui, terminal in zip(u, path):
        if np.isfinite(ui) and np.isfinite(terminal) and float(terminal) <= float(pathogen_ceiling):
            return float(ui)
    return float("nan")


def objective_preferred_umax_from_curve(
    curve: pd.DataFrame,
    *,
    config: Optional[ClosedLoopConfig] = None,
    pauc_feasibility_fraction: Optional[float] = None,
    lr_tolerance: float = 0.0,
    pathogen_ceiling: Optional[float] = None,
) -> Dict[str, float]:
    pauc_frac = (
        float(pauc_feasibility_fraction)
        if pauc_feasibility_fraction is not None
        else (resolve_pauc_feasibility_fraction(config) if config is not None else 0.90)
    )
    ceiling = (
        float(pathogen_ceiling)
        if pathogen_ceiling is not None
        else (resolve_target_terminal_pathogen(config) if config is not None else 4e7)
    )
    u_pauc, pauc_feasible = preferred_u_pauc_constraint_limit(curve, pauc_feasibility_fraction=pauc_frac)
    return {
        "U_dose_reference_limit": preferred_u_dose_reference_limit(curve),
        "U_P_AUC_constraint_limit": u_pauc,
        "P_AUC_constraint_feasible_any": float(pauc_feasible),
        "U_LR_feasibility": preferred_u_lr_feasibility(curve, lr_tolerance=lr_tolerance),
        "U_pathogen_feasibility": preferred_u_pathogen_feasibility(curve, pathogen_ceiling=ceiling),
    }


def _aggregate_candidate_curve(group: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for u_val, u_group in group.groupby("candidate_u_max"):
        row = {"candidate_u_max": float(u_val)}
        for metric in U_LANDSCAPE_METRICS:
            row[metric] = float(_candidate_metric_series(u_group, metric).median())
        if "selected_by_optimizer" in u_group.columns:
            row["selected_by_optimizer"] = bool(u_group["selected_by_optimizer"].any())
        for asp_col in (
            "aspiration_total_dosage",
            "aspiration_P_AUC",
            "aspiration_terminal_pathogen",
            "dose_reference_scale",
        ):
            if asp_col in u_group.columns:
                row[asp_col] = float(u_group[asp_col].iloc[0])
        for i in range(1, N_STRAINS + 1):
            asp_lr = f"aspiration_LR{i}"
            if asp_lr in u_group.columns:
                row[asp_lr] = float(u_group[asp_lr].iloc[0])
            tgt_lr = f"target_LR{i}"
            if tgt_lr in u_group.columns and asp_lr not in row:
                row[tgt_lr] = float(u_group[tgt_lr].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_umax_optima_alignment_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for old, new in U_OPTIMA_LEGACY_RENAMES.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})
    return out


def detect_umax_objective_conflict(align_df: pd.DataFrame, u_grid: np.ndarray) -> bool:
    align_df = normalize_umax_optima_alignment_df(align_df)
    if align_df.empty:
        return False
    cols = [col for col in U_OPTIMA_COLUMNS if col in align_df.columns]
    if not cols:
        return False
    u_span = float(np.max(u_grid) - np.min(u_grid)) if len(u_grid) else 100.0
    threshold = max(0.10 * u_span, 5.0)
    row_spreads = align_df[cols].std(axis=1, skipna=True)
    mean_spread = float(np.nanmean(row_spreads.to_numpy(dtype=float)))
    return bool(np.isfinite(mean_spread) and mean_spread > threshold)


def build_umax_objective_alignment_summary(align_df: pd.DataFrame) -> pd.DataFrame:
    align_df = normalize_umax_optima_alignment_df(align_df)
    rows: List[dict] = []
    for col in U_OBJECTIVE_PREFERRED_COLUMNS:
        if col not in align_df.columns:
            continue
        vals = align_df[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.append(
            {
                "column": col,
                "category": U_OBJECTIVE_CATEGORY_LABELS.get(col, col),
                "median": float(np.median(vals)),
                "q10": float(np.quantile(vals, 0.10)),
                "q25": float(np.quantile(vals, 0.25)),
                "q75": float(np.quantile(vals, 0.75)),
                "q90": float(np.quantile(vals, 0.90)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
        )
    return pd.DataFrame(rows)


def build_umax_objective_alignment(
    candidates_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    landscape_model: str,
    unit: str,
    *,
    config: Optional[ClosedLoopConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    from closed_loop_eval import normalize_u_candidates_df

    sub = normalize_u_candidates_df(candidates_df)
    sub = sub[sub["model"] == landscape_model].copy()
    cases_sub = cases_df[cases_df["model"] == landscape_model].copy()

    if unit == "repeat":
        if "repeat_id" not in sub.columns:
            sub = sub.assign(repeat_id=0)
        group_cols = ["repeat_id"]
    else:
        group_cols = ["case_index"] if "repeat_id" not in sub.columns else ["repeat_id", "case_index"]

    rows: List[dict] = []
    for keys, group in sub.groupby(group_cols):
        key_dict = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        curve = _aggregate_candidate_curve(group)
        preferred = objective_preferred_umax_from_curve(curve, config=config)

        sel_mask = pd.Series(True, index=cases_sub.index)
        for col, val in key_dict.items():
            sel_mask &= cases_sub[col] == val
        sel_rows = cases_sub.loc[sel_mask]
        if len(sel_rows):
            u_comp = float(sel_rows["optimized_u_max"].median())
        elif "selected_by_optimizer" in curve.columns and curve["selected_by_optimizer"].any():
            selected = curve.loc[curve["selected_by_optimizer"], "candidate_u_max"]
            u_comp = float(selected.median())
        else:
            u_comp = float("nan")

        row = {
            **{col: int(val) for col, val in key_dict.items()},
            "landscape_model": landscape_model,
            "unit_type": unit,
            **preferred,
            "U_final_selected": u_comp,
        }
        if unit == "repeat":
            row["unit_id"] = f"repeat_{int(key_dict['repeat_id']):03d}"
        else:
            rid = int(key_dict.get("repeat_id", 0))
            cid = int(key_dict["case_index"])
            row["unit_id"] = f"repeat_{rid:03d}_case_{cid:04d}"
        rows.append(row)

    align_df = normalize_umax_optima_alignment_df(pd.DataFrame(rows))
    conflict = detect_umax_objective_conflict(align_df, np.sort(sub["candidate_u_max"].unique()))
    summary = build_umax_objective_alignment_summary(align_df)
    return align_df, summary, conflict


def umax_derivation_artifacts_complete(outdir: str, n_repeats: int) -> bool:
    """True when step-10a reference + fixed-policy tables exist for all repeats."""
    ref_path = os.path.join(outdir, OPTIMIZER_REFERENCE_BY_REPEAT_CSV)
    policy_path = os.path.join(outdir, FIXED_UMAX_POLICY_BY_REPEAT_CSV)
    if not (os.path.isfile(ref_path) and os.path.isfile(policy_path)):
        return False
    try:
        ref_df = pd.read_csv(ref_path)
        policy_df = pd.read_csv(policy_path)
    except (OSError, pd.errors.ParserError):
        return False
    return len(ref_df) >= int(n_repeats) and len(policy_df) >= int(n_repeats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive training-only Umax optimizer dose references and fixed Umax policies."
    )
    parser.add_argument("--x_csv", required=True)
    parser.add_argument("--y_csv", required=True)
    parser.add_argument("--metadata_csv", default=None)
    parser.add_argument("--predictions_dir", default=None)
    parser.add_argument("--predictions_manifest", default=None)
    parser.add_argument("--predictions_csv", default=None)
    parser.add_argument("--outdir", default="results/umax_optimization")
    parser.add_argument("--u_grid", default=FORMAL_U_GRID_SPEC, help=U_GRID_ARANGE_HELP)
    parser.add_argument("--fixed_representative_umax", type=float, default=FIXED_REPRESENTATIVE_UMAX_DEFAULT)
    parser.add_argument("--dose_reference_quantile", type=float, default=0.90)
    parser.add_argument("--backend", default="auto", choices=["auto", "numba", "python"])
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Parallel workers: across repeats when multiple repeats, else across training rows.",
    )
    parser.add_argument(
        "--derive_fixed_umax_policies_only",
        action="store_true",
        help="Only derive fixed Umax policies (skip dose reference scales).",
    )
    parser.add_argument(
        "--derive_optimizer_references_only",
        action="store_true",
        help="Only derive dose reference scales (skip fixed Umax policies).",
    )
    parser.add_argument(
        "--skip_if_complete",
        action="store_true",
        help="Skip when optimizer references and fixed Umax policies already exist for all repeats.",
    )
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Re-derive even when outputs already exist.",
    )
    args = parser.parse_args()

    if not args.predictions_csv and not args.predictions_dir and not args.predictions_manifest:
        parser.error("Provide --predictions_dir, --predictions_manifest, or --predictions_csv.")

    prediction_jobs = resolve_prediction_csv_jobs(
        predictions_dir=args.predictions_dir,
        predictions_manifest=args.predictions_manifest,
        predictions_csv=args.predictions_csv,
    )
    n_expected_repeats = len(prediction_jobs)
    if (
        args.skip_if_complete
        and not args.force_rerun
        and umax_derivation_artifacts_complete(args.outdir, n_expected_repeats)
    ):
        print(
            f"Skipping step 10a: references and fixed policies already exist for "
            f"{n_expected_repeats} repeats in {args.outdir}"
        )
        return

    config = ClosedLoopConfig(
        u_grid=parse_u_grid(args.u_grid),
        fixed_representative_umax=args.fixed_representative_umax,
        dose_reference_quantile=args.dose_reference_quantile,
    )

    if args.derive_optimizer_references_only:
        df = derive_optimizer_references(
            x_csv=args.x_csv,
            y_csv=args.y_csv,
            metadata_csv=args.metadata_csv,
            outdir=args.outdir,
            predictions_dir=args.predictions_dir,
            predictions_manifest=args.predictions_manifest,
            predictions_csv=args.predictions_csv,
            fixed_representative_umax=args.fixed_representative_umax,
            dose_reference_quantile=args.dose_reference_quantile,
            backend=args.backend,
            n_jobs=args.n_jobs,
        )
        print(f"Wrote optimizer references for {len(df)} repeats to {args.outdir}")
        return

    if args.derive_fixed_umax_policies_only:
        ref_path = os.path.join(args.outdir, OPTIMIZER_REFERENCE_BY_REPEAT_CSV)
        reference_df = pd.read_csv(ref_path) if os.path.isfile(ref_path) else None
        df = derive_fixed_umax_policies(
            x_csv=args.x_csv,
            y_csv=args.y_csv,
            metadata_csv=args.metadata_csv,
            outdir=args.outdir,
            u_grid=config.u_grid,
            config=config,
            predictions_dir=args.predictions_dir,
            predictions_manifest=args.predictions_manifest,
            predictions_csv=args.predictions_csv,
            reference_by_repeat_df=reference_df,
            fixed_representative_umax=args.fixed_representative_umax,
            backend=args.backend,
            n_jobs=args.n_jobs,
        )
        print(f"Wrote fixed Umax policies for {len(df)} repeats to {args.outdir}")
        return

    reference_df = derive_optimizer_references(
        x_csv=args.x_csv,
        y_csv=args.y_csv,
        metadata_csv=args.metadata_csv,
        outdir=args.outdir,
        predictions_dir=args.predictions_dir,
        predictions_manifest=args.predictions_manifest,
        predictions_csv=args.predictions_csv,
        fixed_representative_umax=args.fixed_representative_umax,
        dose_reference_quantile=args.dose_reference_quantile,
        backend=args.backend,
        n_jobs=args.n_jobs,
    )
    policy_df = derive_fixed_umax_policies(
        x_csv=args.x_csv,
        y_csv=args.y_csv,
        metadata_csv=args.metadata_csv,
        outdir=args.outdir,
        u_grid=config.u_grid,
        config=config,
        predictions_dir=args.predictions_dir,
        predictions_manifest=args.predictions_manifest,
        predictions_csv=args.predictions_csv,
        reference_by_repeat_df=reference_df,
        fixed_representative_umax=args.fixed_representative_umax,
        backend=args.backend,
        n_jobs=args.n_jobs,
    )
    print(
        f"Wrote optimizer references and fixed Umax policies for {len(policy_df)} repeats to {args.outdir}"
    )


if __name__ == "__main__":
    main()
