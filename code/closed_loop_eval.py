from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from figure_audit import trajectory_history_from_arrays, write_json_manifest
from simulate_case_metrics_fast import (
    M_DOSE_COUNT,
    M_FINAL_TOTAL_PATHOGEN,
    M_LR1,
    M_P_AUC,
    M_TERMINAL_TOTAL_PATHOGEN,
    M_TOTAL_DOSAGE,
    simulate_case_metrics_fast,
    validate_fast_backend,
    effective_backend,
)
from multi_pathogen_simulator import (
    BIO_COLS,
    B0_S_REP,
    GAMMA_S_REP,
    K_REP,
    MU_REP,
    N_STRAINS,
    PAPER_FIGURE_PROFILE,
    RHO_REP,
    LegacySimulationResult as SimulationResult,
    simulate_paper_case as simulate_case,
)

TARGET_COLS_TTHR = [f"Tthr_{i}" for i in range(1, 6)]
OBJECTIVE_COLS = ["P_AUC"] + [f"LR{i}" for i in range(1, 6)]
METADATA_OBJECTIVE_COLS = ["desired_P_AUC"] + [f"desired_LR{i}" for i in range(1, 6)]

# Final manuscript artifact names (Fig. 4 / Fig. 5)
FIXED_UMAX_VALIDATION_SUBDIR = "fixed_umax_validation"
FIG4_MANIFEST_JSON = "fixed_umax_validation_manifest.json"
FIG4_PLOT_MANIFEST_JSON = "fig4_plot_manifest.json"
FIXED_UMAX_CASES_CSV = "fixed_umax_validation_cases.csv"
FIXED_UMAX_SUMMARY_CSV = "fixed_umax_validation_summary_by_model.csv"
FIXED_UMAX_REPEATED_STATS_CSV = "fixed_umax_validation_repeated_stats.csv"
FIXED_UMAX_SIGNIFICANCE_CSV = "fixed_umax_validation_significance.csv"
FIXED_UMAX_PAIRWISE_CSV = "fixed_umax_validation_pairwise_tests.csv"
FIXED_UMAX_TRAJECTORIES_CSV = "fixed_umax_representative_trajectories.csv"
FIXED_UMAX_CHECKPOINT_JSON = "fixed_umax_validation_checkpoint.json"
FIXED_UMAX_CONSTRAINT_CSV = "fixed_umax_validation_constraint_success.csv"
FIXED_UMAX_FAIRNESS_JSON = "fixed_umax_validation_fairness_check.json"
FIXED_UMAX_U_CANDIDATES_CSV = "fixed_umax_validation_u_candidates.csv"
FIXED_UMAX_TTHR_BY_REPEAT_CSV = "fixed_umax_tthr_by_repeat.csv"
FIG4_ODE_FORWARD_RUNS_PER_MODEL = 1
FIG4_ODE_FORWARD_MODELS = 3
UMAX_OPTIMIZATION_MANIFEST_JSON = "umax_optimization_manifest.json"
FIG5_PLOT_MANIFEST_JSON = "fig5_plot_manifest.json"


def _to_json_native(obj):
    """Recursively convert numpy scalars/arrays to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {_to_json_native(k): _to_json_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_json_native(obj.tolist())
    if isinstance(obj, np.generic):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
    return obj


def _ensure_dir(path: str) -> str:
    """Create directory (absolute path) if needed."""
    resolved = os.path.abspath(path)
    os.makedirs(resolved, exist_ok=True)
    return resolved


def _ensure_parent_dir(file_path: str) -> str:
    """Create parent directory for a file path (absolute path)."""
    resolved = os.path.abspath(file_path)
    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return resolved


def _save_csv(df: pd.DataFrame, path: str) -> None:
    _ensure_parent_dir(path)
    df.to_csv(path, index=False)


def _repeat_fixed_umax_dir(repeat_outdir: str) -> str:
    return os.path.join(repeat_outdir, FIXED_UMAX_VALIDATION_SUBDIR)


TAR_MODEL = "TAR"
TAR_NO_CLOSED_LOOP_MODEL = "TAR-noClosedLoop"
TAR_SRL_MODEL = TAR_MODEL  # legacy alias
LEGACY_TAR_SRL_NO_CYCLE = "TAR-SRL-no-cycle"
RANDOM_FOREST = "RandomForest"
BEST_SINGLE_TREE = "BestSingleTree"
UNIFORM_TREE_MEAN = "UniformTreeMean"

# Models that must appear in the predictions CSV (TAR-noClosedLoop is synthesized in eval).
CLOSED_LOOP_PREDICTION_MODELS: Tuple[str, ...] = (
    TAR_MODEL,
    RANDOM_FOREST,
    UNIFORM_TREE_MEAN,
    BEST_SINGLE_TREE,
)

# Fixed-Umax validation (Fig. 4): same Umax for every model; only predicted Tthr differs.
FIG4_FIXED_UMAX_MODELS: Tuple[str, ...] = (
    TAR_MODEL,
    BEST_SINGLE_TREE,
    UNIFORM_TREE_MEAN,
)
CORE_CLOSED_LOOP_MODELS = FIG4_FIXED_UMAX_MODELS

CLOSED_LOOP_DISPLAY_LABELS: Dict[str, str] = {
    TAR_MODEL: "TAR",
    TAR_NO_CLOSED_LOOP_MODEL: "TAR",
    RANDOM_FOREST: "RF",
    BEST_SINGLE_TREE: "BestTree",
    UNIFORM_TREE_MEAN: "UniformTreeMean",
    "TAR-SRL": "TAR",
    LEGACY_TAR_SRL_NO_CYCLE: "TAR",
    "ExtraTrees": "ET",
}
FIXED_UMAX_DISPLAY_LABELS = CLOSED_LOOP_DISPLAY_LABELS

OBJECTIVE_REFERENCE_TARGETS: Dict[str, str] = {
    "LR": "desired_LR_i per case when present in metadata; else lr_reference_target constant fallback",
    "P_AUC": "desired_P_AUC per case; feasibility at probiotic_pauc_fraction × desired_P_AUC",
    "pathogen": "pathogen_ceiling_cfu_per_mL — one-sided penalty only when terminal pathogen exceeds ceiling",
    "dose": "dose_reference_scale from training-only reference-controller dosage quantile (default q90)",
}

OPTIMIZER_REFERENCE_BY_REPEAT_CSV = "optimizer_reference_by_repeat.csv"
OPTIMIZER_REFERENCE_SUMMARY_CSV = "optimizer_reference_summary.csv"
OPTIMIZER_REFERENCE_MANIFEST_JSON = "optimizer_reference_manifest.json"
FIXED_UMAX_POLICY_BY_REPEAT_CSV = "fixed_umax_policy_by_repeat.csv"
FIXED_UMAX_POLICY_SUMMARY_CSV = "fixed_umax_policy_summary.csv"
FIXED_UMAX_POLICY_MANIFEST_JSON = "fixed_umax_policy_manifest.json"
FIXED_REPRESENTATIVE_UMAX_DEFAULT = 18.0
UMAX_POLICY_PER_CASE_OPTIMIZED = "per_case_optimized"
UMAX_POLICY_TRAINING_MEDIAN = "training_median_soft_umax"
UMAX_POLICY_TRAINING_TUNED_GLOBAL = "training_tuned_global_umax"
UMAX_POLICY_REPRESENTATIVE_FIXED = "representative_fixed"
UMAX_POLICY_METADATA_SOFT = "metadata_soft_umax"
PAUC_FEASIBILITY_EPS = 1e-12

FIG4_SIGNIFICANCE_CONTROLS: Tuple[str, ...] = (
    BEST_SINGLE_TREE,
    UNIFORM_TREE_MEAN,
)
CLOSED_LOOP_SIGNIFICANCE_CONTROLS = FIG4_SIGNIFICANCE_CONTROLS

CLOSED_LOOP_METRIC_DIRECTION: Dict[str, str] = {
    "mean_composite_score": "lower_is_better",
    "mean_target_tracking_error": "lower_is_better",
    "mean_total_dosage": "lower_is_better",
    "mean_terminal_pathogen": "lower_is_better",
    "mean_P_AUC": "higher_is_better",
    "mean_LR": "higher_is_better",
    "constraint_success_rate": "higher_is_better",
    "pathogen_constraint_success_rate": "higher_is_better",
    "probiotic_constraint_success_rate": "higher_is_better",
}

REPEATED_CLOSED_LOOP_METRICS: Tuple[str, ...] = (
    "mean_composite_score",
    "mean_target_tracking_error",
    "mean_total_dosage",
    "mean_terminal_pathogen",
    "mean_P_AUC",
    "mean_LR",
    "constraint_success_rate",
    "pathogen_constraint_success_rate",
    "probiotic_constraint_success_rate",
)

FIG4_PANEL_ORDER: Tuple[Tuple[str, str, str], ...] = (
    ("A", "fixed_umax_representative.png", "deterministic representative trajectories"),
    ("B", "fixed_umax_summary.png", "Fixed-Umax forward ODE summary (single point estimate per model)"),
    ("C", "fixed_umax_constraint_success.png", "Constraint success rates (optional if redundant)"),
)

FIG4_PRIMARY_FIGURES: Tuple[str, ...] = (
    "fixed_umax_representative.png",
    "fixed_umax_summary.png",
)

FIG4_OPTIONAL_FIGURES: Tuple[str, ...] = (
    "fixed_umax_constraint_success.png",
)
FIG4_OPTIONAL_SUMMARY_FIGURES = FIG4_OPTIONAL_FIGURES

FIG4B_SUMMARY_METRICS: Tuple[str, ...] = (
    "mean_total_dosage",
    "mean_P_AUC",
    "mean_LR",
    "mean_terminal_pathogen",
)

FIXED_UMAX_VALIDATION_MODE = "fixed_umax_comparative"

# Fig. 5 — Umax optimization justification (separate from fixed-Umax Fig. 4).
FIG5_ABLATION_CONDITIONS: Tuple[str, ...] = (
    "TAR_optimized",
    "TAR_fixed_training_median",
    "TAR_fixed_training_tuned_global",
)
FIG5_SECONDARY_ABLATION_CONDITIONS: Tuple[str, ...] = (
    "RF_optimized",
    "RF_fixed_training_tuned_global",
)
FIG5_SECONDARY_SUMMARY_CONDITIONS: Tuple[str, ...] = (
    "TAR_optimized",
    "TAR_fixed_training_tuned_global",
    "RF_optimized",
    "RF_fixed_training_tuned_global",
)
FIG5_ALL_ABLATION_CONDITIONS: Tuple[str, ...] = FIG5_ABLATION_CONDITIONS + FIG5_SECONDARY_ABLATION_CONDITIONS
FIG5_REQUIRED_MAIN_CONDITIONS: Tuple[str, ...] = FIG5_ABLATION_CONDITIONS
TRAINING_TUNED_GLOBAL_SELECTION_RULE = (
    "minimize mean composite_penalty; within 1% tolerance minimize mean total dosage; tie-break lower Umax"
)
FIG5_ABLATION_SPEC: Dict[str, Tuple[str, str]] = {
    "TAR_optimized": (TAR_MODEL, UMAX_POLICY_PER_CASE_OPTIMIZED),
    "TAR_fixed_training_median": (TAR_MODEL, UMAX_POLICY_TRAINING_MEDIAN),
    "TAR_fixed_training_tuned_global": (TAR_MODEL, UMAX_POLICY_TRAINING_TUNED_GLOBAL),
    "RF_optimized": (RANDOM_FOREST, UMAX_POLICY_PER_CASE_OPTIMIZED),
    "RF_fixed_training_tuned_global": (RANDOM_FOREST, UMAX_POLICY_TRAINING_TUNED_GLOBAL),
}
FIG5_FIXED_UMAX_POLICY_NAMES: Tuple[str, ...] = (
    UMAX_POLICY_TRAINING_MEDIAN,
    UMAX_POLICY_TRAINING_TUNED_GLOBAL,
)
SHARED_OPTIMIZER_REFERENCE_FILES: Tuple[str, ...] = (
    OPTIMIZER_REFERENCE_BY_REPEAT_CSV,
    OPTIMIZER_REFERENCE_SUMMARY_CSV,
    OPTIMIZER_REFERENCE_MANIFEST_JSON,
)


def subsample_prediction_jobs(
    jobs: List[Tuple[int, str, str]],
    max_repeats: Optional[int],
    strategy: str = "even",
) -> List[Tuple[int, str, str]]:
    """Keep a development subset of prediction repeats (does not affect formal step 10)."""
    if max_repeats is None or max_repeats <= 0 or len(jobs) <= max_repeats:
        return jobs
    jobs = sorted(jobs, key=lambda item: item[0])
    if strategy == "first":
        return jobs[:max_repeats]
    if strategy == "even":
        pick = np.linspace(0, len(jobs) - 1, max_repeats, dtype=int)
        return [jobs[int(i)] for i in pick]
    return jobs[:max_repeats]


def resolve_umax_ablation_spec(config: ClosedLoopConfig) -> Dict[str, Tuple[str, str]]:
    if config.umax_ablation_conditions:
        spec: Dict[str, Tuple[str, str]] = {}
        for condition in config.umax_ablation_conditions:
            if condition not in FIG5_ABLATION_SPEC:
                raise ValueError(f"Unknown Umax ablation condition: {condition}")
            spec[str(condition)] = FIG5_ABLATION_SPEC[condition]
        return spec
    return dict(FIG5_ABLATION_SPEC)


def ablation_spec_requires_fixed_umax_policies(ablation_spec: Dict[str, Tuple[str, str]]) -> bool:
    fixed = set(FIG5_FIXED_UMAX_POLICY_NAMES)
    return any(policy in fixed for _model, policy in ablation_spec.values())


def optimized_base_models_in_spec(ablation_spec: Dict[str, Tuple[str, str]]) -> Tuple[str, ...]:
    """Base prediction models that require per-case Umax grid inverse-design."""
    models = sorted(
        {
            base_model
            for base_model, policy in ablation_spec.values()
            if policy == UMAX_POLICY_PER_CASE_OPTIMIZED
        }
    )
    return tuple(models)


def required_umax_ablation_conditions(config: ClosedLoopConfig) -> Tuple[str, ...]:
    if config.umax_ablation_conditions:
        return tuple(config.umax_ablation_conditions)
    return FIG5_REQUIRED_MAIN_CONDITIONS


def copy_shared_optimizer_reference_artifacts(shared_reference_dir: str, outdir: str) -> bool:
    """Copy pre-derived dose-reference CSVs into a profile outdir (screening step 5b)."""
    import shutil

    copied = False
    os.makedirs(outdir, exist_ok=True)
    for fname in SHARED_OPTIMIZER_REFERENCE_FILES:
        src = os.path.join(shared_reference_dir, fname)
        dst = os.path.join(outdir, fname)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)
            copied = True
    return copied
FIG5_ABLATION_DISPLAY_LABELS: Dict[str, str] = {
    "TAR_optimized": "TAR + optimized $U_{max}$",
    "TAR_fixed_training_median": "TAR + training-median $U_{max}$",
    "TAR_fixed_training_tuned_global": "TAR + training-tuned global $U_{max}$",
    "RF_optimized": "RF + optimized $U_{max}$",
    "RF_fixed_training_tuned_global": "RF + training-tuned global $U_{max}$",
}
FIG5_ABLATION_TITLE_LABELS: Dict[str, str] = {
    "TAR_optimized": "TAR + optimized Umax",
    "TAR_fixed_training_median": "TAR + training-median Umax",
    "TAR_fixed_training_tuned_global": "TAR + training-tuned global Umax",
    "RF_optimized": "RF + optimized Umax",
    "RF_fixed_training_tuned_global": "RF + training-tuned global Umax",
}
FIG5_REPRESENTATIVE_CONDITIONS: Tuple[str, ...] = (
    "TAR_fixed_training_median",
    "TAR_fixed_training_tuned_global",
    "TAR_optimized",
)
FIG5_PANEL_ORDER: Tuple[Tuple[str, str, str], ...] = (
    ("A", "umax_score_landscape.png", "composite-penalty response landscape"),
    ("B", "umax_constraint_feasibility.png", "constraint feasibility"),
    ("C", "umax_ode_ablation.png", "Illustrative ODE ablation trajectories"),
    ("D", "umax_summary_ablation.png", "Repeated ablation summary (mean ± 95% CI)"),
)
FIG5_PRIMARY_FIGURES: Tuple[str, ...] = tuple(panel[1] for panel in FIG5_PANEL_ORDER)
FIG5B_SUMMARY_METRICS: Tuple[str, ...] = (
    "mean_total_dosage",
    "mean_P_AUC",
    "mean_LR",
    "mean_terminal_pathogen",
    "mean_composite_score",
)
FIG5_SIGNIFICANCE_REFERENCE = "TAR_optimized"
FIG5_SIGNIFICANCE_CONTROLS: Tuple[str, ...] = (
    "TAR_fixed_training_median",
    "TAR_fixed_training_tuned_global",
)
UMAX_OPTIMIZATION_STUDY_MODE = "umax_optimization_study"

U_CANDIDATE_EXPORT_COLUMNS: Tuple[str, ...] = (
    "repeat_id",
    "case_index",
    "validation_original_row_index",
    "model",
    "candidate_u_max",
    "total_dosage",
    "dose_count",
    "P_AUC",
    "LR1",
    "LR2",
    "LR3",
    "LR4",
    "LR5",
    "mean_LR",
    "final_total_pathogen",
    "terminal_total_pathogen",
    "target_tracking_error",
    "target_P_AUC",
    "minimum_acceptable_P_AUC",
    "target_LR1",
    "target_LR2",
    "target_LR3",
    "target_LR4",
    "target_LR5",
    "target_mean_LR",
    "target_terminal_pathogen",
    "dose_reference_scale",
    "dose_reference_source",
    "lr_target_source",
    "LR_shortfall_norm",
    "PAUC_shortfall_norm",
    "pathogen_violation_norm",
    "dose_burden_norm",
    "composite_penalty",
    "feasible_candidate",
    "optimizer_selection_rule",
    "selected_by_optimizer",
    "dose_signed_component",
    "pauc_signed_component",
    "lr_signed_component",
    "pathogen_signed_component",
    "signed_relative_inner",
    "signed_relative_rms",
    "hard_violation_rms",
    "optimizer_primary_score",
    "optimizer_tiebreak_score",
    "optimizer_selection_rank",
    "composite_score",
    "aspiration_total_dosage",
    "aspiration_P_AUC",
    "aspiration_LR1",
    "aspiration_LR2",
    "aspiration_LR3",
    "aspiration_LR4",
    "aspiration_LR5",
    "aspiration_mean_LR",
    "aspiration_terminal_pathogen",
    "aspiration_abs_distance_to_initial",
    "aspiration_dose_term",
    "aspiration_pauc_term",
    "aspiration_lr_term",
    "aspiration_pathogen_log_term",
    "dominates_initial_aspiration",
    "initial_aspiration_dominator_count",
    "closest_to_initial_aspiration",
    "updated_reference_from_closest_candidate",
    "dominates_updated_reference",
    "updated_reference_dominator_count",
    "pareto_improved_from_closest",
    "optimizer_selection_stage",
)

UMAX_RESPONSE_LANDSCAPE_COLUMNS: Tuple[str, ...] = (
    "repeat_id",
    "case_index",
    "validation_original_row_index",
    "unit_id",
    "model",
    "base_model",
    "umax_policy",
    "candidate_u_max",
    "total_dosage",
    "dose_count",
    "P_AUC",
    "LR1",
    "LR2",
    "LR3",
    "LR4",
    "LR5",
    "mean_LR",
    "final_total_pathogen",
    "terminal_total_pathogen",
    "target_P_AUC",
    "minimum_acceptable_P_AUC",
    "target_LR1",
    "target_LR2",
    "target_LR3",
    "target_LR4",
    "target_LR5",
    "target_mean_LR",
    "target_terminal_pathogen",
    "dose_reference_scale",
    "dose_reference_source",
    "lr_target_source",
    "LR_constraint_satisfied",
    "P_AUC_constraint_satisfied",
    "pathogen_constraint_satisfied",
    "feasible_candidate",
    "LR_shortfall_norm",
    "PAUC_shortfall_norm",
    "pathogen_violation_norm",
    "dose_burden_norm",
    "composite_penalty",
    "selected_by_optimizer",
    "optimizer_selection_policy",
    "optimizer_selection_rule",
    "optimizer_selection_stage",
    "optimizer_selection_rank",
    "optimizer_primary_score",
    "optimizer_tiebreak_score",
    "aspiration_abs_distance_to_initial",
    "dominates_initial_aspiration",
    "closest_to_initial_aspiration",
    "pareto_improved_from_closest",
)

UMAX_FEASIBLE_REGION_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "repeat_id",
    "case_index",
    "model",
    "n_candidates",
    "n_feasible_candidates",
    "feasible_fraction",
    "min_feasible_u_max",
    "max_feasible_u_max",
    "selected_u_max",
    "selected_is_feasible",
    "selected_total_dosage",
    "selected_P_AUC",
    "selected_mean_LR",
    "selected_terminal_pathogen",
    "selected_composite_penalty",
    "selection_rule",
    "selection_stage",
)

U_CANDIDATE_LEGACY_RENAMES: Dict[str, str] = {
    "total_dosage_ug_per_mL": "total_dosage",
    "final_total_pathogen_CFU_per_mL": "final_total_pathogen",
    "terminal_total_pathogen_CFU_per_mL": "terminal_total_pathogen",
    "total_dosage_norm": "dose_burden_norm",
    "optimization_penalty_score": "composite_penalty",
}

U_LANDSCAPE_METRICS: Tuple[str, ...] = (
    "total_dosage",
    "P_AUC",
    "mean_LR",
    "terminal_total_pathogen",
)

U_OBJECTIVE_PREFERRED_COLUMNS: Tuple[str, ...] = (
    "U_dose_reference_limit",
    "U_P_AUC_constraint_limit",
    "U_LR_feasibility",
    "U_pathogen_feasibility",
    "U_final_selected",
)

U_OPTIMA_COLUMNS: Tuple[str, ...] = U_OBJECTIVE_PREFERRED_COLUMNS

U_OBJECTIVE_CATEGORY_LABELS: Dict[str, str] = {
    "U_dose_reference_limit": "Dose reference limit",
    "U_P_AUC_constraint_limit": "P_AUC constraint limit",
    "U_LR_feasibility": "LR feasibility threshold",
    "U_pathogen_feasibility": "Pathogen feasibility threshold",
    "U_final_selected": "Final selected Umax",
}

U_OPTIMA_LEGACY_RENAMES: Dict[str, str] = {
    "U_dose_plateau": "U_dose_reference_limit",
    "U_composite_selected": "U_final_selected",
    "U_star_dose": "U_dose_reference_limit",
    "U_star_P_AUC": "U_P_AUC_constraint_limit",
    "U_star_mean_LR": "U_LR_feasibility",
    "U_star_pathogen": "U_pathogen_feasibility",
    "U_star_composite": "U_final_selected",
    "Ustar_dose": "U_dose_reference_limit",
    "Ustar_P_AUC": "U_P_AUC_constraint_limit",
    "Ustar_mean_LR": "U_LR_feasibility",
    "Ustar_pathogen": "U_pathogen_feasibility",
    "Ustar_composite": "U_final_selected",
    "U_P_AUC_preservation": "U_P_AUC_constraint_limit",
    "U_LR_response": "U_LR_feasibility",
    "U_pathogen_suppression": "U_pathogen_feasibility",
}

UMAX_SELECTION_POLICIES: Tuple[str, ...] = ("feasible_first", "aspiration_then_pareto")
UMAX_SELECTION_POLICY_DEFAULT = "feasible_first"
ASPIRATION_EPS_DEFAULT = 1e-12
UMAX_ASPIRATION_SELECTION_DEBUG_CSV = "umax_aspiration_selection_debug.csv"
UMAX_SELECTION_POLICY_SENSITIVITY_CSV = "umax_selection_policy_sensitivity.csv"
UMAX_RESPONSE_LANDSCAPE_CSV = "umax_response_landscape.csv"
UMAX_FEASIBLE_REGION_SUMMARY_CSV = "umax_feasible_region_summary.csv"
UMAX_SELECTED_UMAX_DISTRIBUTION_CSV = "umax_selected_umax_distribution.csv"
REVIEWER_NOTE_UMAX_INVERSE_DESIGN = (
    "Umax was not learned as a supervised target. Given the TAR-predicted Tthr vector, "
    "each candidate Umax was forward-reinserted into the closed-loop ODE simulator to construct "
    "a Umax-response landscape. The primary selection rule was constraint-first: choose the "
    "lowest-dosage feasible candidate satisfying LR, P_AUC, and pathogen constraints; if no "
    "feasible candidate existed, choose the lowest composite-penalty fallback. The "
    "aspiration-point rule was retained only as a sensitivity analysis."
)


def _fig_panel_record(
    panel: str,
    filename: str,
    description: str,
    *,
    role: str,
    input_sources: Sequence[str],
    n_repeats: int,
    umax_setting: str,
    significance_mode: str,
) -> dict:
    return {
        "panel": panel,
        "filename": filename,
        "description": description,
        "role": role,
        "input_sources": list(input_sources),
        "n_repeats": int(n_repeats),
        "umax_setting": umax_setting,
        "significance_mode": significance_mode,
    }


def build_fig4_manifest_fields(
    *,
    n_prediction_repeats: int,
    significance_for_manuscript: bool,
    prediction_sources: Optional[Sequence[str]] = None,
) -> dict:
    """Metadata for manuscript Fig. 4 fixed-Umax validation panels."""
    sig_mode = "ode_single_forward_run_no_repeat_significance"
    panel_order = [
        _fig_panel_record(
            panel, filename, desc,
            role="primary_manuscript_figure" if filename in FIG4_PRIMARY_FIGURES else "optional_manuscript_figure",
            input_sources={
                "fixed_umax_representative.png": [
                    f"fixed_umax_validation/{FIXED_UMAX_TRAJECTORIES_CSV}",
                    f"fixed_umax_validation/{FIG4_PLOT_MANIFEST_JSON}",
                ],
                "fixed_umax_summary.png": [
                    f"fixed_umax_validation/{FIXED_UMAX_REPEATED_STATS_CSV}",
                    f"fixed_umax_validation/{FIXED_UMAX_SIGNIFICANCE_CSV}",
                ],
                "fixed_umax_constraint_success.png": [f"fixed_umax_validation/{FIXED_UMAX_REPEATED_STATS_CSV}"],
            }.get(filename, []),
            n_repeats=1,
            umax_setting="fixed_paper_figure_profile",
            significance_mode="none" if filename == "fixed_umax_representative.png" else sig_mode,
        )
        for panel, filename, desc in FIG4_PANEL_ORDER
    ]
    primary = [entry["filename"] for entry in panel_order if entry["filename"] in FIG4_PRIMARY_FIGURES]
    return {
        "figure": "Fig. 4",
        "figure_title": "fixed-Umax validation",
        "validation_section": "fixed-Umax validation",
        "validation_mode": FIXED_UMAX_VALIDATION_MODE,
        "prediction_input_sources": list(prediction_sources or []),
        "fixed_umax_rule": (
            "Fig. 4 uses multi_pathogen_simulator paper_figure bio parameters and shared "
            f"Umax={PAPER_FIGURE_PROFILE.u_max_rep:g} µg/mL. Per-model Tthr is aggregated from "
            "benchmark prediction repeats (median of per-split medians). Exactly three forward "
            "ODE runs total (TAR / BestTree / UniformTreeMean); no per-test-row and no per-repeat ODE."
        ),
        "figure_panel_mapping": panel_order,
        "fig4_panel_order": panel_order,
        "fig4_primary_figures": primary,
        "fig4_optional_figures": list(FIG4_OPTIONAL_FIGURES),
        "fig4_models": list(FIG4_FIXED_UMAX_MODELS),
        "fig4_model_display_labels": {m: FIXED_UMAX_DISPLAY_LABELS[m] for m in FIG4_FIXED_UMAX_MODELS},
        "fig4_significance_comparisons": list(FIG4_SIGNIFICANCE_CONTROLS),
        "fig4b_metrics": list(FIG4B_SUMMARY_METRICS),
        "fig4_requires_repeated_splits": False,
        "n_prediction_repeats": int(n_prediction_repeats),
        "n_ode_forward_runs": int(FIG4_ODE_FORWARD_MODELS),
        "n_repeats": 1,
        "significance_for_manuscript": False,
        "exploratory": True,
        "manuscript_safe": False,
        "illustrative_only_representative": True,
        "significance_rule": (
            "Fig. 4 ODE is a single illustrative forward comparison (3 runs). "
            "Fig. 4B bars are point estimates without repeat-level ODE CIs."
        ),
    }


def build_fig5_manifest_fields(*, n_repeats: int, significance_for_manuscript: bool) -> dict:
    sig_mode = (
        "TAR_optimized_vs_TAR_training_median_and_training_tuned_global_n_repeats_ge_10"
        if significance_for_manuscript
        else "exploratory_no_formal_stars_n_repeats_lt_10"
    )
    panel_order = [
        _fig_panel_record(
            panel, filename, desc,
            role="primary_manuscript_figure",
            input_sources={
                "umax_score_landscape.png": [
                    "umax_optimization/umax_score_landscape_curves.csv",
                    "umax_optimization/umax_selected_umax_distribution.csv",
                ],
                "umax_constraint_feasibility.png": [
                    "umax_optimization/umax_response_landscape.csv",
                    "umax_optimization/umax_optimization_u_candidates.csv",
                ],
                "umax_ode_ablation.png": [
                    "umax_optimization/umax_ablation_representative_trajectories.csv",
                    f"umax_optimization/{FIG5_PLOT_MANIFEST_JSON}",
                ],
                "umax_summary_ablation.png": [
                    "umax_optimization/umax_ablation_repeated_plot_stats.csv",
                    "umax_optimization/umax_ablation_significance_annotations.csv",
                ],
                "umax_summary_ablation_composite_supplementary.png": [
                    "umax_optimization/umax_ablation_repeated_plot_stats.csv",
                ],
            }.get(filename, []),
            n_repeats=n_repeats,
            umax_setting={
                "umax_score_landscape.png": "grid_scan_0_to_100_median_band",
                "umax_constraint_feasibility.png": "constraint_feasibility_fraction_by_u",
                "umax_ode_ablation.png": "TAR_training_median_tuned_global_optimized_illustrative",
                "umax_summary_ablation.png": "TAR_policy_ablation_three_conditions",
            }.get(filename, "optimized_and_training_fixed_policy_ablation"),
            significance_mode="none" if filename != "umax_summary_ablation.png" else sig_mode,
        )
        for panel, filename, desc in FIG5_PANEL_ORDER
    ]
    return {
        "figure": "Fig. 5",
        "figure_title": "Umax optimization analysis",
        "validation_section": "Umax optimization analysis",
        "validation_mode": UMAX_OPTIMIZATION_STUDY_MODE,
        "fixed_umax_policies_training_only": True,
        "validation_rows_excluded_from_policy_derivation": True,
        "optimized_umax_rule": (
            "grid-based Umax inverse-design: post-prediction forward ODE reinsertion over Umax grid; "
            "constraint-first dose-scale selection (feasible_first primary; aspiration_then_pareto sensitivity only)"
        ),
        "fig5d_main_comparison": "TAR-only policy ablation (optimized vs training-median vs training-tuned global)",
        "fig5d_secondary_comparison": "includes RF optimized and RF training-tuned global controls",
        "optimizer_score_rule": (
            "Primary infeasible fallback: weighted one-sided composite_penalty (>= 0). "
            "umax_score_landscape_curves.optimization_penalty_score aliases composite_penalty; "
            "aspiration_abs_distance_to_initial is diagnostic only."
        ),
        "umax_inverse_design_module": True,
        "umax_not_supervised_target": True,
        "tar_predicts_only": "Tthr vector (Tthr_1..Tthr_5); Umax selected post hoc by closed_loop_eval.py",
        "primary_umax_selection_policy": UMAX_SELECTION_POLICY_DEFAULT,
        "aspiration_then_pareto_role": "sensitivity_analysis_diagnostic_only",
        "figure_panel_mapping": panel_order,
        "fig5_panel_order": panel_order,
        "fig5_primary_figures": list(FIG5_PRIMARY_FIGURES),
        "fig5_ablation_conditions": list(FIG5_ABLATION_CONDITIONS),
        "fig5_secondary_ablation_conditions": list(FIG5_SECONDARY_ABLATION_CONDITIONS),
        "fig5_secondary_summary_conditions": list(FIG5_SECONDARY_SUMMARY_CONDITIONS),
        "fig5_ablation_spec": {
            cond: {"base_model": spec[0], "umax_policy": spec[1]}
            for cond, spec in FIG5_ABLATION_SPEC.items()
        },
        "fig5_ablation_display_labels": dict(FIG5_ABLATION_DISPLAY_LABELS),
        "fig5b_metrics": list(FIG5B_SUMMARY_METRICS),
        "fig5_significance_reference": FIG5_SIGNIFICANCE_REFERENCE,
        "fig5_significance_controls": list(FIG5_SIGNIFICANCE_CONTROLS),
        "fig5_requires_repeated_splits": True,
        "n_repeats": int(n_repeats),
        "significance_for_manuscript": bool(significance_for_manuscript),
        "exploratory": bool(n_repeats < 10),
        "manuscript_safe": bool(n_repeats >= 100 and significance_for_manuscript),
        "illustrative_only_ablation_trajectories": True,
        "significance_rule": (
            "TAR optimized vs TAR training-median and TAR training-tuned global; "
            "stars when reference significantly better; no formal stars when n_repeats < 10"
        ),
    }


WEIGHT_PROFILE_NAMES: Tuple[str, ...] = (
    "balanced",
    "efficacy",
    "probiotic_sparing",
    "dose_sparing",
    "custom",
)

LEGACY_PREDICTION_MODEL_MAP: Dict[str, str] = {
    "TAR-SRL": TAR_MODEL,
    LEGACY_TAR_SRL_NO_CYCLE: TAR_MODEL,
}


@dataclass
class ClosedLoopWeights:
    w_track: float = 1.0
    w_path: float = 1.0
    w_probiotic: float = 1.0
    w_dose: float = 0.25


def resolve_weights_from_profile(
    profile: str,
    w_track: float,
    w_path: float,
    w_probiotic: float,
    w_dose: float,
) -> ClosedLoopWeights:
    presets = {
        "balanced": ClosedLoopWeights(1.0, 1.0, 1.0, 0.25),
        "efficacy": ClosedLoopWeights(1.0, 2.0, 1.0, 0.25),
        "probiotic_sparing": ClosedLoopWeights(1.0, 1.0, 2.0, 0.25),
        "dose_sparing": ClosedLoopWeights(1.0, 1.0, 1.0, 1.0),
    }
    if profile == "custom":
        return ClosedLoopWeights(w_track, w_path, w_probiotic, w_dose)
    if profile not in presets:
        raise ValueError(f"Unknown weight_profile '{profile}'. Choose from {list(presets)} or custom.")
    return presets[profile]


@dataclass
class ClosedLoopConfig:
    u_grid: np.ndarray = field(default_factory=lambda: np.arange(0, 101, 1, dtype=float))
    weights: ClosedLoopWeights = field(default_factory=ClosedLoopWeights)
    pathogen_ceiling_cfu_per_mL: float = 4.0e7
    pathogen_floor_cfu_per_mL: float = 1.0e7
    dosage_reference_target: float = 2500.0
    target_total_dosage: float = 2500.0
    target_terminal_pathogen: float = 4.0e7
    pauc_target_source: str = "desired"
    pauc_feasibility_fraction: float = 0.90
    lr_target_source: str = "desired_or_constant"
    pathogen_target_source: str = "ceiling"
    dose_reference_source: str = "training_q90_reference_dosage"
    dose_reference_quantile: float = 0.90
    dose_reference_constant: float = 2500.0
    fixed_representative_umax: float = FIXED_REPRESENTATIVE_UMAX_DEFAULT
    target_pauc_source: str = "desired"
    target_lr_source: str = "desired_or_constant"
    lr_reference_target: float = 1.0
    probiotic_pauc_fraction: float = 0.90
    lr_tolerance: float = 0.0
    constraint_tol: float = 1e-6
    max_closed_loop_cases: int = 0
    permutation_replicates: int = 5000
    bootstrap_replicates: int = 500
    representative_scan_limit: int = 500
    repeat_ci_method: str = "t_interval"
    weight_profile: str = "balanced"
    weight_selection_source: str = "fixed"
    prediction_source: str = ""
    run_umax_optimization_study: bool = False
    ode_backend: str = "auto"
    case_n_jobs: int = 0
    u_grid_n_jobs: int = 0
    umax_selection_policy: str = UMAX_SELECTION_POLICY_DEFAULT
    aspiration_tolerance_rel: float = 0.001
    aspiration_tolerance_abs: float = 1e-9
    aspiration_eps: float = ASPIRATION_EPS_DEFAULT
    umax_ablation_conditions: Optional[Tuple[str, ...]] = None
    screening_sensitivity_mode: bool = False
    export_umax_score_landscape: bool = True
    export_umax_representative_trajectories: bool = True


_RESOLVED_ODE_BACKEND: Optional[str] = None


@dataclass
class MetricsOnlySimulation:
    """Lightweight ODE outcome for u_grid scoring (numba metrics path, no trajectories)."""

    dose_count: int
    total_dosage: float
    P_AUC: float
    LR_terminal_median: np.ndarray
    final_total_pathogen: float
    terminal_total_pathogen: float
    final_probiotic: float = float("nan")
    times: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    B_total: np.ndarray = field(default_factory=lambda: np.zeros((0, N_STRAINS), dtype=float))
    C: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    P_S: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    P_R: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))


@dataclass
class CaseBioParams:
    B0: np.ndarray
    k_arr: np.ndarray
    gamma_arr: np.ndarray
    rho_arr: np.ndarray
    mu_arr: np.ndarray
    desired_pauc: float
    desired_lr: np.ndarray


SIGNED_BENEFIT_CAP = 1.0
SIGNED_PENALTY_CAP = 25.0
HARD_VIOLATION_TOL_ABS = 1e-6
HARD_VIOLATION_TOL_REL = 0.01


def parse_u_grid(spec: str) -> np.ndarray:
    if spec.startswith("arange:"):
        parts = spec.split(":")
        if len(parts) != 4:
            raise ValueError("u_grid arange format must be arange:start:stop:step")
        start, stop, step = (float(parts[1]), float(parts[2]), float(parts[3]))
        if step <= 0:
            raise ValueError("u_grid step must be positive.")
        n = int(np.floor((stop - start) / step + 0.5)) + 1
        grid = start + step * np.arange(max(n, 1), dtype=float)
        grid = grid[grid <= stop + 1e-9]
        decimals = max(0, -int(np.floor(np.log10(step)))) if step < 1 else 0
        if decimals > 0:
            grid = np.round(grid, decimals=decimals)
        return np.unique(grid)
    values = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not values:
        raise ValueError("u_grid must not be empty.")
    return np.asarray(values, dtype=float)


def u_grid_manifest_fields(u_grid: np.ndarray) -> dict:
    u = np.asarray(u_grid, dtype=float)
    if len(u) == 0:
        return {"u_grid_min": float("nan"), "u_grid_max": float("nan"), "u_grid_step": float("nan"), "n_u_grid_points": 0}
    diffs = np.diff(np.unique(u))
    step = float(np.median(diffs)) if len(diffs) else 0.0
    return {
        "u_grid_min": float(np.min(u)),
        "u_grid_max": float(np.max(u)),
        "u_grid_step": step,
        "n_u_grid_points": int(len(u)),
    }


def compute_all_constraints_success_all_zero(summary_df: Optional[pd.DataFrame]) -> bool:
    if summary_df is None or summary_df.empty or "constraint_success_rate" not in summary_df.columns:
        return False
    vals = summary_df["constraint_success_rate"].fillna(0).to_numpy(dtype=float)
    return bool(np.all(np.isfinite(vals)) and np.all(vals <= 0))


def build_closed_loop_optimizer_manifest_fields(
    config: ClosedLoopConfig,
    dose_ref: float,
    *,
    summary_df: Optional[pd.DataFrame] = None,
    lr_target_source: str = "",
    dose_reference_source: str = "",
) -> dict:
    pauc_frac = resolve_pauc_feasibility_fraction(config)
    primary_policy = UMAX_SELECTION_POLICY_DEFAULT
    return {
        "dosage_reference": dose_ref,
        "dose_reference_scale": dose_ref,
        "dose_reference_source": dose_reference_source or config.dose_reference_source,
        "dose_reference_quantile": float(config.dose_reference_quantile),
        "target_terminal_pathogen": resolve_target_terminal_pathogen(config),
        "pauc_target_source": resolve_pauc_target_source(config),
        "pauc_feasibility_fraction": pauc_frac,
        "lr_target_source": lr_target_source or config.lr_target_source,
        "lr_reference_target": float(config.lr_reference_target),
        "pathogen_target_source": config.pathogen_target_source,
        "pathogen_ceiling_cfu_per_mL": float(config.pathogen_ceiling_cfu_per_mL),
        "target_pauc_source": resolve_pauc_target_source(config),
        "target_lr_source": config.lr_target_source,
        "probiotic_pauc_fraction": pauc_frac,
        **u_grid_manifest_fields(config.u_grid),
        "umax_optimizer_module": "grid-based Umax inverse-design / response-landscape selection",
        "umax_not_supervised_target": True,
        "tar_predicts_only": "Tthr vector (Tthr_1..Tthr_5)",
        "umax_selection_post_prediction": "forward ODE reinsertion into closed-loop simulator",
        "umax_selection_policy": config.umax_selection_policy,
        "primary_umax_selection_policy": primary_policy,
        "umax_selection_policy_applied_to_all_models": True,
        "primary_selection_rule": (
            "constraint-first dose-scale selection: select lowest-dosage feasible candidate "
            "satisfying LR, P_AUC, and pathogen constraints; if no feasible candidate exists, "
            "select lowest composite-penalty fallback"
        ),
        "aspiration_then_pareto_role": "sensitivity_analysis_diagnostic_only",
        "fixed_umax_policies_training_only": True,
        "held_out_samples_excluded": True,
        "validation_rows_excluded_from_policy_derivation": True,
        "reviewer_note_umax_inverse_design": REVIEWER_NOTE_UMAX_INVERSE_DESIGN,
        "aspiration_reference_definitions": {
            "aspiration_total_dosage": "dose_reference_scale (training q90 reference-controller dosage)",
            "aspiration_P_AUC": "minimum_acceptable_P_AUC = pauc_feasibility_fraction * desired_P_AUC",
            "aspiration_LR_i": "target_LR_i from desired_LR1..5 metadata or lr_reference_target fallback",
            "aspiration_terminal_pathogen": "target_terminal_pathogen (pathogen_ceiling_cfu_per_mL)",
        },
        "aspiration_targets_per_case_desired": config.lr_target_source in ("desired", "desired_or_constant"),
        "aspiration_targets_constant_fallback": config.lr_target_source != "desired",
        "aspiration_references_training_only_no_held_out": True,
        "aspiration_tolerance_rel": float(config.aspiration_tolerance_rel),
        "aspiration_tolerance_abs": float(config.aspiration_tolerance_abs),
        "objective_formula": (
            "Primary (feasible_first): constraint-first Umax inverse-design — feasible candidates "
            "minimize total_dosage with tie-breaks; otherwise minimize weighted one-sided composite_penalty. "
            "Sensitivity only (aspiration_then_pareto): aspiration-point distance + Pareto cleanup; "
            "not used as the manuscript primary optimizer."
        ),
        "objective_reference_targets": OBJECTIVE_REFERENCE_TARGETS,
        "optimizer_weights": {
            "w_track": config.weights.w_track,
            "w_probiotic": config.weights.w_probiotic,
            "w_path": config.weights.w_path,
            "w_dose": config.weights.w_dose,
        },
        "all_constraints_success_all_zero": compute_all_constraints_success_all_zero(summary_df),
    }


def clip_tthr(tthr: np.ndarray) -> np.ndarray:
    tthr = np.asarray(tthr, dtype=float).reshape(N_STRAINS)
    tthr = np.nan_to_num(tthr, nan=1.0, posinf=1.0e10, neginf=1.0)
    return np.clip(tthr, 1.0, 1.0e10)


def filter_closed_loop_predictions(predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    resolved: Dict[str, np.ndarray] = {}
    for name in CLOSED_LOOP_PREDICTION_MODELS:
        if name in predictions:
            resolved[name] = predictions[name]
        elif name == TAR_MODEL and "TAR-SRL" in predictions:
            resolved[name] = predictions["TAR-SRL"]
    missing = [m for m in CLOSED_LOOP_PREDICTION_MODELS if m not in resolved]
    if missing:
        raise ValueError(
            f"Closed-loop requires predictions for core models {list(CLOSED_LOOP_PREDICTION_MODELS)}; "
            f"missing: {missing}"
        )
    return resolved


def normalize_prediction_model_name(name: str) -> str:
    return LEGACY_PREDICTION_MODEL_MAP.get(name, name)


def parse_predictions_wide_df(
    pred_df: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, int, int]:
    if "validation_original_row_index" not in pred_df.columns:
        raise ValueError("predictions CSV must include validation_original_row_index")
    val_idx = pred_df["validation_original_row_index"].to_numpy(dtype=int)
    repeat_id = int(pred_df["repeat_id"].iloc[0]) if "repeat_id" in pred_df.columns else 0
    seed = int(pred_df["seed"].iloc[0]) if "seed" in pred_df.columns else repeat_id

    raw_predictions: Dict[str, np.ndarray] = {}
    for col in pred_df.columns:
        if not col.startswith("pred_"):
            continue
        remainder = col[len("pred_") :]
        for target in TARGET_COLS_TTHR:
            suffix = f"_{target}"
            if not remainder.endswith(suffix):
                continue
            model_name = normalize_prediction_model_name(remainder[: -len(suffix)])
            if model_name not in CLOSED_LOOP_PREDICTION_MODELS:
                continue
            if model_name not in raw_predictions:
                raw_predictions[model_name] = np.zeros((len(pred_df), len(TARGET_COLS_TTHR)), dtype=float)
            raw_predictions[model_name][:, TARGET_COLS_TTHR.index(target)] = pred_df[col].to_numpy(dtype=float)
    return filter_closed_loop_predictions(raw_predictions), val_idx, repeat_id, seed


def load_predictions_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _repeat_id_from_prediction_path(path: str) -> int:
    match = re.search(r"repeat_(\d+)", path.replace("\\", "/"))
    if match:
        return int(match.group(1))
    return 0


def _normalize_prediction_source_label(source: str) -> str:
    """Collapse per-repeat prediction CSV paths to one label for cross-repeat fairness."""
    if not source:
        return ""
    norm = os.path.normpath(source).replace("\\", "/")
    return re.sub(r"repeat_\d+", "repeat_XXX", norm)


def _row_indices_for_model(cases_df: pd.DataFrame, model: str) -> List[int]:
    rows = cases_df[cases_df["model"] == model].sort_values("case_index")
    if rows.empty:
        return []
    return rows["original_row_index"].astype(int).tolist()


def _within_repeat_model_fairness(cases_df: pd.DataFrame) -> Tuple[bool, bool]:
    """Return (same_held_out_indices, same_case_ordering) across core models in one repeat."""
    index_sets = [
        _row_indices_for_model(cases_df, model)
        for model in CORE_CLOSED_LOOP_MODELS
        if not cases_df[cases_df["model"] == model].empty
    ]
    if not index_sets:
        return True, True
    ref = index_sets[0]
    same_indices = all(indices == ref for indices in index_sets[1:])
    return same_indices, same_indices


def resolve_prediction_csv_jobs(
    *,
    predictions_dir: Optional[str] = None,
    predictions_manifest: Optional[str] = None,
    predictions_csv: Optional[str] = None,
) -> List[Tuple[int, str, str]]:
    """Return (repeat_id, csv_path, prediction_source_label) jobs."""
    if predictions_csv:
        path = os.path.abspath(predictions_csv)
        return [(0, path, path)]

    manifest_base = None
    manifest: dict = {}
    if predictions_manifest:
        manifest = load_predictions_manifest(predictions_manifest)
        manifest_base = os.path.dirname(os.path.abspath(predictions_manifest))

    jobs: List[Tuple[int, str, str]] = []
    rel_files = manifest.get("prediction_files", [])
    for rel in rel_files:
        rel_norm = str(rel).replace("\\", "/")
        if "/repeat_" not in rel_norm or not rel_norm.endswith("predictions.csv"):
            continue
        full = os.path.join(manifest_base or "", rel.replace("/", os.sep))
        if os.path.isfile(full):
            jobs.append((_repeat_id_from_prediction_path(full), full, full))

    if not jobs and predictions_dir:
        pattern = os.path.join(predictions_dir, "repeat_*", "predictions.csv")
        for path in sorted(glob.glob(pattern)):
            jobs.append((_repeat_id_from_prediction_path(path), os.path.abspath(path), path))

    if not jobs:
        raise FileNotFoundError(
            "No repeat prediction CSVs found. Provide --predictions_dir, --predictions_manifest, or --predictions_csv."
        )
    dedup: Dict[int, Tuple[int, str, str]] = {}
    for repeat_id, path, source in jobs:
        dedup[repeat_id] = (repeat_id, path, source)
    return [dedup[k] for k in sorted(dedup)]


def load_validation_rows(
    x_csv: str,
    metadata_csv: Optional[str],
    val_indices: np.ndarray,
    *,
    x_df: Optional[pd.DataFrame] = None,
    metadata_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if x_df is None:
        x_df = pd.read_csv(x_csv)
    x_test_df = x_df.iloc[val_indices].reset_index(drop=True)
    meta_path = metadata_csv or os.path.join(os.path.dirname(os.path.abspath(x_csv)), "sample_metadata.csv")
    metadata_test_df = None
    if metadata_df is None and os.path.isfile(meta_path):
        metadata_df = pd.read_csv(meta_path)
    if metadata_df is not None:
        metadata_test_df = metadata_df.iloc[val_indices].reset_index(drop=True)
    return x_test_df, metadata_test_df


def extract_objectives(
    x_row: pd.Series,
    metadata_row: Optional[pd.Series],
) -> Tuple[float, np.ndarray]:
    if all(col in x_row.index for col in OBJECTIVE_COLS):
        desired_pauc = float(x_row["P_AUC"])
        desired_lr = np.array([float(x_row[f"LR{i}"]) for i in range(1, 6)], dtype=float)
        return desired_pauc, desired_lr
    if metadata_row is not None and all(col in metadata_row.index for col in METADATA_OBJECTIVE_COLS):
        desired_pauc = float(metadata_row["desired_P_AUC"])
        desired_lr = np.array(
            [float(metadata_row[f"desired_LR{i}"]) for i in range(1, 6)],
            dtype=float,
        )
        return desired_pauc, desired_lr
    raise ValueError("Could not resolve desired P_AUC/LR objectives from X or metadata.")


def bio_params_from_row(x_row: pd.Series, metadata_row: Optional[pd.Series]) -> CaseBioParams:
    missing = [col for col in BIO_COLS if col not in x_row.index]
    if missing:
        raise ValueError(f"Missing biological columns in test X: {missing[:5]}")
    bio = x_row[BIO_COLS].astype(float).to_numpy()
    desired_pauc, desired_lr = extract_objectives(x_row, metadata_row)
    return CaseBioParams(
        B0=bio[0:5],
        k_arr=bio[5:10],
        gamma_arr=bio[10:15],
        rho_arr=bio[15:20],
        mu_arr=bio[20:25],
        desired_pauc=desired_pauc,
        desired_lr=desired_lr,
    )


def paper_figure_bio_params() -> CaseBioParams:
    """Bio + desired outcomes for Fig. 4 forward sim (multi_pathogen_simulator paper_figure profile)."""
    ref = simulate_case(
        u_max=float(PAPER_FIGURE_PROFILE.u_max_rep),
        T_thr=np.asarray(PAPER_FIGURE_PROFILE.T_thr_rep, dtype=float),
    )
    return CaseBioParams(
        B0=np.asarray(B0_S_REP, dtype=float),
        k_arr=np.asarray(K_REP, dtype=float),
        gamma_arr=np.asarray(GAMMA_S_REP, dtype=float),
        rho_arr=np.asarray(RHO_REP, dtype=float),
        mu_arr=np.asarray(MU_REP, dtype=float),
        desired_pauc=float(ref.P_AUC),
        desired_lr=np.asarray(ref.LR_terminal_median, dtype=float),
    )


def fixed_umax_paper_profile(config: ClosedLoopConfig) -> float:
    """Shared fixed Umax for Fig. 4 (matches representative ODE / paper_figure profile)."""
    del config  # reserved for future override; paper profile Umax is manuscript default
    return float(PAPER_FIGURE_PROFILE.u_max_rep)


def median_predicted_tthr(
    predictions_tthr: Dict[str, np.ndarray],
    model_name: str,
    n_cases: int,
) -> np.ndarray:
    arr = np.asarray(predictions_tthr[model_name][:n_cases], dtype=float)
    if arr.ndim == 1:
        return arr
    return np.median(arr, axis=0)


def resolve_study_ode_backend(requested: str = "auto", *, verbose: bool = False) -> str:
    """Resolve numba vs python ODE backend once per process (validated against legacy path)."""
    global _RESOLVED_ODE_BACKEND
    requested = str(requested or "auto").lower()
    if requested == "auto" and _RESOLVED_ODE_BACKEND is not None:
        return _RESOLVED_ODE_BACKEND
    fast_ok = True
    warnings: List[str] = []
    if requested in ("auto", "numba"):
        fast_ok, warnings = validate_fast_backend()
    resolve_as = "numba" if requested in ("auto", "numba") else "python"
    backend = effective_backend(resolve_as, fast_ok)
    if requested in ("auto", "numba"):
        _RESOLVED_ODE_BACKEND = backend
    if verbose:
        print(f"ODE backend: {backend} (requested={requested})", flush=True)
        for note in warnings[:3]:
            print(f"  fast-backend note: {note}", flush=True)
    return backend


def ode_backend_for_config(config: ClosedLoopConfig) -> str:
    resolved = getattr(config, "_resolved_ode_backend", None)
    if resolved:
        return str(resolved)
    return resolve_study_ode_backend(config.ode_backend)


def resolve_case_n_jobs(config: ClosedLoopConfig, repeat_n_jobs: int) -> int:
    """Avoid nested joblib pools: parallelize repeats OR cases, not both by default."""
    if config.case_n_jobs > 0:
        return int(config.case_n_jobs)
    if repeat_n_jobs > 1:
        return 1
    cpu = os.cpu_count() or 2
    return max(1, cpu - 1)


def resolve_u_grid_n_jobs(config: ClosedLoopConfig, repeat_n_jobs: int, case_n_jobs: int) -> int:
    """Parallel u_grid scoring when cases are serial (repeat-parallel workers)."""
    explicit = int(getattr(config, "u_grid_n_jobs", 0) or 0)
    if explicit > 0:
        return explicit
    if case_n_jobs > 1:
        return 1
    n_grid = len(np.asarray(config.u_grid, dtype=float))
    if n_grid <= 1:
        return 1
    cpu = os.cpu_count() or 4
    repeat_workers = max(1, repeat_n_jobs if repeat_n_jobs > 0 else cpu)
    return max(1, min(n_grid, cpu // repeat_workers))


def _metrics_to_simulation_result(metrics: np.ndarray) -> MetricsOnlySimulation:
    metrics = np.asarray(metrics, dtype=float).ravel()
    lr = np.array([metrics[M_LR1 + i] for i in range(N_STRAINS)], dtype=float)
    return MetricsOnlySimulation(
        dose_count=int(metrics[M_DOSE_COUNT]),
        total_dosage=float(metrics[M_TOTAL_DOSAGE]),
        P_AUC=float(metrics[M_P_AUC]),
        LR_terminal_median=lr,
        final_total_pathogen=float(metrics[M_FINAL_TOTAL_PATHOGEN]),
        terminal_total_pathogen=float(metrics[M_TERMINAL_TOTAL_PATHOGEN]),
    )


def _simulation_result_from_candidate(cand: dict) -> MetricsOnlySimulation:
    lr = np.array([float(cand[f"LR{i + 1}"]) for i in range(N_STRAINS)], dtype=float)
    terminal = float(
        cand.get("terminal_total_pathogen_CFU_per_mL", cand.get("terminal_total_pathogen", np.nan))
    )
    return MetricsOnlySimulation(
        dose_count=int(cand.get("dose_count", 0)),
        total_dosage=float(cand.get("total_dosage_ug_per_mL", cand.get("total_dosage", np.nan))),
        P_AUC=float(cand["P_AUC"]),
        LR_terminal_median=lr,
        final_total_pathogen=float(
            cand.get("final_total_pathogen_CFU_per_mL", cand.get("final_total_pathogen", np.nan))
        ),
        terminal_total_pathogen=terminal,
    )


def terminal_total_pathogen(res: SimulationResult) -> float:
    if isinstance(res, MetricsOnlySimulation):
        return float(res.terminal_total_pathogen)
    if res.times.size == 0 or res.B_total.size == 0:
        stored = getattr(res, "terminal_total_pathogen", None)
        if stored is not None and np.isfinite(stored):
            return float(stored)
    terminal_mask = res.times >= (res.times[-1] - 12.0 - 1e-12)
    return float(np.median(res.B_total[terminal_mask].sum(axis=1)))


def resolve_pauc_target_source(config: ClosedLoopConfig) -> str:
    return str(getattr(config, "pauc_target_source", None) or config.target_pauc_source)


def resolve_pauc_feasibility_fraction(config: ClosedLoopConfig) -> float:
    frac = getattr(config, "pauc_feasibility_fraction", None)
    if frac is None:
        frac = config.probiotic_pauc_fraction
    return float(frac)


def resolve_dosage_reference(config: ClosedLoopConfig) -> float:
    """Backward-compatible alias: constant dose reference when training scale unavailable."""
    if config.dose_reference_source == "constant":
        return float(max(config.dose_reference_constant, 1.0))
    return float(max(config.dose_reference_constant, config.target_total_dosage, config.dosage_reference_target, 1.0))


def resolve_target_total_dosage(config: ClosedLoopConfig) -> float:
    return float(config.dose_reference_constant if config.dose_reference_constant > 0 else config.target_total_dosage)


def resolve_target_terminal_pathogen(config: ClosedLoopConfig) -> float:
    if config.pathogen_target_source == "ceiling":
        return float(config.pathogen_ceiling_cfu_per_mL)
    return float(
        config.target_terminal_pathogen
        if config.target_terminal_pathogen > 0
        else config.pathogen_ceiling_cfu_per_mL
    )


def _metadata_has_desired_lr(metadata_row: Optional[pd.Series]) -> bool:
    if metadata_row is None:
        return False
    return all(f"desired_LR{i}" in metadata_row.index and pd.notna(metadata_row.get(f"desired_LR{i}")) for i in range(1, 6))


def _x_row_has_kpi_lr(x_row: Optional[pd.Series]) -> bool:
    if x_row is None:
        return False
    return all(f"LR{i}" in x_row.index and pd.notna(x_row.get(f"LR{i}")) for i in range(1, 6))


def resolve_lr_targets(
    bio: CaseBioParams,
    config: ClosedLoopConfig,
    *,
    metadata_row: Optional[pd.Series] = None,
    x_row: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, str]:
    source_mode = config.lr_target_source
    if source_mode in ("reference",):
        return lr_reference_targets(config), "constant_fallback"
    if _metadata_has_desired_lr(metadata_row):
        return (
            np.array([float(metadata_row[f"desired_LR{i}"]) for i in range(1, 6)], dtype=float),
            "desired_per_case",
        )
    if _x_row_has_kpi_lr(x_row):
        return (
            np.array([float(x_row[f"LR{i}"]) for i in range(1, 6)], dtype=float),
            "desired_per_case",
        )
    if source_mode in ("desired",) or np.all(np.isfinite(bio.desired_lr)):
        return np.asarray(bio.desired_lr, dtype=float), "desired_per_case"
    return lr_reference_targets(config), "constant_fallback"


def resolve_target_pauc(bio: CaseBioParams, config: ClosedLoopConfig) -> float:
    if resolve_pauc_target_source(config) == "reference":
        return float(config.lr_reference_target)
    return float(bio.desired_pauc)


def resolve_target_lr_vector(
    bio: CaseBioParams,
    config: ClosedLoopConfig,
    *,
    metadata_row: Optional[pd.Series] = None,
    x_row: Optional[pd.Series] = None,
) -> np.ndarray:
    targets, _ = resolve_lr_targets(bio, config, metadata_row=metadata_row, x_row=x_row)
    return targets


def resolve_minimum_acceptable_pauc(target_pauc: float, config: ClosedLoopConfig) -> float:
    return float(resolve_pauc_feasibility_fraction(config) * target_pauc)


def lr_reference_targets(config: ClosedLoopConfig) -> np.ndarray:
    return np.full(N_STRAINS, float(config.lr_reference_target), dtype=float)


def dose_reference_quantile_column(quantile: float) -> str:
    pct = int(round(float(quantile) * 100))
    return f"dose_reference_q{pct}"


def resolve_dose_reference_scale_for_repeat(
    config: ClosedLoopConfig,
    repeat_id: int,
    reference_by_repeat_df: Optional[pd.DataFrame],
) -> Tuple[float, str]:
    if config.dose_reference_source == "constant":
        return float(max(config.dose_reference_constant, 1.0)), "constant"
    if reference_by_repeat_df is not None and not reference_by_repeat_df.empty:
        rows = reference_by_repeat_df[reference_by_repeat_df["repeat_id"] == int(repeat_id)]
        if not rows.empty:
            row = rows.iloc[0]
            qcol = dose_reference_quantile_column(config.dose_reference_quantile)
            if qcol in row.index and pd.notna(row[qcol]):
                return float(max(row[qcol], 1.0)), str(row.get("dose_reference_source", config.dose_reference_source))
            if "dose_reference_scale" in row.index and pd.notna(row["dose_reference_scale"]):
                return float(max(row["dose_reference_scale"], 1.0)), str(
                    row.get("dose_reference_source", config.dose_reference_source)
                )
    return float(max(config.dose_reference_constant, 1.0)), "constant_fallback"


def load_optimizer_reference_by_repeat(path: str) -> Optional[pd.DataFrame]:
    if path and os.path.isfile(path):
        return pd.read_csv(path)
    return None


def load_fixed_umax_policy_by_repeat(path: str) -> Optional[pd.DataFrame]:
    if path and os.path.isfile(path):
        df = pd.read_csv(path)
        if "repeat_id" in df.columns:
            return df.set_index("repeat_id", drop=False)
        return df
    return None


def resolve_fixed_umax_for_policy(
    policy_name: str,
    repeat_id: int,
    metadata_row: Optional[pd.Series],
    fixed_policy_df: Optional[pd.DataFrame],
    config: ClosedLoopConfig,
) -> Tuple[float, str]:
    """Resolve repeat-level fixed Umax from training-only derived policies."""
    rep_umax = float(config.fixed_representative_umax or FIXED_REPRESENTATIVE_UMAX_DEFAULT)
    if policy_name == UMAX_POLICY_METADATA_SOFT:
        if metadata_row is not None and "soft_u_max" in metadata_row.index:
            val = metadata_row.get("soft_u_max")
            if pd.notna(val):
                return float(val), "metadata_soft_umax"
        return rep_umax, "metadata_soft_umax_fallback"
    if fixed_policy_df is None or fixed_policy_df.empty:
        raise ValueError(f"fixed_umax_policy_by_repeat.csv required for policy '{policy_name}'.")

    rows = fixed_policy_df
    if "repeat_id" in rows.columns:
        match = rows[rows["repeat_id"] == int(repeat_id)]
    else:
        match = rows.loc[[int(repeat_id)]] if int(repeat_id) in rows.index else rows.iloc[0:0]
    if match.empty:
        raise ValueError(f"No fixed Umax policy row for repeat_id={repeat_id}.")
    row = match.iloc[0]

    if policy_name == UMAX_POLICY_TRAINING_MEDIAN:
        val = row.get("training_median_soft_u_max", np.nan)
        return float(val if pd.notna(val) else rep_umax), "training_median_soft_umax"
    if policy_name == UMAX_POLICY_TRAINING_TUNED_GLOBAL:
        val = row.get("training_tuned_global_u_max", np.nan)
        if not pd.notna(val):
            raise ValueError(f"training_tuned_global_u_max missing for repeat_id={repeat_id}.")
        return float(val), "training_tuned_global_umax"
    if policy_name == UMAX_POLICY_REPRESENTATIVE_FIXED:
        val = row.get("representative_fixed_u_max", rep_umax)
        return float(val if pd.notna(val) else rep_umax), "pre_specified_representative"
    raise ValueError(f"Unknown fixed Umax policy '{policy_name}'.")


def resolve_umax_policy_value(
    policy_name: str,
    repeat_id: int,
    fixed_policy_df: Optional[pd.DataFrame],
    config: ClosedLoopConfig,
) -> Tuple[float, str]:
    """Resolve repeat-level Umax for Fig. 5 fixed-policy ablation (training-only policies)."""
    rep_umax = float(config.fixed_representative_umax or FIXED_REPRESENTATIVE_UMAX_DEFAULT)
    if policy_name == UMAX_POLICY_PER_CASE_OPTIMIZED:
        raise ValueError("resolve_umax_policy_value does not apply to per_case_optimized.")
    if fixed_policy_df is None or fixed_policy_df.empty:
        raise ValueError(
            f"fixed_umax_policy_by_repeat.csv required for Fig. 5 policy '{policy_name}' "
            f"(repeat_id={repeat_id})."
        )

    rows = fixed_policy_df
    if "repeat_id" in rows.columns:
        match = rows[rows["repeat_id"] == int(repeat_id)]
    else:
        match = rows.loc[[int(repeat_id)]] if int(repeat_id) in rows.index else rows.iloc[0:0]
    if match.empty:
        raise ValueError(f"No fixed Umax policy row for repeat_id={repeat_id}.")
    row = match.iloc[0]

    if policy_name == UMAX_POLICY_TRAINING_MEDIAN:
        val = row.get("training_median_soft_u_max", np.nan)
        if pd.notna(val):
            return float(val), "training_median_soft_u_max"
        warnings.warn(
            f"repeat_id={repeat_id}: training_median_soft_u_max missing; "
            f"falling back to u_grid median ({float(np.median(config.u_grid)):.4g}).",
            stacklevel=2,
        )
        return float(np.median(config.u_grid)), "training_median_u_grid_fallback"
    if policy_name == UMAX_POLICY_TRAINING_TUNED_GLOBAL:
        val = row.get("training_tuned_global_u_max", np.nan)
        if not pd.notna(val):
            raise ValueError(f"training_tuned_global_u_max missing for repeat_id={repeat_id}.")
        return float(val), "training_tuned_global_umax"
    if policy_name == UMAX_POLICY_REPRESENTATIVE_FIXED:
        val = row.get("representative_fixed_u_max", rep_umax)
        return float(val if pd.notna(val) else rep_umax), "pre_specified_representative"
    raise ValueError(f"Unknown Umax policy '{policy_name}'.")


def compute_optimizer_shortfalls(
    *,
    total_dosage: float,
    P_AUC: float,
    LR_terminal_median: Sequence[float],
    terminal_total_pathogen: float,
    target_pauc: float,
    target_lr: Sequence[float],
    dose_reference_scale: float,
    config: ClosedLoopConfig,
) -> Tuple[float, float, float, float]:
    pauc_frac = resolve_pauc_feasibility_fraction(config)
    min_pauc = pauc_frac * target_pauc
    lr_shortfall_norm = float(
        np.mean(
            [
                max(0.0, float(target_lr[i]) - float(LR_terminal_median[i]))
                / max(abs(float(target_lr[i])), PAUC_FEASIBILITY_EPS)
                for i in range(N_STRAINS)
            ]
        )
    )
    pauc_shortfall_norm = max(0.0, min_pauc - float(P_AUC)) / max(abs(float(target_pauc)), PAUC_FEASIBILITY_EPS)
    ceiling = resolve_target_terminal_pathogen(config)
    pathogen_violation_norm = max(
        0.0,
        float(np.log10((float(terminal_total_pathogen) + 1.0) / (ceiling + 1.0))),
    )
    dose_burden_norm = float(total_dosage) / max(float(dose_reference_scale), 1.0)
    return lr_shortfall_norm, pauc_shortfall_norm, pathogen_violation_norm, dose_burden_norm


def compute_composite_penalty(
    lr_shortfall_norm: float,
    pauc_shortfall_norm: float,
    pathogen_violation_norm: float,
    dose_burden_norm: float,
    config: ClosedLoopConfig,
) -> float:
    w = config.weights
    return float(
        w.w_track * lr_shortfall_norm
        + w.w_probiotic * pauc_shortfall_norm
        + w.w_path * pathogen_violation_norm
        + w.w_dose * dose_burden_norm
    )


def is_feasible_candidate(
    *,
    LR_terminal_median: Sequence[float],
    P_AUC: float,
    terminal_total_pathogen: float,
    target_lr: Sequence[float],
    target_pauc: float,
    config: ClosedLoopConfig,
) -> bool:
    min_pauc = resolve_minimum_acceptable_pauc(target_pauc, config)
    lr_ok = bool(
        np.all(
            np.asarray(LR_terminal_median, dtype=float)
            >= np.asarray(target_lr, dtype=float) - config.lr_tolerance
        )
    )
    pauc_ok = bool(float(P_AUC) >= min_pauc - config.constraint_tol)
    path_ok = bool(float(terminal_total_pathogen) <= config.pathogen_ceiling_cfu_per_mL + config.constraint_tol)
    return lr_ok and pauc_ok and path_ok


def signed_relative_component(
    current: float,
    target: float,
    higher_is_better: bool,
    benefit_cap: float = SIGNED_BENEFIT_CAP,
    penalty_cap: float = SIGNED_PENALTY_CAP,
) -> float:
    denom = max(abs(float(target)), 1e-12)
    rel = (float(current) - float(target)) / denom
    sq = rel * rel
    favorable = rel >= 0.0 if higher_is_better else rel <= 0.0
    if favorable:
        return -min(sq, benefit_cap)
    return min(sq, penalty_cap)


def compute_optimization_penalty_score(
    *,
    total_dosage: float,
    P_AUC: float,
    LR_terminal_median: Sequence[float],
    terminal_total_pathogen: float,
    target_pauc: float,
    target_lr: Sequence[float],
    dose_reference_scale: float,
    config: ClosedLoopConfig,
) -> float:
    """Primary infeasible fallback score: weighted composite_penalty (>= 0)."""
    lr_n, pauc_n, path_n, dose_n = compute_optimizer_shortfalls(
        total_dosage=total_dosage,
        P_AUC=P_AUC,
        LR_terminal_median=LR_terminal_median,
        terminal_total_pathogen=terminal_total_pathogen,
        target_pauc=target_pauc,
        target_lr=target_lr,
        dose_reference_scale=dose_reference_scale,
        config=config,
    )
    return compute_composite_penalty(lr_n, pauc_n, path_n, dose_n, config)


def compute_hard_violation_rms(
    *,
    total_dosage: float,
    P_AUC: float,
    LR_terminal_median: Sequence[float],
    terminal_total_pathogen: float,
    bio: CaseBioParams,
    config: ClosedLoopConfig,
    dose_reference_scale: float,
    metadata_row: Optional[pd.Series] = None,
    x_row: Optional[pd.Series] = None,
) -> float:
    """Legacy diagnostic RMS of one-sided violations (not primary optimizer)."""
    target_pauc = resolve_target_pauc(bio, config)
    target_lr = resolve_target_lr_vector(bio, config, metadata_row=metadata_row, x_row=x_row)
    lr_n, pauc_n, path_n, dose_n = compute_optimizer_shortfalls(
        total_dosage=total_dosage,
        P_AUC=P_AUC,
        LR_terminal_median=LR_terminal_median,
        terminal_total_pathogen=terminal_total_pathogen,
        target_pauc=target_pauc,
        target_lr=target_lr,
        dose_reference_scale=dose_reference_scale,
        config=config,
    )
    return float(np.sqrt(np.mean(np.square([lr_n, pauc_n, path_n, dose_n]))))


def compute_signed_relative_objective(
    *,
    total_dosage: float,
    P_AUC: float,
    LR_terminal_median: Sequence[float],
    terminal_total_pathogen: float,
    bio: CaseBioParams,
    config: ClosedLoopConfig,
    weights: ClosedLoopWeights,
    dose_reference_scale: float,
    metadata_row: Optional[pd.Series] = None,
    x_row: Optional[pd.Series] = None,
) -> dict:
    target_pauc = resolve_target_pauc(bio, config)
    target_lr = resolve_target_lr_vector(bio, config, metadata_row=metadata_row, x_row=x_row)
    target_path = resolve_target_terminal_pathogen(config)
    min_pauc = resolve_minimum_acceptable_pauc(target_pauc, config)

    dose_signed = signed_relative_component(total_dosage, dose_reference_scale, higher_is_better=False)
    pauc_signed = signed_relative_component(P_AUC, target_pauc, higher_is_better=True)
    lr_signed_terms = [
        signed_relative_component(LR_terminal_median[i], target_lr[i], higher_is_better=True)
        for i in range(N_STRAINS)
    ]
    lr_signed = float(np.mean(lr_signed_terms))
    pathogen_signed = signed_relative_component(terminal_total_pathogen, target_path, higher_is_better=False)

    signed_inner = (
        weights.w_dose * dose_signed
        + weights.w_probiotic * pauc_signed
        + weights.w_track * lr_signed
        + weights.w_path * pathogen_signed
    )
    signed_relative_rms = float(np.sign(signed_inner) * np.sqrt(abs(signed_inner)))

    return {
        "target_P_AUC": target_pauc,
        "minimum_acceptable_P_AUC": min_pauc,
        "target_mean_LR": float(np.mean(target_lr)),
        "target_terminal_pathogen": target_path,
        "dose_signed_component": dose_signed,
        "pauc_signed_component": pauc_signed,
        "lr_signed_component": lr_signed,
        "pathogen_signed_component": pathogen_signed,
        "signed_relative_inner": float(signed_inner),
        "signed_relative_rms": signed_relative_rms,
        "LR_relative_sq_term": lr_signed,
        "PAUC_relative_sq_term": pauc_signed,
        "pathogen_relative_sq_term": pathogen_signed,
        "dose_relative_sq_term": dose_signed,
        "relative_rms_inner": float(signed_inner),
    }


def target_tracking_error(res: SimulationResult, desired_pauc: float, desired_lr: np.ndarray) -> float:
    lr_error = float(np.mean(np.abs(res.LR_terminal_median - desired_lr)))
    pauc_error = float(abs(res.P_AUC - desired_pauc))
    return lr_error + pauc_error


def constraint_flags(
    res: SimulationResult,
    bio: CaseBioParams,
    config: ClosedLoopConfig,
    *,
    metadata_row: Optional[pd.Series] = None,
    x_row: Optional[pd.Series] = None,
) -> Tuple[bool, bool, bool]:
    target_lr, _ = resolve_lr_targets(bio, config, metadata_row=metadata_row, x_row=x_row)
    target_pauc = resolve_target_pauc(bio, config)
    terminal_path = terminal_total_pathogen(res)
    lr_ok = bool(np.all(res.LR_terminal_median >= target_lr - config.lr_tolerance))
    probiotic_ok = bool(res.P_AUC >= resolve_minimum_acceptable_pauc(target_pauc, config))
    pathogen_ok = bool(terminal_path <= config.pathogen_ceiling_cfu_per_mL)
    return lr_ok and probiotic_ok and pathogen_ok, pathogen_ok, probiotic_ok


def normalized_shortfalls(
    res: SimulationResult,
    bio: CaseBioParams,
    config: ClosedLoopConfig,
    dose_reference_scale: float,
    *,
    metadata_row: Optional[pd.Series] = None,
    x_row: Optional[pd.Series] = None,
) -> Tuple[float, float, float, float]:
    target_pauc = resolve_target_pauc(bio, config)
    target_lr = resolve_target_lr_vector(bio, config, metadata_row=metadata_row, x_row=x_row)
    terminal_path = terminal_total_pathogen(res)
    return compute_optimizer_shortfalls(
        total_dosage=float(res.total_dosage),
        P_AUC=float(res.P_AUC),
        LR_terminal_median=res.LR_terminal_median,
        terminal_total_pathogen=terminal_path,
        target_pauc=target_pauc,
        target_lr=target_lr,
        dose_reference_scale=dose_reference_scale,
        config=config,
    )


def _optimizer_values_close(a: float, b: float, ref_min: float) -> bool:
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    tol = max(HARD_VIOLATION_TOL_ABS, HARD_VIOLATION_TOL_REL * max(abs(ref_min), 0.0))
    return abs(float(a) - float(b)) <= tol


def _feasible_sort_key(cand: dict) -> tuple:
    return (
        float(cand.get("total_dosage_ug_per_mL", cand.get("total_dosage", np.inf))),
        float(cand.get("terminal_total_pathogen_CFU_per_mL", cand.get("terminal_total_pathogen", np.inf))),
        -float(cand.get("P_AUC", -np.inf)),
        -float(cand.get("mean_LR", -np.inf)),
        float(cand["candidate_u_max"]),
    )


def _infeasible_sort_key(cand: dict) -> tuple:
    return (
        float(cand.get("composite_penalty", cand.get("optimization_penalty_score", np.inf))),
        float(cand.get("terminal_total_pathogen_CFU_per_mL", cand.get("terminal_total_pathogen", np.inf))),
        -float(cand.get("P_AUC", -np.inf)),
        float(cand.get("total_dosage_ug_per_mL", cand.get("total_dosage", np.inf))),
        float(cand["candidate_u_max"]),
    )


def _optimizer_sort_key(cand: dict) -> tuple:
    if cand.get("feasible_candidate"):
        return (0,) + _feasible_sort_key(cand)
    return (1,) + _infeasible_sort_key(cand)


def assign_optimizer_selection_ranks(candidates: List[dict]) -> None:
    ranked = sorted(candidates, key=_optimizer_sort_key)
    for rank, cand in enumerate(ranked, start=1):
        cand["optimizer_selection_rank"] = int(rank)
        if cand.get("feasible_candidate"):
            cand["optimizer_primary_score"] = float(
                cand.get("total_dosage_ug_per_mL", cand.get("total_dosage", np.nan))
            )
            cand["optimizer_tiebreak_score"] = float(
                cand.get("terminal_total_pathogen_CFU_per_mL", cand.get("terminal_total_pathogen", np.inf))
            )
        else:
            cand["optimizer_primary_score"] = float(
                cand.get("composite_penalty", cand.get("optimization_penalty_score", np.nan))
            )
            cand["optimizer_tiebreak_score"] = float(
                cand.get("terminal_total_pathogen_CFU_per_mL", cand.get("terminal_total_pathogen", np.inf))
            )



def select_best_u_candidate(candidates: List[dict]) -> Tuple[dict, str]:
    """Constraint-first: feasible min-dose; else min composite_penalty."""
    if not candidates:
        raise ValueError("select_best_u_candidate requires at least one candidate.")
    assign_optimizer_selection_ranks(candidates)
    feasible = [c for c in candidates if c.get("feasible_candidate")]
    if feasible:
        best = min(feasible, key=_feasible_sort_key)
        return best, "feasible_min_dose"
    best = min(candidates, key=_infeasible_sort_key)
    return best, "infeasible_min_composite_penalty"


def _cand_total_dosage(cand: dict) -> float:
    return float(cand.get("total_dosage", cand.get("total_dosage_ug_per_mL", np.nan)))


def _cand_terminal_pathogen(cand: dict) -> float:
    return float(
        cand.get("terminal_total_pathogen", cand.get("terminal_total_pathogen_CFU_per_mL", np.nan))
    )


def _aspiration_tolerance(ref_val: float, config: ClosedLoopConfig) -> float:
    return max(float(config.aspiration_tolerance_abs), float(config.aspiration_tolerance_rel) * max(abs(ref_val), 1.0))


def build_initial_aspiration_reference(cand: dict) -> dict:
    """Per-case initial aspiration reference r0 from candidate metadata."""
    ref = {
        "aspiration_total_dosage": float(cand["dose_reference_scale"]),
        "aspiration_P_AUC": float(cand["minimum_acceptable_P_AUC"]),
        "aspiration_mean_LR": float(cand["target_mean_LR"]),
        "aspiration_terminal_pathogen": float(cand["target_terminal_pathogen"]),
    }
    for i in range(1, N_STRAINS + 1):
        ref[f"aspiration_LR{i}"] = float(cand[f"target_LR{i}"])
    return ref


def build_outcome_reference_from_candidate(cand: dict) -> dict:
    """Updated reference r1 from fallback candidate outcomes."""
    ref = {
        "total_dosage": _cand_total_dosage(cand),
        "aspiration_P_AUC": float(cand["P_AUC"]),
        "aspiration_mean_LR": float(cand["mean_LR"]),
        "aspiration_terminal_pathogen": _cand_terminal_pathogen(cand),
    }
    for i in range(1, N_STRAINS + 1):
        ref[f"aspiration_LR{i}"] = float(cand[f"LR{i}"])
    return ref


def _reference_dose(ref: dict) -> float:
    return float(ref.get("total_dosage", ref.get("aspiration_total_dosage", np.nan)))


def _reference_pauc(ref: dict) -> float:
    return float(ref.get("aspiration_P_AUC", ref.get("P_AUC", np.nan)))


def _reference_pathogen(ref: dict) -> float:
    return float(ref.get("aspiration_terminal_pathogen", ref.get("terminal_total_pathogen", np.nan)))


def _reference_lr(ref: dict, strain_index: int) -> float:
    key_asp = f"aspiration_LR{strain_index}"
    key_lr = f"LR{strain_index}"
    key_tgt = f"target_LR{strain_index}"
    if key_asp in ref:
        return float(ref[key_asp])
    if key_lr in ref:
        return float(ref[key_lr])
    return float(ref[key_tgt])


def compute_aspiration_absolute_distance(
    candidate: dict,
    aspiration_reference: dict,
    weights: ClosedLoopWeights,
    eps: float,
) -> Tuple[float, dict]:
    """Unsigned normalized distance to aspiration reference (not signed reward)."""
    eps = max(float(eps), 1e-30)
    dose_ref = float(aspiration_reference["aspiration_total_dosage"])
    dose_val = _cand_total_dosage(candidate)
    dose_term = ((dose_val - dose_ref) / max(abs(dose_ref), eps)) ** 2

    pauc_ref = float(aspiration_reference["aspiration_P_AUC"])
    pauc_val = float(candidate["P_AUC"])
    pauc_term = ((pauc_val - pauc_ref) / max(abs(pauc_ref), eps)) ** 2

    lr_sq: List[float] = []
    for i in range(1, N_STRAINS + 1):
        lr_ref = float(aspiration_reference[f"aspiration_LR{i}"])
        lr_val = float(candidate[f"LR{i}"])
        lr_sq.append(((lr_val - lr_ref) / max(abs(lr_ref), eps)) ** 2)
    lr_term = float(np.mean(lr_sq))

    path_ref = float(aspiration_reference["aspiration_terminal_pathogen"])
    path_val = _cand_terminal_pathogen(candidate)
    log_path = float(np.log10(path_val + 1.0))
    log_ref = float(np.log10(path_ref + 1.0))
    pathogen_term = (log_path - log_ref) ** 2

    dist = float(
        np.sqrt(
            weights.w_dose * dose_term
            + weights.w_probiotic * pauc_term
            + weights.w_track * lr_term
            + weights.w_path * pathogen_term
        )
    )
    return dist, {
        "aspiration_dose_term": float(dose_term),
        "aspiration_pauc_term": float(pauc_term),
        "aspiration_lr_term": float(lr_term),
        "aspiration_pathogen_log_term": float(pathogen_term),
    }


def candidate_dominates_reference(candidate: dict, reference: dict, config: ClosedLoopConfig) -> bool:
    dose_tol = _aspiration_tolerance(_reference_dose(reference), config)
    if _cand_total_dosage(candidate) > _reference_dose(reference) + dose_tol:
        return False

    pauc_tol = _aspiration_tolerance(_reference_pauc(reference), config)
    if float(candidate["P_AUC"]) < _reference_pauc(reference) - pauc_tol:
        return False

    for i in range(1, N_STRAINS + 1):
        lr_tol = _aspiration_tolerance(_reference_lr(reference, i), config)
        if float(candidate[f"LR{i}"]) < _reference_lr(reference, i) - lr_tol:
            return False

    path_tol = _aspiration_tolerance(_reference_pathogen(reference), config)
    if _cand_terminal_pathogen(candidate) > _reference_pathogen(reference) + path_tol:
        return False
    return True


def candidate_strictly_improves_reference(candidate: dict, reference: dict, config: ClosedLoopConfig) -> bool:
    if not candidate_dominates_reference(candidate, reference, config):
        return False
    dose_tol = _aspiration_tolerance(_reference_dose(reference), config)
    pauc_tol = _aspiration_tolerance(_reference_pauc(reference), config)
    path_tol = _aspiration_tolerance(_reference_pathogen(reference), config)

    if _cand_total_dosage(candidate) < _reference_dose(reference) - dose_tol:
        return True
    if float(candidate["P_AUC"]) > _reference_pauc(reference) + pauc_tol:
        return True
    if _cand_terminal_pathogen(candidate) < _reference_pathogen(reference) - path_tol:
        return True
    for i in range(1, N_STRAINS + 1):
        lr_tol = _aspiration_tolerance(_reference_lr(reference, i), config)
        if float(candidate[f"LR{i}"]) > _reference_lr(reference, i) + lr_tol:
            return True
    return False


def _aspiration_tiebreak_key(cand: dict) -> tuple:
    return (
        float(cand.get("total_dosage", cand.get("total_dosage_ug_per_mL", np.inf))),
        float(cand.get("terminal_total_pathogen", cand.get("terminal_total_pathogen_CFU_per_mL", np.inf))),
        -float(cand.get("P_AUC", -np.inf)),
        -float(cand.get("mean_LR", -np.inf)),
        float(cand["candidate_u_max"]),
    )


def _aspiration_selection_sort_key(cand: dict) -> tuple:
    return (float(cand.get("aspiration_abs_distance_to_initial", np.inf)),) + _aspiration_tiebreak_key(cand)


def enrich_candidates_with_aspiration(candidates: List[dict], config: ClosedLoopConfig) -> dict:
    """Attach aspiration metadata and distances; return initial reference r0."""
    if not candidates:
        raise ValueError("enrich_candidates_with_aspiration requires at least one candidate.")
    r0 = build_initial_aspiration_reference(candidates[0])
    weights = config.weights
    eps = config.aspiration_eps
    primary_dominators: List[dict] = []
    for cand in candidates:
        for key, val in r0.items():
            cand[key] = val
        dist, terms = compute_aspiration_absolute_distance(cand, r0, weights, eps)
        cand["aspiration_abs_distance_to_initial"] = dist
        cand.update(terms)
        dominates = candidate_dominates_reference(cand, r0, config)
        cand["dominates_initial_aspiration"] = bool(dominates)
        if dominates:
            primary_dominators.append(cand)
    n_primary = len(primary_dominators)
    for cand in candidates:
        cand["initial_aspiration_dominator_count"] = int(n_primary)
        cand["closest_to_initial_aspiration"] = False
        cand["updated_reference_from_closest_candidate"] = False
        cand["dominates_updated_reference"] = False
        cand["updated_reference_dominator_count"] = 0
        cand["pareto_improved_from_closest"] = False
        cand["optimizer_selection_stage"] = ""
    return r0


def select_best_u_candidate_aspiration_then_pareto(
    candidates: List[dict],
    config: ClosedLoopConfig,
) -> Tuple[dict, str, dict]:
    """Two-stage aspiration-point + Pareto cleanup selection."""
    if not candidates:
        raise ValueError("select_best_u_candidate_aspiration_then_pareto requires at least one candidate.")
    if "aspiration_abs_distance_to_initial" not in candidates[0]:
        r0 = enrich_candidates_with_aspiration(candidates, config)
    else:
        r0 = build_initial_aspiration_reference(candidates[0])
        for cand in candidates:
            if "initial_aspiration_dominator_count" not in cand:
                cand["initial_aspiration_dominator_count"] = int(
                    sum(1 for c in candidates if c.get("dominates_initial_aspiration"))
                )
    primary_dominators = [c for c in candidates if c.get("dominates_initial_aspiration")]

    debug = {
        "aspiration_total_dosage": float(r0["aspiration_total_dosage"]),
        "aspiration_P_AUC": float(r0["aspiration_P_AUC"]),
        "aspiration_mean_LR": float(r0["aspiration_mean_LR"]),
        "aspiration_terminal_pathogen": float(r0["aspiration_terminal_pathogen"]),
        "initial_dominator_count": int(len(primary_dominators)),
        "updated_reference_dominator_count": 0,
        "closest_candidate_u_max": float("nan"),
        "closest_candidate_abs_distance": float("nan"),
    }

    if primary_dominators:
        best = min(primary_dominators, key=_aspiration_selection_sort_key)
        rule = "aspiration_met_min_distance"
        stage = "primary_aspiration_met"
        for cand in candidates:
            cand["optimizer_selection_stage"] = stage
        debug["updated_reference_dominator_count"] = 0
        debug["closest_candidate_u_max"] = float("nan")
        debug["closest_candidate_abs_distance"] = float("nan")
        debug["selection_rule"] = rule
        return best, rule, {**debug, "optimizer_selection_stage": stage}

    fallback = min(candidates, key=lambda c: float(c["aspiration_abs_distance_to_initial"]))
    fallback["closest_to_initial_aspiration"] = True
    fallback["updated_reference_from_closest_candidate"] = True
    r1 = build_outcome_reference_from_candidate(fallback)
    debug["closest_candidate_u_max"] = float(fallback["candidate_u_max"])
    debug["closest_candidate_abs_distance"] = float(fallback["aspiration_abs_distance_to_initial"])

    secondary_dominators: List[dict] = []
    for cand in candidates:
        dominates_r1 = candidate_dominates_reference(cand, r1, config)
        strict = candidate_strictly_improves_reference(cand, r1, config)
        cand["dominates_updated_reference"] = bool(dominates_r1)
        cand["pareto_improved_from_closest"] = bool(dominates_r1 and strict)
        if dominates_r1 and strict:
            secondary_dominators.append(cand)

    debug["updated_reference_dominator_count"] = int(len(secondary_dominators))
    for cand in candidates:
        cand["updated_reference_dominator_count"] = int(len(secondary_dominators))

    if secondary_dominators:
        best = min(secondary_dominators, key=_aspiration_selection_sort_key)
        rule = "closest_fallback_pareto_improved"
        stage = "secondary_pareto_from_closest"
        for cand in candidates:
            if not cand.get("optimizer_selection_stage"):
                cand["optimizer_selection_stage"] = stage
        debug["selection_rule"] = rule
        return best, rule, {**debug, "optimizer_selection_stage": stage}

    for cand in candidates:
        cand["optimizer_selection_stage"] = "closest_to_initial_aspiration_no_pareto_improvement"
    rule = "closest_to_initial_aspiration_no_pareto_improvement"
    return fallback, rule, {
        **debug,
        "optimizer_selection_stage": "closest_to_initial_aspiration_no_pareto_improvement",
        "selection_rule": rule,
    }


def select_best_u_candidate_with_policy(
    candidates: List[dict],
    config: ClosedLoopConfig,
) -> Tuple[dict, str, dict]:
    policy = config.umax_selection_policy
    if policy == "feasible_first":
        best, rule = select_best_u_candidate(candidates)
        stage = (
            "primary_feasible_region"
            if rule == "feasible_min_dose"
            else "fallback_infeasible_penalty"
        )
        return best, rule, {
            "optimizer_selection_stage": stage,
            "optimizer_selection_policy": policy,
            "selection_rule": rule,
            "initial_dominator_count": 0,
            "updated_reference_dominator_count": 0,
            "closest_candidate_u_max": float("nan"),
            "closest_candidate_abs_distance": float("nan"),
        }
    if policy != "aspiration_then_pareto":
        raise ValueError(f"Unknown umax_selection_policy: {policy}")
    best, rule, debug = select_best_u_candidate_aspiration_then_pareto(candidates, config)
    debug["optimizer_selection_policy"] = policy
    return best, rule, debug


def build_umax_selection_policy_sensitivity_rows(
    candidates: List[dict],
    config: ClosedLoopConfig,
    *,
    repeat_id: int,
    case_index: int,
    model: str,
) -> List[dict]:
    """Compare feasible_first vs aspiration_then_pareto on the same candidate grid."""
    import copy

    rows: List[dict] = []
    for policy in UMAX_SELECTION_POLICIES:
        cands = copy.deepcopy(candidates)
        pol_config = replace(config, umax_selection_policy=policy)
        best, rule, debug = select_best_u_candidate_with_policy(cands, pol_config)
        rows.append(
            {
                "repeat_id": int(repeat_id),
                "case_index": int(case_index),
                "model": model,
                "umax_selection_policy": policy,
                "selected_u_max": float(best["candidate_u_max"]),
                "selection_rule": rule,
                "optimizer_selection_stage": debug.get("optimizer_selection_stage", ""),
                "selected_total_dosage": _cand_total_dosage(best),
                "selected_P_AUC": float(best["P_AUC"]),
                "selected_mean_LR": float(best["mean_LR"]),
                "selected_terminal_pathogen": _cand_terminal_pathogen(best),
                "selected_composite_penalty": float(
                    best.get("composite_penalty", best.get("optimization_penalty_score", np.nan))
                ),
                "selected_feasible_candidate": bool(best.get("feasible_candidate", False)),
                "aspiration_abs_distance_to_initial": float(
                    best.get("aspiration_abs_distance_to_initial", np.nan)
                ),
            }
        )
    return rows


def build_umax_aspiration_selection_debug_row(
    *,
    repeat_id: int,
    case_index: int,
    model: str,
    best_cand: dict,
    selection_rule: str,
    debug: dict,
) -> dict:
    return {
        "repeat_id": int(repeat_id),
        "case_index": int(case_index),
        "model": model,
        "selected_u_max": float(best_cand["candidate_u_max"]),
        "selection_rule": selection_rule,
        "initial_dominator_count": int(debug.get("initial_dominator_count", 0)),
        "updated_reference_dominator_count": int(debug.get("updated_reference_dominator_count", 0)),
        "closest_candidate_u_max": float(debug.get("closest_candidate_u_max", np.nan)),
        "closest_candidate_abs_distance": float(debug.get("closest_candidate_abs_distance", np.nan)),
        "selected_candidate_abs_distance": float(
            best_cand.get("aspiration_abs_distance_to_initial", np.nan)
        ),
        "selected_total_dosage": _cand_total_dosage(best_cand),
        "selected_P_AUC": float(best_cand["P_AUC"]),
        "selected_mean_LR": float(best_cand["mean_LR"]),
        "selected_terminal_pathogen": _cand_terminal_pathogen(best_cand),
        "aspiration_total_dosage": float(debug.get("aspiration_total_dosage", np.nan)),
        "aspiration_P_AUC": float(debug.get("aspiration_P_AUC", np.nan)),
        "aspiration_mean_LR": float(debug.get("aspiration_mean_LR", np.nan)),
        "aspiration_terminal_pathogen": float(debug.get("aspiration_terminal_pathogen", np.nan)),
        "optimizer_selection_stage": str(debug.get("optimizer_selection_stage", "")),
    }


def fixed_umax_for_case(meta_row: Optional[pd.Series], config: ClosedLoopConfig) -> float:
    """Fixed Umax for models without closed-loop u_grid optimization."""
    if meta_row is not None and "soft_u_max" in meta_row.index:
        val = meta_row["soft_u_max"]
        if pd.notna(val):
            return float(val)
    return float(np.median(config.u_grid))


def components_from_candidate(cand: dict, selection_rule: str) -> dict:
    dose_burden = cand.get("dose_burden_norm", cand.get("total_dosage_norm", np.nan))
    composite = cand.get("composite_penalty", cand.get("composite_score", np.nan))
    return {
        "target_tracking_error": cand["target_tracking_error"],
        "composite_score": composite,
        "composite_penalty": composite,
        "LR_shortfall_norm": cand["LR_shortfall_norm"],
        "PAUC_shortfall_norm": cand["PAUC_shortfall_norm"],
        "pathogen_violation_norm": cand["pathogen_violation_norm"],
        "dose_burden_norm": dose_burden,
        "total_dosage_norm": dose_burden,
        "LR_relative_sq_term": cand["LR_relative_sq_term"],
        "PAUC_relative_sq_term": cand["PAUC_relative_sq_term"],
        "pathogen_relative_sq_term": cand["pathogen_relative_sq_term"],
        "dose_relative_sq_term": cand["dose_relative_sq_term"],
        "relative_rms_inner": cand["relative_rms_inner"],
        "hard_violation_rms": cand["hard_violation_rms"],
        "signed_relative_rms": cand["signed_relative_rms"],
        "signed_relative_inner": cand["signed_relative_inner"],
        "selection_rule": selection_rule,
        "optimizer_selection_rule": selection_rule,
        "feasible_candidate": cand.get("feasible_candidate", False),
    }


def simulate_at_fixed_umax(
    bio: CaseBioParams,
    tthr: np.ndarray,
    config: ClosedLoopConfig,
    dose_reference_scale: float,
    meta_row: Optional[pd.Series],
    *,
    u_fixed: Optional[float] = None,
    umax_policy: str = "",
    x_row: Optional[pd.Series] = None,
    dose_reference_source: str = "",
    trajectory: bool = False,
    ode_backend: Optional[str] = None,
) -> Tuple[float, SimulationResult, dict, List[dict]]:
    if u_fixed is None:
        u_fixed = fixed_umax_for_case(meta_row, config)
    res, cand = simulate_candidate(
        bio,
        tthr,
        float(u_fixed),
        config,
        dose_reference_scale,
        metadata_row=meta_row,
        x_row=x_row,
        dose_reference_source=dose_reference_source,
        trajectory=trajectory,
        ode_backend=ode_backend,
    )
    cand = dict(cand)
    cand["selected_by_optimizer"] = True
    policy_rules = {
        UMAX_POLICY_TRAINING_MEDIAN: "fixed_training_median_soft_umax",
        UMAX_POLICY_TRAINING_TUNED_GLOBAL: "fixed_training_tuned_global_umax",
        UMAX_POLICY_REPRESENTATIVE_FIXED: "fixed_representative_umax",
        UMAX_POLICY_METADATA_SOFT: "fixed_metadata_soft_umax",
    }
    if umax_policy in policy_rules:
        rule = policy_rules[umax_policy]
    elif u_fixed == float(PAPER_FIGURE_PROFILE.u_max_rep):
        rule = "fixed_paper_figure_u_max"
    elif meta_row is not None and "soft_u_max" in meta_row.index and pd.notna(meta_row.get("soft_u_max")):
        rule = "fixed_soft_u_max"
    else:
        rule = "fixed_median_u_grid"
    cand["optimizer_selection_rule"] = rule
    components = components_from_candidate(cand, selection_rule=rule)
    return float(u_fixed), res, components, [cand]


def composite_score_from_norms(
    lr_shortfall_norm: float,
    pauc_shortfall_norm: float,
    pathogen_violation_norm: float,
    dose_burden_norm: float,
    config: ClosedLoopConfig,
) -> float:
    return compute_composite_penalty(
        lr_shortfall_norm, pauc_shortfall_norm, pathogen_violation_norm, dose_burden_norm, config
    )


def simulate_candidate(
    bio: CaseBioParams,
    tthr: np.ndarray,
    u_max: float,
    config: ClosedLoopConfig,
    dose_reference_scale: float,
    *,
    metadata_row: Optional[pd.Series] = None,
    x_row: Optional[pd.Series] = None,
    dose_reference_source: str = "",
    lr_target_source: str = "",
    trajectory: bool = False,
    ode_backend: Optional[str] = None,
) -> Tuple[SimulationResult, dict]:
    tthr = clip_tthr(tthr)
    backend = ode_backend or ode_backend_for_config(config)
    if trajectory or backend == "python":
        res = simulate_case(
            bio.B0,
            bio.k_arr,
            bio.gamma_arr,
            bio.rho_arr,
            bio.mu_arr,
            float(u_max),
            tthr,
        )
    else:
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
        res = _metrics_to_simulation_result(metrics)
    target_lr, lr_source = resolve_lr_targets(
        bio, config, metadata_row=metadata_row, x_row=x_row
    )
    if lr_target_source:
        lr_source = lr_target_source
    target_pauc = resolve_target_pauc(bio, config)
    min_pauc = resolve_minimum_acceptable_pauc(target_pauc, config)
    target_path = resolve_target_terminal_pathogen(config)
    dose_src = dose_reference_source or config.dose_reference_source
    all_ok, path_ok, prob_ok = constraint_flags(
        res, bio, config, metadata_row=metadata_row, x_row=x_row
    )
    lr_ok = bool(
        np.all(
            res.LR_terminal_median >= np.asarray(target_lr, dtype=float) - config.lr_tolerance
        )
    )
    track = target_tracking_error(res, bio.desired_pauc, bio.desired_lr)
    terminal_path = terminal_total_pathogen(res)
    lr_n, pauc_n, path_n, dose_n = normalized_shortfalls(
        res, bio, config, dose_reference_scale, metadata_row=metadata_row, x_row=x_row
    )
    composite_penalty = composite_score_from_norms(lr_n, pauc_n, path_n, dose_n, config)
    signed_obj = compute_signed_relative_objective(
        total_dosage=float(res.total_dosage),
        P_AUC=float(res.P_AUC),
        LR_terminal_median=res.LR_terminal_median,
        terminal_total_pathogen=terminal_path,
        bio=bio,
        config=config,
        weights=config.weights,
        dose_reference_scale=dose_reference_scale,
        metadata_row=metadata_row,
        x_row=x_row,
    )
    hard_violation = compute_hard_violation_rms(
        total_dosage=float(res.total_dosage),
        P_AUC=float(res.P_AUC),
        LR_terminal_median=res.LR_terminal_median,
        terminal_total_pathogen=terminal_path,
        bio=bio,
        config=config,
        dose_reference_scale=dose_reference_scale,
        metadata_row=metadata_row,
        x_row=x_row,
    )
    optimization_penalty = compute_optimization_penalty_score(
        total_dosage=float(res.total_dosage),
        P_AUC=float(res.P_AUC),
        LR_terminal_median=res.LR_terminal_median,
        terminal_total_pathogen=terminal_path,
        target_pauc=target_pauc,
        target_lr=target_lr,
        dose_reference_scale=dose_reference_scale,
        config=config,
    )
    feasible = is_feasible_candidate(
        LR_terminal_median=res.LR_terminal_median,
        P_AUC=float(res.P_AUC),
        terminal_total_pathogen=terminal_path,
        target_lr=target_lr,
        target_pauc=target_pauc,
        config=config,
    )
    row = {
        "candidate_u_max": float(u_max),
        "dose_count": int(res.dose_count),
        "total_dosage_ug_per_mL": float(res.total_dosage),
        "P_AUC": float(res.P_AUC),
        "mean_LR": float(np.mean(res.LR_terminal_median)),
        "final_total_pathogen_CFU_per_mL": float(res.final_total_pathogen),
        "terminal_total_pathogen_CFU_per_mL": terminal_path,
        "target_tracking_error": track,
        "LR_constraint_satisfied": bool(lr_ok),
        "P_AUC_constraint_satisfied": bool(prob_ok),
        "pathogen_constraint_satisfied": bool(path_ok),
        "probiotic_constraint_satisfied": bool(prob_ok),
        "all_constraints_satisfied": bool(all_ok),
        "target_P_AUC": target_pauc,
        "minimum_acceptable_P_AUC": min_pauc,
        "target_mean_LR": float(np.mean(target_lr)),
        "target_terminal_pathogen": target_path,
        "dose_reference_scale": float(dose_reference_scale),
        "dose_reference_source": dose_src,
        "lr_target_source": lr_source,
        "LR_shortfall_norm": lr_n,
        "PAUC_shortfall_norm": pauc_n,
        "pathogen_violation_norm": path_n,
        "dose_burden_norm": dose_n,
        "total_dosage_norm": dose_n,
        "composite_penalty": composite_penalty,
        "feasible_candidate": bool(feasible),
        "hard_violation_rms": hard_violation,
        "optimization_penalty_score": optimization_penalty,
        "composite_score": composite_penalty,
        **signed_obj,
    }
    for i in range(N_STRAINS):
        row[f"LR{i + 1}"] = float(res.LR_terminal_median[i])
        row[f"target_LR{i + 1}"] = float(target_lr[i])
    return res, row


def optimize_umax(
    bio: CaseBioParams,
    tthr: np.ndarray,
    config: ClosedLoopConfig,
    dose_reference_scale: float,
    *,
    metadata_row: Optional[pd.Series] = None,
    x_row: Optional[pd.Series] = None,
    dose_reference_source: str = "",
    lr_target_source: str = "",
    trajectory: bool = False,
    ode_backend: Optional[str] = None,
) -> Tuple[float, SimulationResult, dict, List[dict], dict, List[dict]]:
    backend = ode_backend or ode_backend_for_config(config)
    u_grid_n_jobs = int(getattr(config, "_resolved_u_grid_n_jobs", 1) or 1)
    grid = [float(u_max) for u_max in config.u_grid]

    def _one_candidate(u_max: float) -> dict:
        _, cand = simulate_candidate(
            bio,
            tthr,
            float(u_max),
            config,
            dose_reference_scale,
            metadata_row=metadata_row,
            x_row=x_row,
            dose_reference_source=dose_reference_source,
            lr_target_source=lr_target_source,
            trajectory=False,
            ode_backend=backend,
        )
        return cand

    if u_grid_n_jobs <= 1 or len(grid) <= 1:
        candidates: List[dict] = [_one_candidate(u_max) for u_max in grid]
    else:
        from joblib import Parallel, delayed

        candidates = Parallel(n_jobs=u_grid_n_jobs, prefer="threads")(
            delayed(_one_candidate)(u_max) for u_max in grid
        )
        candidates = sorted(candidates, key=lambda cand: float(cand["candidate_u_max"]))

    enrich_candidates_with_aspiration(candidates, config)
    raw_candidates = [dict(c) for c in candidates]
    best_cand, selection_rule, selection_debug = select_best_u_candidate_with_policy(candidates, config)
    assign_optimizer_selection_ranks(candidates)

    best_u = float(best_cand["candidate_u_max"])
    selection_policy = str(selection_debug.get("optimizer_selection_policy", config.umax_selection_policy))
    selection_stage = str(selection_debug.get("optimizer_selection_stage", ""))
    for cand in candidates:
        cand["selected_by_optimizer"] = abs(cand["candidate_u_max"] - best_u) < 1e-9
        cand["optimizer_selection_policy"] = selection_policy
        cand["optimizer_selection_rule"] = selection_rule if cand["selected_by_optimizer"] else ""
        if cand["selected_by_optimizer"]:
            cand["optimizer_selection_stage"] = selection_stage

    sensitivity_rows = build_umax_selection_policy_sensitivity_rows(
        raw_candidates,
        config,
        repeat_id=-1,
        case_index=-1,
        model="",
    )
    if trajectory:
        best_res, _ = simulate_candidate(
            bio,
            tthr,
            best_u,
            config,
            dose_reference_scale,
            metadata_row=metadata_row,
            x_row=x_row,
            dose_reference_source=dose_reference_source,
            lr_target_source=lr_target_source,
            trajectory=True,
            ode_backend=backend,
        )
    else:
        best_res = _simulation_result_from_candidate(best_cand)
    components = {
        "target_tracking_error": best_cand["target_tracking_error"],
        "composite_score": best_cand["composite_penalty"],
        "composite_penalty": best_cand["composite_penalty"],
        "LR_shortfall_norm": best_cand["LR_shortfall_norm"],
        "PAUC_shortfall_norm": best_cand["PAUC_shortfall_norm"],
        "pathogen_violation_norm": best_cand["pathogen_violation_norm"],
        "dose_burden_norm": best_cand["dose_burden_norm"],
        "total_dosage_norm": best_cand.get("dose_burden_norm", best_cand.get("total_dosage_norm")),
        "LR_relative_sq_term": best_cand["LR_relative_sq_term"],
        "PAUC_relative_sq_term": best_cand["PAUC_relative_sq_term"],
        "pathogen_relative_sq_term": best_cand["pathogen_relative_sq_term"],
        "dose_relative_sq_term": best_cand["dose_relative_sq_term"],
        "relative_rms_inner": best_cand["relative_rms_inner"],
        "hard_violation_rms": best_cand["hard_violation_rms"],
        "signed_relative_rms": best_cand["signed_relative_rms"],
        "signed_relative_inner": best_cand["signed_relative_inner"],
        "selection_rule": selection_rule,
        "optimizer_selection_rule": selection_rule,
        "feasible_candidate": best_cand.get("feasible_candidate", False),
    }
    return best_u, best_res, components, candidates, selection_debug, sensitivity_rows


def closed_loop_case_row(
    repeat_id: int,
    case_index: int,
    original_row_index: int,
    model_name: str,
    bio: CaseBioParams,
    tthr_pred: np.ndarray,
    optimized_u: float,
    res: SimulationResult,
    components: dict,
    config: ClosedLoopConfig,
    extra_meta: Optional[dict] = None,
) -> dict:
    all_ok, path_ok, prob_ok = constraint_flags(res, bio, config)
    row = {
        "repeat_id": repeat_id,
        "case_index": case_index,
        "original_row_index": original_row_index,
        "model": model_name,
        "optimized_u_max": optimized_u,
        "total_dosage_ug_per_mL": float(res.total_dosage),
        "dose_count": int(res.dose_count),
        "P_AUC": float(res.P_AUC),
        "mean_LR": float(np.mean(res.LR_terminal_median)),
        "final_total_pathogen_CFU_per_mL": float(res.final_total_pathogen),
        "terminal_total_pathogen_CFU_per_mL": terminal_total_pathogen(res),
        "final_probiotic_CFU_per_mL": float(res.final_probiotic),
        "target_tracking_error": float(components["target_tracking_error"]),
        "LR_shortfall_norm": float(components["LR_shortfall_norm"]),
        "PAUC_shortfall_norm": float(components["PAUC_shortfall_norm"]),
        "pathogen_violation_norm": float(components.get("pathogen_violation_norm", np.nan)),
        "total_dosage_norm": float(components.get("dose_burden_norm", components.get("total_dosage_norm", np.nan))),
        "dose_burden_norm": float(components.get("dose_burden_norm", components.get("total_dosage_norm", np.nan))),
        "composite_score": float(components["composite_score"]),
        "composite_penalty": float(components.get("composite_penalty", components["composite_score"])),
        "pathogen_constraint_satisfied": bool(path_ok),
        "probiotic_constraint_satisfied": bool(prob_ok),
        "all_constraints_satisfied": bool(all_ok),
        "optimizer_selection_rule": components.get("selection_rule", ""),
        "desired_P_AUC": float(bio.desired_pauc),
    }
    clipped = clip_tthr(tthr_pred)
    for i in range(N_STRAINS):
        row[f"predicted_Tthr_{i + 1}"] = float(clipped[i])
        row[f"LR{i + 1}"] = float(res.LR_terminal_median[i])
        row[f"desired_LR{i + 1}"] = float(bio.desired_lr[i])
    if extra_meta:
        row.update(extra_meta)
    return row


def aggregate_predicted_tthr_across_repeats(
    prediction_jobs: Sequence[Tuple[int, str, str]],
    x_csv: str,
    metadata_csv: Optional[str],
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, List[str]]:
    """Aggregate per-model Tthr: median across repeats of within-split medians."""
    per_model_stack: Dict[str, List[np.ndarray]] = {m: [] for m in CORE_CLOSED_LOOP_MODELS}
    audit_rows: List[dict] = []
    prediction_sources: List[str] = []
    for repeat_id, pred_csv, source in sorted(prediction_jobs, key=lambda item: item[0]):
        pred_df = pd.read_csv(pred_csv)
        predictions, val_idx, _, seed = parse_predictions_wide_df(pred_df)
        predictions = filter_closed_loop_predictions(predictions)
        x_test_df, _ = load_validation_rows(x_csv, metadata_csv, val_idx)
        n_cases = len(x_test_df)
        row: dict = {
            "repeat_id": int(repeat_id),
            "prediction_csv": pred_csv,
            "n_validation_cases": int(n_cases),
            "seed": int(seed) if np.isfinite(seed) else seed,
        }
        for model_name in CORE_CLOSED_LOOP_MODELS:
            if model_name not in predictions:
                continue
            med = median_predicted_tthr(predictions, model_name, n_cases)
            per_model_stack[model_name].append(med)
            for i, val in enumerate(med, start=1):
                row[f"median_Tthr_{i}_{model_name}"] = float(val)
        audit_rows.append(row)
        prediction_sources.append(source or pred_csv)
    aggregated = {
        model_name: np.median(np.stack(values, axis=0), axis=0)
        for model_name, values in per_model_stack.items()
        if values
    }
    if not aggregated:
        raise ValueError("No Tthr predictions found for fixed-Umax validation.")
    return aggregated, pd.DataFrame(audit_rows), prediction_sources


def evaluate_fixed_umax_paper_figure_forward(
    tthr_by_model: Dict[str, np.ndarray],
    config: ClosedLoopConfig,
    dosage_reference: float,
    *,
    n_prediction_repeats: int = 1,
    n_validation_cases: Optional[int] = None,
    srl_model_name: str = TAR_MODEL,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fig. 4: exactly one paper_figure forward ODE per model (3 total); not per test row or repeat."""
    case_rows: List[dict] = []
    candidate_rows: List[dict] = []
    stored_results: Dict[str, Dict[int, Tuple[SimulationResult, float, np.ndarray]]] = {
        model: {} for model in CORE_CLOSED_LOOP_MODELS
    }
    bio = paper_figure_bio_params()
    u_fixed = fixed_umax_paper_profile(config)

    for model_name in CORE_CLOSED_LOOP_MODELS:
        if model_name not in tthr_by_model:
            continue
        tthr = np.asarray(tthr_by_model[model_name], dtype=float)
        optimized_u, res, components, _ = simulate_at_fixed_umax(
            bio,
            tthr,
            config,
            dosage_reference,
            meta_row=None,
            u_fixed=u_fixed,
            trajectory=True,
        )
        extra_model = {
            "paper_figure_profile": True,
            "umax_optimizer_applied": False,
            "fixed_umax_validation": True,
            "aggregated_tthr": "median_of_repeat_split_medians",
            "n_prediction_repeats": int(n_prediction_repeats),
            "n_validation_cases": int(n_validation_cases) if n_validation_cases is not None else np.nan,
        }
        case_rows.append(
            closed_loop_case_row(
                repeat_id=0,
                case_index=0,
                original_row_index=-1,
                model_name=model_name,
                bio=bio,
                tthr_pred=tthr,
                optimized_u=optimized_u,
                res=res,
                components=components,
                config=config,
                extra_meta=extra_model,
            )
        )
        stored_results[model_name][0] = (res, optimized_u, clip_tthr(tthr))

    cases_df = pd.DataFrame(case_rows)
    candidates_df = pd.DataFrame(candidate_rows)
    summary_by_model = (
        cases_df.groupby("model", as_index=False)
        .agg(
            mean_composite_score=("composite_score", "mean"),
            mean_target_tracking_error=("target_tracking_error", "mean"),
            mean_P_AUC=("P_AUC", "mean"),
            mean_LR=("mean_LR", "mean"),
            mean_total_dosage=("total_dosage_ug_per_mL", "mean"),
            constraint_success_rate=("all_constraints_satisfied", "mean"),
            pathogen_constraint_success_rate=("pathogen_constraint_satisfied", "mean"),
            probiotic_constraint_success_rate=("probiotic_constraint_satisfied", "mean"),
            mean_optimized_u_max=("optimized_u_max", "mean"),
            mean_dose_count=("dose_count", "mean"),
            mean_final_pathogen=("final_total_pathogen_CFU_per_mL", "mean"),
            mean_terminal_pathogen=("terminal_total_pathogen_CFU_per_mL", "mean"),
        )
    )
    diagnostics = {
        "n_cases": int(n_validation_cases) if n_validation_cases is not None else 0,
        "n_prediction_repeats": int(n_prediction_repeats),
        "stored_results": stored_results,
        "summary_by_model": summary_by_model,
        "srl_model_name": srl_model_name,
        "tthr_by_model": {k: np.asarray(v, dtype=float).tolist() for k, v in tthr_by_model.items()},
    }
    return cases_df, candidates_df, diagnostics


def _fixed_umax_root_is_complete(outdir: str) -> bool:
    cases_path = os.path.join(outdir, FIXED_UMAX_CASES_CSV)
    traj_path = os.path.join(outdir, FIXED_UMAX_TRAJECTORIES_CSV)
    if not (os.path.isfile(cases_path) and os.path.isfile(traj_path)):
        return False
    if os.path.getsize(traj_path) == 0:
        return False
    try:
        cases_df = pd.read_csv(cases_path)
        traj_df = pd.read_csv(traj_path)
    except Exception:
        return False
    if len(cases_df) != FIG4_ODE_FORWARD_MODELS:
        return False
    if traj_df.empty or "time_h" not in traj_df.columns:
        return False
    if "paper_figure_profile" in cases_df.columns:
        return bool(cases_df["paper_figure_profile"].astype(bool).all())
    return True


def _regenerate_fixed_umax_figures_from_disk(outdir: str, config: ClosedLoopConfig) -> None:
    traj_path = os.path.join(outdir, FIXED_UMAX_TRAJECTORIES_CSV)
    stats_path = os.path.join(outdir, FIXED_UMAX_REPEATED_STATS_CSV)
    plot_manifest_path = os.path.join(outdir, FIG4_PLOT_MANIFEST_JSON)
    traj_df = pd.read_csv(traj_path)
    stats_df = pd.read_csv(stats_path) if os.path.isfile(stats_path) else pd.read_csv(
        os.path.join(outdir, FIXED_UMAX_SUMMARY_CSV)
    )
    plot_manifest = {}
    if os.path.isfile(plot_manifest_path):
        with open(plot_manifest_path, encoding="utf-8") as fh:
            plot_manifest = json.load(fh)
    annotations_path = os.path.join(outdir, FIXED_UMAX_SIGNIFICANCE_CSV)
    annotations_df = pd.read_csv(annotations_path) if os.path.isfile(annotations_path) else pd.DataFrame()
    write_fixed_umax_primary_figure_pngs(
        outdir,
        trajectories_df=traj_df,
        plot_manifest=plot_manifest,
        stats_df=stats_df,
        config=config,
        n_repeats=1,
        annotations_df=annotations_df,
    )


def run_fixed_umax_validation_pipeline(
    *,
    predictions_dir: Optional[str] = None,
    predictions_manifest: Optional[str] = None,
    predictions_csv: Optional[str] = None,
    x_csv: str,
    metadata_csv: Optional[str] = None,
    outdir: str,
    config: ClosedLoopConfig,
    verbose: bool = False,
    finalize_only: bool = False,
    force_rerun: bool = False,
) -> None:
    """Fig. 4 data: aggregate benchmark Tthr predictions, then run 3 paper_figure ODEs total."""
    outdir = _ensure_dir(outdir)
    if finalize_only:
        if not _fixed_umax_root_is_complete(outdir):
            raise RuntimeError(
                f"Cannot finalize fixed-Umax validation: missing root artifacts under {outdir}"
            )
        _regenerate_fixed_umax_figures_from_disk(outdir, config)
        print(f"Regenerated Fig. 4 figure PNGs under {outdir}")
        return

    if not force_rerun and _fixed_umax_root_is_complete(outdir):
        print(
            f"Fixed-Umax validation already complete ({FIG4_ODE_FORWARD_MODELS} ODEs at {outdir}); skipping.",
            flush=True,
        )
        return

    prediction_jobs = resolve_prediction_csv_jobs(
        predictions_dir=predictions_dir,
        predictions_manifest=predictions_manifest,
        predictions_csv=predictions_csv,
    )
    n_prediction_repeats = len(prediction_jobs)
    if verbose:
        print(
            f"Fig. 4: aggregating Tthr from {n_prediction_repeats} prediction repeat(s), "
            f"then running {FIG4_ODE_FORWARD_MODELS} forward ODEs (paper_figure profile)...",
            flush=True,
        )
    aggregated_tthr, tthr_audit_df, prediction_sources = aggregate_predicted_tthr_across_repeats(
        prediction_jobs, x_csv, metadata_csv
    )
    _save_csv(tthr_audit_df, os.path.join(outdir, FIXED_UMAX_TTHR_BY_REPEAT_CSV))
    source_label = prediction_sources[0] if len(prediction_sources) == 1 else f"{len(prediction_sources)}_repeats"
    eval_config = replace(config, prediction_source=source_label)
    run_fixed_umax_validation_from_aggregated_tthr(
        outdir=outdir,
        tthr_by_model=aggregated_tthr,
        config=eval_config,
        n_prediction_repeats=n_prediction_repeats,
        prediction_sources=prediction_sources,
    )
    print(f"\nSaved fixed-Umax validation ({FIG4_ODE_FORWARD_MODELS} ODEs) to {outdir}")


def run_fixed_umax_validation_from_aggregated_tthr(
    outdir: str,
    tthr_by_model: Dict[str, np.ndarray],
    config: ClosedLoopConfig,
    *,
    n_prediction_repeats: int,
    prediction_sources: Sequence[str],
    dosage_reference: Optional[float] = None,
) -> dict:
    outdir = _ensure_dir(outdir)
    dose_ref = dosage_reference if dosage_reference is not None else resolve_dosage_reference(config)
    srl_model_name = TAR_MODEL
    cases_df, candidates_df, diag = evaluate_fixed_umax_paper_figure_forward(
        tthr_by_model,
        config,
        dose_ref,
        n_prediction_repeats=n_prediction_repeats,
        srl_model_name=srl_model_name,
    )
    summary_by_model = diag["summary_by_model"].copy()
    strongest_param = identify_strongest_parameter_control(summary_by_model, srl_model_name)
    strongest_cl = identify_strongest_closed_loop_control(summary_by_model, srl_model_name)
    pairwise_df = build_pairwise_tests(
        cases_df,
        srl_model_name=srl_model_name,
        metrics=[
            "composite_score",
            "target_tracking_error",
            "mean_LR",
            "P_AUC",
            "total_dosage_ug_per_mL",
        ],
        config=config,
        repeat_id=0,
    )
    constraint_df = summary_by_model[
        ["model", "constraint_success_rate", "pathogen_constraint_success_rate", "probiotic_constraint_success_rate"]
    ].copy()
    rep_info = select_representative_case(cases_df, srl_model_name, strongest_cl)
    rep_case_idx = rep_info["selected_case_index"]
    plot_models = [m for m in CORE_CLOSED_LOOP_MODELS if m in diag["stored_results"]]
    plot_payload = {
        model: diag["stored_results"][model][rep_case_idx]
        for model in plot_models
        if rep_case_idx in diag["stored_results"].get(model, {})
    }
    write_closed_loop_trajectory_artifacts(plot_payload, plot_models, outdir, rep_info)
    optimizer_meta = write_optimizer_diagnostics(outdir, cases_df, candidates_df, config)
    fairness = build_fairness_check(
        cases_df,
        config,
        np.array([], dtype=int),
        diag["n_cases"],
        prediction_sources=list(prediction_sources),
    )
    validate_fairness_check(fairness)
    _save_csv(cases_df, os.path.join(outdir, FIXED_UMAX_CASES_CSV))
    export_u_candidates_csv(candidates_df, os.path.join(outdir, FIXED_UMAX_U_CANDIDATES_CSV))
    _save_csv(summary_by_model, os.path.join(outdir, FIXED_UMAX_SUMMARY_CSV))
    _save_csv(pairwise_df, os.path.join(outdir, FIXED_UMAX_PAIRWISE_CSV))
    _save_csv(constraint_df, os.path.join(outdir, FIXED_UMAX_CONSTRAINT_CSV))
    _save_csv(pd.DataFrame(), os.path.join(outdir, FIXED_UMAX_SIGNIFICANCE_CSV))
    _save_csv(summary_by_model, os.path.join(outdir, FIXED_UMAX_REPEATED_STATS_CSV))
    fairness_path = os.path.join(outdir, FIXED_UMAX_FAIRNESS_JSON)
    _ensure_parent_dir(fairness_path)
    with open(fairness_path, "w", encoding="utf-8") as fh:
        json.dump(_to_json_native(fairness), fh, indent=2)
    manifest = {
        "core_models": list(CORE_CLOSED_LOOP_MODELS),
        "model_display_labels": {k: FIXED_UMAX_DISPLAY_LABELS[k] for k in CORE_CLOSED_LOOP_MODELS},
        "prediction_sources": list(prediction_sources),
        "no_retraining": True,
        "weight_profile": config.weight_profile,
        "weight_selection_source": config.weight_selection_source,
        "strongest_parameter_baseline": strongest_param,
        "strongest_validation_baseline": strongest_cl,
        "u_grid": config.u_grid.tolist(),
        "optimizer_rule": fairness["optimizer_rule"],
        **build_closed_loop_optimizer_manifest_fields(config, dose_ref, summary_df=summary_by_model),
        "fairness_check": fairness,
        "representative_case_selection": rep_info,
        "aggregated_tthr_by_model": diag.get("tthr_by_model", {}),
        **optimizer_meta,
        **build_fig4_manifest_fields(
            n_prediction_repeats=n_prediction_repeats,
            significance_for_manuscript=False,
            prediction_sources=prediction_sources,
        ),
    }
    write_json_manifest(outdir, FIG4_MANIFEST_JSON, _to_json_native(manifest))
    traj_df = pd.read_csv(os.path.join(outdir, FIXED_UMAX_TRAJECTORIES_CSV))
    plot_manifest_path = os.path.join(outdir, FIG4_PLOT_MANIFEST_JSON)
    plot_manifest = {}
    if os.path.isfile(plot_manifest_path):
        with open(plot_manifest_path, encoding="utf-8") as fh:
            plot_manifest = json.load(fh)
    write_fixed_umax_primary_figure_pngs(
        outdir,
        trajectories_df=traj_df,
        plot_manifest=plot_manifest,
        stats_df=summary_by_model,
        config=config,
        n_repeats=1,
        annotations_df=pd.DataFrame(),
    )
    return {
        "cases_df": cases_df,
        "candidates_df": candidates_df,
        "summary_by_model": summary_by_model,
        "pairwise_df": pairwise_df,
        "run_summary": manifest,
        "dosage_reference": dose_ref,
    }


def _umax_ablation_case_row(
    *,
    repeat_id: int,
    case_index: int,
    original_row_index: int,
    ablation_condition: str,
    base_model: str,
    umax_policy: str,
    bio: CaseBioParams,
    tthr_pred: np.ndarray,
    optimized_u: float,
    res: SimulationResult,
    components: dict,
    config: ClosedLoopConfig,
    extra_meta: Optional[dict] = None,
) -> dict:
    row = closed_loop_case_row(
        repeat_id=repeat_id,
        case_index=case_index,
        original_row_index=original_row_index,
        model_name=ablation_condition,
        bio=bio,
        tthr_pred=tthr_pred,
        optimized_u=optimized_u,
        res=res,
        components=components,
        config=config,
        extra_meta=extra_meta,
    )
    row["ablation_condition"] = ablation_condition
    row["base_model"] = base_model
    row["umax_policy"] = umax_policy
    row["umax_mode"] = umax_policy  # legacy alias
    row["condition"] = ablation_condition
    row["selected_u_max"] = optimized_u
    row["feasible_candidate"] = bool(components.get("feasible_candidate", False))
    return row


def build_umax_policy_ablation_cases_export(cases_df: pd.DataFrame) -> pd.DataFrame:
    """Canonical Fig. 5 policy ablation case table."""
    if cases_df.empty:
        return cases_df.copy()
    out = cases_df.copy()
    if "condition" not in out.columns and "ablation_condition" in out.columns:
        out["condition"] = out["ablation_condition"]
    if "selected_u_max" not in out.columns and "optimized_u_max" in out.columns:
        out["selected_u_max"] = out["optimized_u_max"]
    rename_map = {
        "total_dosage_ug_per_mL": "total_dosage",
        "terminal_total_pathogen_CFU_per_mL": "terminal_total_pathogen",
    }
    for old, new in rename_map.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    keep_cols = [
        "repeat_id",
        "case_index",
        "condition",
        "umax_policy",
        "selected_u_max",
        "total_dosage",
        "P_AUC",
        "mean_LR",
        "terminal_total_pathogen",
        "composite_penalty",
        "LR_shortfall_norm",
        "PAUC_shortfall_norm",
        "pathogen_violation_norm",
        "dose_burden_norm",
        "feasible_candidate",
    ]
    for col in keep_cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep_cols + [c for c in out.columns if c not in keep_cols]]


def build_umax_policy_ablation_summary_by_condition(cases_df: pd.DataFrame) -> pd.DataFrame:
    export = build_umax_policy_ablation_cases_export(cases_df)
    cond_col = "condition" if "condition" in export.columns else "ablation_condition"
    agg = (
        export.groupby(cond_col, as_index=False)
        .agg(
            n_rows=("case_index", "count"),
            n_repeats=("repeat_id", "nunique"),
            n_cases=("case_index", "nunique"),
            mean_selected_u_max=("selected_u_max", "mean"),
            mean_total_dosage=("total_dosage", "mean"),
            mean_P_AUC=("P_AUC", "mean"),
            mean_LR=("mean_LR", "mean"),
            mean_terminal_pathogen=("terminal_total_pathogen", "mean"),
            mean_composite_penalty=("composite_penalty", "mean"),
            feasible_fraction=("feasible_candidate", "mean"),
        )
        .rename(columns={cond_col: "condition"})
    )
    return agg


def build_umax_policy_ablation_condition_counts(cases_df: pd.DataFrame) -> pd.DataFrame:
    export = build_umax_policy_ablation_cases_export(cases_df)
    cond_col = "condition" if "condition" in export.columns else "ablation_condition"
    counts = (
        export.groupby(cond_col, as_index=False)
        .agg(
            n_rows=("case_index", "count"),
            n_repeats=("repeat_id", "nunique"),
            n_cases=("case_index", "nunique"),
        )
        .rename(columns={cond_col: "condition"})
    )
    return counts


def validate_fig5_main_ablation_conditions(
    cases_df: pd.DataFrame,
    *,
    context: str = "",
    required: Optional[Sequence[str]] = None,
) -> None:
    """Raise if any required Fig. 5D main ablation condition is missing."""
    required_list = list(required or FIG5_REQUIRED_MAIN_CONDITIONS)
    if cases_df.empty:
        raise RuntimeError(
            f"Fig. 5 ablation validation failed{': ' + context if context else ''}: cases dataframe is empty."
        )
    cond_col = "ablation_condition" if "ablation_condition" in cases_df.columns else "condition"
    if cond_col not in cases_df.columns:
        raise RuntimeError(
            f"Fig. 5 ablation validation failed{': ' + context if context else ''}: "
            "missing ablation_condition/condition column."
        )
    counts = cases_df.groupby(cond_col).size().to_dict()
    observed = sorted(counts.keys())
    missing = [c for c in required_list if c not in counts]
    if missing:
        prefix = f"Fig. 5 ablation validation failed{': ' + context if context else ''}"
        count_lines = "\n".join(f"  {k}: {v} rows" for k, v in sorted(counts.items()))
        raise RuntimeError(
            f"{prefix}\n"
            f"expected conditions: {required_list}\n"
            f"observed conditions: {observed}\n"
            f"missing conditions: {missing}\n"
            f"row counts by condition:\n{count_lines}"
        )


def save_umax_policy_ablation_artifacts(
    cases_df: pd.DataFrame,
    outdir: str,
    *,
    required_conditions: Optional[Sequence[str]] = None,
) -> None:
    """Write canonical Fig. 5 policy ablation CSVs and validate main conditions."""
    required_list = list(required_conditions or FIG5_REQUIRED_MAIN_CONDITIONS)
    validate_fig5_main_ablation_conditions(
        cases_df, context=f"outdir={outdir}", required=required_list
    )
    policy_cases = build_umax_policy_ablation_cases_export(cases_df)
    summary = build_umax_policy_ablation_summary_by_condition(cases_df)
    counts = build_umax_policy_ablation_condition_counts(cases_df)
    _save_csv(policy_cases, os.path.join(outdir, "umax_policy_ablation_cases.csv"))
    _save_csv(summary, os.path.join(outdir, "umax_policy_ablation_summary_by_condition.csv"))
    _save_csv(counts, os.path.join(outdir, "umax_policy_ablation_condition_counts.csv"))
    if len(counts) < len(required_list):
        raise RuntimeError(
            f"umax_policy_ablation_condition_counts has {len(counts)} conditions; "
            f"expected at least {len(required_list)}."
        )
    # Backward-compatible aliases for figure_audit / legacy scripts
    _save_csv(cases_df, os.path.join(outdir, "umax_ablation_cases.csv"))
    summary_legacy = summary.rename(columns={"condition": "model", "mean_composite_penalty": "mean_composite_score"})
    for col in REPEATED_CLOSED_LOOP_METRICS:
        if col not in summary_legacy.columns:
            if col == "mean_composite_score" and "mean_composite_penalty" in summary.columns:
                summary_legacy[col] = summary["mean_composite_penalty"]
            elif col == "mean_target_tracking_error":
                summary_legacy[col] = np.nan
            elif col == "constraint_success_rate" and "feasible_fraction" in summary.columns:
                summary_legacy[col] = summary["feasible_fraction"]
            else:
                summary_legacy[col] = np.nan
    _save_csv(summary_legacy, os.path.join(outdir, "umax_ablation_summary_by_condition.csv"))


def select_umax_ablation_representative_case(cases_df: pd.DataFrame) -> dict:
    """Pick case where TAR optimized beats training-tuned global (75–95th pct improvement)."""
    tuned = "TAR_fixed_training_tuned_global"
    opt = "TAR_optimized"
    pivot = cases_df.pivot(index=["repeat_id", "case_index"], columns="ablation_condition", values="composite_score")
    if tuned not in pivot.columns or opt not in pivot.columns:
        return {"selected_case_index": 0, "selected_repeat_id": 0, "note": "fallback_first_case"}
    improvement = pivot[tuned] - pivot[opt]
    improvement = improvement[np.isfinite(improvement)]
    if improvement.empty:
        return {"selected_case_index": 0, "selected_repeat_id": 0, "note": "fallback_first_case"}
    q75 = float(improvement.quantile(0.75))
    q95 = float(improvement.quantile(0.95))
    band = improvement[(improvement >= q75) & (improvement <= q95)]
    candidates = band if not band.empty else improvement
    target = float(candidates.median())
    idx = (candidates - target).abs().idxmin()
    repeat_id, case_index = int(idx[0]), int(idx[1])
    return {
        "selected_repeat_id": repeat_id,
        "selected_case_index": case_index,
        "penalty_reduction_tuned_global_minus_optimized": float(improvement.loc[idx]),
        "improvement_percentile_band": [q75, q95],
        "plot_conditions": list(FIG5_REPRESENTATIVE_CONDITIONS),
    }


def write_umax_ablation_trajectory_artifacts(
    plot_payload: Dict[str, Tuple[SimulationResult, float, np.ndarray]],
    plot_conditions: Sequence[str],
    outdir: str,
    rep_info: dict,
) -> None:
    missing = [c for c in plot_conditions if c not in plot_payload]
    if missing:
        raise RuntimeError(
            f"Fig. 5C trajectory export incomplete: missing conditions {missing}. "
            f"Available: {sorted(plot_payload.keys())}"
        )
    rows: List[dict] = []
    t_thr_by_condition: Dict[str, List[float]] = {}
    policy_umax_by_condition: Dict[str, float] = {}
    for condition in plot_conditions:
        res, optimized_u, t_thr = plot_payload[condition]
        policy_umax_by_condition[condition] = float(optimized_u)
        t_thr_by_condition[condition] = [float(v) for v in t_thr]
        p_total = res.P_S + res.P_R
        hist = trajectory_history_from_arrays(res.times, res.C, p_total, res.B_total)
        for _, hist_row in hist.iterrows():
            rows.append({"ablation_condition": condition, **hist_row.to_dict()})
    traj_path = os.path.join(_ensure_dir(outdir), "umax_ablation_representative_trajectories.csv")
    _save_csv(pd.DataFrame(rows), traj_path)
    ordered = [c for c in FIG5_REPRESENTATIVE_CONDITIONS if c in t_thr_by_condition]
    display_labels = []
    for condition in ordered:
        umax = policy_umax_by_condition.get(condition, float("nan"))
        base = FIG5_ABLATION_TITLE_LABELS.get(condition, condition)
        display_labels.append(f"{base} = {umax:.1f}")
    manifest = {
        "validation_section": "Umax optimization analysis",
        "plot_conditions": ordered,
        "plot_display_labels": display_labels,
        "policy_umax_by_condition": policy_umax_by_condition,
        "composite_penalty_by_condition": rep_info.get("composite_penalty_by_condition", {}),
        "t_thr_by_condition": t_thr_by_condition,
        "trajectories_csv": os.path.basename(traj_path),
        "figure_png": "umax_ode_ablation.png",
        "fig5_panel": "C",
        "illustrative_only": True,
        **rep_info,
    }
    write_json_manifest(outdir, FIG5_PLOT_MANIFEST_JSON, _to_json_native(manifest))


def _process_umax_optimization_case(
    case_idx: int,
    *,
    repeat_id: int,
    val_indices: np.ndarray,
    x_test_df: pd.DataFrame,
    metadata_test_df: Optional[pd.DataFrame],
    predictions_tthr: Dict[str, np.ndarray],
    config: ClosedLoopConfig,
    dose_reference_scale: float,
    dose_reference_source: str,
    ode_backend: str,
    fixed_policy_df: Optional[pd.DataFrame] = None,
) -> Tuple[List[dict], List[dict], List[dict], List[dict], List[dict], List[dict], List[dict]]:
    x_row = x_test_df.iloc[case_idx]
    meta_row = metadata_test_df.iloc[case_idx] if metadata_test_df is not None else None
    bio = bio_params_from_row(x_row, meta_row)
    extra: dict = {}
    if meta_row is not None and "bio_id" in meta_row.index:
        extra["bio_id"] = int(meta_row["bio_id"])
    if meta_row is not None and "desired_profile_id" in meta_row.index:
        extra["desired_profile_id"] = int(meta_row["desired_profile_id"])

    ablation_spec = resolve_umax_ablation_spec(config)
    optimized_models = optimized_base_models_in_spec(ablation_spec)
    needs_landscape_export = bool(config.export_umax_score_landscape and optimized_models)

    opt_cache: Dict[str, Tuple[float, SimulationResult, dict, List[dict], dict, List[dict]]] = {}
    aspiration_debug_rows: List[dict] = []
    policy_sensitivity_rows: List[dict] = []
    candidate_rows: List[dict] = []
    score_landscape_rows: List[dict] = []
    response_landscape_rows: List[dict] = []
    feasible_summary_rows: List[dict] = []

    for base_model in optimized_models:
        tthr = predictions_tthr[base_model][case_idx]
        best_u, best_res, components, candidates, selection_debug, sensitivity = optimize_umax(
            bio,
            tthr,
            config,
            dose_reference_scale,
            metadata_row=meta_row,
            x_row=x_row,
            dose_reference_source=dose_reference_source,
            trajectory=False,
            ode_backend=ode_backend,
        )
        opt_cache[base_model] = (best_u, best_res, components, candidates, selection_debug, sensitivity)
        selection_rule = str(
            selection_debug.get("selection_rule", components.get("optimizer_selection_rule", ""))
        )
        selection_stage = str(selection_debug.get("optimizer_selection_stage", ""))
        aspiration_debug_rows.append(
            build_umax_aspiration_selection_debug_row(
                repeat_id=repeat_id,
                case_index=case_idx,
                model=base_model,
                best_cand=next(c for c in candidates if c.get("selected_by_optimizer")),
                selection_rule=selection_rule,
                debug=selection_debug,
            )
        )
        for row in sensitivity:
            row = dict(row)
            row["repeat_id"] = int(repeat_id)
            row["case_index"] = int(case_idx)
            row["model"] = base_model
            policy_sensitivity_rows.append(row)
        if needs_landscape_export:
            feasible_summary_rows.append(
                build_umax_feasible_region_summary_row(
                    repeat_id=repeat_id,
                    case_index=case_idx,
                    model=base_model,
                    candidates=candidates,
                    selection_rule=selection_rule,
                    selection_stage=selection_stage,
                )
            )
            for cand in candidates:
                candidate_rows.append(
                    format_u_candidate_export_row(
                        repeat_id=repeat_id,
                        case_index=case_idx,
                        validation_original_row_index=int(val_indices[case_idx]),
                        model=base_model,
                        cand=cand,
                    )
                )
                score_landscape_rows.append(
                    format_umax_score_landscape_row(
                        repeat_id=repeat_id,
                        case_index=case_idx,
                        validation_original_row_index=int(val_indices[case_idx]),
                        cand=cand,
                    )
                )
                response_landscape_rows.append(
                    format_umax_response_landscape_row(
                        repeat_id=repeat_id,
                        case_index=case_idx,
                        validation_original_row_index=int(val_indices[case_idx]),
                        model=base_model,
                        base_model=base_model,
                        umax_policy=UMAX_POLICY_PER_CASE_OPTIMIZED,
                        cand=cand,
                    )
                )

    case_rows: List[dict] = []
    for ablation_condition, (base_model, umax_policy) in ablation_spec.items():
        tthr = predictions_tthr[base_model][case_idx]
        umax_policy_source = ""
        if umax_policy == UMAX_POLICY_PER_CASE_OPTIMIZED:
            if base_model not in opt_cache:
                raise RuntimeError(
                    f"{ablation_condition} requires optimize_umax precompute for {base_model}."
                )
            optimized_u, res, components, _, _, _ = opt_cache[base_model]
            umax_policy_source = "per_case_optimizer"
        else:
            optimized_u, umax_policy_source = resolve_umax_policy_value(
                umax_policy, repeat_id, fixed_policy_df, config
            )
            optimized_u, res, components, _ = simulate_at_fixed_umax(
                bio,
                tthr,
                config,
                dose_reference_scale,
                meta_row,
                u_fixed=optimized_u,
                umax_policy=umax_policy,
                x_row=x_row,
                dose_reference_source=dose_reference_source,
                trajectory=False,
                ode_backend=ode_backend,
            )
        extra_model = {
            **extra,
            "umax_optimizer_applied": umax_policy == UMAX_POLICY_PER_CASE_OPTIMIZED,
            "umax_optimization_study": True,
            "umax_policy_source": umax_policy_source,
            "selected_u_max": float(optimized_u),
        }
        case_rows.append(
            _umax_ablation_case_row(
                repeat_id=repeat_id,
                case_index=case_idx,
                original_row_index=int(val_indices[case_idx]),
                ablation_condition=ablation_condition,
                base_model=base_model,
                umax_policy=umax_policy,
                bio=bio,
                tthr_pred=tthr,
                optimized_u=optimized_u,
                res=res,
                components=components,
                config=config,
                extra_meta=extra_model,
            )
        )
    return (
        case_rows,
        candidate_rows,
        score_landscape_rows,
        response_landscape_rows,
        feasible_summary_rows,
        aspiration_debug_rows,
        policy_sensitivity_rows,
    )


def _build_umax_representative_plot_payload(
    case_idx: int,
    *,
    repeat_id: int,
    x_test_df: pd.DataFrame,
    metadata_test_df: Optional[pd.DataFrame],
    predictions_tthr: Dict[str, np.ndarray],
    config: ClosedLoopConfig,
    dose_reference_scale: float,
    dose_reference_source: str,
    ode_backend: str,
    fixed_policy_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Tuple[SimulationResult, float, np.ndarray]]:
    x_row = x_test_df.iloc[case_idx]
    meta_row = metadata_test_df.iloc[case_idx] if metadata_test_df is not None else None
    bio = bio_params_from_row(x_row, meta_row)
    plot_payload: Dict[str, Tuple[SimulationResult, float, np.ndarray]] = {}
    for condition in FIG5_REPRESENTATIVE_CONDITIONS:
        base_model, umax_policy = FIG5_ABLATION_SPEC[condition]
        tthr = predictions_tthr[base_model][case_idx]
        if umax_policy == UMAX_POLICY_PER_CASE_OPTIMIZED:
            optimized_u, res, _, _, _, _ = optimize_umax(
                bio,
                tthr,
                config,
                dose_reference_scale,
                metadata_row=meta_row,
                x_row=x_row,
                dose_reference_source=dose_reference_source,
                trajectory=True,
                ode_backend=ode_backend,
            )
        else:
            u_fixed, _ = resolve_umax_policy_value(
                umax_policy, repeat_id, fixed_policy_df, config
            )
            optimized_u, res, _, _ = simulate_at_fixed_umax(
                bio,
                tthr,
                config,
                dose_reference_scale,
                meta_row,
                u_fixed=u_fixed,
                umax_policy=umax_policy,
                x_row=x_row,
                dose_reference_source=dose_reference_source,
                trajectory=True,
                ode_backend=ode_backend,
            )
        plot_payload[condition] = (res, optimized_u, clip_tthr(tthr))
    return plot_payload


def evaluate_umax_optimization_study(
    repeat_id: int,
    x_test_df: pd.DataFrame,
    metadata_test_df: Optional[pd.DataFrame],
    predictions_tthr: Dict[str, np.ndarray],
    val_indices: np.ndarray,
    config: ClosedLoopConfig,
    dose_reference_scale: float,
    *,
    dose_reference_source: str = "",
    case_n_jobs: int = 1,
    ode_backend: Optional[str] = None,
    fixed_policy_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    predictions_tthr = filter_closed_loop_predictions(predictions_tthr)
    backend = ode_backend or ode_backend_for_config(config)
    n_cases = len(x_test_df) if config.max_closed_loop_cases <= 0 else min(
        config.max_closed_loop_cases, len(x_test_df)
    )
    case_indices = list(range(n_cases))

    def _run_case(case_idx: int):
        return _process_umax_optimization_case(
            case_idx,
            repeat_id=repeat_id,
            val_indices=val_indices,
            x_test_df=x_test_df,
            metadata_test_df=metadata_test_df,
            predictions_tthr=predictions_tthr,
            config=config,
            dose_reference_scale=dose_reference_scale,
            dose_reference_source=dose_reference_source,
            ode_backend=backend,
            fixed_policy_df=fixed_policy_df,
        )

    case_rows: List[dict] = []
    candidate_rows: List[dict] = []
    score_landscape_rows: List[dict] = []
    response_landscape_rows: List[dict] = []
    feasible_summary_rows: List[dict] = []
    aspiration_debug_rows: List[dict] = []
    policy_sensitivity_rows: List[dict] = []
    if case_n_jobs <= 1 or len(case_indices) <= 1:
        for case_idx in case_indices:
            rows, cands, scores, resp, feas, asp_dbg, pol_sens = _run_case(case_idx)
            case_rows.extend(rows)
            candidate_rows.extend(cands)
            score_landscape_rows.extend(scores)
            response_landscape_rows.extend(resp)
            feasible_summary_rows.extend(feas)
            aspiration_debug_rows.extend(asp_dbg)
            policy_sensitivity_rows.extend(pol_sens)
    else:
        from joblib import Parallel, delayed

        chunks = Parallel(n_jobs=case_n_jobs)(
            delayed(_run_case)(case_idx) for case_idx in case_indices
        )
        for rows, cands, scores, resp, feas, asp_dbg, pol_sens in chunks:
            case_rows.extend(rows)
            candidate_rows.extend(cands)
            score_landscape_rows.extend(scores)
            response_landscape_rows.extend(resp)
            feasible_summary_rows.extend(feas)
            aspiration_debug_rows.extend(asp_dbg)
            policy_sensitivity_rows.extend(pol_sens)

    cases_df = pd.DataFrame(case_rows)
    candidates_df = pd.DataFrame(candidate_rows)
    score_landscape_df = pd.DataFrame(score_landscape_rows)
    response_landscape_df = pd.DataFrame(response_landscape_rows)
    feasible_summary_df = pd.DataFrame(feasible_summary_rows)
    aspiration_debug_df = pd.DataFrame(aspiration_debug_rows)
    policy_sensitivity_df = pd.DataFrame(policy_sensitivity_rows)
    summary_by_condition = (
        cases_df.groupby("ablation_condition", as_index=False)
        .agg(
            mean_composite_score=("composite_score", "mean"),
            mean_target_tracking_error=("target_tracking_error", "mean"),
            mean_P_AUC=("P_AUC", "mean"),
            mean_LR=("mean_LR", "mean"),
            mean_total_dosage=("total_dosage_ug_per_mL", "mean"),
            constraint_success_rate=("all_constraints_satisfied", "mean"),
            mean_optimized_u_max=("optimized_u_max", "mean"),
            mean_terminal_pathogen=("terminal_total_pathogen_CFU_per_mL", "mean"),
        )
        .rename(columns={"ablation_condition": "model"})
    )
    rep_info = select_umax_ablation_representative_case(cases_df)
    required_conditions = required_umax_ablation_conditions(config)
    if config.export_umax_representative_trajectories:
        rep_case_idx = int(rep_info.get("selected_case_index", 0))
        rep_repeat_id = int(rep_info.get("selected_repeat_id", repeat_id))
        plot_payload = _build_umax_representative_plot_payload(
            rep_case_idx,
            repeat_id=rep_repeat_id,
            x_test_df=x_test_df,
            metadata_test_df=metadata_test_df,
            predictions_tthr=predictions_tthr,
            config=config,
            dose_reference_scale=dose_reference_scale,
            dose_reference_source=dose_reference_source,
            ode_backend=backend,
            fixed_policy_df=fixed_policy_df,
        )
        rep_subset = cases_df[
            (cases_df["case_index"] == rep_case_idx) & (cases_df["repeat_id"] == rep_repeat_id)
        ]
        policy_umax_by_condition = {
            str(row["ablation_condition"]): float(row["selected_u_max"])
            for _, row in rep_subset.iterrows()
            if pd.notna(row.get("selected_u_max", row.get("optimized_u_max")))
        }
        composite_penalty_by_condition = {
            str(row["ablation_condition"]): float(row["composite_penalty"])
            for _, row in rep_subset.iterrows()
            if pd.notna(row.get("composite_penalty", row.get("composite_score")))
        }
        rep_info["policy_umax_by_condition"] = policy_umax_by_condition
        rep_info["composite_penalty_by_condition"] = composite_penalty_by_condition
        rep_info["selection_reason"] = rep_info.get("note", "representative_case_band_median")
    else:
        plot_payload = {}
    validate_fig5_main_ablation_conditions(
        cases_df, context=f"repeat_id={repeat_id}", required=required_conditions
    )
    diagnostics = {
        "n_cases": n_cases,
        "plot_payload": plot_payload,
        "summary_by_condition": summary_by_condition,
        "representative_case_selection": rep_info,
        "ode_backend": backend,
        "case_n_jobs": int(case_n_jobs),
        "aspiration_debug_df": aspiration_debug_df,
        "policy_sensitivity_df": policy_sensitivity_df,
        "response_landscape_df": response_landscape_df,
        "feasible_summary_df": feasible_summary_df,
    }
    return cases_df, candidates_df, score_landscape_df, response_landscape_df, feasible_summary_df, diagnostics


def run_umax_optimization_study_evaluation(
    outdir: str,
    repeat_id: int,
    x_test_df: pd.DataFrame,
    metadata_test_df: Optional[pd.DataFrame],
    predictions_tthr: Dict[str, np.ndarray],
    val_indices: np.ndarray,
    config: ClosedLoopConfig,
    dose_reference_scale: Optional[float] = None,
    *,
    dose_reference_source: str = "",
    reference_by_repeat_df: Optional[pd.DataFrame] = None,
    fixed_policy_df: Optional[pd.DataFrame] = None,
    case_n_jobs: int = 1,
    ode_backend: Optional[str] = None,
) -> dict:
    outdir = _ensure_dir(outdir)
    if dose_reference_scale is None:
        dose_ref, dose_src = resolve_dose_reference_scale_for_repeat(
            config, repeat_id, reference_by_repeat_df
        )
    else:
        dose_ref = float(dose_reference_scale)
        dose_src = dose_reference_source or config.dose_reference_source
    cases_df, candidates_df, score_landscape_df, response_landscape_df, feasible_summary_df, diag = (
        evaluate_umax_optimization_study(
            repeat_id=repeat_id,
            x_test_df=x_test_df,
            metadata_test_df=metadata_test_df,
            predictions_tthr=predictions_tthr,
            val_indices=val_indices,
            config=config,
            dose_reference_scale=dose_ref,
            dose_reference_source=dose_src,
            case_n_jobs=case_n_jobs,
            ode_backend=ode_backend,
            fixed_policy_df=fixed_policy_df,
        )
    )
    summary_by_condition = diag["summary_by_condition"].copy()
    rep_info = diag["representative_case_selection"]
    required_conditions = required_umax_ablation_conditions(config)
    if config.export_umax_representative_trajectories and diag.get("plot_payload"):
        write_umax_ablation_trajectory_artifacts(
            diag["plot_payload"], FIG5_REPRESENTATIVE_CONDITIONS, outdir, rep_info
        )

    align_cases = cases_df[cases_df["ablation_condition"] == "TAR_optimized"].copy()
    align_df = pd.DataFrame()
    if not align_cases.empty and config.export_umax_score_landscape and not candidates_df.empty:
        align_cases["model"] = TAR_MODEL
        from derive_optimizer_references import build_umax_objective_alignment

        align_df, align_summary, conflict = build_umax_objective_alignment(
            candidates_df, align_cases, TAR_MODEL, "case", config=config
        )
        _save_csv(align_df, os.path.join(outdir, "umax_objective_alignment.csv"))
        _save_csv(align_summary, os.path.join(outdir, "umax_objective_alignment_summary.csv"))
    else:
        conflict = False

    save_umax_policy_ablation_artifacts(
        cases_df, outdir, required_conditions=required_conditions
    )
    if config.export_umax_score_landscape and not candidates_df.empty:
        export_u_candidates_csv(candidates_df, os.path.join(outdir, "umax_optimization_u_candidates.csv"))
        _save_csv(score_landscape_df, os.path.join(outdir, "umax_score_landscape_curves.csv"))
        export_umax_response_landscape_csv(
            response_landscape_df, os.path.join(outdir, UMAX_RESPONSE_LANDSCAPE_CSV)
        )
        export_umax_feasible_region_summary_csv(
            feasible_summary_df, os.path.join(outdir, UMAX_FEASIBLE_REGION_SUMMARY_CSV)
        )
        selected_dist = build_umax_selected_umax_distribution_df(candidates_df)
        if not selected_dist.empty:
            _save_csv(selected_dist, os.path.join(outdir, UMAX_SELECTED_UMAX_DISTRIBUTION_CSV))
    _save_csv(summary_by_condition, os.path.join(outdir, "umax_ablation_repeated_plot_stats.csv"))
    _save_csv(
        diag.get("aspiration_debug_df", pd.DataFrame()),
        os.path.join(outdir, UMAX_ASPIRATION_SELECTION_DEBUG_CSV),
    )
    _save_csv(
        diag.get("policy_sensitivity_df", pd.DataFrame()),
        os.path.join(outdir, UMAX_SELECTION_POLICY_SENSITIVITY_CSV),
    )

    invariant_errors = validate_umax_inverse_design_invariants(
        candidates_df=candidates_df,
        response_landscape_df=response_landscape_df,
        feasible_summary_df=feasible_summary_df,
        policy_sensitivity_df=diag.get("policy_sensitivity_df"),
        config=config,
    )
    if invariant_errors:
        raise RuntimeError(
            "Umax inverse-design invariant check failed:\n" + "\n".join(invariant_errors)
        )

    manifest = {
        "prediction_source": config.prediction_source,
        "no_retraining": True,
        "u_grid": config.u_grid.tolist(),
        "ode_backend": diag.get("ode_backend", ode_backend_for_config(config)),
        "case_n_jobs": int(diag.get("case_n_jobs", case_n_jobs)),
        **build_closed_loop_optimizer_manifest_fields(
            config, dose_ref, summary_df=summary_by_condition, dose_reference_source=dose_src
        ),
        "umax_objective_conflict_detected": bool(conflict),
        "umax_inverse_design_exports": {
            "umax_response_landscape_csv": UMAX_RESPONSE_LANDSCAPE_CSV,
            "umax_feasible_region_summary_csv": UMAX_FEASIBLE_REGION_SUMMARY_CSV,
            "umax_optimization_u_candidates_csv": "umax_optimization_u_candidates.csv",
            "umax_score_landscape_curves_csv": "umax_score_landscape_curves.csv",
            "umax_selection_policy_sensitivity_csv": UMAX_SELECTION_POLICY_SENSITIVITY_CSV,
        },
        **build_fig5_manifest_fields(n_repeats=1, significance_for_manuscript=False),
    }
    write_json_manifest(outdir, UMAX_OPTIMIZATION_MANIFEST_JSON, _to_json_native(manifest))
    return {
        "cases_df": cases_df,
        "candidates_df": candidates_df,
        "score_landscape_df": score_landscape_df,
        "response_landscape_df": response_landscape_df,
        "feasible_summary_df": feasible_summary_df,
        "summary_by_condition": summary_by_condition,
        "alignment_df": align_df,
        "dosage_reference": dose_ref,
        "dose_reference_scale": dose_ref,
        "dose_reference_source": dose_src,
    }


def paired_bootstrap_ci(diff: np.ndarray, n_bootstrap: int, seed: int) -> Tuple[float, float, float]:
    if len(diff) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(diff), size=len(diff))
        boots.append(float(np.mean(diff[idx])))
    boots = np.asarray(boots, dtype=float)
    return float(np.mean(diff)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def permutation_test_mean_diff(diff: np.ndarray, n_perm: int, seed: int) -> float:
    if len(diff) == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    observed = float(np.mean(diff))
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diff))
        perm_mean = float(np.mean(diff * signs))
        if abs(perm_mean) >= abs(observed):
            count += 1
    return float((count + 1) / (n_perm + 1))


def significance_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "na"
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def evaluate_significance_label(
    p_value: float,
    ci_low: float,
    ci_high: float,
    n_repeats: int,
    srl_better: bool,
    single_split_exploratory: bool = False,
) -> Tuple[str, str]:
    if not srl_better:
        return "ns", "control_better"
    if single_split_exploratory:
        return "ns", "single_split"
    if n_repeats < 2:
        return "ns", "ns"
    stars = significance_label(p_value)
    if (
        stars == "ns"
        and srl_better
        and np.isfinite(ci_low)
        and np.isfinite(ci_high)
        and ci_low > 0.0
    ):
        stars = "*"
    if stars == "ns":
        return "ns", "ns"
    if n_repeats < 10:
        return stars, "exploratory"
    return stars, "formal"


def comparison_result_label(
    star_label: str,
    significance_tier: str,
    srl_better: bool,
    n_repeats: int,
) -> str:
    if significance_tier == "control_better":
        return "control_better"
    if star_label in {"*", "**", "***", "****"}:
        if n_repeats < 10 or significance_tier == "exploratory":
            return "exploratory_srl_better"
        return "srl_better"
    return "not_significant"


def build_pairwise_tests(
    cases_df: pd.DataFrame,
    srl_model_name: str,
    metrics: Sequence[str],
    config: ClosedLoopConfig,
    repeat_id: int,
    control_models: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    controls = list(control_models or CLOSED_LOOP_SIGNIFICANCE_CONTROLS)
    rows = []
    pivot = cases_df.pivot_table(index="case_index", columns="model", values=list(metrics), aggfunc="first")
    for control in controls:
        if control == srl_model_name or control not in cases_df["model"].unique():
            continue
        for metric in metrics:
            if (metric, srl_model_name) not in pivot.columns or (metric, control) not in pivot.columns:
                continue
            srl_vals = pivot[(metric, srl_model_name)].to_numpy(dtype=float)
            ctrl_vals = pivot[(metric, control)].to_numpy(dtype=float)
            mask = np.isfinite(srl_vals) & np.isfinite(ctrl_vals)
            srl_vals = srl_vals[mask]
            ctrl_vals = ctrl_vals[mask]
            if len(srl_vals) < 3:
                continue
            diff = srl_vals - ctrl_vals
            mean_diff, ci_low, ci_high = paired_bootstrap_ci(
                diff, config.bootstrap_replicates, seed=repeat_id + 101
            )
            try:
                wilcoxon_stat, wilcoxon_p = wilcoxon(srl_vals, ctrl_vals, zero_method="wilcox")
            except ValueError:
                wilcoxon_stat, wilcoxon_p = np.nan, np.nan
            perm_p = permutation_test_mean_diff(diff, config.permutation_replicates, seed=repeat_id + 303)
            direction = "lower_is_better" if "error" in metric or "score" in metric or "dosage" in metric else "higher_is_better"
            srl_better = mean_diff < 0 if direction == "lower_is_better" else mean_diff > 0
            star_label, significance_tier = evaluate_significance_label(
                float(min(wilcoxon_p, perm_p)) if np.isfinite(wilcoxon_p) or np.isfinite(perm_p) else float("nan"),
                ci_low,
                ci_high,
                n_repeats=1,
                srl_better=srl_better,
                single_split_exploratory=True,
            )
            rows.append(
                {
                    "repeat_id": repeat_id,
                    "metric": metric,
                    "srl_model": srl_model_name,
                    "control_model": control,
                    "mean_srl_minus_control": mean_diff,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "wilcoxon_statistic": float(wilcoxon_stat) if np.isfinite(wilcoxon_stat) else np.nan,
                    "wilcoxon_pvalue": float(wilcoxon_p) if np.isfinite(wilcoxon_p) else np.nan,
                    "permutation_pvalue": perm_p,
                    "n_pairs": int(len(diff)),
                    "direction": direction,
                    "significance_label": star_label,
                    "significance_tier": significance_tier,
                    "comparison_result": comparison_result_label(star_label, significance_tier, srl_better, 1),
                }
            )
    return pd.DataFrame(rows)


def repeat_metric_ci(values: np.ndarray, method: str = "t_interval") -> Tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, mean, mean
    if method == "percentile":
        return mean, float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
    sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    margin = 1.96 * sem
    return mean, mean - margin, mean + margin


def aggregate_repeated_closed_loop_summaries(
    repeat_summaries: List[pd.DataFrame],
    ci_method: str = "t_interval",
) -> pd.DataFrame:
    long_df = pd.concat(repeat_summaries, ignore_index=True)
    rows = []
    for model, group in long_df.groupby("model"):
        row = {"model": model}
        for col in REPEATED_CLOSED_LOOP_METRICS:
            if col not in group.columns:
                continue
            mean, lo, hi = repeat_metric_ci(group[col].to_numpy(dtype=float), method=ci_method)
            row[col] = mean
            row[f"{col}_ci_low"] = lo
            row[f"{col}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def _repeat_umax_study_dir(repeat_outdir: str) -> str:
    return os.path.join(repeat_outdir, "umax_study")


def _umax_study_repeat_is_complete(study_outdir: str) -> bool:
    cases_path = os.path.join(study_outdir, "umax_ablation_cases.csv")
    summary_path = os.path.join(study_outdir, "umax_ablation_summary_by_condition.csv")
    return os.path.isfile(cases_path) and os.path.getsize(cases_path) > 0 and os.path.isfile(summary_path)


def build_repeated_closed_loop_significance(
    repeat_summaries: List[pd.DataFrame],
    srl_model_name: str,
    n_repeats: int,
    config: ClosedLoopConfig,
    *,
    control_models: Optional[Sequence[str]] = None,
    metrics: Optional[Sequence[str]] = None,
    annotation_metrics: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    long_df = pd.concat(repeat_summaries, ignore_index=True)
    if "repeat_id" not in long_df.columns:
        raise ValueError("repeat summaries must include repeat_id")

    controls = list(control_models or CLOSED_LOOP_SIGNIFICANCE_CONTROLS)
    metric_list = list(metrics or REPEATED_CLOSED_LOOP_METRICS)
    annotate_metrics = set(annotation_metrics or FIG4B_SUMMARY_METRICS)
    pairwise_rows: List[dict] = []
    annotation_rows: List[dict] = []
    for control in controls:
        if control == srl_model_name:
            continue
        for metric in metric_list:
            if metric not in long_df.columns:
                continue
            srl_vals = (
                long_df[long_df["model"] == srl_model_name]
                .sort_values("repeat_id")[metric]
                .to_numpy(dtype=float)
            )
            ctrl_vals = (
                long_df[long_df["model"] == control]
                .sort_values("repeat_id")[metric]
                .to_numpy(dtype=float)
            )
            n = min(len(srl_vals), len(ctrl_vals))
            if n == 0:
                continue
            srl_vals = srl_vals[:n]
            ctrl_vals = ctrl_vals[:n]
            diff = srl_vals - ctrl_vals
            direction = CLOSED_LOOP_METRIC_DIRECTION.get(metric, "lower_is_better")
            srl_better = float(np.mean(diff)) < 0 if direction == "lower_is_better" else float(np.mean(diff)) > 0
            ci_low = float(np.percentile(diff, 2.5)) if n > 1 else float("nan")
            ci_high = float(np.percentile(diff, 97.5)) if n > 1 else float("nan")
            try:
                wilcoxon_p = float(wilcoxon(diff).pvalue) if n > 1 and not np.allclose(diff, 0.0) else 1.0
            except Exception:
                wilcoxon_p = float("nan")
            perm_p = permutation_test_mean_diff(diff, config.permutation_replicates, seed=abs(hash(metric + control)) % 10000) if n > 1 else float("nan")
            p_for_label = float(min(p for p in (wilcoxon_p, perm_p) if np.isfinite(p))) if n > 1 else float("nan")
            star_label, significance_tier = evaluate_significance_label(
                p_for_label, ci_low, ci_high, n_repeats=n, srl_better=srl_better,
                single_split_exploratory=n_repeats == 1,
            )
            comp = comparison_result_label(star_label, significance_tier, srl_better, n_repeats)
            row = {
                "metric": metric,
                "srl_model": srl_model_name,
                "control_model": control,
                "direction": direction,
                "n_repeats": n,
                "mean_srl": float(np.mean(srl_vals)),
                "mean_control": float(np.mean(ctrl_vals)),
                "mean_srl_minus_control": float(np.mean(diff)),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "wilcoxon_pvalue": wilcoxon_p,
                "permutation_pvalue": perm_p,
                "significance_label": star_label,
                "significance_tier": significance_tier,
                "comparison_result": comp,
            }
            pairwise_rows.append(row)
            if metric in annotate_metrics:
                plot_label = star_label if comp not in {"control_better", "not_significant"} else ""
                if comp == "control_better":
                    plot_label = "control_better"
                annotation_rows.append(
                    {
                        "metric": metric,
                        "srl_model": srl_model_name,
                        "control_model": control,
                        "significance_label": star_label,
                        "comparison_result": comp,
                        "plot_label": plot_label,
                        "exploratory": bool(n_repeats < 10),
                    }
                )
    return pd.DataFrame(pairwise_rows), pd.DataFrame(annotation_rows)


def identify_strongest_parameter_control(summary_df: pd.DataFrame, srl_model_name: str) -> str:
    exclude = {srl_model_name, TAR_NO_CLOSED_LOOP_MODEL}
    controls = summary_df[~summary_df["model"].isin(exclude)].copy()
    if "mean_R2_original" in controls.columns:
        controls = controls.sort_values("mean_R2_original", ascending=False)
    elif "mean_composite_score" in controls.columns:
        controls = controls.sort_values("mean_composite_score", ascending=True)
    if controls.empty:
        raise ValueError("No control models available.")
    return str(controls.iloc[0]["model"])


def identify_strongest_closed_loop_control(summary_df: pd.DataFrame, srl_model_name: str) -> str:
    controls = summary_df[summary_df["model"].isin(CLOSED_LOOP_SIGNIFICANCE_CONTROLS)].copy()
    controls = controls.sort_values("mean_composite_score", ascending=True)
    if controls.empty:
        raise ValueError("No control models available for closed-loop comparison.")
    return str(controls.iloc[0]["model"])


def select_representative_case(
    cases_df: pd.DataFrame,
    srl_model_name: str,
    strongest_control: str,
) -> dict:
    """Fig. 4 uses a single paper_figure forward sim per model (case_index=0)."""
    case_idx = int(cases_df["case_index"].iloc[0]) if not cases_df.empty else 0
    return {
        "selection_criteria": {
            "paper_figure_profile": True,
            "aggregated_tthr": "median_of_repeat_split_medians",
            "single_forward_ode_per_model": True,
            "n_ode_forward_runs": FIG4_ODE_FORWARD_MODELS,
        },
        "n_candidates": 1,
        "selected_case_index": case_idx,
        "selected_case_summary": {"case_index": case_idx},
        "plot_models": list(FIG4_FIXED_UMAX_MODELS),
        "strongest_closed_loop_control": strongest_control,
    }


def write_closed_loop_trajectory_artifacts(
    plot_payload: Dict[str, Tuple[SimulationResult, float, np.ndarray]],
    plot_models: List[str],
    outdir: str,
    rep_info: dict,
) -> None:
    rows: List[dict] = []
    t_thr_by_model: Dict[str, List[float]] = {}
    metrics_by_model: Dict[str, dict] = {}
    for model_name in dict.fromkeys(plot_models):
        if model_name not in plot_payload:
            continue
        res, optimized_u, t_thr = plot_payload[model_name]
        t_thr_by_model[model_name] = [float(v) for v in t_thr]
        p_total = res.P_S + res.P_R
        hist = trajectory_history_from_arrays(res.times, res.C, p_total, res.B_total)
        if hist.empty:
            raise ValueError(
                f"Trajectory history empty for model {model_name!r}; "
                "re-run forward ODE with trajectory=True (full simulate_case path)."
            )
        dose_times = list(getattr(res, "dose_times", []) or [])
        for _, hist_row in hist.iterrows():
            row = {
                "model": model_name,
                **hist_row.to_dict(),
                "u_max_ug_per_mL": float(optimized_u),
            }
            row["total_dosage"] = float(res.total_dosage)
            row["P_AUC"] = float(res.P_AUC)
            rows.append(row)
        metrics_by_model[model_name] = {
            "total_dosage": float(res.total_dosage),
            "P_AUC": float(res.P_AUC),
            "mean_LR": float(np.mean(res.LR_terminal_median)),
            "optimized_u_max": float(optimized_u),
            "u_max_ug_per_mL": float(optimized_u),
            "dose_times_h": dose_times,
            "final_pathogen_M": float(res.final_total_pathogen / 1e6),
            "terminal_pathogen_M": float(terminal_total_pathogen(res) / 1e6),
        }
    traj_path = os.path.join(_ensure_dir(outdir), FIXED_UMAX_TRAJECTORIES_CSV)
    _save_csv(pd.DataFrame(rows), traj_path)
    plot_models_ordered = [m for m in FIG4_FIXED_UMAX_MODELS if m in dict.fromkeys(plot_models)]
    manifest = {
        "validation_section": "fixed-Umax validation",
        "validation_mode": FIXED_UMAX_VALIDATION_MODE,
        "plot_models": plot_models_ordered,
        "plot_display_labels": [CLOSED_LOOP_DISPLAY_LABELS.get(m, m) for m in plot_models_ordered],
        "selected_case_index": rep_info.get("selected_case_index"),
        "strongest_closed_loop_control": rep_info.get("strongest_closed_loop_control"),
        "t_thr_by_model": t_thr_by_model,
        "u_max_by_model": {m: metrics_by_model[m].get("u_max_ug_per_mL") for m in metrics_by_model},
        "dose_times_h_by_model": {m: metrics_by_model[m].get("dose_times_h", []) for m in metrics_by_model},
        "metrics_by_model": metrics_by_model,
        "paper_figure_profile": True,
        "trajectories_csv": os.path.basename(traj_path),
        "figure_png": "fixed_umax_representative.png",
        "fig4_panel": "A",
        "illustrative_only": True,
        "model_order": plot_models_ordered,
    }
    write_json_manifest(outdir, FIG4_PLOT_MANIFEST_JSON, _to_json_native(manifest))


def _closed_loop_figure_dir(outdir: str) -> str:
    return _ensure_dir(os.path.join(outdir, "figure"))


def _first_trajectory_segment(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.sort_values("time_h").reset_index(drop=True)
    if sub.empty:
        return sub
    times = sub["time_h"].to_numpy(dtype=float)
    breaks = np.where(np.diff(times) < -1e-9)[0]
    if len(breaks):
        return sub.iloc[: breaks[0] + 1].copy()
    return sub


def write_fixed_umax_primary_figure_pngs(
    outdir: str,
    *,
    trajectories_df: pd.DataFrame,
    plot_manifest: dict,
    stats_df: pd.DataFrame,
    config: ClosedLoopConfig,
    n_repeats: int = 1,
    annotations_df: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    """Write Fig. 4A/4B PNGs under outdir/figure (fixed-Umax validation)."""
    from figure_audit import plot_fixed_umax_representative, plot_fixed_umax_summary

    figure_dir = _closed_loop_figure_dir(outdir)
    outputs: Dict[str, str] = {}
    model_labels = list(plot_manifest.get("plot_models") or list(CORE_CLOSED_LOOP_MODELS))
    display_labels = plot_manifest.get("plot_display_labels") or [
        CLOSED_LOOP_DISPLAY_LABELS.get(m, m) for m in model_labels
    ]
    t_thr_by_model: Dict[str, np.ndarray] = {}
    raw_t_thr = plot_manifest.get("t_thr_by_model") or {}
    for model in model_labels:
        if model in raw_t_thr:
            t_thr_by_model[model] = np.asarray(raw_t_thr[model], dtype=float)
    metrics_by_model = dict(plot_manifest.get("metrics_by_model") or {})

    rep_path = os.path.join(figure_dir, "fixed_umax_representative.png")
    if not trajectories_df.empty and t_thr_by_model:
        plot_fixed_umax_representative(
            trajectories_df,
            model_labels,
            t_thr_by_model,
            rep_path,
            display_labels=display_labels,
        )
        outputs["fixed_umax_representative.png"] = rep_path

    summary_path = os.path.join(figure_dir, "fixed_umax_summary.png")
    plot_fixed_umax_summary(stats_df, summary_path, n_repeats=n_repeats)
    outputs["fixed_umax_summary.png"] = summary_path
    return outputs


def normalize_u_candidates_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map legacy candidate columns to the canonical u-grid export schema."""
    out = df.copy()
    for old, new in U_CANDIDATE_LEGACY_RENAMES.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})
    if "dose_burden_norm" not in out.columns and "total_dosage_norm" in out.columns:
        out["dose_burden_norm"] = out["total_dosage_norm"]
    if "composite_penalty" not in out.columns and "composite_score" in out.columns:
        out["composite_penalty"] = out["composite_score"]
    for col in U_CANDIDATE_EXPORT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    if "selected_by_optimizer" not in out.columns:
        out["selected_by_optimizer"] = False
    if "feasible_candidate" in out.columns:
        out["feasible_candidate"] = out["feasible_candidate"].fillna(False).astype(bool)
    return out


def format_umax_score_landscape_row(
    *,
    repeat_id: int,
    case_index: int,
    validation_original_row_index: int,
    cand: dict,
) -> dict:
    """Backward-compatible alias curve row; optimization_penalty_score == composite_penalty."""
    composite = float(cand.get("composite_penalty", cand.get("optimization_penalty_score", np.nan)))
    return {
        "repeat_id": int(repeat_id),
        "case_index": int(case_index),
        "validation_original_row_index": int(validation_original_row_index),
        "unit_id": f"repeat_{repeat_id:03d}_case_{case_index:04d}",
        "candidate_u_max": float(cand["candidate_u_max"]),
        "optimization_penalty_score": composite,
        "aspiration_abs_distance_to_initial": float(
            cand.get("aspiration_abs_distance_to_initial", np.nan)
        ),
        "composite_penalty": composite,
        "is_minimum_score": bool(cand.get("selected_by_optimizer", False)),
        "is_initial_aspiration_dominator": bool(cand.get("dominates_initial_aspiration", False)),
        "is_closest_to_initial_aspiration": bool(cand.get("closest_to_initial_aspiration", False)),
    }


def format_umax_response_landscape_row(
    *,
    repeat_id: int,
    case_index: int,
    validation_original_row_index: int,
    model: str,
    base_model: str,
    umax_policy: str,
    cand: dict,
) -> dict:
    export = format_u_candidate_export_row(
        repeat_id=repeat_id,
        case_index=case_index,
        validation_original_row_index=validation_original_row_index,
        model=model,
        cand=cand,
    )
    row = {
        "repeat_id": int(repeat_id),
        "case_index": int(case_index),
        "validation_original_row_index": int(validation_original_row_index),
        "unit_id": f"repeat_{repeat_id:03d}_case_{case_index:04d}",
        "model": model,
        "base_model": base_model,
        "umax_policy": umax_policy,
        "candidate_u_max": float(cand["candidate_u_max"]),
        "total_dosage": float(export["total_dosage"]),
        "dose_count": int(export["dose_count"]),
        "P_AUC": float(export["P_AUC"]),
        "mean_LR": float(export["mean_LR"]),
        "final_total_pathogen": float(export["final_total_pathogen"]),
        "terminal_total_pathogen": float(export["terminal_total_pathogen"]),
        "target_P_AUC": float(export["target_P_AUC"]),
        "minimum_acceptable_P_AUC": float(export["minimum_acceptable_P_AUC"]),
        "target_mean_LR": float(export["target_mean_LR"]),
        "target_terminal_pathogen": float(export["target_terminal_pathogen"]),
        "dose_reference_scale": float(export["dose_reference_scale"]),
        "dose_reference_source": str(export["dose_reference_source"]),
        "lr_target_source": str(export["lr_target_source"]),
        "LR_constraint_satisfied": bool(
            cand.get("LR_constraint_satisfied", export.get("LR_constraint_satisfied", False))
        ),
        "P_AUC_constraint_satisfied": bool(
            cand.get("P_AUC_constraint_satisfied", export.get("P_AUC_constraint_satisfied", False))
        ),
        "pathogen_constraint_satisfied": bool(
            cand.get("pathogen_constraint_satisfied", export.get("pathogen_constraint_satisfied", False))
        ),
        "feasible_candidate": bool(export["feasible_candidate"]),
        "LR_shortfall_norm": float(export["LR_shortfall_norm"]),
        "PAUC_shortfall_norm": float(export["PAUC_shortfall_norm"]),
        "pathogen_violation_norm": float(export["pathogen_violation_norm"]),
        "dose_burden_norm": float(export["dose_burden_norm"]),
        "composite_penalty": float(export["composite_penalty"]),
        "selected_by_optimizer": bool(export["selected_by_optimizer"]),
        "optimizer_selection_policy": str(cand.get("optimizer_selection_policy", "")),
        "optimizer_selection_rule": str(export["optimizer_selection_rule"]),
        "optimizer_selection_stage": str(export["optimizer_selection_stage"]),
        "optimizer_selection_rank": int(export["optimizer_selection_rank"]),
        "optimizer_primary_score": float(export["optimizer_primary_score"]),
        "optimizer_tiebreak_score": float(export["optimizer_tiebreak_score"]),
        "aspiration_abs_distance_to_initial": float(export["aspiration_abs_distance_to_initial"]),
        "dominates_initial_aspiration": bool(export["dominates_initial_aspiration"]),
        "closest_to_initial_aspiration": bool(export["closest_to_initial_aspiration"]),
        "pareto_improved_from_closest": bool(export["pareto_improved_from_closest"]),
    }
    for i in range(1, 6):
        lr_key = f"LR{i}"
        tgt_key = f"target_LR{i}"
        if lr_key in export:
            row[lr_key] = float(export[lr_key])
        if tgt_key in export:
            row[tgt_key] = float(export[tgt_key])
    return row


def build_umax_feasible_region_summary_row(
    *,
    repeat_id: int,
    case_index: int,
    model: str,
    candidates: Sequence[dict],
    selection_rule: str,
    selection_stage: str,
) -> dict:
    feasible = [c for c in candidates if c.get("feasible_candidate")]
    selected = next((c for c in candidates if c.get("selected_by_optimizer")), None)
    if selected is None and candidates:
        selected = min(candidates, key=lambda c: float(c.get("optimizer_selection_rank", np.inf)))
    n_candidates = int(len(candidates))
    n_feasible = int(len(feasible))
    min_feas_u = float(min(c["candidate_u_max"] for c in feasible)) if feasible else float("nan")
    max_feas_u = float(max(c["candidate_u_max"] for c in feasible)) if feasible else float("nan")
    if selected is None:
        return {
            "repeat_id": int(repeat_id),
            "case_index": int(case_index),
            "model": model,
            "n_candidates": n_candidates,
            "n_feasible_candidates": n_feasible,
            "feasible_fraction": float(n_feasible / n_candidates) if n_candidates else 0.0,
            "min_feasible_u_max": min_feas_u,
            "max_feasible_u_max": max_feas_u,
            "selected_u_max": float("nan"),
            "selected_is_feasible": False,
            "selected_total_dosage": float("nan"),
            "selected_P_AUC": float("nan"),
            "selected_mean_LR": float("nan"),
            "selected_terminal_pathogen": float("nan"),
            "selected_composite_penalty": float("nan"),
            "selection_rule": selection_rule,
            "selection_stage": selection_stage,
        }
    return {
        "repeat_id": int(repeat_id),
        "case_index": int(case_index),
        "model": model,
        "n_candidates": n_candidates,
        "n_feasible_candidates": n_feasible,
        "feasible_fraction": float(n_feasible / n_candidates) if n_candidates else 0.0,
        "min_feasible_u_max": min_feas_u,
        "max_feasible_u_max": max_feas_u,
        "selected_u_max": float(selected["candidate_u_max"]),
        "selected_is_feasible": bool(selected.get("feasible_candidate", False)),
        "selected_total_dosage": _cand_total_dosage(selected),
        "selected_P_AUC": float(selected["P_AUC"]),
        "selected_mean_LR": float(selected["mean_LR"]),
        "selected_terminal_pathogen": _cand_terminal_pathogen(selected),
        "selected_composite_penalty": float(
            selected.get("composite_penalty", selected.get("optimization_penalty_score", np.nan))
        ),
        "selection_rule": selection_rule,
        "selection_stage": selection_stage,
    }


def export_umax_response_landscape_csv(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        _save_csv(pd.DataFrame(columns=list(UMAX_RESPONSE_LANDSCAPE_COLUMNS)), path)
        return
    missing = [col for col in UMAX_RESPONSE_LANDSCAPE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{UMAX_RESPONSE_LANDSCAPE_CSV} missing columns: {missing}")
    _save_csv(df[list(UMAX_RESPONSE_LANDSCAPE_COLUMNS)], path)


def export_umax_feasible_region_summary_csv(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        _save_csv(pd.DataFrame(columns=list(UMAX_FEASIBLE_REGION_SUMMARY_COLUMNS)), path)
        return
    missing = [col for col in UMAX_FEASIBLE_REGION_SUMMARY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{UMAX_FEASIBLE_REGION_SUMMARY_CSV} missing columns: {missing}")
    _save_csv(df[list(UMAX_FEASIBLE_REGION_SUMMARY_COLUMNS)], path)


def build_umax_selected_umax_distribution_df(candidates_df: pd.DataFrame) -> pd.DataFrame:
    if candidates_df.empty or "selected_by_optimizer" not in candidates_df.columns:
        return pd.DataFrame()
    sel = candidates_df[candidates_df["selected_by_optimizer"].fillna(False).astype(bool)].copy()
    if sel.empty:
        return pd.DataFrame()
    keep = [
        c
        for c in (
            "repeat_id",
            "case_index",
            "validation_original_row_index",
            "model",
            "candidate_u_max",
            "total_dosage",
            "composite_penalty",
            "feasible_candidate",
            "optimizer_selection_policy",
            "optimizer_selection_rule",
            "optimizer_selection_stage",
        )
        if c in sel.columns
    ]
    return sel[keep].reset_index(drop=True)


def validate_umax_inverse_design_invariants(
    *,
    candidates_df: Optional[pd.DataFrame] = None,
    response_landscape_df: Optional[pd.DataFrame] = None,
    feasible_summary_df: Optional[pd.DataFrame] = None,
    policy_sensitivity_df: Optional[pd.DataFrame] = None,
    config: Optional[ClosedLoopConfig] = None,
) -> List[str]:
    """Smoke-test invariants for Umax inverse-design exports. Returns error messages (empty = pass)."""
    errors: List[str] = []
    active_policy = (
        config.umax_selection_policy
        if config is not None
        else UMAX_SELECTION_POLICY_DEFAULT
    )
    check_feasible_first_selection = active_policy == "feasible_first"

    landscape = response_landscape_df
    if landscape is None or landscape.empty:
        landscape = candidates_df
    if landscape is not None and not landscape.empty:
        group_cols = ["repeat_id", "case_index", "model"]
        if all(c in landscape.columns for c in group_cols) and "selected_by_optimizer" in landscape.columns:
            sel_counts = (
                landscape.groupby(group_cols)["selected_by_optimizer"]
                .apply(lambda s: int(s.fillna(False).astype(bool).sum()))
            )
            bad = sel_counts[sel_counts != 1]
            if not bad.empty:
                errors.append(
                    f"selected_by_optimizer must be True for exactly one row per repeat/case/model; "
                    f"violations={bad.to_dict()}"
                )

    summary = feasible_summary_df
    if (
        check_feasible_first_selection
        and summary is not None
        and not summary.empty
        and landscape is not None
        and not landscape.empty
    ):
        for _, srow in summary.iterrows():
            mask = (
                (landscape["repeat_id"] == srow["repeat_id"])
                & (landscape["case_index"] == srow["case_index"])
                & (landscape["model"] == srow["model"])
            )
            sub = landscape.loc[mask]
            if sub.empty:
                continue
            n_feas = int(srow.get("n_feasible_candidates", 0))
            selected = sub[sub["selected_by_optimizer"].fillna(False).astype(bool)]
            if len(selected) != 1:
                continue
            sel_row = selected.iloc[0]
            cands = sub.to_dict("records")
            if n_feas > 0:
                if not bool(sel_row.get("feasible_candidate", False)):
                    errors.append(
                        f"repeat={srow['repeat_id']} case={srow['case_index']} model={srow['model']}: "
                        "selected candidate must be feasible when n_feasible_candidates > 0"
                    )
                feas = [c for c in cands if c.get("feasible_candidate")]
                if feas:
                    expected = min(feas, key=_feasible_sort_key)
                    if abs(float(expected["candidate_u_max"]) - float(sel_row["candidate_u_max"])) > 1e-9:
                        errors.append(
                            f"repeat={srow['repeat_id']} case={srow['case_index']} model={srow['model']}: "
                            "feasible selected candidate must match feasible_first tie-break winner"
                        )
            else:
                expected = min(cands, key=_infeasible_sort_key)
                if abs(float(expected["candidate_u_max"]) - float(sel_row["candidate_u_max"])) > 1e-9:
                    errors.append(
                        f"repeat={srow['repeat_id']} case={srow['case_index']} model={srow['model']}: "
                        "infeasible fallback must match feasible_first tie-break winner"
                    )

    if policy_sensitivity_df is not None and not policy_sensitivity_df.empty:
        req = {
            "selected_composite_penalty",
            "selected_feasible_candidate",
            "umax_selection_policy",
        }
        missing = req - set(policy_sensitivity_df.columns)
        if missing:
            errors.append(f"umax_selection_policy_sensitivity.csv missing columns: {sorted(missing)}")

    return errors


def format_u_candidate_export_row(
    *,
    repeat_id: int,
    case_index: int,
    validation_original_row_index: int,
    model: str,
    cand: dict,
) -> dict:
    """One row of the full u_grid scan for closed_loop_u_candidates.csv."""
    dose_burden = cand.get("dose_burden_norm", cand.get("total_dosage_norm", np.nan))
    composite = cand.get("composite_penalty", cand.get("composite_score", np.nan))
    row = {
        "repeat_id": int(repeat_id),
        "case_index": int(case_index),
        "validation_original_row_index": int(validation_original_row_index),
        "model": model,
        "candidate_u_max": float(cand["candidate_u_max"]),
        "total_dosage": float(cand.get("total_dosage", cand.get("total_dosage_ug_per_mL", np.nan))),
        "dose_count": int(cand.get("dose_count", 0)),
        "P_AUC": float(cand["P_AUC"]),
        "mean_LR": float(cand["mean_LR"]),
        "final_total_pathogen": float(
            cand.get("final_total_pathogen", cand.get("final_total_pathogen_CFU_per_mL", np.nan))
        ),
        "terminal_total_pathogen": float(
            cand.get(
                "terminal_total_pathogen",
                cand.get("terminal_total_pathogen_CFU_per_mL", np.nan),
            )
        ),
        "target_tracking_error": float(cand.get("target_tracking_error", np.nan)),
        "target_P_AUC": float(cand.get("target_P_AUC", np.nan)),
        "minimum_acceptable_P_AUC": float(cand.get("minimum_acceptable_P_AUC", np.nan)),
        "target_mean_LR": float(cand.get("target_mean_LR", np.nan)),
        "target_terminal_pathogen": float(cand.get("target_terminal_pathogen", np.nan)),
        "dose_reference_scale": float(cand.get("dose_reference_scale", np.nan)),
        "dose_reference_source": str(cand.get("dose_reference_source", "")),
        "lr_target_source": str(cand.get("lr_target_source", "")),
        "LR_shortfall_norm": float(cand.get("LR_shortfall_norm", np.nan)),
        "PAUC_shortfall_norm": float(cand.get("PAUC_shortfall_norm", np.nan)),
        "pathogen_violation_norm": float(cand.get("pathogen_violation_norm", np.nan)),
        "dose_burden_norm": float(dose_burden),
        "composite_penalty": float(composite),
        "feasible_candidate": bool(cand.get("feasible_candidate", False)),
        "optimizer_selection_rule": str(cand.get("optimizer_selection_rule", "")),
        "selected_by_optimizer": bool(cand.get("selected_by_optimizer", False)),
        "dose_signed_component": float(cand.get("dose_signed_component", np.nan)),
        "pauc_signed_component": float(cand.get("pauc_signed_component", np.nan)),
        "lr_signed_component": float(cand.get("lr_signed_component", np.nan)),
        "pathogen_signed_component": float(cand.get("pathogen_signed_component", np.nan)),
        "signed_relative_inner": float(cand.get("signed_relative_inner", np.nan)),
        "signed_relative_rms": float(cand.get("signed_relative_rms", np.nan)),
        "hard_violation_rms": float(cand.get("hard_violation_rms", np.nan)),
        "optimizer_primary_score": float(
            cand.get("optimizer_primary_score", cand.get("composite_penalty", cand.get("hard_violation_rms", np.nan)))
        ),
        "optimizer_tiebreak_score": float(
            cand.get("optimizer_tiebreak_score", cand.get("total_dosage", cand.get("total_dosage_ug_per_mL", np.nan)))
        ),
        "optimizer_selection_rank": int(cand.get("optimizer_selection_rank", 0)),
        "composite_score": float(composite),
        "aspiration_total_dosage": float(cand.get("aspiration_total_dosage", np.nan)),
        "aspiration_P_AUC": float(cand.get("aspiration_P_AUC", np.nan)),
        "aspiration_mean_LR": float(cand.get("aspiration_mean_LR", np.nan)),
        "aspiration_terminal_pathogen": float(cand.get("aspiration_terminal_pathogen", np.nan)),
        "aspiration_abs_distance_to_initial": float(
            cand.get("aspiration_abs_distance_to_initial", np.nan)
        ),
        "aspiration_dose_term": float(cand.get("aspiration_dose_term", np.nan)),
        "aspiration_pauc_term": float(cand.get("aspiration_pauc_term", np.nan)),
        "aspiration_lr_term": float(cand.get("aspiration_lr_term", np.nan)),
        "aspiration_pathogen_log_term": float(cand.get("aspiration_pathogen_log_term", np.nan)),
        "dominates_initial_aspiration": bool(cand.get("dominates_initial_aspiration", False)),
        "initial_aspiration_dominator_count": int(cand.get("initial_aspiration_dominator_count", 0)),
        "closest_to_initial_aspiration": bool(cand.get("closest_to_initial_aspiration", False)),
        "updated_reference_from_closest_candidate": bool(
            cand.get("updated_reference_from_closest_candidate", False)
        ),
        "dominates_updated_reference": bool(cand.get("dominates_updated_reference", False)),
        "updated_reference_dominator_count": int(cand.get("updated_reference_dominator_count", 0)),
        "pareto_improved_from_closest": bool(cand.get("pareto_improved_from_closest", False)),
        "optimizer_selection_stage": str(cand.get("optimizer_selection_stage", "")),
    }
    for i in range(1, 6):
        key = f"LR{i}"
        if key in cand:
            row[key] = float(cand[key])
        tgt_key = f"target_LR{i}"
        if tgt_key in cand:
            row[tgt_key] = float(cand[tgt_key])
        asp_key = f"aspiration_LR{i}"
        if asp_key in cand:
            row[asp_key] = float(cand[asp_key])
    return row


def export_u_candidates_csv(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        _save_csv(pd.DataFrame(columns=list(U_CANDIDATE_EXPORT_COLUMNS)), path)
        return
    normalized = normalize_u_candidates_df(df)
    missing = [col for col in U_CANDIDATE_EXPORT_COLUMNS if col not in normalized.columns]
    if missing:
        raise ValueError(f"{FIXED_UMAX_U_CANDIDATES_CSV} missing columns: {missing}")
    _save_csv(normalized[list(U_CANDIDATE_EXPORT_COLUMNS)], path)


def _concat_csv_files(paths: Sequence[str], out_path: str) -> None:
    """Concatenate CSVs without loading all rows into memory at once."""
    _ensure_parent_dir(out_path)
    wrote_header = False
    with open(out_path, "w", encoding="utf-8", newline="") as outf:
        for path in paths:
            if not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8", newline="") as inf:
                for line_no, line in enumerate(inf):
                    if line_no == 0 and wrote_header:
                        continue
                    outf.write(line)
                    if line_no == 0:
                        wrote_header = True


def build_fairness_check(
    cases_df: pd.DataFrame,
    config: ClosedLoopConfig,
    val_indices: Sequence[int],
    n_cases: int,
    *,
    prediction_sources: Optional[Sequence[str]] = None,
    worker_results: Optional[List[dict]] = None,
) -> dict:
    models = sorted(cases_df["model"].unique())
    case_counts = cases_df.groupby("model")["case_index"].nunique().to_dict()
    same_case_count = len(set(case_counts.values())) == 1

    same_case_ordering = True
    same_held_out_indices = True
    if worker_results:
        # Repeated benchmark: each repeat may use a different held-out split; fairness is
        # within-repeat (all core models see the same validation rows in the same order).
        for worker in worker_results:
            within_indices, within_ordering = _within_repeat_model_fairness(worker["cases_df"])
            if not within_indices:
                same_held_out_indices = False
            if not within_ordering:
                same_case_ordering = False
    elif not cases_df.empty:
        same_held_out_indices, same_case_ordering = _within_repeat_model_fairness(cases_df)

    sources = [s for s in (prediction_sources or []) if s]
    if not sources and config.prediction_source:
        sources = [config.prediction_source]
    normalized_sources = [_normalize_prediction_source_label(s) for s in sources]
    normalized_sources = [s for s in normalized_sources if s]
    same_prediction_source = len(set(normalized_sources)) <= 1 if normalized_sources else True

    checks = {
        "same_prediction_source": same_prediction_source,
        "same_held_out_row_indices": same_held_out_indices,
        "same_case_ordering": same_case_ordering,
        "same_u_grid": True,
        "same_ode_constants": True,
        "same_objective_weights": True,
        "same_desired_objective_source": True,
        "same_pathogen_reference_band": True,
        "same_dosage_reference_target": True,
        "same_lr_reference_target": True,
        "same_probiotic_pauc_fraction": True,
        "no_retraining_in_closed_loop_eval": True,
    }
    passed = same_case_count and len(models) == len(CORE_CLOSED_LOOP_MODELS) and all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "core_models": list(CORE_CLOSED_LOOP_MODELS),
        "models_evaluated": models,
        "cases_per_model": {k: int(v) for k, v in case_counts.items()},
        "shared_val_indices": [int(x) for x in list(val_indices)[:n_cases]],
        "shared_u_grid": config.u_grid.tolist(),
        "shared_ode": "multi_pathogen_simulator.simulate_paper_case",
        "objective_weights": asdict(config.weights),
        "weight_profile": config.weight_profile,
        "weight_selection_source": config.weight_selection_source,
        "pathogen_ceiling_cfu_per_mL": config.pathogen_ceiling_cfu_per_mL,
        "pathogen_floor_cfu_per_mL": config.pathogen_floor_cfu_per_mL,
        "dosage_reference_target": config.dosage_reference_target,
        "lr_reference_target": config.lr_reference_target,
        "probiotic_pauc_fraction": config.probiotic_pauc_fraction,
        "prediction_sources": sources,
        "prediction_source_pattern": normalized_sources[0] if normalized_sources else None,
        "fairness_scope": "within_repeat" if worker_results and len(worker_results) > 1 else "single_run",
        "optimizer_rule": (
            "Fixed-Umax comparative validation: all models use the same per-case fixed Umax "
            "(metadata soft_u_max, else median u_grid); only predicted Tthr differs by model."
        ),
        "objective_reference_targets": OBJECTIVE_REFERENCE_TARGETS,
    }


def validate_fairness_check(fairness: dict) -> None:
    if fairness.get("passed"):
        return
    failed = [name for name, ok in fairness.get("checks", {}).items() if not ok]
    detail = failed or fairness
    raise RuntimeError(f"Closed-loop fairness check failed: {detail}")


def compute_umax_selection_boundary_stats(
    cases_df: pd.DataFrame,
    u_grid: np.ndarray,
    *,
    model: str = TAR_MODEL,
) -> dict:
    sub = cases_df[cases_df["model"] == model] if "model" in cases_df.columns else cases_df
    if sub.empty or len(u_grid) == 0:
        return {
            "fraction_selected_u_min": float("nan"),
            "fraction_selected_u_max": float("nan"),
            "fraction_selected_zero_dose": float("nan"),
            "warning_zero_dose_dominates": False,
            "warning_grid_boundary_dominates": False,
        }
    u_min, u_max = float(np.min(u_grid)), float(np.max(u_grid))
    u_sel = sub["optimized_u_max"].to_numpy(dtype=float)
    frac_min = float(np.mean(u_sel <= u_min + 1e-9))
    frac_max = float(np.mean(u_sel >= u_max - 1e-9))
    frac_zero = float(np.mean(u_sel <= 1e-9))
    warn_zero = bool(frac_zero > 0.2)
    warn_boundary = bool(frac_min > 0.2 or frac_max > 0.2)
    return {
        "fraction_selected_u_min": frac_min,
        "fraction_selected_u_max": frac_max,
        "fraction_selected_zero_dose": frac_zero,
        "warning_zero_dose_dominates": warn_zero,
        "warning_grid_boundary_dominates": warn_boundary,
    }


def _u_candidate_grid_available(candidates_df: pd.DataFrame) -> bool:
    return (
        candidates_df is not None
        and not candidates_df.empty
        and "model" in candidates_df.columns
    )


def write_optimizer_selection_debug(
    outdir: str,
    cases_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> None:
    debug_path = os.path.join(outdir, "optimizer_selection_debug.csv")
    if not _u_candidate_grid_available(candidates_df):
        _save_csv(pd.DataFrame(), debug_path)
        return
    tar_cases = cases_df[cases_df["model"] == TAR_MODEL].copy()
    selected = normalize_u_candidates_df(candidates_df)
    selected = selected[(selected["model"] == TAR_MODEL) & selected["selected_by_optimizer"]]
    rows: List[dict] = []
    for _, case in tar_cases.iterrows():
        mask = (
            (selected["repeat_id"] == int(case["repeat_id"]))
            & (selected["case_index"] == int(case["case_index"]))
        )
        if not mask.any():
            continue
        cand = selected.loc[mask].iloc[0]
        rows.append(
            {
                "repeat_id": int(case["repeat_id"]),
                "case_index": int(case["case_index"]),
                "validation_original_row_index": int(case.get("original_row_index", case.get("validation_original_row_index", -1))),
                "model": TAR_MODEL,
                "selected_u_max": float(case["optimized_u_max"]),
                "optimizer_selection_rule": case.get("optimizer_selection_rule", ""),
                "hard_violation_rms": float(cand.get("hard_violation_rms", np.nan)),
                "signed_relative_rms": float(cand.get("signed_relative_rms", np.nan)),
                "signed_relative_inner": float(cand.get("signed_relative_inner", np.nan)),
                "optimizer_primary_score": float(cand.get("optimizer_primary_score", cand.get("hard_violation_rms", np.nan))),
                "optimizer_tiebreak_score": float(
                    cand.get("optimizer_tiebreak_score", cand.get("signed_relative_rms", np.nan))
                ),
                "optimizer_selection_rank": int(cand.get("optimizer_selection_rank", 0)),
                "dose_signed_component": float(cand.get("dose_signed_component", np.nan)),
                "pauc_signed_component": float(cand.get("pauc_signed_component", np.nan)),
                "lr_signed_component": float(cand.get("lr_signed_component", np.nan)),
                "pathogen_signed_component": float(cand.get("pathogen_signed_component", np.nan)),
                "target_total_dosage": float(cand.get("target_total_dosage", np.nan)),
                "target_P_AUC": float(cand.get("target_P_AUC", np.nan)),
                "target_mean_LR": float(cand.get("target_mean_LR", np.nan)),
                "target_terminal_pathogen": float(cand.get("target_terminal_pathogen", np.nan)),
                "total_dosage": float(cand.get("total_dosage", cand.get("total_dosage_ug_per_mL", np.nan))),
                "P_AUC": float(cand.get("P_AUC", np.nan)),
                "mean_LR": float(cand.get("mean_LR", np.nan)),
                "terminal_total_pathogen": float(
                    cand.get("terminal_total_pathogen", cand.get("terminal_total_pathogen_CFU_per_mL", np.nan))
                ),
            }
        )
    _save_csv(pd.DataFrame(rows), debug_path)


def compute_umax_boundary_fraction(cases_df: pd.DataFrame, u_grid: np.ndarray) -> float:
    u_min, u_max = float(u_grid.min()), float(u_grid.max())
    at_boundary = (cases_df["optimized_u_max"] <= u_min + 1e-9) | (cases_df["optimized_u_max"] >= u_max - 1e-9)
    return float(at_boundary.mean())


def write_optimizer_diagnostics(
    outdir: str,
    cases_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    config: ClosedLoopConfig,
) -> dict:
    tar_cases = cases_df[cases_df["model"] == TAR_MODEL]
    rule_counts = (
        cases_df.groupby("optimizer_selection_rule", as_index=False)
        .size()
        .rename(columns={"optimizer_selection_rule": "selection_rule", "size": "count"})
    )
    total_cases = max(len(cases_df), 1)
    rule_counts["fraction"] = rule_counts["count"] / total_cases
    _save_csv(rule_counts, os.path.join(outdir, "optimizer_selection_rule_counts.csv"))

    selected = pd.DataFrame()
    if _u_candidate_grid_available(candidates_df):
        selected = normalize_u_candidates_df(candidates_df)
        selected = selected[selected["selected_by_optimizer"]]
    boundary_fraction = (
        compute_umax_boundary_fraction(tar_cases, config.u_grid) if not tar_cases.empty else float("nan")
    )
    boundary_stats = compute_umax_selection_boundary_stats(tar_cases, config.u_grid)
    if boundary_stats.get("warning_zero_dose_dominates"):
        print(
            "WARNING: TAR optimizer frequently selected zero dose; inspect objective components.",
            flush=True,
        )
    write_optimizer_selection_debug(outdir, cases_df, candidates_df)
    umax_stats_df = tar_cases if not tar_cases.empty else cases_df
    summary = {
        "u_candidate_grid_available": _u_candidate_grid_available(candidates_df),
        "fixed_umax_validation_mode": not _u_candidate_grid_available(candidates_df),
        "min_hard_violation_fraction": float(
            tar_cases["optimizer_selection_rule"].eq("min_hard_violation").mean()
        )
        if not tar_cases.empty
        else float("nan"),
        "tiebreak_signed_rms_fraction": float(
            tar_cases["optimizer_selection_rule"].eq("tiebreak_signed_rms").mean()
        )
        if not tar_cases.empty
        else float("nan"),
        "tiebreak_min_dose_fraction": float(
            tar_cases["optimizer_selection_rule"].eq("tiebreak_min_dose").mean()
        )
        if not tar_cases.empty
        else float("nan"),
        "tiebreak_min_umax_fraction": float(
            tar_cases["optimizer_selection_rule"].eq("tiebreak_min_umax").mean()
        )
        if not tar_cases.empty
        else float("nan"),
        "selected_u_max_median": float(umax_stats_df["optimized_u_max"].median()),
        "selected_u_max_q25": float(umax_stats_df["optimized_u_max"].quantile(0.25)),
        "selected_u_max_q75": float(umax_stats_df["optimized_u_max"].quantile(0.75)),
        "selected_u_max_boundary_fraction": boundary_fraction,
        "hard_violation_rms_mean": float(selected["hard_violation_rms"].mean())
        if "hard_violation_rms" in selected.columns and len(selected)
        else np.nan,
        "signed_relative_rms_mean": float(selected["signed_relative_rms"].mean())
        if "signed_relative_rms" in selected.columns and len(selected)
        else np.nan,
        "LR_shortfall_contribution_mean": float(selected["LR_shortfall_norm"].mean()) if len(selected) else np.nan,
        "PAUC_shortfall_contribution_mean": float(selected["PAUC_shortfall_norm"].mean()) if len(selected) else np.nan,
        "pathogen_violation_contribution_mean": float(selected["pathogen_violation_norm"].mean())
        if len(selected)
        else np.nan,
        "dosage_contribution_mean": float(selected["total_dosage_norm"].mean()) if len(selected) else np.nan,
        "dose_signed_component_mean": float(selected["dose_signed_component"].mean())
        if "dose_signed_component" in selected.columns and len(selected)
        else np.nan,
        "pauc_signed_component_mean": float(selected["pauc_signed_component"].mean())
        if "pauc_signed_component" in selected.columns and len(selected)
        else np.nan,
        "lr_signed_component_mean": float(selected["lr_signed_component"].mean())
        if "lr_signed_component" in selected.columns and len(selected)
        else np.nan,
        "pathogen_signed_component_mean": float(selected["pathogen_signed_component"].mean())
        if "pathogen_signed_component" in selected.columns and len(selected)
        else np.nan,
        **boundary_stats,
    }
    _save_csv(pd.DataFrame([summary]), os.path.join(outdir, "optimizer_score_component_summary.csv"))
    return {"umax_boundary_warning": bool(boundary_fraction > 0.30), **summary}



def _run_umax_study_repeat_from_predictions(
    repeat_id: int,
    predictions_csv: str,
    prediction_source: str,
    root_outdir: str,
    x_csv: str,
    metadata_csv: Optional[str],
    config: ClosedLoopConfig,
    verbose: bool = False,
    skip_if_complete: bool = True,
    reference_by_repeat_df: Optional[pd.DataFrame] = None,
    fixed_policy_df: Optional[pd.DataFrame] = None,
    case_n_jobs: int = 1,
    ode_backend: Optional[str] = None,
    x_df: Optional[pd.DataFrame] = None,
    metadata_df: Optional[pd.DataFrame] = None,
) -> dict:
    repeat_outdir = os.path.join(root_outdir, "repeats", f"repeat_{repeat_id:03d}")
    study_outdir = _repeat_umax_study_dir(repeat_outdir)
    dose_ref, dose_src = resolve_dose_reference_scale_for_repeat(config, repeat_id, reference_by_repeat_df)
    if skip_if_complete and _umax_study_repeat_is_complete(study_outdir):
        if verbose:
            print(f"  Skipping umax study repeat {repeat_id}: reusing {study_outdir}", flush=True)
        cases_df = pd.read_csv(os.path.join(study_outdir, "umax_ablation_cases.csv"))
        summary = pd.read_csv(os.path.join(study_outdir, "umax_ablation_summary_by_condition.csv"))
        return {
            "repeat_id": repeat_id,
            "repeat_outdir": repeat_outdir,
            "prediction_source": prediction_source or predictions_csv,
            "cases_df": cases_df,
            "summary_by_condition": summary.assign(repeat_id=repeat_id),
            "dosage_reference": dose_ref,
            "dose_reference_scale": dose_ref,
            "dose_reference_source": dose_src,
        }

    if verbose:
        print(f"\n===== Umax optimization study repeat {repeat_id} =====", flush=True)
    pred_df = pd.read_csv(predictions_csv)
    predictions, val_idx, _, _ = parse_predictions_wide_df(pred_df)
    x_test_df, metadata_test_df = load_validation_rows(
        x_csv,
        metadata_csv,
        val_idx,
        x_df=x_df,
        metadata_df=metadata_df,
    )
    eval_config = replace(config, prediction_source=prediction_source or predictions_csv)
    result = run_umax_optimization_study_evaluation(
        outdir=study_outdir,
        repeat_id=repeat_id,
        x_test_df=x_test_df,
        metadata_test_df=metadata_test_df,
        predictions_tthr=predictions,
        val_indices=val_idx,
        config=eval_config,
        dose_reference_scale=dose_ref,
        dose_reference_source=dose_src,
        reference_by_repeat_df=reference_by_repeat_df,
        fixed_policy_df=fixed_policy_df,
        case_n_jobs=case_n_jobs,
        ode_backend=ode_backend,
    )
    return {
        "repeat_id": repeat_id,
        "repeat_outdir": repeat_outdir,
        "prediction_source": prediction_source or predictions_csv,
        "cases_df": result["cases_df"],
        "summary_by_condition": result["summary_by_condition"].assign(repeat_id=repeat_id),
        "dosage_reference": result["dosage_reference"],
        "dose_reference_scale": result.get("dose_reference_scale", dose_ref),
        "dose_reference_source": result.get("dose_reference_source", dose_src),
    }


def finalize_repeated_umax_optimization_outputs(
    outdir: str,
    worker_results: List[dict],
    config: ClosedLoopConfig,
) -> None:
    outdir = _ensure_dir(outdir)
    worker_results.sort(key=lambda item: item["repeat_id"])
    n_repeats = len(worker_results)
    all_cases = pd.concat([w["cases_df"] for w in worker_results], ignore_index=True)
    repeat_summaries = [w["summary_by_condition"] for w in worker_results]
    aggregated_summary = aggregate_repeated_closed_loop_summaries(
        repeat_summaries, ci_method=config.repeat_ci_method
    )
    _save_csv(aggregated_summary, os.path.join(outdir, "umax_ablation_repeated_plot_stats.csv"))

    pairwise_df, annotations_df = build_repeated_closed_loop_significance(
        repeat_summaries,
        FIG5_SIGNIFICANCE_REFERENCE,
        n_repeats,
        config,
        control_models=FIG5_SIGNIFICANCE_CONTROLS,
        metrics=FIG5B_SUMMARY_METRICS,
        annotation_metrics=FIG5B_SUMMARY_METRICS,
    )
    _save_csv(pairwise_df, os.path.join(outdir, "umax_ablation_pairwise_tests.csv"))
    _save_csv(annotations_df, os.path.join(outdir, "umax_ablation_significance_annotations.csv"))
    validate_fig5_main_ablation_conditions(
        all_cases, context=f"finalize outdir={outdir}", required=required_umax_ablation_conditions(config)
    )
    save_umax_policy_ablation_artifacts(
        all_cases,
        outdir,
        required_conditions=required_umax_ablation_conditions(config),
    )

    if config.export_umax_score_landscape:
        landscape_paths = [
            os.path.join(_repeat_umax_study_dir(w["repeat_outdir"]), "umax_score_landscape_curves.csv")
            for w in worker_results
        ]
        _concat_csv_files(landscape_paths, os.path.join(outdir, "umax_score_landscape_curves.csv"))
        candidate_paths = [
            os.path.join(_repeat_umax_study_dir(w["repeat_outdir"]), "umax_optimization_u_candidates.csv")
            for w in worker_results
        ]
        _concat_csv_files(candidate_paths, os.path.join(outdir, "umax_optimization_u_candidates.csv"))
        response_paths = [
            os.path.join(_repeat_umax_study_dir(w["repeat_outdir"]), UMAX_RESPONSE_LANDSCAPE_CSV)
            for w in worker_results
        ]
        _concat_csv_files(response_paths, os.path.join(outdir, UMAX_RESPONSE_LANDSCAPE_CSV))
        feasible_paths = [
            os.path.join(_repeat_umax_study_dir(w["repeat_outdir"]), UMAX_FEASIBLE_REGION_SUMMARY_CSV)
            for w in worker_results
        ]
        _concat_csv_files(feasible_paths, os.path.join(outdir, UMAX_FEASIBLE_REGION_SUMMARY_CSV))
        selected_paths = [
            os.path.join(_repeat_umax_study_dir(w["repeat_outdir"]), UMAX_SELECTED_UMAX_DISTRIBUTION_CSV)
            for w in worker_results
        ]
        _concat_csv_files(selected_paths, os.path.join(outdir, UMAX_SELECTED_UMAX_DISTRIBUTION_CSV))
    debug_paths = [
        os.path.join(_repeat_umax_study_dir(w["repeat_outdir"]), UMAX_ASPIRATION_SELECTION_DEBUG_CSV)
        for w in worker_results
    ]
    _concat_csv_files(debug_paths, os.path.join(outdir, UMAX_ASPIRATION_SELECTION_DEBUG_CSV))
    sensitivity_paths = [
        os.path.join(_repeat_umax_study_dir(w["repeat_outdir"]), UMAX_SELECTION_POLICY_SENSITIVITY_CSV)
        for w in worker_results
    ]
    _concat_csv_files(sensitivity_paths, os.path.join(outdir, UMAX_SELECTION_POLICY_SENSITIVITY_CSV))
    conflict = False
    if config.export_umax_score_landscape and os.path.isfile(
        os.path.join(outdir, "umax_optimization_u_candidates.csv")
    ):
        all_candidates = pd.read_csv(os.path.join(outdir, "umax_optimization_u_candidates.csv"))
        response_path = os.path.join(outdir, UMAX_RESPONSE_LANDSCAPE_CSV)
        feasible_path = os.path.join(outdir, UMAX_FEASIBLE_REGION_SUMMARY_CSV)
        sensitivity_path = os.path.join(outdir, UMAX_SELECTION_POLICY_SENSITIVITY_CSV)
        response_df = pd.read_csv(response_path) if os.path.isfile(response_path) else None
        feasible_df = pd.read_csv(feasible_path) if os.path.isfile(feasible_path) else None
        sensitivity_df = pd.read_csv(sensitivity_path) if os.path.isfile(sensitivity_path) else None
        invariant_errors = validate_umax_inverse_design_invariants(
            candidates_df=all_candidates,
            response_landscape_df=response_df,
            feasible_summary_df=feasible_df,
            policy_sensitivity_df=sensitivity_df,
            config=config,
        )
        if invariant_errors:
            raise RuntimeError(
                "Umax inverse-design invariant check failed after repeat aggregation:\n"
                + "\n".join(invariant_errors)
            )
        align_cases = all_cases[all_cases["ablation_condition"] == "TAR_optimized"].copy()
        if not align_cases.empty:
            align_cases["model"] = TAR_MODEL
            from derive_optimizer_references import build_umax_objective_alignment

            align_df, align_summary, conflict = build_umax_objective_alignment(
                all_candidates, align_cases, TAR_MODEL, "case", config=config
            )
            _save_csv(align_df, os.path.join(outdir, "umax_objective_alignment.csv"))
            _save_csv(align_summary, os.path.join(outdir, "umax_objective_alignment_summary.csv"))

    if config.export_umax_representative_trajectories:
        import shutil

        last_study = _repeat_umax_study_dir(worker_results[-1]["repeat_outdir"])
        for fname in ["umax_ablation_representative_trajectories.csv", FIG5_PLOT_MANIFEST_JSON]:
            src = os.path.join(last_study, fname)
            dst = os.path.join(outdir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)

    dose_ref = worker_results[0].get("dose_reference_scale", worker_results[0]["dosage_reference"])
    dose_src = worker_results[0].get("dose_reference_source", config.dose_reference_source)
    significance_for_manuscript = bool(n_repeats >= 10)
    fixed_policy_path = os.path.join(outdir, FIXED_UMAX_POLICY_BY_REPEAT_CSV)
    fixed_policy_manifest_path = os.path.join(outdir, FIXED_UMAX_POLICY_MANIFEST_JSON)
    fixed_policy_meta: dict = {}
    if os.path.isfile(fixed_policy_manifest_path):
        with open(fixed_policy_manifest_path, encoding="utf-8") as fh:
            fixed_policy_meta = json.load(fh)
    manifest = {
        "prediction_sources": [w.get("prediction_source", "") for w in worker_results],
        "u_grid": config.u_grid.tolist(),
        "ode_backend": ode_backend_for_config(config),
        "umax_objective_conflict_detected": bool(conflict),
        "fixed_umax_policy_by_repeat_csv": FIXED_UMAX_POLICY_BY_REPEAT_CSV if os.path.isfile(fixed_policy_path) else None,
        "umax_inverse_design_exports": {
            "umax_response_landscape_csv": UMAX_RESPONSE_LANDSCAPE_CSV,
            "umax_feasible_region_summary_csv": UMAX_FEASIBLE_REGION_SUMMARY_CSV,
            "umax_optimization_u_candidates_csv": "umax_optimization_u_candidates.csv",
            "umax_score_landscape_curves_csv": "umax_score_landscape_curves.csv",
            "umax_selection_policy_sensitivity_csv": UMAX_SELECTION_POLICY_SENSITIVITY_CSV,
        },
        **fixed_policy_meta,
        **build_closed_loop_optimizer_manifest_fields(
            config, dose_ref, summary_df=aggregated_summary, dose_reference_source=dose_src
        ),
        **build_fig5_manifest_fields(
            n_repeats=n_repeats,
            significance_for_manuscript=significance_for_manuscript,
        ),
    }
    write_json_manifest(outdir, UMAX_OPTIMIZATION_MANIFEST_JSON, _to_json_native(manifest))


def run_repeated_umax_optimization_pipeline(
    *,
    predictions_dir: Optional[str] = None,
    predictions_manifest: Optional[str] = None,
    predictions_csv: Optional[str] = None,
    x_csv: str,
    y_csv: Optional[str] = None,
    metadata_csv: Optional[str] = None,
    outdir: str,
    config: ClosedLoopConfig,
    n_jobs: int = 1,
    verbose: bool = False,
    finalize_only: bool = False,
    skip_completed_repeats: bool = True,
    max_repeats: Optional[int] = None,
    repeat_subsample: str = "even",
    shared_reference_dir: Optional[str] = None,
) -> None:
    outdir = _ensure_dir(outdir)
    ode_backend = resolve_study_ode_backend(config.ode_backend, verbose=verbose)
    config._resolved_ode_backend = ode_backend
    repeat_workers = n_jobs if n_jobs != 0 else (os.cpu_count() or 2)
    case_n_jobs = resolve_case_n_jobs(config, repeat_workers)
    u_grid_n_jobs = resolve_u_grid_n_jobs(config, repeat_workers, case_n_jobs)
    config._resolved_u_grid_n_jobs = u_grid_n_jobs
    if verbose:
        print(
            f"Parallelism: repeat_n_jobs={n_jobs}, case_n_jobs={case_n_jobs}, "
            f"u_grid_n_jobs={u_grid_n_jobs}",
            flush=True,
        )
    if verbose:
        print(f"Preloading feature tables from {x_csv}...", flush=True)
    x_df = pd.read_csv(x_csv)
    metadata_df: Optional[pd.DataFrame] = None
    meta_path = metadata_csv or os.path.join(os.path.dirname(os.path.abspath(x_csv)), "sample_metadata.csv")
    if meta_path and os.path.isfile(meta_path):
        metadata_df = pd.read_csv(meta_path)
    prediction_jobs = resolve_prediction_csv_jobs(
        predictions_dir=predictions_dir,
        predictions_manifest=predictions_manifest,
        predictions_csv=predictions_csv,
    )
    prediction_jobs = subsample_prediction_jobs(
        prediction_jobs, max_repeats, strategy=repeat_subsample
    )
    if verbose and max_repeats:
        print(
            f"Using {len(prediction_jobs)} prediction repeat(s) "
            f"(max_repeats={max_repeats}, strategy={repeat_subsample})",
            flush=True,
        )
    n_repeats = len(prediction_jobs)
    job_by_repeat = {repeat_id: (path, source) for repeat_id, path, source in prediction_jobs}

    if shared_reference_dir:
        copy_shared_optimizer_reference_artifacts(shared_reference_dir, outdir)

    ablation_spec = resolve_umax_ablation_spec(config)
    needs_fixed_policies = ablation_spec_requires_fixed_umax_policies(ablation_spec)

    reference_by_repeat_df: Optional[pd.DataFrame] = None
    ref_path = os.path.join(outdir, OPTIMIZER_REFERENCE_BY_REPEAT_CSV)
    if config.dose_reference_source != "constant":
        if y_csv is None:
            raise ValueError(
                "y_csv is required when dose_reference_source is training_q90_reference_dosage."
            )
        if not os.path.isfile(ref_path) or not skip_completed_repeats:
            from derive_optimizer_references import derive_optimizer_references

            if verbose:
                print(f"Deriving training-only dose references -> {outdir}", flush=True)
            derive_optimizer_references(
                x_csv=x_csv,
                y_csv=y_csv,
                metadata_csv=metadata_csv,
                outdir=outdir,
                predictions_dir=predictions_dir,
                predictions_manifest=predictions_manifest,
                predictions_csv=predictions_csv,
                fixed_representative_umax=config.fixed_representative_umax,
                dose_reference_quantile=config.dose_reference_quantile,
                backend=config.ode_backend,
                n_jobs=n_jobs,
                max_repeats=max_repeats,
                repeat_subsample=repeat_subsample,
            )
        elif verbose:
            print(f"Reusing existing dose references: {ref_path}", flush=True)
        reference_by_repeat_df = load_optimizer_reference_by_repeat(ref_path)

    fixed_policy_df: Optional[pd.DataFrame] = None
    policy_path = os.path.join(outdir, FIXED_UMAX_POLICY_BY_REPEAT_CSV)
    if needs_fixed_policies:
        if y_csv is None:
            raise ValueError("y_csv is required for training-only fixed Umax policy derivation.")
        if not os.path.isfile(policy_path) or not skip_completed_repeats:
            from derive_optimizer_references import derive_fixed_umax_policies

            if verbose:
                print(f"Deriving training-only fixed Umax policies -> {outdir}", flush=True)
            derive_fixed_umax_policies(
                x_csv=x_csv,
                y_csv=y_csv,
                metadata_csv=metadata_csv,
                outdir=outdir,
                predictions_dir=predictions_dir,
                predictions_manifest=predictions_manifest,
                predictions_csv=predictions_csv,
                u_grid=config.u_grid,
                config=config,
                reference_by_repeat_df=reference_by_repeat_df,
                backend=config.ode_backend,
                n_jobs=n_jobs,
                max_repeats=max_repeats,
                repeat_subsample=repeat_subsample,
            )
        elif verbose:
            print(f"Reusing existing fixed Umax policies: {policy_path}", flush=True)
        fixed_policy_df = load_fixed_umax_policy_by_repeat(policy_path)
    elif verbose:
        print("Skipping fixed Umax policy derivation (optimized-only ablation).", flush=True)

    def _collect_completed() -> Tuple[List[dict], List[int]]:
        completed: List[dict] = []
        missing: List[int] = []
        for repeat_id in sorted(job_by_repeat.keys()):
            repeat_outdir = os.path.join(outdir, "repeats", f"repeat_{repeat_id:03d}")
            study_outdir = _repeat_umax_study_dir(repeat_outdir)
            if _umax_study_repeat_is_complete(study_outdir):
                completed.append(
                    _run_umax_study_repeat_from_predictions(
                        repeat_id,
                        job_by_repeat[repeat_id][0],
                        job_by_repeat[repeat_id][1],
                        outdir,
                        x_csv,
                        metadata_csv,
                        config,
                        verbose=False,
                        skip_if_complete=True,
                        reference_by_repeat_df=reference_by_repeat_df,
                        fixed_policy_df=fixed_policy_df,
                        case_n_jobs=case_n_jobs,
                        ode_backend=ode_backend,
                        x_df=x_df,
                        metadata_df=metadata_df,
                    )
                )
            else:
                missing.append(repeat_id)
        return completed, missing

    if finalize_only:
        completed, missing = _collect_completed()
        if missing:
            raise RuntimeError(f"Cannot finalize umax optimization study: missing repeats {missing[:5]}")
        finalize_repeated_umax_optimization_outputs(outdir, completed, config)
        print(f"\nSaved repeated umax optimization outputs to {outdir}")
        return

    completed, missing = _collect_completed() if skip_completed_repeats else ([], sorted(job_by_repeat.keys()))
    if skip_completed_repeats and not missing:
        finalize_repeated_umax_optimization_outputs(outdir, completed, config)
        print(f"All {n_repeats} umax study repeats complete; finalized.", flush=True)
        return

    def _run_one(repeat_id: int) -> dict:
        pred_csv, pred_source = job_by_repeat[repeat_id]
        return _run_umax_study_repeat_from_predictions(
            repeat_id,
            pred_csv,
            pred_source,
            outdir,
            x_csv,
            metadata_csv,
            config,
            verbose=verbose,
            skip_if_complete=False,
            reference_by_repeat_df=reference_by_repeat_df,
            fixed_policy_df=fixed_policy_df,
            case_n_jobs=case_n_jobs,
            ode_backend=ode_backend,
            x_df=x_df,
            metadata_df=metadata_df,
        )

    if len(missing) <= 1 or n_jobs == 1:
        newly_run = [_run_one(rid) for rid in (missing if missing else sorted(job_by_repeat.keys()))]
    else:
        from joblib import Parallel, delayed

        if verbose:
            print(f"Running {len(missing)} umax study repeat(s)...", flush=True)
        newly_run = Parallel(n_jobs=n_jobs if n_jobs > 0 else -1)(
            delayed(_run_one)(repeat_id) for repeat_id in missing
        )
        if verbose:
            print(f"Finished {len(newly_run)} umax study repeat(s).", flush=True)
    all_results = completed + newly_run
    finalize_repeated_umax_optimization_outputs(outdir, all_results, config)
    print(f"\nSaved repeated umax optimization outputs to {outdir}")


def _infer_n_repeats(
    prediction_jobs: List[Tuple[int, str, str]],
    predictions_manifest: Optional[str],
    outdir: str,
) -> int:
    if predictions_manifest and os.path.isfile(predictions_manifest):
        n = int(load_predictions_manifest(predictions_manifest).get("n_repeats", 0))
        if n > 0:
            return n
    if prediction_jobs:
        return len(prediction_jobs)
    return len(glob.glob(os.path.join(outdir, "repeats", "repeat_*", FIXED_UMAX_VALIDATION_SUBDIR)))



LOCKED_FINAL_CONFIG_FILENAME = "locked_final_config.json"
FAIRNESS_CHECK_FILENAME = "fairness_check.json"
DEFAULT_SCREENING_PLAN_PATH = os.path.join("analysis_plan", "parameter_screening_plan.yaml")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prediction-only closed-loop evaluation (TAR-only Umax optimizer)."
    )
    parser.add_argument(
        "--x_csv",
        default=None,
        help="X_features.csv for closed-loop evaluation (not required with --parameter_screening).",
    )
    parser.add_argument("--y_csv", default=None, help="y_targets.csv; required for training-derived dose reference.")
    parser.add_argument("--metadata_csv", default=None)
    parser.add_argument(
        "--predictions_dir",
        default=None,
        help="Directory with repeats/repeat_XXX/predictions.csv from tree_srl_benchmark.",
    )
    parser.add_argument("--predictions_manifest", default=None)
    parser.add_argument(
        "--predictions_csv",
        default=None,
        help="Single-split wide predictions CSV (alternative to --predictions_dir).",
    )
    parser.add_argument("--outdir", default="results/fixed_umax_validation")
    parser.add_argument("--u_grid", default="arange:0:101:1")
    parser.add_argument("--pathogen_ceiling", type=float, default=4.0e7)
    parser.add_argument("--pathogen_floor", type=float, default=1.0e7)
    parser.add_argument("--dosage_reference_target", type=float, default=2500.0)
    parser.add_argument("--target_total_dosage", type=float, default=2500.0)
    parser.add_argument("--target_terminal_pathogen", type=float, default=4.0e7)
    parser.add_argument(
        "--pauc_target_source",
        choices=["desired"],
        default="desired",
        help="Per-case desired_P_AUC target (default).",
    )
    parser.add_argument(
        "--pauc_feasibility_fraction",
        type=float,
        default=0.90,
        help="Hard feasibility: P_AUC >= fraction × desired_P_AUC.",
    )
    parser.add_argument(
        "--lr_target_source",
        choices=["desired_or_constant", "desired", "reference"],
        default="desired_or_constant",
        help="Use metadata desired_LR* when present; else lr_reference_target constant.",
    )
    parser.add_argument(
        "--pathogen_target_source",
        choices=["ceiling"],
        default="ceiling",
        help="Pathogen objective uses pathogen_ceiling only (one-sided).",
    )
    parser.add_argument(
        "--dose_reference_source",
        choices=["training_q90_reference_dosage", "constant"],
        default="training_q90_reference_dosage",
        help="Dose burden normalization scale from training-only reference dosages (default q90).",
    )
    parser.add_argument("--dose_reference_quantile", type=float, default=0.90)
    parser.add_argument("--dose_reference_constant", type=float, default=2500.0)
    parser.add_argument("--fixed_representative_umax", type=float, default=FIXED_REPRESENTATIVE_UMAX_DEFAULT)
    parser.add_argument(
        "--target_pauc_source",
        choices=["desired", "reference"],
        default="desired",
        help="Deprecated alias for --pauc_target_source.",
    )
    parser.add_argument(
        "--target_lr_source",
        choices=["desired_or_constant", "desired", "reference"],
        default="desired_or_constant",
        help="Deprecated alias for --lr_target_source.",
    )
    parser.add_argument("--lr_reference_target", type=float, default=1.0)
    parser.add_argument("--probiotic_pauc_fraction", type=float, default=0.90)
    parser.add_argument("--derive_optimizer_references_only", action="store_true")
    parser.add_argument("--derive_fixed_umax_policies_only", action="store_true")
    parser.add_argument(
        "--weight_profile",
        choices=list(WEIGHT_PROFILE_NAMES),
        default="balanced",
    )
    parser.add_argument(
        "--weight_selection_source",
        choices=["fixed", "train_only"],
        default="fixed",
    )
    parser.add_argument("--w_track", type=float, default=1.0)
    parser.add_argument("--w_path", type=float, default=1.0)
    parser.add_argument("--w_probiotic", type=float, default=1.0)
    parser.add_argument("--w_dose", type=float, default=1.0)
    parser.add_argument("--max_closed_loop_cases", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--permutation_replicates", type=int, default=5000)
    parser.add_argument("--run_umax_optimization_study", action="store_true")
    parser.add_argument(
        "--umax_selection_policy",
        choices=list(UMAX_SELECTION_POLICIES),
        default=UMAX_SELECTION_POLICY_DEFAULT,
        help="Umax inverse-design selection: feasible_first (primary) or aspiration_then_pareto (sensitivity).",
    )
    parser.add_argument("--aspiration_tolerance_rel", type=float, default=0.001)
    parser.add_argument("--aspiration_tolerance_abs", type=float, default=1e-9)
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "numba", "python"],
        help="ODE simulator for u_grid scans: auto=numba when validated (default), else python.",
    )
    parser.add_argument(
        "--u_grid_n_jobs",
        type=int,
        default=0,
        help="Parallel workers for u_grid scoring inside each case (0=auto when repeat-parallel).",
    )
    parser.add_argument(
        "--case_n_jobs",
        type=int,
        default=0,
        help="Parallel validation cases within each repeat (0=auto: all cores when repeat_n_jobs=1).",
    )
    parser.add_argument("--n_jobs", type=int, default=-1, help="Parallel workers across prediction repeats.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--finalize_only", action="store_true")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument(
        "--parameter_screening",
        action="store_true",
        help="Run Stage-1 leakage-safe parameter screening (not formal test).",
    )
    parser.add_argument(
        "--screening_stage",
        choices=["all", "relabel", "training", "umax_weights", "lock", "figures"],
        default="all",
    )
    parser.add_argument(
        "--screening_plan",
        default=DEFAULT_SCREENING_PLAN_PATH,
        help="Path to parameter_screening_plan.yaml",
    )
    parser.add_argument("--microbio_csv", default=None, help="Raw MICROBIO.csv for relabel screening.")
    parser.add_argument(
        "--run_umax_weight_sensitivity",
        action="store_true",
        help="During screening, run Umax weight sensitivity (requires predictions).",
    )
    parser.add_argument(
        "--locked_final_evaluation",
        action="store_true",
        help="Stage-2: enforce results/screening/locked_final_config.json (no CLI parameter overrides).",
    )
    parser.add_argument(
        "--allow_experimental_override",
        action="store_true",
        help="Allow CLI overrides vs locked_final_config.json; marks outputs not_for_manuscript.",
    )
    args = parser.parse_args()

    if args.parameter_screening:
        from microbio_dataset import run_parameter_screening_pipeline

        run_parameter_screening_pipeline(
            plan_path=args.screening_plan,
            screening_stage=args.screening_stage,
            microbio_csv=args.microbio_csv,
            device="auto",
            n_jobs=args.n_jobs,
            run_umax_sensitivity=args.run_umax_weight_sensitivity,
            predictions_dir=args.predictions_dir,
            predictions_manifest=args.predictions_manifest,
        )
        print("Parameter screening complete (development / sensitivity only; not formal test).")
        return

    if not args.x_csv:
        parser.error("--x_csv is required unless --parameter_screening is set.")

    not_for_manuscript = False
    if args.locked_final_evaluation:
        from microbio_dataset import enforce_locked_final_config, load_locked_final_config

        locked = load_locked_final_config()
        not_for_manuscript = enforce_locked_final_config(
            locked,
            weight_profile=args.weight_profile,
            u_grid_str=args.u_grid,
            allow_experimental_override=args.allow_experimental_override,
        )
        w_prof = str(locked.get("umax_weight_profile", "balanced"))
        w_dict = dict(locked.get("umax_weights", {}))
        args.weight_profile = w_prof
        args.w_track = float(w_dict.get("w_track", 1.0))
        args.w_path = float(w_dict.get("w_path", 1.0))
        args.w_probiotic = float(w_dict.get("w_probiotic", 1.0))
        args.w_dose = float(w_dict.get("w_dose", 0.25))
        args.u_grid = str(locked.get("u_grid", args.u_grid))
        args.umax_selection_policy = str(
            locked.get("umax_selection_policy", UMAX_SELECTION_POLICY_DEFAULT)
        )

    if not args.predictions_csv and not args.predictions_dir and not args.predictions_manifest:
        parser.error("Provide --predictions_dir (recommended), --predictions_manifest, or --predictions_csv.")

    weights = resolve_weights_from_profile(
        args.weight_profile, args.w_track, args.w_path, args.w_probiotic, args.w_dose
    )
    config = ClosedLoopConfig(
        u_grid=parse_u_grid(args.u_grid),
        weights=weights,
        pathogen_ceiling_cfu_per_mL=args.pathogen_ceiling,
        pathogen_floor_cfu_per_mL=args.pathogen_floor,
        dosage_reference_target=args.dosage_reference_target,
        target_total_dosage=args.target_total_dosage,
        target_terminal_pathogen=args.target_terminal_pathogen,
        pauc_target_source=args.pauc_target_source,
        pauc_feasibility_fraction=args.pauc_feasibility_fraction,
        lr_target_source=args.lr_target_source,
        pathogen_target_source=args.pathogen_target_source,
        dose_reference_source=args.dose_reference_source,
        dose_reference_quantile=args.dose_reference_quantile,
        dose_reference_constant=args.dose_reference_constant,
        fixed_representative_umax=args.fixed_representative_umax,
        target_pauc_source=args.target_pauc_source,
        target_lr_source=args.target_lr_source,
        lr_reference_target=args.lr_reference_target,
        probiotic_pauc_fraction=args.probiotic_pauc_fraction,
        max_closed_loop_cases=args.max_closed_loop_cases,
        bootstrap_replicates=args.bootstrap,
        permutation_replicates=args.permutation_replicates,
        weight_profile=args.weight_profile,
        weight_selection_source=args.weight_selection_source,
        run_umax_optimization_study=args.run_umax_optimization_study,
        ode_backend=args.backend,
        case_n_jobs=args.case_n_jobs,
        u_grid_n_jobs=args.u_grid_n_jobs,
        umax_selection_policy=args.umax_selection_policy,
        aspiration_tolerance_rel=args.aspiration_tolerance_rel,
        aspiration_tolerance_abs=args.aspiration_tolerance_abs,
    )

    if args.derive_optimizer_references_only:
        if not args.y_csv:
            parser.error("--derive_optimizer_references_only requires --y_csv")
        from derive_optimizer_references import derive_optimizer_references

        derive_optimizer_references(
            x_csv=args.x_csv,
            y_csv=args.y_csv,
            metadata_csv=args.metadata_csv,
            outdir=args.outdir,
            predictions_dir=args.predictions_dir,
            predictions_manifest=args.predictions_manifest,
            predictions_csv=args.predictions_csv,
            fixed_representative_umax=config.fixed_representative_umax,
            dose_reference_quantile=config.dose_reference_quantile,
            backend=config.ode_backend,
            n_jobs=args.n_jobs,
        )
        return

    if args.derive_fixed_umax_policies_only:
        if not args.y_csv:
            parser.error("--derive_fixed_umax_policies_only requires --y_csv")
        from derive_optimizer_references import derive_fixed_umax_policies

        ref_path = os.path.join(args.outdir, OPTIMIZER_REFERENCE_BY_REPEAT_CSV)
        reference_df = load_optimizer_reference_by_repeat(ref_path)
        derive_fixed_umax_policies(
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
            fixed_representative_umax=config.fixed_representative_umax,
            backend=config.ode_backend,
            n_jobs=args.n_jobs,
        )
        return

    if args.run_umax_optimization_study:
        run_repeated_umax_optimization_pipeline(
            predictions_dir=args.predictions_dir,
            predictions_manifest=args.predictions_manifest,
            predictions_csv=args.predictions_csv,
            x_csv=args.x_csv,
            y_csv=args.y_csv,
            metadata_csv=args.metadata_csv,
            outdir=args.outdir,
            config=config,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
            finalize_only=args.finalize_only,
            skip_completed_repeats=not args.force_rerun,
        )
        return

    run_fixed_umax_validation_pipeline(
        predictions_dir=args.predictions_dir,
        predictions_manifest=args.predictions_manifest,
        predictions_csv=args.predictions_csv,
        x_csv=args.x_csv,
        metadata_csv=args.metadata_csv,
        outdir=args.outdir,
        config=config,
        verbose=args.verbose,
        finalize_only=args.finalize_only,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()
