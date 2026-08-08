"""Figure audit, unified styling, and benchmark plot generation."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Unified manuscript palette (user-provided; red = experimental, blue = control)
PALETTE_RED_DARK = "#934B43"
PALETTE_RED_MID = "#D76364"
PALETTE_RED_LIGHT = "#EF7A6D"

PALETTE_NEUTRAL = "#F1D77E"
PALETTE_GREEN_MID = "#B1CE46"
PALETTE_GREEN_LIGHT = "#63E398"

PALETTE_BLUE_LIGHT = "#9DC3E7"
PALETTE_BLUE_MID = "#5F97D2"
PALETTE_BLUE_PERI = "#9394E7"

PALETTE_WHITE = "#FFFFFF"
COLOR_BG = "#F7F9FB"

# Manuscript typography — adjust here to scale all figures together.
FS_BASE = 12
FS_TICK = 11
FS_TICK_SM = 10
FS_TITLE = 13
FS_PANEL = 12
FS_LEGEND = 11
FS_LEGEND_SM = 10
FS_ANNOT = 11
FS_SIG = 12
FS_NOTE = 10
FS_TRAJECTORY_TITLE = 14
FS_ODE_BACK = 13
LAYOUT_PAD = 1.5
LAYOUT_H_PAD = 1.6
LAYOUT_W_PAD = 1.2

# ---------------------------------------------------------------------------
# Color usage (single source of truth for all manuscript figures)
#   Diverging heatmaps (correlation, gain):  blue = -1 / negative, white = 0, red = +1 / positive
#   Sequential heatmaps (RMSE, weights):     white = low, blue = high
#   Model bar charts:                        experimental = red ramp, control = blue ramp
#   Trajectory / closed-loop lines:          manuscript palette, hue-spread for legibility
# ---------------------------------------------------------------------------

# Control vs experimental bar families
CONTROL_BLUE_RAMP: List[str] = [PALETTE_BLUE_LIGHT, PALETTE_BLUE_PERI, PALETTE_BLUE_MID]
EXPERIMENTAL_RED_RAMP: List[str] = [PALETTE_RED_LIGHT, PALETTE_RED_MID, PALETTE_RED_DARK]
# Same-category diagnostic / gain bars: blue shades only (keeps figures calm)
ANALYSIS_BAR_RAMP: List[str] = CONTROL_BLUE_RAMP

BAR_CYCLE: List[str] = CONTROL_BLUE_RAMP + EXPERIMENTAL_RED_RAMP

# Heatmaps — diverging: blue (-1) → white (0) → red (+1); sequential: white (low) → blue (high)
HEATMAP_DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    "igem_div",
    [PALETTE_BLUE_MID, PALETTE_BLUE_LIGHT, PALETTE_WHITE, PALETTE_RED_LIGHT, PALETTE_RED_MID],
    N=256,
)
HEATMAP_SEQUENTIAL = mcolors.LinearSegmentedColormap.from_list(
    "igem_seq",
    [PALETTE_WHITE, PALETTE_BLUE_LIGHT, PALETTE_BLUE_MID],
    N=256,
)
HEATMAP_ERROR = HEATMAP_SEQUENTIAL
HEATMAP_GAIN = HEATMAP_DIVERGING

# Trajectory lines: spread hues across the manuscript palette for legibility (not all blue).
TRAJECTORY_AMP = PALETTE_BLUE_MID
TRAJECTORY_PROBIOTIC = PALETTE_GREEN_MID
TRAJECTORY_PATHOGENS: List[str] = [
    PALETTE_BLUE_MID,
    PALETTE_RED_MID,
    PALETTE_BLUE_PERI,
    PALETTE_GREEN_LIGHT,
    PALETTE_NEUTRAL,
]


def apply_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": FS_BASE,
            "axes.titlesize": FS_PANEL,
            "axes.labelsize": FS_BASE,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_LEGEND,
            "axes.prop_cycle": plt.cycler(color=BAR_CYCLE),
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#333333",
            "axes.linewidth": 0.8,
            "text.color": "#333333",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#E0E0E0",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def finalize_figure_layout(
    fig,
    *,
    rect: Optional[Tuple[float, float, float, float]] = None,
    pad: float = LAYOUT_PAD,
    h_pad: float = LAYOUT_H_PAD,
    w_pad: float = LAYOUT_W_PAD,
) -> None:
    """Tighten subplot spacing without changing figure aspect ratio."""
    kwargs = {"pad": pad, "h_pad": h_pad, "w_pad": w_pad}
    if rect is not None:
        kwargs["rect"] = rect
    fig.tight_layout(**kwargs)


def place_figure_legend_below(
    fig,
    handles,
    labels,
    *,
    ncol: int,
    y: float = -0.02,
    frameon: bool = True,
    bottom_rect: float = 0.12,
) -> None:
    """Legend below all panels; reserves bottom margin so labels do not overlap."""
    fig.legend(
        handles,
        labels,
        ncol=ncol,
        fontsize=FS_LEGEND,
        frameon=frameon,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
    )
    finalize_figure_layout(fig, rect=(0.0, bottom_rect, 1.0, 0.98))


def companion_svg_path(path: str) -> str:
    base, _ = os.path.splitext(path)
    return base + ".svg"


def save_figure(
    fig,
    outpath: str,
    *,
    dpi: int = 300,
    bbox_inches: str = "tight",
    **kwargs,
) -> Tuple[str, str]:
    """Save matplotlib figure as PNG and SVG; return ``(png_path, svg_path)``."""
    base, _ = os.path.splitext(outpath)
    png_path = base + ".png"
    svg_path = base + ".svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches, format="png", **kwargs)
    fig.savefig(svg_path, bbox_inches=bbox_inches, format="svg", **kwargs)
    return png_path, svg_path


def expand_figure_outputs(outputs: Dict[str, str]) -> Dict[str, str]:
    """Add ``.svg`` siblings for each ``.png`` entry when the SVG file exists."""
    expanded = dict(outputs)
    for key, path in list(outputs.items()):
        if not key.endswith(".png"):
            continue
        svg_path = companion_svg_path(path)
        if os.path.isfile(svg_path):
            expanded[key[:-4] + ".svg"] = svg_path
    return expanded


def bar_colors(n: int) -> Sequence[str]:
    if n <= len(BAR_CYCLE):
        return BAR_CYCLE[:n]
    return [BAR_CYCLE[i % len(BAR_CYCLE)] for i in range(n)]


def _spread_ramp(ramp: Sequence[str], n: int) -> List[str]:
    if n <= 0:
        return []
    if n == 1:
        return [ramp[len(ramp) // 2]]
    idx = np.linspace(0, len(ramp) - 1, n)
    return [ramp[int(round(i))] for i in idx]


def _symmetric_heatmap_limits(values: np.ndarray, fallback: float = 0.05) -> Tuple[float, float]:
    """Symmetric vmin/vmax for diverging heatmaps (blue = negative, red = positive)."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        lim = fallback
    else:
        lim = float(np.max(np.abs(finite)))
        if lim <= 1e-12:
            lim = fallback
    return -lim, lim


def diverging_heatmap_kwargs(
    values: np.ndarray,
    *,
    fixed_limits: Optional[Tuple[float, float]] = None,
) -> dict:
    """Shared imshow kwargs for all signed heatmaps (correlation, gain, …)."""
    if fixed_limits is not None:
        vmin, vmax = fixed_limits
    else:
        vmin, vmax = _symmetric_heatmap_limits(values)
    return {"cmap": HEATMAP_DIVERGING, "vmin": vmin, "vmax": vmax}


def sequential_heatmap_kwargs(values: np.ndarray) -> dict:
    """Shared imshow kwargs for magnitude-only heatmaps (RMSE, weights, …)."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return {"cmap": HEATMAP_SEQUENTIAL, "vmin": 0.0, "vmax": vmax}


# --- benchmark plots ---
TAR_MODEL = "TAR"
TAR_NO_CLOSED_LOOP_MODEL = "TAR-noClosedLoop"
BEST_SINGLE_TREE = "BestSingleTree"
UNIFORM_TREE_MEAN = "UniformTreeMean"
RANDOM_FOREST = "RandomForest"
TARGET_COLS_TTHR = [f"Tthr_{i}" for i in range(1, 6)]

LEGACY_MODEL_NAME_MAP = {
    "TAR-SRL": TAR_MODEL,
    "TAR-SRL-no-cycle": TAR_MODEL,
    "TAR-SRL-rerank": "TAR-rerank_legacy_not_main",
}

MODEL_DISPLAY_LABELS = {
    TAR_MODEL: "TAR",
    RANDOM_FOREST: "RF",
    BEST_SINGLE_TREE: "BestTree",
    UNIFORM_TREE_MEAN: "UniformTreeMean",
    "ExtraTrees": "ET",
}

MAIN_COMPARE_ORDER = [TAR_MODEL, RANDOM_FOREST, UNIFORM_TREE_MEAN, BEST_SINGLE_TREE]
ODE_BACK_BAR_MODEL_ORDER = [TAR_MODEL, RANDOM_FOREST, BEST_SINGLE_TREE, UNIFORM_TREE_MEAN]
MANUSCRIPT_BAR_MODEL_ORDER = MAIN_COMPARE_ORDER

FIG3_ARCHITECTURE_TEXT = (
    "bio features + desired outcomes -> single-tree experts -> target-wise OOF stack -> predicted Tthr"
)
FIG3_PANEL_ORDER: List[Tuple[str, Optional[str], str]] = [
    ("A", None, "Model architecture schematic"),
    ("B", "model_compare_r2.png", "threshold benchmark panel"),
    ("C", "prediction_error_heatmap.png", "Per-target RMSE heatmap"),
    ("D", "ode_back_outcome_heatmap.png", "ODE-back functional outcome R² heatmap"),
]

# Per-file manuscript roles and source groups (figures live under different result roots).
MANUSCRIPT_SOURCE_MAPPING: Dict[str, Dict[str, str]] = {
    "model_compare_r2.png": {
        "description": "threshold benchmark panel",
        "source_group": "tree_srl_benchmark",
    },
    "target_weight_heatmap.png": {
        "description": "signed target-wise stacking coefficients",
        "source_group": "tree_srl_benchmark",
    },
    "uncertainty_decomposition_by_target.png": {
        "description": "auxiliary uncertainty fraction",
        "source_group": "tree_srl_benchmark",
    },
    "ode_back_r2_barplot.png": {
        "description": "repeated ODE-back functional R2",
        "source_group": "ode_back_validation",
    },
    "fixed_umax_representative.png": {
        "description": "deterministic representative trajectories",
        "source_group": "fixed_umax_validation",
    },
    "umax_constraint_feasibility.png": {
        "description": "constraint feasibility",
        "source_group": "umax_optimization",
    },
    "umax_score_landscape.png": {
        "description": "composite-penalty response landscape",
        "source_group": "umax_optimization",
    },
    "umax_summary_ablation_composite_supplementary.png": {
        "description": "composite-penalty policy ablation",
        "source_group": "umax_optimization",
    },
}
FIG3_PRIMARY_FIGURES: Tuple[str, ...] = (
    "model_compare_r2.png",
    "prediction_error_heatmap.png",
    "ode_back_outcome_heatmap.png",
)
FIG3_SUPPLEMENTARY_FIGURES: Tuple[str, ...] = (
    "target_weight_heatmap.png",
    "uncertainty_decomposition.png",
    "uncertainty_decomposition_by_target.png",
    "ode_back_r2_barplot.png",
    "ode_back_pred_vs_ref_scatter.png",
)

ODE_BACK_VALIDATION_SUBDIR = "ode_back_validation"
ODE_BACK_MANIFEST_JSON = "ode_back_validation_manifest.json"
ODE_BACK_SUMMARY_CSV = "ode_back_summary_by_model.csv"
ODE_BACK_PER_OUTCOME_CSV = "ode_back_per_outcome_metrics.csv"
ODE_BACK_CASE_CSV = "ode_back_case_results.csv"
ODE_BACK_PAIRWISE_CSV = "ode_back_pairwise_significance.csv"
ODE_BACK_TRAJECTORY_PAIRWISE_CSV = "ode_back_trajectory_pairwise_significance.csv"
ODE_BACK_BAR_METRIC = "mean_outcome_R2"
ODE_BACK_TRAJECTORY_R2_METRIC = "trajectory_R2"

FIG4_FIXED_UMAX_MODELS = [TAR_MODEL, BEST_SINGLE_TREE, UNIFORM_TREE_MEAN]
FIG4_SIGNIFICANCE_CONTROLS = [BEST_SINGLE_TREE, UNIFORM_TREE_MEAN]

FIG4_PANEL_ORDER: List[Tuple[str, str, str]] = [
    ("A", "fixed_umax_representative.png", "deterministic representative trajectories"),
    ("B", "fixed_umax_summary.png", "Fixed-Umax forward ODE summary (single point estimate per model)"),
    ("C", "fixed_umax_constraint_success.png", "Constraint success rates (optional if redundant)"),
]
FIG4_PRIMARY_FIGURES: Tuple[str, ...] = (
    "fixed_umax_representative.png",
    "fixed_umax_summary.png",
)
FIG4_OPTIONAL_FIGURES: Tuple[str, ...] = (
    "fixed_umax_constraint_success.png",
)

FIG4_SIGNIFICANCE_RULE_TEXT = (
    "TAR leftmost; compare TAR vs BestTree and TAR vs UniformTreeMean only; "
    "bidirectional stars (↑ = TAR better, ↓ = control better) when formal significance supports either direction; "
    "exact p-values and paired differences retained in CSV; no stars when n_repeats < 10"
)

FIG5_PANEL_ORDER: List[Tuple[str, str, str]] = [
    ("A", "umax_score_landscape.png", "composite-penalty response landscape"),
    ("B", "umax_constraint_feasibility.png", "constraint feasibility"),
    ("C", "umax_ode_ablation.png", "Representative ODE ablation trajectories"),
    ("D", "umax_summary_ablation.png", "TAR-only Umax policy ablation summary"),
]
FIG5_PRIMARY_FIGURES: Tuple[str, ...] = tuple(panel[1] for panel in FIG5_PANEL_ORDER)

FIG5_MAIN_SUMMARY_METRICS: Tuple[Tuple[str, str], ...] = (
    ("mean_total_dosage", "Total dosage"),
    ("mean_P_AUC", r"$P_{AUC}$"),
    ("mean_LR", "Mean LR"),
    ("mean_terminal_pathogen", "Terminal pathogen"),
)

FIG5_ABLATION_SHORT_LABELS: Dict[str, str] = {
    "TAR_optimized": "Optimized",
    "TAR_fixed_training_median": "Median fixed",
    "TAR_fixed_training_tuned_global": "Tuned global",
}

FIG5_ABLATION_BAR_COLORS: Dict[str, str] = {
    "TAR_optimized": PALETTE_RED_MID,
    "TAR_fixed_training_median": PALETTE_BLUE_MID,
    "TAR_fixed_training_tuned_global": PALETTE_BLUE_LIGHT,
}

FIG5_ODE_POLICY_SHORT_LABELS: Dict[str, str] = {
    "TAR_fixed_training_median": "Training median",
    "TAR_fixed_training_tuned_global": "Training-tuned global",
    "TAR_optimized": "Optimized",
}

FIXED_UMAX_VALIDATION_SUBDIR = "fixed_umax_validation"
FIG4_MANIFEST_JSON = "fixed_umax_validation_manifest.json"
FIG4_PLOT_MANIFEST_JSON = "fig4_plot_manifest.json"
FIXED_UMAX_REPEATED_STATS_CSV = "fixed_umax_validation_repeated_stats.csv"
FIXED_UMAX_SIGNIFICANCE_CSV = "fixed_umax_validation_significance.csv"
FIXED_UMAX_SUMMARY_CSV = "fixed_umax_validation_summary_by_model.csv"
FIXED_UMAX_TRAJECTORIES_CSV = "fixed_umax_representative_trajectories.csv"
UMAX_OPTIMIZATION_MANIFEST_JSON = "umax_optimization_manifest.json"
FIG5_PLOT_MANIFEST_JSON = "fig5_plot_manifest.json"

SIGNIFICANCE_RULE_TEXT = (
    "TAR leftmost; compare TAR vs each control only; "
    "bidirectional stars (↑ = TAR better, ↓ = control better) when formal significance supports either direction; "
    "exact p-values and paired differences retained in CSV; no stars when n_repeats < 10"
)

MODEL_BAR_COLORS: Dict[str, str] = {
    TAR_MODEL: PALETTE_RED_MID,
    TAR_NO_CLOSED_LOOP_MODEL: PALETTE_RED_LIGHT,
    RANDOM_FOREST: PALETTE_BLUE_MID,
    BEST_SINGLE_TREE: PALETTE_BLUE_LIGHT,
    UNIFORM_TREE_MEAN: PALETTE_BLUE_PERI,
    "ExtraTrees": PALETTE_BLUE_PERI,
}

EXPERIMENTAL_MODELS = frozenset({TAR_MODEL, TAR_NO_CLOSED_LOOP_MODEL, "TAR-SRL", "TAR-SRL-no-cycle"})

CORE_BENCHMARK_MODELS = frozenset(MAIN_COMPARE_ORDER)

SIGNIFICANCE_PLOT_SPECS = [
    (TAR_MODEL, RANDOM_FOREST),
    (TAR_MODEL, UNIFORM_TREE_MEAN),
    (TAR_MODEL, BEST_SINGLE_TREE),
]


def normalize_model_name(model_name: str) -> str:
    return LEGACY_MODEL_NAME_MAP.get(model_name, model_name)


def color_for_model(model: str) -> str:
    """Experimental models = red family; controls = blue family."""
    if model in MODEL_BAR_COLORS:
        return MODEL_BAR_COLORS[model]
    if model in EXPERIMENTAL_MODELS:
        return PALETTE_RED_MID
    return CONTROL_BLUE_RAMP[hash(model) % len(CONTROL_BLUE_RAMP)]


def colors_for_models(models: Sequence[str]) -> List[str]:
    return [color_for_model(model) for model in models]


def model_display_label(model_name: str) -> str:
    return MODEL_DISPLAY_LABELS.get(model_name, model_name)


def _assign_significance_bracket_levels(
    pairs: List[Tuple[str, str, str]],
    model_to_x: Dict[str, int],
) -> List[int]:
    """Assign vertical tier per bracket; overlapping x-ranges get different levels."""
    levels: List[int] = []
    spans: List[Tuple[float, float, int]] = []
    for model_a, model_b, label in pairs:
        if label in {"ns", "na", "nan", ""}:
            levels.append(-1)
            continue
        if model_a not in model_to_x or model_b not in model_to_x:
            levels.append(-1)
            continue
        x1, x2 = sorted([float(model_to_x[model_a]), float(model_to_x[model_b])])
        level = 0
        while any(not (x2 < lo or x1 > hi) and used == level for lo, hi, used in spans):
            level += 1
        levels.append(level)
        spans.append((x1, x2, level))
    return levels


def plot_metric_barplot(
    summary_df: pd.DataFrame,
    outpath: str,
    metric_col: str,
    ylabel: str,
    title: str,
    model_order: Optional[List[str]] = None,
    pin_first: Optional[List[str]] = None,
    significance_pairs: Optional[List[Tuple[str, str, str]]] = None,
    single_split_exploratory: bool = False,
    significance_exploratory_note: bool = False,
    exploratory_note: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    significance_bracket_step_scale: float = 1.0,
) -> Optional[plt.Axes]:
    apply_matplotlib_style()
    plot_df = summary_df.copy()
    if pin_first:
        pinned_names = [m for m in pin_first if m in plot_df["model"].values]
        rest = plot_df[~plot_df["model"].isin(pinned_names)]
        if metric_col in plot_df.columns:
            rest = rest.sort_values(metric_col, ascending=False)
        plot_df = pd.concat([plot_df[plot_df["model"].isin(pinned_names)], rest], ignore_index=True)
    elif model_order:
        plot_df = _reorder_bar_plot_df(plot_df, model_order)
    elif metric_col in plot_df.columns:
        plot_df = plot_df.sort_values(metric_col, ascending=False).reset_index(drop=True)

    display_labels = [model_display_label(m) for m in plot_df["model"]]
    model_names = plot_df["model"].tolist()
    model_to_x_pre = {model: idx for idx, model in enumerate(plot_df["model"])}
    valid_pairs_pre = [
        (a, b, lbl)
        for a, b, lbl in (significance_pairs or [])
        if lbl not in {"ns", "na", "nan", ""} and a in model_to_x_pre and b in model_to_x_pre
    ]
    levels_pre = _assign_significance_bracket_levels(valid_pairs_pre, model_to_x_pre)
    max_bracket_level = max((level for level in levels_pre if level >= 0), default=-1)
    fig_height = 5.8 + 0.9 * max(max_bracket_level + 1, 0)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(7, 0.62 * len(plot_df)), fig_height))
    else:
        fig = ax.figure
    x = np.arange(len(plot_df))
    y = plot_df[metric_col].to_numpy(dtype=float)
    colors = colors_for_models(model_names)
    bars = ax.bar(x, y, edgecolor="white", linewidth=1.0, color=colors, width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=35, ha="right")
    ax.tick_params(axis="x", pad=5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=2)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ci_low_col = f"{metric_col}_ci_low"
    ci_high_col = f"{metric_col}_ci_high"
    bar_tops = y.copy()
    bar_bottoms = y.copy()
    if {ci_low_col, ci_high_col}.issubset(plot_df.columns):
        lo = plot_df[ci_low_col].to_numpy(dtype=float)
        hi = plot_df[ci_high_col].to_numpy(dtype=float)
        if np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
            yerr = np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])
            centers = [bar.get_x() + bar.get_width() / 2 for bar in bars]
            ax.errorbar(centers, y, yerr=yerr, fmt="none", capsize=4, linewidth=1.0, color="black")
            finite_lo = np.isfinite(lo)
            finite_hi = np.isfinite(hi)
            bar_tops = np.where(finite_hi, hi, bar_tops)
            bar_bottoms = np.where(finite_lo, lo, bar_bottoms)

    if significance_pairs:
        model_to_x = {model: idx for idx, model in enumerate(plot_df["model"])}
        valid_pairs = [
            (a, b, lbl) for a, b, lbl in significance_pairs
            if lbl not in {"ns", "na", "nan", ""} and a in model_to_x and b in model_to_x
        ]
        levels = _assign_significance_bracket_levels(valid_pairs, model_to_x)
        ymin = float(np.nanmin(bar_bottoms)) if np.any(np.isfinite(bar_bottoms)) else 0.0
        ymax = float(np.nanmax(bar_tops)) if np.any(np.isfinite(bar_tops)) else 0.0
        yspan = max(ymax - ymin, 0.05) if np.any(np.isfinite(y)) else 0.1
        bracket_padding = 0.12 * yspan
        bracket_step = 0.20 * yspan * significance_bracket_step_scale
        bracket_base = ymax + bracket_padding
        max_bracket_top = ymax
        tick_h = 0.04 * yspan
        for (model_a, model_b, label), level in zip(valid_pairs, levels):
            if level < 0:
                continue
            x1, x2 = sorted([model_to_x[model_a], model_to_x[model_b]])
            y_bracket = bracket_base + level * bracket_step
            ax.plot(
                [x1, x1, x2, x2],
                [y_bracket, y_bracket + tick_h, y_bracket + tick_h, y_bracket],
                color="black", linewidth=1.0,
            )
            ax.text(
                (x1 + x2) / 2.0, y_bracket + tick_h + 0.015 * yspan,
                label, ha="center", va="bottom", fontsize=13, fontweight="bold",
            )
            max_bracket_top = max(max_bracket_top, y_bracket + tick_h + 0.08 * yspan)
        if valid_pairs:
            ax.set_ylim(bottom=min(ymin - 0.05 * yspan, ax.get_ylim()[0]), top=max_bracket_top + 0.08 * yspan)

    if own_fig:
        finalize_figure_layout(fig, rect=(0.0, 0.16, 1.0, 1.0))
        save_figure(fig, outpath)
        plt.close(fig)
    return ax


def plot_r2_barplot(
    summary_df: pd.DataFrame,
    outpath: str,
    title: str,
    model_order: Optional[List[str]] = None,
    pin_first: Optional[List[str]] = None,
    significance_pairs: Optional[List[Tuple[str, str, str]]] = None,
    single_split_exploratory: bool = False,
    significance_exploratory_note: bool = False,
    exploratory_note: Optional[str] = None,
) -> None:
    plot_metric_barplot(
        summary_df=summary_df,
        outpath=outpath,
        metric_col="mean_R2_original",
        ylabel="Mean target-wise $R^2$ (original scale)",
        title=title,
        model_order=model_order,
        pin_first=pin_first,
        significance_pairs=significance_pairs,
        single_split_exploratory=single_split_exploratory,
        significance_exploratory_note=significance_exploratory_note,
        exploratory_note=exploratory_note,
    )


def select_main_controls_plot(
    available: set,
    main_order: Optional[List[str]] = None,
    **_kwargs,
) -> List[str]:
    order = main_order or MAIN_COMPARE_ORDER
    normalized = {normalize_model_name(m) for m in available}
    return [name for name in order if name in normalized]


def build_significance_plot_specs(available: set, **_kwargs) -> List[Tuple[str, str]]:
    specs: List[Tuple[str, str]] = []
    normalized = {normalize_model_name(m) for m in available}
    for srl_model, control_model in SIGNIFICANCE_PLOT_SPECS:
        if srl_model in normalized and control_model in normalized:
            specs.append((srl_model, control_model))
    return specs


def significance_pairs_for_plot(
    pairwise_df: pd.DataFrame,
    srl_model: str,
    control_models: Sequence[str],
    metric: str = "mean_R2_original",
    significance_for_manuscript: bool = False,
    bidirectional: bool = False,
) -> List[Tuple[str, str, str]]:
    """Build (srl, control, star_label) triples for formal TAR-vs-control brackets.

    When ``bidirectional`` is True, statistically supported control-better comparisons are
    annotated with ↓ and TAR-better with ↑ so direction is unambiguous. CSV p-values and
    paired differences are never modified here.
    """
    pairs: List[Tuple[str, str, str]] = []
    if pairwise_df.empty or not significance_for_manuscript:
        return pairs
    control_better_tiers = {
        "control_better",
        "formal_control_better",
        "exploratory_control_better",
    }
    sub = pairwise_df[(pairwise_df["metric"] == metric) & (pairwise_df["srl_model"] == srl_model)]
    for control_model in control_models:
        row = sub[sub["control_model"] == control_model]
        if row.empty:
            continue
        row0 = row.iloc[0]
        if bool(row0.get("exploratory", False)):
            continue
        label = str(row0["significance_label"])
        tier = str(row0.get("significance_tier", ""))
        comparison = str(row0.get("comparison_result", ""))
        is_control_better = tier in control_better_tiers or comparison in {
            "control_better",
            "exploratory_control_better",
        }
        if label in {"ns", "na", "nan", ""}:
            if not (bidirectional and is_control_better):
                continue
            from tree_srl_benchmark import significance_label

            p_candidates = [
                float(p)
                for p in (row0.get("permutation_p"), row0.get("wilcoxon_p"))
                if pd.notna(p) and np.isfinite(float(p))
            ]
            if p_candidates:
                label = significance_label(min(p_candidates))
            ci_high = float(row0.get("CI_high", np.nan))
            if label == "ns" and np.isfinite(ci_high) and ci_high < 0.0:
                label = "*"
        if label in {"ns", "na", "nan", ""}:
            continue
        if not bidirectional and is_control_better:
            continue
        if bidirectional:
            # Unambiguous direction markers; omit all stars rather than show ambiguous ****.
            label = f"{label}↓" if is_control_better else f"{label}↑"
        pairs.append((srl_model, control_model, label))
    return pairs


def expert_display_label(expert_id: str) -> str:
    """Short display-only expert label; CSV / manifest retain the original expert ID."""
    s = str(expert_id)
    if s.startswith("ExtraTree_"):
        s = "ET_" + s[len("ExtraTree_") :]
    elif s.startswith("CART_"):
        s = s[5:]
    return (
        s.replace("friedman", "fr")
        .replace("poisson", "pois")
        .replace("shallow", "sh")
        .replace("single", "1")
        .replace("features", "f")
        .replace("depth", "d")
        .replace("leaf", "L")
        .replace("deep", "dp")
        .replace("log2_", "log2")
        .replace("sqrt_", "sqrt")
    )


def plot_target_weight_heatmap(weights_df: pd.DataFrame, outpath: str, stacker_type: str = "ridge") -> None:
    apply_matplotlib_style()
    if weights_df.empty:
        return

    if "stacker_type" in weights_df.columns:
        stacker_types = sorted(
            {str(t).strip().lower() for t in weights_df["stacker_type"].dropna().unique() if str(t).strip()}
        )
    else:
        stacker_types = [str(stacker_type).strip().lower()] if str(stacker_type).strip() else ["ridge"]

    if len(stacker_types) > 1:
        raise ValueError(
            "plot_target_weight_heatmap: mixed stacker_type values "
            f"{stacker_types}; refuse to average incomparable coefficient scales. "
            "Filter to one stacker_type or produce separate facets."
        )
    resolved = stacker_types[0] if stacker_types else str(stacker_type).strip().lower() or "ridge"
    ridge_like = resolved == "ridge" or resolved.startswith("ridge")
    convex_like = resolved in {
        "convex",
        "nonneg",
        "non-negative",
        "non_negative",
        "nnls",
        "positive",
    } or "convex" in resolved or "nonneg" in resolved

    pivot = (
        weights_df.groupby(["target", "expert"], as_index=False)["weight"]
        .mean()
        .pivot(index="target", columns="expert", values="weight")
    )
    display_experts = [expert_display_label(c) for c in pivot.columns]
    n_experts = max(pivot.shape[1], 1)
    fig_w = max(12.0, 0.70 * n_experts)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))
    if ridge_like:
        im = ax.imshow(pivot.values, aspect="auto", **diverging_heatmap_kwargs(pivot.values))
        cbar_label = "Ridge coefficient"
    elif convex_like:
        im = ax.imshow(pivot.values, aspect="auto", **sequential_heatmap_kwargs(pivot.values))
        cbar_label = "Convex weight"
    else:
        raise ValueError(
            f"plot_target_weight_heatmap: unrecognized stacker_type {resolved!r}; "
            "expected ridge or convex/non-negative."
        )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(display_experts, rotation=60, ha="right", fontsize=9)
    ax.tick_params(axis="x", pad=2)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Target")
    ax.set_title("Target-wise stacking coefficients")
    finalize_figure_layout(fig, rect=(0.0, 0.18, 1.0, 1.0))
    save_figure(fig, outpath)
    plt.close(fig)


def plot_per_target_metric_heatmap(
    per_target_df: pd.DataFrame,
    outpath: str,
    models: List[str],
    value_col: str,
    cbar_label: str,
    *,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Axes]:
    apply_matplotlib_style()
    if per_target_df.empty:
        return ax
    sub = per_target_df[per_target_df["model"].isin(models)].copy()
    if sub.empty or value_col not in sub.columns:
        return ax
    pivot = (
        sub.groupby(["model", "target"], as_index=False)[value_col]
        .mean()
        .pivot(index="model", columns="target", values=value_col)
    )
    order = [m for m in models if m in pivot.index]
    pivot = pivot.loc[order]
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(7, 0.55 * pivot.shape[1]), max(4, 0.45 * pivot.shape[0])))
    im = ax.imshow(pivot.values, aspect="auto", **sequential_heatmap_kwargs(pivot.values))
    if own_fig:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    else:
        fig = ax.figure
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([model_display_label(m) for m in pivot.index], fontsize=11)
    ax.set_xlabel("Target")
    ax.set_ylabel("Model")
    if own_fig:
        fig.tight_layout()
        save_figure(fig, outpath)
        plt.close(fig)
    return ax


ODE_BACK_HEATMAP_OUTCOMES = [
    "P_AUC",
    "LR1",
    "LR2",
    "LR3",
    "LR4",
    "LR5",
    "mean_LR",
    "log10_terminal_total_pathogen",
    "total_dosage",
]


def _normalize_ode_back_summary_df(summary_df: pd.DataFrame, metric_col: str = ODE_BACK_BAR_METRIC) -> pd.DataFrame:
    """Resolve legacy duplicate CI column suffixes from repeated merges."""
    df = summary_df.copy()
    for base in (f"{metric_col}_ci_low", f"{metric_col}_ci_high"):
        if base not in df.columns:
            for suffix in ("_y", "_x", ""):
                cand = base if not suffix else f"{base}{suffix}"
                if cand in df.columns:
                    df[base] = df[cand]
                    break
    if metric_col == ODE_BACK_BAR_METRIC and metric_col not in df.columns and "mean_outcome_R2_repeat" in df.columns:
        df[metric_col] = df["mean_outcome_R2_repeat"]
    return df


def plot_ode_back_r2_barplot(
    summary_df: pd.DataFrame,
    outpath: str,
    *,
    model_order: Optional[List[str]] = None,
    title: str = "ODE-back functional outcome $R^2$ (100 repeats, mean $\\pm$ 95% CI)",
    significance_pairs: Optional[List[Tuple[str, str, str]]] = None,
    n_repeats: int = 0,
    metric_col: str = ODE_BACK_BAR_METRIC,
    ylabel: str = "Mean functional outcome $R^2$",
) -> None:
    """Bar plot of repeat-averaged ODE-back R² with 95% CI and TAR significance brackets."""
    order = model_order or list(ODE_BACK_BAR_MODEL_ORDER)
    plot_df = _normalize_ode_back_summary_df(summary_df, metric_col=metric_col)
    plot_df = plot_df[plot_df["model"].isin(order)].copy()
    if plot_df.empty or metric_col not in plot_df.columns:
        return
    ci_low_col = f"{metric_col}_ci_low"
    ci_high_col = f"{metric_col}_ci_high"
    apply_matplotlib_style()
    fig_height = 5.8 + 0.9 * max(len(significance_pairs or []), 1)
    fig, ax = plt.subplots(figsize=(max(7, 0.62 * len(plot_df)), fig_height))
    plot_metric_barplot(
        plot_df,
        outpath,
        metric_col=metric_col,
        ylabel=ylabel,
        title=title,
        model_order=order,
        pin_first=[TAR_MODEL],
        significance_pairs=significance_pairs,
        ax=ax,
    )
    ax.set_title(title, fontsize=FS_ODE_BACK, pad=2)
    ax.set_ylabel(ylabel, fontsize=FS_ODE_BACK)
    ax.tick_params(axis="both", labelsize=FS_TICK + 1)
    for text in ax.texts:
        if text.get_fontweight() == "bold":
            text.set_fontsize(FS_SIG + 1)
    if {ci_low_col, ci_high_col}.issubset(plot_df.columns):
        lo = float(np.nanmin(plot_df[ci_low_col].to_numpy(dtype=float)))
        hi = float(np.nanmax(plot_df[ci_high_col].to_numpy(dtype=float)))
        yvals = plot_df[metric_col].to_numpy(dtype=float)
        if np.any(np.isfinite(yvals)):
            lo = min(lo, float(np.nanmin(yvals))) if np.isfinite(lo) else float(np.nanmin(yvals))
            hi = max(hi, float(np.nanmax(yvals))) if np.isfinite(hi) else float(np.nanmax(yvals))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            yspan = max(hi - lo, 0.005)
            cur_lo, cur_hi = ax.get_ylim()
            ax.set_ylim(lo - 0.25 * yspan, max(cur_hi, hi + 0.35 * yspan))
    finalize_figure_layout(fig, rect=(0.0, 0.16, 1.0, 1.0))
    save_figure(fig, outpath)
    plt.close(fig)


def plot_ode_back_trajectory_r2_barplot(
    summary_df: pd.DataFrame,
    outpath: str,
    *,
    model_order: Optional[List[str]] = None,
    significance_pairs: Optional[List[Tuple[str, str, str]]] = None,
    n_repeats: int = 0,
) -> None:
    plot_ode_back_r2_barplot(
        summary_df,
        outpath,
        model_order=model_order,
        title="ODE-back trajectory $R^2$ (pred vs ref $T_{thr}$, 100 repeats, mean $\\pm$ 95% CI)",
        significance_pairs=significance_pairs,
        n_repeats=n_repeats,
        metric_col=ODE_BACK_TRAJECTORY_R2_METRIC,
        ylabel="Trajectory $R^2$ (pred vs ref ODE)",
    )


def plot_ode_back_outcome_heatmap(
    per_outcome_df: pd.DataFrame,
    outpath: str,
    *,
    model_order: Optional[List[str]] = None,
    outcomes: Optional[Sequence[str]] = None,
) -> None:
    """Heatmap: models × ODE-back outcome R²."""
    apply_matplotlib_style()
    if per_outcome_df.empty:
        return
    order = model_order or list(MAIN_COMPARE_ORDER)
    outcome_list = list(outcomes or ODE_BACK_HEATMAP_OUTCOMES)
    sub = per_outcome_df[
        per_outcome_df["model"].isin(order) & per_outcome_df["outcome"].isin(outcome_list)
    ].copy()
    if sub.empty:
        return
    pivot = (
        sub.groupby(["model", "outcome"], as_index=False)["R2"]
        .mean()
        .pivot(index="model", columns="outcome", values="R2")
    )
    cols = [c for c in outcome_list if c in pivot.columns]
    pivot = pivot.reindex(index=[m for m in order if m in pivot.index], columns=cols)
    fig, ax = plt.subplots(figsize=(max(8.5, 0.65 * len(cols)), max(4.2, 0.55 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", **sequential_heatmap_kwargs(pivot.values))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$R^2$")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([model_display_label(m) for m in pivot.index], fontsize=11)
    ax.set_xlabel("Functional outcome")
    ax.set_ylabel("Model")
    ax.set_title("ODE-back functional outcome $R^2$")
    fig.tight_layout()
    save_figure(fig, outpath)
    plt.close(fig)


def plot_ode_back_pred_vs_ref_scatter(
    case_df: pd.DataFrame,
    outpath: str,
    *,
    model_order: Optional[List[str]] = None,
    outcomes: Optional[Sequence[str]] = None,
) -> None:
    """Small multiples: predicted vs reference ODE outcomes."""
    apply_matplotlib_style()
    if case_df.empty:
        return
    order = model_order or list(MAIN_COMPARE_ORDER)
    panels = list(outcomes or ("P_AUC", "mean_LR", "log10_terminal_total_pathogen"))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, outcome in zip(axes, panels):
        ref_col = f"ref_{outcome}"
        pred_col = f"pred_{outcome}"
        if ref_col not in case_df.columns or pred_col not in case_df.columns:
            ax.set_visible(False)
            continue
        for model in order:
            sub = case_df[case_df["model"] == model]
            if sub.empty:
                continue
            x = sub[ref_col].to_numpy(dtype=float)
            y = sub[pred_col].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if not np.any(mask):
                continue
            ax.scatter(
                x[mask],
                y[mask],
                s=14,
                alpha=0.55,
                label=model_display_label(model),
                color=color_for_model(model),
                edgecolors="none",
            )
        lo = ax.get_xlim()
        hi = ax.get_ylim()
        lim_lo = min(lo[0], hi[0])
        lim_hi = max(lo[1], hi[1])
        if np.isfinite(lim_lo) and np.isfinite(lim_hi) and lim_hi > lim_lo:
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="#888888", linewidth=0.9)
        ax.set_xlabel(f"Reference {outcome}")
        ax.set_ylabel(f"Predicted {outcome}")
        ax.set_title(outcome)
        ax.grid(True, linestyle="--", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(order), 4), bbox_to_anchor=(0.5, 0.99), fontsize=FS_LEGEND)
    finalize_figure_layout(fig, rect=(0, 0, 1, 0.90))
    save_figure(fig, outpath)
    plt.close(fig)


def plot_model_metric_combined_panel(
    summary_df: pd.DataFrame,
    per_target_df: pd.DataFrame,
    outpath: str,
    *,
    summary_metric_col: str,
    summary_ylabel: str,
    summary_title: str,
    heatmap_value_col: str,
    heatmap_cbar_label: str,
    model_order: List[str],
    significance_pairs: Optional[List[Tuple[str, str, str]]] = None,
) -> None:
    """Left: aggregate bar (mean ± 95% CI); right: per-target heatmap."""
    apply_matplotlib_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.2), gridspec_kw={"width_ratios": [1.05, 1.15]})
    plot_metric_barplot(
        summary_df[summary_df["model"].isin(model_order)],
        outpath,
        metric_col=summary_metric_col,
        ylabel=summary_ylabel,
        title=summary_title,
        model_order=model_order,
        significance_pairs=significance_pairs,
        ax=axes[0],
        significance_bracket_step_scale=1.35,
    )
    plot_per_target_metric_heatmap(
        per_target_df,
        outpath,
        model_order,
        heatmap_value_col,
        heatmap_cbar_label,
        ax=axes[1],
    )
    axes[1].set_title(f"Per-target {heatmap_cbar_label}")
    axes[0].set_title(summary_title, pad=2)
    finalize_figure_layout(fig, rect=(0.0, 0.14, 1.0, 1.0))
    save_figure(fig, outpath)
    plt.close(fig)


def plot_prediction_error_heatmap(per_target_df: pd.DataFrame, outpath: str, models: List[str]) -> None:
    plot_per_target_metric_heatmap(
        per_target_df,
        outpath,
        models,
        "RMSE_original",
        "RMSE (original scale)",
    )



def _aggregate_uncertainty_by_target(
    summary_df: pd.DataFrame,
    case_df: pd.DataFrame,
) -> pd.DataFrame:
    """Mean uncertainty stats per Tthr target; 95% CI across repeats when available."""
    value_cols = (
        "mean_aleatoric_std",
        "mean_epistemic_std",
        "mean_total_std",
        "mean_epistemic_fraction",
    )
    if not summary_df.empty and "target" in summary_df.columns:
        df = summary_df.copy()
        cols = [c for c in value_cols if c in df.columns]
        if not cols:
            df = case_df.groupby("target", as_index=False).agg(
                mean_aleatoric_std=("aleatoric_std", "mean"),
                mean_epistemic_std=("epistemic_std", "mean"),
                mean_total_std=("total_std", "mean"),
                mean_epistemic_fraction=("epistemic_fraction", "mean"),
            )
            cols = list(value_cols)
        elif "repeat_id" in df.columns and df["repeat_id"].nunique() > 1:
            rows: List[dict] = []
            for target, grp in df.groupby("target"):
                row: dict = {"target": str(target)}
                for col in cols:
                    vals = grp[col].astype(float).dropna()
                    row[col] = float(vals.mean()) if len(vals) else float("nan")
                    if len(vals) > 1:
                        row[f"{col}_ci_half"] = float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))
                    else:
                        row[f"{col}_ci_half"] = 0.0
                rows.append(row)
            df = pd.DataFrame(rows)
        elif "repeat_id" in df.columns and df.groupby("target").size().max() > 1:
            df = df.groupby("target", as_index=False)[cols].mean(numeric_only=True)
    elif not case_df.empty and "target" in case_df.columns:
        df = case_df.groupby("target", as_index=False).agg(
            mean_aleatoric_std=("aleatoric_std", "mean"),
            mean_epistemic_std=("epistemic_std", "mean"),
            mean_total_std=("total_std", "mean"),
            mean_epistemic_fraction=("epistemic_fraction", "mean"),
        )
    else:
        return pd.DataFrame()
    preferred = [f"Tthr_{i}" for i in range(1, 6)]
    order = {t: i for i, t in enumerate(preferred)}
    df = df[df["target"].astype(str).str.startswith("Tthr")].copy()
    if df.empty:
        return df
    df["_ord"] = df["target"].map(lambda t: order.get(str(t), 99))
    return df.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(x, y)
        return float(rho) if np.isfinite(rho) else float("nan")
    except Exception:
        return float("nan")


def _pearson_r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    return r * r if np.isfinite(r) else float("nan")


def _scatter_uncertainty_trend(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    *,
    spearman_rho: Optional[float] = None,
    pearson_r2: Optional[float] = None,
    show_pearson_r2: bool = True,
    stats_x: Optional[np.ndarray] = None,
    stats_y: Optional[np.ndarray] = None,
    trend_x: Optional[np.ndarray] = None,
    trend_y: Optional[np.ndarray] = None,
) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        return
    ax.scatter(x, y, s=8, alpha=0.35, color=color, edgecolors="none")

    sx = stats_x if stats_x is not None else x
    sy = stats_y if stats_y is not None else y
    rho = spearman_rho if spearman_rho is not None else _spearman_rho(sx, sy)
    stat_lines: List[str] = []
    if show_pearson_r2:
        r2 = pearson_r2 if pearson_r2 is not None else _pearson_r2(sx, sy)
        if np.isfinite(r2):
            stat_lines.append(f"Pearson $r^2$={r2:.2f}")
    if np.isfinite(rho):
        stat_lines.append(f"Spearman $\\rho$ = {rho:.2f}")
    if stat_lines:
        ax.text(
            0.03,
            0.97,
            "\n".join(stat_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            color="0.25",
        )

    bx = np.asarray(trend_x if trend_x is not None else x, dtype=float)
    by = np.asarray(trend_y if trend_y is not None else y, dtype=float)
    bmask = np.isfinite(bx) & np.isfinite(by)
    bx, by = bx[bmask], by[bmask]
    if bx.size < 8:
        return
    order = np.argsort(bx)
    bx, by = bx[order], by[order]
    n_bins = min(15, max(5, bx.size // 40))
    edges = np.linspace(float(bx.min()), float(bx.max()), n_bins + 1)
    centers: List[float] = []
    means: List[float] = []
    stds: List[float] = []
    for i in range(n_bins):
        bin_mask = (bx >= edges[i]) & (bx < edges[i + 1] if i < n_bins - 1 else bx <= edges[i + 1])
        if int(bin_mask.sum()) < 3:
            continue
        centers.append(float(0.5 * (edges[i] + edges[i + 1])))
        means.append(float(np.mean(by[bin_mask])))
        stds.append(float(np.std(by[bin_mask])))
    if centers:
        centers_arr = np.asarray(centers, dtype=float)
        means_arr = np.asarray(means, dtype=float)
        stds_arr = np.asarray(stds, dtype=float)
        ax.fill_between(
            centers_arr,
            means_arr - stds_arr,
            means_arr + stds_arr,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(centers_arr, means_arr, color=color, linewidth=1.2, linestyle="--", alpha=0.9)


def _plot_uncertainty_by_target_bars(
    per_target: pd.DataFrame,
    targets: List[str],
    outpath: str,
    *,
    n_repeats: int = 1,
) -> None:
    if per_target.empty or len(targets) <= 1:
        return
    fig_bar, axes_bar = plt.subplots(
        1, 2, figsize=(7.4, 3.8), gridspec_kw={"width_ratios": [1.15, 0.85]},
    )
    x = np.arange(len(targets))
    ale = [float(per_target.loc[per_target["target"] == t, "mean_aleatoric_std"].iloc[0]) for t in targets]
    epi = [float(per_target.loc[per_target["target"] == t, "mean_epistemic_std"].iloc[0]) for t in targets]
    frac = [float(per_target.loc[per_target["target"] == t, "mean_epistemic_fraction"].iloc[0]) for t in targets]
    ale_err = [
        float(per_target.loc[per_target["target"] == t, "mean_aleatoric_std_ci_half"].iloc[0])
        if "mean_aleatoric_std_ci_half" in per_target.columns
        else 0.0
        for t in targets
    ]
    epi_err = [
        float(per_target.loc[per_target["target"] == t, "mean_epistemic_std_ci_half"].iloc[0])
        if "mean_epistemic_std_ci_half" in per_target.columns
        else 0.0
        for t in targets
    ]
    frac_err = [
        float(per_target.loc[per_target["target"] == t, "mean_epistemic_fraction_ci_half"].iloc[0])
        if "mean_epistemic_fraction_ci_half" in per_target.columns
        else 0.0
        for t in targets
    ]
    has_ci = n_repeats > 1 and any(v > 0 for v in ale_err + epi_err + frac_err)
    axes_bar[0].bar(
        x - 0.18,
        ale,
        width=0.36,
        label="Aleatoric",
        color=PALETTE_BLUE_LIGHT,
        yerr=ale_err if has_ci else None,
        capsize=3,
        error_kw={"linewidth": 0.9, "ecolor": "black"},
    )
    axes_bar[0].bar(
        x + 0.18,
        epi,
        width=0.36,
        label="Epistemic",
        color=PALETTE_RED_MID,
        yerr=epi_err if has_ci else None,
        capsize=3,
        error_kw={"linewidth": 0.9, "ecolor": "black"},
    )
    axes_bar[0].set_xticks(x)
    axes_bar[0].set_xticklabels(targets)
    axes_bar[0].set_ylabel("Mean std")
    axes_bar[0].set_title("Aleatoric vs epistemic by target")
    axes_bar[0].legend(frameon=False, fontsize=FS_LEGEND, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2)
    axes_bar[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes_bar[1].bar(
        x,
        frac,
        color=PALETTE_GREEN_MID,
        alpha=0.85,
        edgecolor="white",
        yerr=frac_err if has_ci else None,
        capsize=3,
        error_kw={"linewidth": 0.9, "ecolor": "black"},
    )
    axes_bar[1].set_xticks(x)
    axes_bar[1].set_xticklabels(targets)
    axes_bar[1].set_ylim(0, 1)
    axes_bar[1].set_ylabel("Epistemic / total")
    axes_bar[1].set_title("Epistemic fraction by target")
    axes_bar[1].grid(axis="y", linestyle="--", alpha=0.35)
    fig_bar.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_figure(fig_bar, outpath)
    plt.close(fig_bar)


def _ordered_tthr_targets(targets: Sequence[str]) -> List[str]:
    preferred = [f"Tthr_{i}" for i in range(1, 6)]
    found = [t for t in preferred if t in set(targets)]
    return found or sorted(set(targets))


def _subsample_for_scatter(df: pd.DataFrame, n: int = 1200, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n, random_state=seed)


def _render_uncertainty_scatter(
    ax,
    *,
    kind: str,
    full_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    show_ylabel: bool,
    show_xlabel: bool,
    title: Optional[str] = None,
) -> None:
    """Render one aleatoric or epistemic uncertainty scatter (data/stats unchanged)."""
    x_col = "aleatoric_std" if kind == "aleatoric" else "epistemic_std"
    color = PALETTE_BLUE_MID if kind == "aleatoric" else PALETTE_RED_MID
    xlabel = "Aleatoric std" if kind == "aleatoric" else "Epistemic std"
    if not sub_df.empty and x_col in full_df.columns and "abs_error" in full_df.columns:
        stats_x = full_df[x_col].to_numpy(dtype=float)
        stats_y = full_df["abs_error"].to_numpy(dtype=float)
        _scatter_uncertainty_trend(
            ax,
            sub_df[x_col].to_numpy(dtype=float),
            sub_df["abs_error"].to_numpy(dtype=float),
            color,
            spearman_rho=_spearman_rho(stats_x, stats_y),
            show_pearson_r2=False,
            stats_x=stats_x,
            stats_y=stats_y,
            trend_x=stats_x,
            trend_y=stats_y,
        )
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("|error|", fontsize=10)
    if title:
        ax.set_title(title, fontsize=10, pad=4)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="both", linestyle="--", alpha=0.25)
    for text in ax.texts:
        text.set_fontsize(9)


def plot_uncertainty_decomposition(
    case_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    outpath: str,
    *,
    method: str = "mc_dropout",
) -> None:
    """Epistemic vs aleatoric uncertainty decomposition (exploratory diagnostic).

    Layout is 4×3:
      row 1 aleatoric Tthr_1–3; row 2 epistemic Tthr_1–3;
      row 3 aleatoric Tthr_4–5 + pooled; row 4 epistemic Tthr_4–5 + pooled.
    """
    apply_matplotlib_style()
    if case_df.empty:
        return

    n_repeats = int(summary_df["repeat_id"].nunique()) if "repeat_id" in summary_df.columns else 1
    per_target = _aggregate_uncertainty_by_target(summary_df, case_df)
    tthr_case = case_df[case_df["target"].astype(str).str.startswith("Tthr")].copy()
    if tthr_case.empty and per_target.empty:
        return

    targets = _ordered_tthr_targets(
        per_target["target"].tolist() if not per_target.empty else tthr_case["target"].unique()
    )
    if not targets:
        return

    # Keep the same statistical aggregation side-plot as before.
    if not per_target.empty and len(targets) > 1:
        stem, ext = os.path.splitext(outpath)
        per_target_path = f"{stem}_by_target{ext}"
        _plot_uncertainty_by_target_bars(per_target, targets, per_target_path, n_repeats=n_repeats)

    scatter_n = 1200
    scatter_seed = 42
    target_payloads: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    ale_subs: List[pd.DataFrame] = []
    epi_subs: List[pd.DataFrame] = []
    for target in targets:
        sub = tthr_case[tthr_case["target"] == target]
        plot_full = sub[["aleatoric_std", "epistemic_std", "abs_error"]].dropna()
        plot_sub = _subsample_for_scatter(plot_full, n=scatter_n, seed=scatter_seed)
        ale_full = plot_full[["aleatoric_std", "abs_error"]]
        epi_full = plot_full[["epistemic_std", "abs_error"]]
        ale_sub = plot_sub[["aleatoric_std", "abs_error"]]
        epi_sub = plot_sub[["epistemic_std", "abs_error"]]
        target_payloads[target] = (ale_full, epi_full, ale_sub, epi_sub)
        ale_subs.append(ale_sub)
        epi_subs.append(epi_sub)

    pooled_label = "pooled"
    if len(targets) > 1:
        ale_all_full = tthr_case[["aleatoric_std", "abs_error"]].dropna()
        epi_all_full = tthr_case[["epistemic_std", "abs_error"]].dropna()
        ale_all_sub = pd.concat(ale_subs, ignore_index=True) if ale_subs else ale_all_full
        epi_all_sub = pd.concat(epi_subs, ignore_index=True) if epi_subs else epi_all_full
        target_payloads[pooled_label] = (ale_all_full, epi_all_full, ale_all_sub, epi_all_sub)

    # Column groups: [Tthr_1..3], [Tthr_4, Tthr_5, pooled]
    top_targets = list(targets[:3])
    while len(top_targets) < 3:
        top_targets.append("")
    bottom_targets = list(targets[3:5])
    while len(bottom_targets) < 2:
        bottom_targets.append("")
    bottom_targets.append(pooled_label if pooled_label in target_payloads else "")

    fig, axes = plt.subplots(4, 3, figsize=(11.2, 10.4), constrained_layout=True)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=PALETTE_BLUE_MID, label="Aleatoric"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=PALETTE_RED_MID, label="Epistemic"),
    ]

    def _draw_row(row_idx: int, kind: str, col_targets: Sequence[str]) -> None:
        for col_idx, target in enumerate(col_targets):
            ax = axes[row_idx, col_idx]
            if not target or target not in target_payloads:
                ax.set_visible(False)
                continue
            ale_full, epi_full, ale_sub, epi_sub = target_payloads[target]
            full_df = ale_full if kind == "aleatoric" else epi_full
            sub_df = ale_sub if kind == "aleatoric" else epi_sub
            title = "pooled" if target == pooled_label else target
            _render_uncertainty_scatter(
                ax,
                kind=kind,
                full_df=full_df,
                sub_df=sub_df,
                show_ylabel=(col_idx == 0),
                show_xlabel=True,
                title=title if row_idx in {0, 2} else None,
            )

    _draw_row(0, "aleatoric", top_targets)
    _draw_row(1, "epistemic", top_targets)
    _draw_row(2, "aleatoric", bottom_targets)
    _draw_row(3, "epistemic", bottom_targets)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("Uncertainty decomposition", fontsize=FS_TITLE, y=1.04)
    save_figure(fig, outpath)
    plt.close(fig)


# --- ODE / closed-loop trajectory plots ---

N_ODE_STRAINS = 5
TRAJECTORY_HISTORY_COLS = ["time_h", "C_ug_per_mL", "P_total_CFU_per_mL"] + [
    f"B_total_{i}_CFU_per_mL" for i in range(1, N_ODE_STRAINS + 1)
]


def trajectory_history_from_arrays(
    times: np.ndarray,
    C: np.ndarray,
    P_total: np.ndarray,
    B_total: np.ndarray,
) -> pd.DataFrame:
    data: dict = {
        "time_h": np.asarray(times, dtype=float),
        "C_ug_per_mL": np.asarray(C, dtype=float),
        "P_total_CFU_per_mL": np.asarray(P_total, dtype=float),
    }
    b_total = np.asarray(B_total, dtype=float)
    for i in range(N_ODE_STRAINS):
        data[f"B_total_{i + 1}_CFU_per_mL"] = b_total[:, i]
    return pd.DataFrame(data)


def read_t_thr_from_manifest(manifest: dict, model: Optional[str] = None) -> np.ndarray:
    if model and isinstance(manifest.get("t_thr_by_model"), dict):
        values = manifest["t_thr_by_model"].get(model)
        if values is not None:
            return np.asarray(values, dtype=float)
    values = manifest.get("T_thr") or manifest.get("t_thr")
    if values is None:
        raise KeyError("Manifest missing T_thr / t_thr (or t_thr_by_model).")
    return np.asarray(values, dtype=float)


def plot_representative_ode_trajectory(
    history_df: pd.DataFrame,
    t_thr: np.ndarray,
    outpath: str,
    *,
    total_dosage: float,
    probiotic_two_compartment: bool = False,
) -> None:
    """Fig. 3-style single-controller ODE trajectory (representative forward simulation)."""
    apply_matplotlib_style()
    times = history_df["time_h"].to_numpy(dtype=float)
    C = history_df["C_ug_per_mL"].to_numpy(dtype=float)
    P_total = history_df["P_total_CFU_per_mL"].to_numpy(dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(times, C, color=TRAJECTORY_AMP, linewidth=1.8)
    axes[0].set_ylabel(r"AMP ($\mu$g mL$^{-1}$)", labelpad=10)
    axes[0].set_title(f"AMP concentration ({total_dosage:.1f} µg/mL total)", fontsize=FS_TRAJECTORY_TITLE)

    axes[1].plot(times, P_total / 1e6, color=TRAJECTORY_PROBIOTIC, linewidth=1.8)
    axes[1].set_ylabel(r"Probiotic ($\times 10^6$ CFU mL$^{-1}$)", labelpad=10)
    axes[1].set_title(
        "Probiotic (total, S/R)" if probiotic_two_compartment else "Probiotic",
        fontsize=FS_TRAJECTORY_TITLE,
    )

    pathogen_colors = TRAJECTORY_PATHOGENS[:N_ODE_STRAINS]
    for i in range(N_ODE_STRAINS):
        color_i = pathogen_colors[i]
        b_col = f"B_total_{i + 1}_CFU_per_mL"
        axes[2].plot(times, history_df[b_col].to_numpy(dtype=float) / 1e6, color=color_i, linewidth=1.8, label=f"B{i + 1}")
        axes[2].hlines(
            t_thr[i] / 1e6,
            times[0],
            times[-1],
            colors=color_i,
            linestyles="dashed",
            linewidth=0.9,
            alpha=0.55,
            label="_nolegend_",
        )
    axes[2].set_xlabel("Time (h)")
    axes[2].set_ylabel(r"Pathogen strain ($\times 10^6$ CFU mL$^{-1}$)", labelpad=10)
    axes[2].set_title("Pathogens", fontsize=FS_TRAJECTORY_TITLE)
    axes[2].legend(ncol=5, fontsize=FS_LEGEND, frameon=False, loc="upper right")
    finalize_figure_layout(fig, h_pad=2.4)
    save_figure(fig, outpath)
    plt.close(fig)




def _closed_loop_display_label(model_name: str) -> str:
    try:
        from closed_loop_eval import CLOSED_LOOP_DISPLAY_LABELS
        return CLOSED_LOOP_DISPLAY_LABELS.get(model_name, model_name)
    except ImportError:
        return model_name


def _first_trajectory_segment(sub: pd.DataFrame) -> pd.DataFrame:
    """Keep one ODE trajectory when CSV accidentally concatenates duplicate segments."""
    sub = sub.sort_values("time_h").reset_index(drop=True)
    if sub.empty:
        return sub
    times = sub["time_h"].to_numpy(dtype=float)
    breaks = np.where(np.diff(times) < -1e-9)[0]
    if len(breaks):
        return sub.iloc[: breaks[0] + 1].copy()
    return sub


def plot_fixed_umax_representative(
    trajectories_df: pd.DataFrame,
    model_labels: Sequence[str],
    t_thr_by_model: Dict[str, np.ndarray],
    outpath: str,
    *,
    display_labels: Optional[Sequence[str]] = None,
) -> None:
    """Fig. 4A: TAR / BestTree / UniformTreeMean at shared fixed Umax; only Tthr differs."""
    order = [m for m in FIG4_FIXED_UMAX_MODELS if m in model_labels or m in trajectories_df["model"].unique()]
    model_labels = order[:3]
    if display_labels is None:
        display_labels = [_closed_loop_display_label(m) for m in model_labels]
    else:
        display_labels = list(display_labels)[: len(model_labels)]
    apply_matplotlib_style()
    n_cols = min(3, len(model_labels))
    fig, axes = plt.subplots(3, n_cols, figsize=(5.5 * n_cols, 8), sharex="col")
    if n_cols == 1:
        axes = np.array(axes).reshape(3, 1)
    pathogen_colors = TRAJECTORY_PATHOGENS[:N_ODE_STRAINS]
    legend_handles = None
    legend_labels = None

    for col, model_name in enumerate(model_labels[:n_cols]):
        sub = _first_trajectory_segment(trajectories_df[trajectories_df["model"] == model_name])
        if sub.empty:
            continue
        times = sub["time_h"].to_numpy(dtype=float)
        C = sub["C_ug_per_mL"].to_numpy(dtype=float)
        P_total = sub["P_total_CFU_per_mL"].to_numpy(dtype=float)
        t_thr = np.asarray(t_thr_by_model[model_name], dtype=float)
        title = display_labels[col] if col < len(display_labels) else _closed_loop_display_label(model_name)

        axes[0, col].plot(times, C, linewidth=1.8, color=TRAJECTORY_AMP)
        axes[0, col].set_title(title, fontsize=FS_TRAJECTORY_TITLE, pad=8)
        if col == 0:
            axes[0, col].set_ylabel(r"AMP ($\mu$g mL$^{-1}$)", labelpad=10)

        axes[1, col].plot(times, P_total / 1e6, linewidth=1.8, color=TRAJECTORY_PROBIOTIC)
        if col == 0:
            axes[1, col].set_ylabel(r"Probiotic ($\times10^6$ CFU mL$^{-1}$)", labelpad=10)

        for i in range(N_ODE_STRAINS):
            color_i = pathogen_colors[i]
            b_col = f"B_total_{i + 1}_CFU_per_mL"
            axes[2, col].plot(
                times,
                sub[b_col].to_numpy(dtype=float) / 1e6,
                linewidth=1.6,
                color=color_i,
                label=f"B{i + 1}" if col == 0 else "_nolegend_",
            )
            axes[2, col].hlines(
                t_thr[i] / 1e6, times[0], times[-1], colors=color_i, linestyles="dashed", linewidth=0.9, alpha=0.45
            )
        axes[2, col].set_xlabel("Time (h)")
        if col == 0:
            axes[2, col].set_ylabel(r"Pathogen strain ($\times10^6$ CFU mL$^{-1}$)", labelpad=10)
            legend_handles, legend_labels = axes[2, col].get_legend_handles_labels()

    for row in range(3):
        ylims = [axes[row, c].get_ylim() for c in range(n_cols)]
        ymin = min(0.0, *(y[0] for y in ylims))
        ymax = max(y[1] for y in ylims)
        for c in range(n_cols):
            axes[row, c].set_ylim(ymin, ymax * 1.03)
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            ncol=5,
            fontsize=FS_LEGEND,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
        )
    finalize_figure_layout(fig, rect=(0.0, 0.08, 1.0, 0.98), h_pad=2.0, w_pad=1.4)
    save_figure(fig, outpath)
    plt.close(fig)




def closed_loop_significance_pairs_for_plot(
    annotations_df: pd.DataFrame,
    srl_model: str,
    metric: str = "mean_composite_score",
) -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []
    if annotations_df.empty:
        return pairs
    sub = annotations_df[
        (annotations_df["metric"] == metric) & (annotations_df["srl_model"] == srl_model)
    ]
    for _, row in sub.iterrows():
        if bool(row.get("exploratory", False)):
            continue
        plot_label = str(row.get("plot_label", ""))
        if plot_label in {"", "control_better", "ns", "na", "nan"}:
            continue
        pairs.append((srl_model, str(row["control_model"]), plot_label))
    return pairs


def _bar_ci_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    if not all(np.isfinite(v) for v in (lo_a, hi_a, lo_b, hi_b)):
        return True
    return not (hi_a < lo_b or hi_b < lo_a)


def _bracket_yspan(ymin: float, ymax: float) -> float:
    span = max(float(ymax - ymin), 0.0)
    if span <= 0:
        return max(0.05 * max(abs(ymax), 1.0), 1e-12)
    ref = max(abs(ymax), abs(ymin), 1e-12)
    if span / ref < 0.02:
        return span
    return max(span, 0.05)


def closed_loop_significance_pairs_from_bar_stats(
    plot_df: pd.DataFrame,
    metrics: Sequence[str],
    *,
    srl_model: str,
    repeat_pairs_by_metric: Optional[Dict[str, List[Tuple[str, str, str]]]] = None,
) -> Dict[str, List[Tuple[str, str, str]]]:
    """Gate repeat-level stars with non-overlapping aggregate bar CIs (Fig. 4B small-effect metrics)."""
    try:
        from closed_loop_eval import CLOSED_LOOP_METRIC_DIRECTION, CLOSED_LOOP_SIGNIFICANCE_CONTROLS
    except ImportError:
        return {}
    direction_map = dict(CLOSED_LOOP_METRIC_DIRECTION)
    repeat_pairs_by_metric = repeat_pairs_by_metric or {}
    srl_row = plot_df[plot_df["model"] == srl_model]
    if srl_row.empty:
        return {}
    srl_row = srl_row.iloc[0]
    pairs_by_metric: Dict[str, List[Tuple[str, str, str]]] = {}

    for metric in metrics:
        if metric not in plot_df.columns:
            continue
        ci_lo_col = f"{metric}_ci_low"
        ci_hi_col = f"{metric}_ci_high"
        if ci_lo_col not in plot_df.columns or ci_hi_col not in plot_df.columns:
            continue
        direction = direction_map.get(metric, "lower_is_better")
        tar_lo = float(srl_row[ci_lo_col])
        tar_hi = float(srl_row[ci_hi_col])
        tar_mean = float(srl_row[metric])
        repeat_pairs = {
            (a, b): label for a, b, label in repeat_pairs_by_metric.get(metric, []) if a == srl_model
        }
        metric_pairs: List[Tuple[str, str, str]] = []
        for control in CLOSED_LOOP_SIGNIFICANCE_CONTROLS:
            if control == srl_model or control not in plot_df["model"].values:
                continue
            if (srl_model, control) not in repeat_pairs:
                continue
            ctrl_row = plot_df[plot_df["model"] == control].iloc[0]
            ctrl_lo = float(ctrl_row[ci_lo_col])
            ctrl_hi = float(ctrl_row[ci_hi_col])
            ctrl_mean = float(ctrl_row[metric])
            if _bar_ci_overlap(tar_lo, tar_hi, ctrl_lo, ctrl_hi):
                continue
            tar_better = tar_mean > ctrl_mean if direction == "higher_is_better" else tar_mean < ctrl_mean
            if not tar_better:
                continue
            metric_pairs.append((srl_model, control, repeat_pairs[(srl_model, control)]))
        if metric_pairs:
            pairs_by_metric[metric] = metric_pairs
    return pairs_by_metric


def closed_loop_significance_pairs_from_repeats(
    outdir: str,
    metrics: Sequence[str],
    *,
    srl_model: str,
    n_repeats: int,
    permutation_replicates: int = 5000,
) -> Dict[str, List[Tuple[str, str, str]]]:
    """Plot-only: derive TAR-vs-control significance for extra Fig. 4B metrics."""
    if n_repeats < 10:
        return {}
    import re

    try:
        from closed_loop_eval import (
            CLOSED_LOOP_METRIC_DIRECTION,
            CLOSED_LOOP_SIGNIFICANCE_CONTROLS,
            comparison_result_label,
            evaluate_significance_label,
            permutation_test_mean_diff,
        )
    except ImportError:
        return {}

    paths = sorted(
        glob.glob(
            os.path.join(
                outdir, "repeats", "repeat_*", FIXED_UMAX_VALIDATION_SUBDIR,
                "fixed_umax_validation_summary_by_model.csv",
            )
        )
    )
    if not paths:
        return {}
    frames: List[pd.DataFrame] = []
    for path in paths:
        match = re.search(r"repeat_(\d+)", path.replace("\\", "/"))
        repeat_id = int(match.group(1)) if match else 0
        frame = pd.read_csv(path)
        frame["repeat_id"] = repeat_id
        frames.append(frame)
    long_df = pd.concat(frames, ignore_index=True)
    direction_map = dict(CLOSED_LOOP_METRIC_DIRECTION)
    pairs_by_metric: Dict[str, List[Tuple[str, str, str]]] = {}

    for metric in metrics:
        if metric not in long_df.columns:
            continue
        metric_pairs: List[Tuple[str, str, str]] = []
        for control in CLOSED_LOOP_SIGNIFICANCE_CONTROLS:
            if control == srl_model:
                continue
            srl_vals = (
                long_df[long_df["model"] == srl_model]
                .sort_values("repeat_id")[metric]
                .to_numpy(dtype=float)
            )
            ctrl_vals = (
                long_df[long_df["model"] == control]
                .sort_values("repeat_id")[metric]
                .to_numpy(dtype=float)
            )
            n = min(len(srl_vals), len(ctrl_vals))
            if n <= 1:
                continue
            srl_vals = srl_vals[:n]
            ctrl_vals = ctrl_vals[:n]
            diff = srl_vals - ctrl_vals
            direction = direction_map.get(metric, "lower_is_better")
            srl_better = float(np.mean(diff)) < 0 if direction == "lower_is_better" else float(np.mean(diff)) > 0
            ci_low = float(np.percentile(diff, 2.5))
            ci_high = float(np.percentile(diff, 97.5))
            try:
                from scipy.stats import wilcoxon

                wilcoxon_p = float(wilcoxon(diff).pvalue) if not np.allclose(diff, 0.0) else 1.0
            except Exception:
                wilcoxon_p = float("nan")
            perm_p = permutation_test_mean_diff(
                diff,
                permutation_replicates,
                seed=abs(hash(metric + control)) % 10000,
            )
            p_for_label = float(min(p for p in (wilcoxon_p, perm_p) if np.isfinite(p)))
            star_label, significance_tier = evaluate_significance_label(
                p_for_label,
                ci_low,
                ci_high,
                n_repeats=n,
                srl_better=srl_better,
                single_split_exploratory=n_repeats == 1,
            )
            comp = comparison_result_label(star_label, significance_tier, srl_better, n_repeats)
            if comp in {"control_better", "not_significant"}:
                continue
            if star_label in {"", "ns", "na", "nan"}:
                continue
            metric_pairs.append((srl_model, control, star_label))
        if metric_pairs:
            pairs_by_metric[metric] = metric_pairs
    return pairs_by_metric


def _reorder_bar_plot_df(plot_df: pd.DataFrame, order: Optional[List[str]] = None) -> pd.DataFrame:
    """Keep only models in manuscript order: TAR, RF, UniformTreeMean, Best tree."""
    order = order or list(MANUSCRIPT_BAR_MODEL_ORDER)
    available = plot_df["model"].tolist()
    ordered = [m for m in order if m in available]
    if not ordered:
        return plot_df
    return plot_df.set_index("model").loc[ordered].reset_index()


def _plot_metric_panel_with_ci(
    ax,
    plot_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    *,
    significance_pairs: Optional[List[Tuple[str, str, str]]] = None,
    significance_bracket_step_scale: float = 1.25,
    focus_ylim: bool = False,
    ylim_pad_fraction: float = 0.18,
    baseline_focus_bars: bool = False,
    ylim_fixed: Optional[Tuple[float, float]] = None,
    display_labels: Optional[Sequence[str]] = None,
    bar_colors: Optional[Sequence[str]] = None,
    horizontal: bool = False,
    horizontal_bar_height: float = 0.92,
) -> None:
    model_names = plot_df["model"].tolist()
    if display_labels is None:
        display_labels = [_closed_loop_display_label(m) for m in model_names]
    else:
        display_labels = list(display_labels)
    x = np.arange(len(plot_df))
    y = plot_df[metric_col].to_numpy(dtype=float)
    colors = list(bar_colors) if bar_colors is not None else colors_for_models(model_names)

    ci_low_col = f"{metric_col}_ci_low"
    ci_high_col = f"{metric_col}_ci_high"
    bar_tops = y.copy()
    bar_bottoms = y.copy()
    lo = hi = None
    if {ci_low_col, ci_high_col}.issubset(plot_df.columns):
        lo = plot_df[ci_low_col].to_numpy(dtype=float)
        hi = plot_df[ci_high_col].to_numpy(dtype=float)
        finite_lo = np.isfinite(lo)
        finite_hi = np.isfinite(hi)
        bar_tops = np.where(finite_hi, hi, bar_tops)
        bar_bottoms = np.where(finite_lo, lo, bar_bottoms)

    bar_baseline = 0.0
    if baseline_focus_bars:
        data_bottom = float(np.nanmin(bar_bottoms)) if np.any(np.isfinite(bar_bottoms)) else float(np.nanmin(y))
        data_top = float(np.nanmax(bar_tops)) if np.any(np.isfinite(bar_tops)) else float(np.nanmax(y))
        span = max(data_top - data_bottom, 1e-12 * max(abs(data_top), 1.0))
        bar_baseline = data_bottom - ylim_pad_fraction * span

    bar_heights = y - bar_baseline if baseline_focus_bars else y
    if horizontal:
        if significance_pairs:
            raise ValueError("_plot_metric_panel_with_ci: horizontal bars do not support significance brackets")
        y_pos = x
        bars = ax.barh(
            y_pos,
            bar_heights,
            left=bar_baseline if baseline_focus_bars else 0.0,
            edgecolor="white",
            linewidth=1.0,
            color=colors,
            height=horizontal_bar_height,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_labels, fontsize=10)
        ax.set_xlabel(ylabel)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(y=0.06)
        if lo is not None and hi is not None and np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
            xerr = np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])
            centers = [bar.get_y() + bar.get_height() / 2 for bar in bars]
            ax.errorbar(y, centers, xerr=xerr, fmt="none", capsize=4, linewidth=1.0, color="black")
        if ylim_fixed is not None:
            ax.set_xlim(ylim_fixed[0], ylim_fixed[1])
        elif baseline_focus_bars:
            data_left = float(np.nanmin(bar_bottoms)) if np.any(np.isfinite(bar_bottoms)) else float(np.nanmin(y))
            data_right = float(np.nanmax(bar_tops)) if np.any(np.isfinite(bar_tops)) else float(np.nanmax(y))
            span = max(data_right - data_left, 1e-12 * max(abs(data_right), 1.0))
            pad = 0.28 * span
            ax.set_xlim(bar_baseline, data_right + pad)
        elif focus_ylim:
            data_left = float(np.nanmin(bar_bottoms)) if np.any(np.isfinite(bar_bottoms)) else float(np.nanmin(y))
            data_right = float(np.nanmax(bar_tops)) if np.any(np.isfinite(bar_tops)) else float(np.nanmax(y))
            if not np.isfinite(data_left) or not np.isfinite(data_right):
                return
            span = max(data_right - data_left, 1e-9 * max(abs(data_right), 1.0))
            pad = ylim_pad_fraction * span
            ax.set_xlim(data_left - pad, data_right + pad)
        return

    bars = ax.bar(
        x,
        bar_heights,
        bottom=bar_baseline if baseline_focus_bars else 0.0,
        edgecolor="white",
        linewidth=1.0,
        color=colors,
        width=0.72,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=35, ha="right", fontsize=FS_TICK_SM)
    ax.tick_params(axis="x", pad=5)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if lo is not None and hi is not None and np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
        yerr = np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])
        centers = [bar.get_x() + bar.get_width() / 2 for bar in bars]
        ax.errorbar(centers, y, yerr=yerr, fmt="none", capsize=4, linewidth=1.0, color="black")

    if significance_pairs:
        model_to_x = {model: idx for idx, model in enumerate(plot_df["model"])}
        valid_pairs = [
            (a, b, lbl)
            for a, b, lbl in significance_pairs
            if lbl not in {"ns", "na", ""} and a in model_to_x and b in model_to_x
        ]
        levels = _assign_significance_bracket_levels(valid_pairs, model_to_x)
        ymin = float(np.nanmin(bar_bottoms)) if np.any(np.isfinite(bar_bottoms)) else 0.0
        ymax = float(np.nanmax(bar_tops)) if np.any(np.isfinite(bar_tops)) else 0.0
        yspan = _bracket_yspan(ymin, ymax)
        bracket_padding = 0.14 * yspan
        bracket_step = 0.22 * yspan * significance_bracket_step_scale
        bracket_base = ymax + bracket_padding
        max_bracket_top = ymax
        tick_h = 0.04 * yspan
        for (model_a, model_b, label), level in zip(valid_pairs, levels):
            if level < 0:
                continue
            x1, x2 = sorted([model_to_x[model_a], model_to_x[model_b]])
            y_bracket = bracket_base + level * bracket_step
            ax.plot(
                [x1, x1, x2, x2],
                [y_bracket, y_bracket + tick_h, y_bracket + tick_h, y_bracket],
                color="black",
                linewidth=1.0,
            )
            ax.text(
                (x1 + x2) / 2.0,
                y_bracket + tick_h + 0.015 * yspan,
                label,
                ha="center",
                va="bottom",
                fontsize=FS_SIG,
                fontweight="bold",
            )
            max_bracket_top = max(max_bracket_top, y_bracket + tick_h + 0.08 * yspan)
        if valid_pairs:
            ax.set_ylim(bottom=ymin - 0.05 * yspan, top=max_bracket_top + 0.05 * yspan)

    if ylim_fixed is not None:
        ax.set_ylim(ylim_fixed[0], ylim_fixed[1])
    elif baseline_focus_bars:
        data_bottom = float(np.nanmin(bar_bottoms)) if np.any(np.isfinite(bar_bottoms)) else float(np.nanmin(y))
        data_top = float(np.nanmax(bar_tops)) if np.any(np.isfinite(bar_tops)) else float(np.nanmax(y))
        span = max(data_top - data_bottom, 1e-12 * max(abs(data_top), 1.0))
        pad = 0.28 * span
        ylim_low = bar_baseline
        ylim_high = data_top + pad
        if significance_pairs:
            _, bracket_top = ax.get_ylim()
            if np.isfinite(bracket_top) and bracket_top > ylim_high:
                ylim_high = bracket_top + 0.06 * span
        ax.set_ylim(ylim_low, ylim_high)
    elif focus_ylim:
        data_bottom = float(np.nanmin(bar_bottoms)) if np.any(np.isfinite(bar_bottoms)) else float(np.nanmin(y))
        data_top = float(np.nanmax(bar_tops)) if np.any(np.isfinite(bar_tops)) else float(np.nanmax(y))
        if not np.isfinite(data_bottom) or not np.isfinite(data_top):
            return
        span = max(data_top - data_bottom, 1e-9 * max(abs(data_top), 1.0))
        pad = ylim_pad_fraction * span
        bottom = data_bottom - pad
        top = data_top + pad
        if significance_pairs:
            _, bracket_top = ax.get_ylim()
            if np.isfinite(bracket_top) and bracket_top > top:
                top = bracket_top + 0.04 * span
        ax.set_ylim(bottom, top)


def _augment_closed_loop_plot_stats(stats_df: pd.DataFrame, outdir: str) -> pd.DataFrame:
    """Add plot-only metrics (e.g. mean_P_AUC, mean_LR) from per-repeat summaries when missing."""
    extra_metrics = ("mean_P_AUC", "mean_LR")
    if all(col in stats_df.columns for col in extra_metrics):
        return stats_df
    import re

    paths = sorted(
        glob.glob(
            os.path.join(
                outdir, "repeats", "repeat_*", FIXED_UMAX_VALIDATION_SUBDIR,
                "fixed_umax_validation_summary_by_model.csv",
            )
        )
    )
    if not paths:
        return stats_df
    frames: List[pd.DataFrame] = []
    for path in paths:
        match = re.search(r"repeat_(\d+)", path.replace("\\", "/"))
        repeat_id = int(match.group(1)) if match else 0
        frame = pd.read_csv(path)
        frame["repeat_id"] = repeat_id
        frames.append(frame)
    long_df = pd.concat(frames, ignore_index=True)
    try:
        from closed_loop_eval import repeat_metric_ci
    except ImportError:
        return stats_df
    rows: List[dict] = []
    for model, group in long_df.groupby("model"):
        row: dict = {"model": model}
        for col in extra_metrics:
            if col not in group.columns:
                continue
            mean, lo, hi = repeat_metric_ci(group[col].to_numpy(dtype=float))
            row[col] = mean
            row[f"{col}_ci_low"] = lo
            row[f"{col}_ci_high"] = hi
        rows.append(row)
    if not rows:
        return stats_df
    extra_df = pd.DataFrame(rows)
    merged = stats_df.merge(extra_df, on="model", how="left")
    return merged


def plot_fixed_umax_summary(
    stats_df: pd.DataFrame,
    outpath: str,
    *,
    n_repeats: int = 1,
    significance_pairs: Optional[object] = None,
    model_order: Optional[List[str]] = None,
    outdir: Optional[str] = None,
) -> None:
    """Fig. 4B: fixed-Umax forward ODE summary (point estimates; no repeat-level ODE CI)."""
    apply_matplotlib_style()
    if outdir:
        stats_df = _augment_closed_loop_plot_stats(stats_df, outdir)
    order = model_order or list(FIG4_FIXED_UMAX_MODELS)
    plot_df = _reorder_bar_plot_df(stats_df.copy(), order)

    metric_specs = [
        ("mean_total_dosage", "Total dosage ($\\mu$g mL$^{-1}$)"),
        ("mean_P_AUC", "Mean $P_{AUC}$"),
        ("mean_LR", "Mean LR"),
        ("mean_terminal_pathogen", "Terminal pathogen (CFU mL$^{-1}$)"),
    ]
    n_panels = sum(1 for col, _ in metric_specs if col in plot_df.columns)
    fig, axes = plt.subplots(1, max(n_panels, 1), figsize=(3.6 * max(n_panels, 1), 5.2))
    if n_panels == 1:
        axes = np.array([axes])
    significance_pairs_by_metric = significance_pairs if isinstance(significance_pairs, dict) else {}
    panel_idx = 0
    repeat_sig = closed_loop_significance_pairs_from_repeats(
        outdir,
        tuple(col for col, _ in metric_specs),
        srl_model=TAR_MODEL,
        n_repeats=n_repeats,
    ) if outdir and n_repeats >= 10 else {}
    gated_extra = closed_loop_significance_pairs_from_bar_stats(
        plot_df,
        tuple(col for col, _ in metric_specs),
        srl_model=TAR_MODEL,
        repeat_pairs_by_metric=repeat_sig,
    )
    significance_pairs_by_metric = {**significance_pairs_by_metric, **gated_extra}

    for col, ylabel in metric_specs:
        if col not in plot_df.columns:
            continue
        ax = axes[panel_idx]
        panel_idx += 1
        pairs = significance_pairs_by_metric.get(col)
        _plot_metric_panel_with_ci(
            ax,
            plot_df,
            col,
            ylabel,
            significance_pairs=pairs,
            focus_ylim=True,
            ylim_pad_fraction=0.18,
            baseline_focus_bars=(col == "mean_P_AUC"),
            significance_bracket_step_scale=1.35,
        )

    for ax in axes[panel_idx:]:
        ax.set_visible(False)

    if n_repeats <= 1:
        fig.suptitle(
            "Fixed Umax forward comparison (3 ODE runs; point estimates, no repeat-level CI)",
            fontsize=FS_TITLE,
            y=0.99,
        )

    finalize_figure_layout(fig, rect=(0.0, 0.14, 1.0, 0.96 if n_repeats <= 1 else 1.0))
    save_figure(fig, outpath)
    plt.close(fig)





_LANDSCAPE_CURVE_COLS = (
    "repeat_id",
    "case_index",
    "validation_original_row_index",
    "unit_id",
    "candidate_u_max",
    "optimization_penalty_score",
    "aspiration_abs_distance_to_initial",
    "composite_penalty",
    "is_initial_aspiration_dominator",
    "is_closest_to_initial_aspiration",
    "is_minimum_score",
)

_CANDIDATES_PLOT_COLS = (
    "repeat_id",
    "case_index",
    "validation_original_row_index",
    "unit_id",
    "model",
    "base_model",
    "ablation_condition",
    "candidate_u_max",
    "aspiration_dose_term",
    "aspiration_pauc_term",
    "aspiration_lr_term",
    "aspiration_pathogen_log_term",
    "LR_shortfall_norm",
    "PAUC_shortfall_norm",
    "pathogen_violation_norm",
    "dose_burden_norm",
    "composite_penalty",
    "optimization_penalty_score",
    "selected_by_optimizer",
    "is_minimum_score",
    "LR_constraint_satisfied",
    "P_AUC_constraint_satisfied",
    "pathogen_constraint_satisfied",
    "feasible_candidate",
    "LR1",
    "LR2",
    "LR3",
    "LR4",
    "LR5",
    "target_LR1",
    "target_LR2",
    "target_LR3",
    "target_LR4",
    "target_LR5",
    "P_AUC",
    "minimum_acceptable_P_AUC",
    "terminal_total_pathogen",
    "target_terminal_pathogen",
)

_RESPONSE_LANDSCAPE_PLOT_COLS = _CANDIDATES_PLOT_COLS + (
    "umax_policy",
    "aspiration_abs_distance_to_initial",
)


def _filter_tar_optimized_landscape(df: pd.DataFrame) -> pd.DataFrame:
    """Keep TAR optimized landscape rows using the best available identifier column."""
    if df.empty:
        return df
    work = df.copy()
    if "model" in work.columns:
        return work[work["model"] == "TAR"].copy()
    if "base_model" in work.columns:
        return work[work["base_model"] == "TAR"].copy()
    if "ablation_condition" in work.columns:
        return work[work["ablation_condition"] == "TAR_optimized"].copy()
    return work


def _resolve_landscape_score_col(df: pd.DataFrame) -> str:
    for col in ("composite_penalty", "optimization_penalty_score", "aspiration_abs_distance_to_initial"):
        if col in df.columns and df[col].notna().any():
            return col
    return ""


def _aggregate_penalty_landscape_summary(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return (
        df.groupby("candidate_u_max", as_index=False)[score_col]
        .agg(
            median="median",
            q25=lambda s: float(np.quantile(s, 0.25)),
            q75=lambda s: float(np.quantile(s, 0.75)),
            q10=lambda s: float(np.quantile(s, 0.10)),
            q90=lambda s: float(np.quantile(s, 0.90)),
        )
        .sort_values("candidate_u_max")
    )


def _extract_selected_umax_values(df: pd.DataFrame) -> np.ndarray:
    work = _ensure_umax_unit_id(df)
    group_col = "unit_id" if "unit_id" in work.columns else None
    for flag_col in ("selected_by_optimizer", "is_minimum_score"):
        if flag_col not in work.columns:
            continue
        flagged = work[work[flag_col].astype(bool)]
        if flagged.empty:
            continue
        if group_col:
            vals = flagged.groupby(group_col, sort=False)["candidate_u_max"].first().dropna().astype(float)
        else:
            vals = flagged["candidate_u_max"].dropna().astype(float)
        if not vals.empty:
            return vals.to_numpy(dtype=float)
    return np.array([], dtype=float)


def _assign_umax_constraint_flags(df: pd.DataFrame, *, lr_tolerance: float = 0.0) -> pd.DataFrame:
    work = df.copy()

    def _row_lr_ok(row: pd.Series) -> bool:
        if "LR_constraint_satisfied" in row.index and pd.notna(row["LR_constraint_satisfied"]):
            return bool(row["LR_constraint_satisfied"])
        for i in range(1, 6):
            lr_col, tgt_col = f"LR{i}", f"target_LR{i}"
            if lr_col not in row.index or tgt_col not in row.index:
                return False
            if not np.isfinite(row[lr_col]) or not np.isfinite(row[tgt_col]):
                return False
            if float(row[lr_col]) < float(row[tgt_col]) - lr_tolerance:
                return False
        return True

    def _row_pauc_ok(row: pd.Series) -> bool:
        if "P_AUC_constraint_satisfied" in row.index and pd.notna(row["P_AUC_constraint_satisfied"]):
            return bool(row["P_AUC_constraint_satisfied"])
        if "P_AUC" not in row.index or "minimum_acceptable_P_AUC" not in row.index:
            return False
        if not np.isfinite(row["P_AUC"]) or not np.isfinite(row["minimum_acceptable_P_AUC"]):
            return False
        return float(row["P_AUC"]) >= float(row["minimum_acceptable_P_AUC"])

    def _row_pathogen_ok(row: pd.Series) -> bool:
        if "pathogen_constraint_satisfied" in row.index and pd.notna(row["pathogen_constraint_satisfied"]):
            return bool(row["pathogen_constraint_satisfied"])
        term_col = "terminal_total_pathogen"
        if term_col not in row.index and "final_total_pathogen" in row.index:
            term_col = "final_total_pathogen"
        if term_col not in row.index or "target_terminal_pathogen" not in row.index:
            return False
        if not np.isfinite(row[term_col]) or not np.isfinite(row["target_terminal_pathogen"]):
            return False
        return float(row[term_col]) <= float(row["target_terminal_pathogen"])

    work["_lr_ok"] = work.apply(_row_lr_ok, axis=1)
    work["_pauc_ok"] = work.apply(_row_pauc_ok, axis=1)
    work["_pathogen_ok"] = work.apply(_row_pathogen_ok, axis=1)
    if "feasible_candidate" in work.columns and work["feasible_candidate"].notna().any():
        work["_all_ok"] = work["feasible_candidate"].astype(bool)
    else:
        work["_all_ok"] = work["_lr_ok"] & work["_pauc_ok"] & work["_pathogen_ok"]
    return work


def _constraint_feasibility_by_u(df: pd.DataFrame) -> pd.DataFrame:
    work = _assign_umax_constraint_flags(df)
    return (
        work.groupby("candidate_u_max", as_index=False)
        .agg(
            lr_fraction=("_lr_ok", "mean"),
            pauc_fraction=("_pauc_ok", "mean"),
            pathogen_fraction=("_pathogen_ok", "mean"),
            all_fraction=("_all_ok", "mean"),
        )
        .sort_values("candidate_u_max")
    )


def _load_umax_response_landscape_csv_for_plot(
    path: str,
    *,
    max_background_cases: int = 30,
    sampled_units: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    usecols = _umax_csv_usecols(path, _RESPONSE_LANDSCAPE_PLOT_COLS)
    if usecols is None:
        return pd.DataFrame()
    if not sampled_units:
        sampled_units = _sample_umax_unit_ids_chunked(path, max_units=max_background_cases, seed=42)
    unit_set = set(sampled_units)
    score_cols = [c for c in ("composite_penalty", "optimization_penalty_score") if c in usecols]
    flag_cols = [c for c in ("selected_by_optimizer", "is_minimum_score") if c in usecols]
    chunks: List[pd.DataFrame] = []
    flag_chunks: List[pd.DataFrame] = []
    summary_chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=400_000):
        chunk = _filter_tar_optimized_landscape(_ensure_umax_unit_id(chunk))
        if chunk.empty:
            continue
        if unit_set:
            bg = chunk[chunk["unit_id"].isin(unit_set)]
            if not bg.empty:
                chunks.append(bg)
        if flag_cols:
            flagged = chunk[chunk[flag_cols].astype(bool).any(axis=1)]
            if not flagged.empty:
                flag_chunks.append(flagged)
        score_usecols = [c for c in ("candidate_u_max", *score_cols) if c in chunk.columns]
        if score_usecols:
            summary_chunks.append(chunk[score_usecols])
    if not chunks and not summary_chunks and not flag_chunks:
        return pd.DataFrame()
    work = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    if flag_chunks:
        work.attrs["_flag_rows"] = pd.concat(flag_chunks, ignore_index=True).drop_duplicates()
    if summary_chunks and score_cols:
        summary_src = pd.concat(summary_chunks, ignore_index=True)
        score_col = _resolve_landscape_score_col(summary_src)
        if score_col:
            work.attrs["_precomputed_summary"] = {
                score_col: _aggregate_penalty_landscape_summary(summary_src, score_col)
            }
    return work


def _load_umax_landscape_for_fig5(
    outdir: str,
    *,
    max_background_cases: int = 30,
    sampled_units: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, str]:
    response_path = os.path.join(outdir, "umax_response_landscape.csv")
    if os.path.exists(response_path):
        df = _load_umax_response_landscape_csv_for_plot(
            response_path, max_background_cases=max_background_cases, sampled_units=sampled_units,
        )
        return df, "response"
    curves_path = os.path.join(outdir, "umax_score_landscape_curves.csv")
    if os.path.exists(curves_path):
        df = _load_umax_curve_csv_for_plot(
            curves_path, max_background_cases=max_background_cases, sampled_units=sampled_units,
        )
        return df, "curves"
    return pd.DataFrame(), ""


def _load_umax_feasibility_df(
    outdir: str,
    landscape_df: pd.DataFrame,
    landscape_source: str,
) -> pd.DataFrame:
    if not landscape_df.empty and landscape_source in ("response", "curves"):
        return _filter_tar_optimized_landscape(landscape_df)
    candidates_path = os.path.join(outdir, "umax_optimization_u_candidates.csv")
    if not os.path.exists(candidates_path):
        return pd.DataFrame()
    usecols = _umax_csv_usecols(candidates_path, _CANDIDATES_PLOT_COLS)
    if usecols is None:
        return pd.DataFrame()
    chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(candidates_path, usecols=usecols, chunksize=400_000):
        chunk = _filter_tar_optimized_landscape(_ensure_umax_unit_id(chunk))
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _load_umax_feasibility_df_full(outdir: str, landscape_source: str) -> pd.DataFrame:
    """Load all rows for constraint-fraction aggregation (not background-sampled)."""

    def _load_all_filtered(path: str, allowed: Sequence[str]) -> pd.DataFrame:
        usecols = _umax_csv_usecols(path, allowed)
        if usecols is None:
            return pd.DataFrame()
        chunks: List[pd.DataFrame] = []
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=400_000):
            chunk = _filter_tar_optimized_landscape(_ensure_umax_unit_id(chunk))
            if not chunk.empty:
                chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    response_path = os.path.join(outdir, "umax_response_landscape.csv")
    if os.path.exists(response_path):
        return _load_all_filtered(response_path, _RESPONSE_LANDSCAPE_PLOT_COLS)

    candidates_path = os.path.join(outdir, "umax_optimization_u_candidates.csv")
    if os.path.exists(candidates_path):
        return _load_all_filtered(candidates_path, _CANDIDATES_PLOT_COLS)

    if landscape_source == "curves":
        curves_path = os.path.join(outdir, "umax_score_landscape_curves.csv")
        if os.path.exists(curves_path):
            df = _load_all_filtered(curves_path, _RESPONSE_LANDSCAPE_PLOT_COLS)
            if not df.empty and (
                {"LR_constraint_satisfied", "feasible_candidate"} & set(df.columns)
                or {"LR1", "P_AUC", "terminal_total_pathogen"} <= set(df.columns)
            ):
                return df
    return pd.DataFrame()


def _umax_csv_usecols(path: str, allowed: Sequence[str]) -> Optional[List[str]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    picked = [col for col in allowed if col in header]
    return picked or None


def _sample_umax_unit_ids(path: str, *, max_units: int, seed: int) -> List[str]:
    usecols = _umax_csv_usecols(path, ("unit_id", "repeat_id", "case_index"))
    if usecols is None:
        return []
    meta = pd.read_csv(path, usecols=usecols).drop_duplicates()
    if "unit_id" not in meta.columns and {"repeat_id", "case_index"}.issubset(meta.columns):
        meta["unit_id"] = (
            "repeat_"
            + meta["repeat_id"].astype(int).astype(str).str.zfill(3)
            + "_case_"
            + meta["case_index"].astype(int).astype(str).str.zfill(4)
        )
    if "unit_id" not in meta.columns:
        return []
    unit_ids = meta["unit_id"].drop_duplicates().tolist()
    if len(unit_ids) <= max_units:
        return unit_ids
    rng = np.random.default_rng(seed)
    return rng.choice(unit_ids, size=max_units, replace=False).tolist()


def _sample_umax_unit_ids_chunked(path: str, *, max_units: int, seed: int) -> List[str]:
    usecols = _umax_csv_usecols(path, ("unit_id", "repeat_id", "case_index"))
    if usecols is None:
        return []
    seen: set = set()
    unit_ids: List[str] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000):
        chunk = _ensure_umax_unit_id(chunk)
        for uid in chunk["unit_id"].drop_duplicates():
            uid_s = str(uid)
            if uid_s in seen:
                continue
            seen.add(uid_s)
            unit_ids.append(uid_s)
    if len(unit_ids) <= max_units:
        return unit_ids
    rng = np.random.default_rng(seed)
    return rng.choice(unit_ids, size=max_units, replace=False).tolist()


def _sample_umax_unit_ids_from_outdir(outdir: str, *, max_units: int, seed: int) -> List[str]:
    for name in ("umax_objective_alignment.csv", "umax_ablation_cases.csv", "umax_policy_ablation_cases.csv"):
        ref_path = os.path.join(outdir, name)
        if not os.path.exists(ref_path):
            continue
        meta = pd.read_csv(ref_path)
        if "unit_id" not in meta.columns and {"repeat_id", "case_index"}.issubset(meta.columns):
            meta = meta.copy()
            meta["unit_id"] = (
                "repeat_"
                + meta["repeat_id"].astype(int).astype(str).str.zfill(3)
                + "_case_"
                + meta["case_index"].astype(int).astype(str).str.zfill(4)
            )
        if "unit_id" not in meta.columns:
            continue
        unit_ids = meta["unit_id"].drop_duplicates().astype(str).tolist()
        if len(unit_ids) <= max_units:
            return unit_ids
        rng = np.random.default_rng(seed)
        return rng.choice(unit_ids, size=max_units, replace=False).tolist()
    return []


def _ensure_umax_unit_id(df: pd.DataFrame) -> pd.DataFrame:
    if "unit_id" in df.columns or not {"repeat_id", "case_index"}.issubset(df.columns):
        return df
    out = df.copy()
    out["unit_id"] = (
        "repeat_"
        + out["repeat_id"].astype(int).astype(str).str.zfill(3)
        + "_case_"
        + out["case_index"].astype(int).astype(str).str.zfill(4)
    )
    return out


def _aggregate_by_candidate_u(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        df.groupby("candidate_u_max", as_index=False)[value_col]
        .agg(
            median="median",
            q10=lambda s: float(np.quantile(s, 0.10)),
            q90=lambda s: float(np.quantile(s, 0.90)),
        )
        .sort_values("candidate_u_max")
    )


def _load_umax_curve_csv_for_plot(
    path: str,
    *,
    max_background_cases: int = 150,
    sampled_units: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    usecols = _umax_csv_usecols(path, _LANDSCAPE_CURVE_COLS)
    if usecols is None:
        return pd.DataFrame()
    if not sampled_units:
        sampled_units = _sample_umax_unit_ids_chunked(path, max_units=max_background_cases, seed=42)
    unit_set = set(sampled_units)
    score_cols = [c for c in ("aspiration_abs_distance_to_initial", "optimization_penalty_score") if c in usecols]
    summary_usecols = [c for c in ("candidate_u_max", *score_cols) if c in usecols]
    chunks: List[pd.DataFrame] = []
    summary_chunks: List[pd.DataFrame] = []
    flag_chunks: List[pd.DataFrame] = []
    flag_cols = [
        c
        for c in (
            "is_initial_aspiration_dominator",
            "is_closest_to_initial_aspiration",
            "is_minimum_score",
        )
        if c in usecols
    ]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=400_000):
        chunk = _ensure_umax_unit_id(chunk)
        if unit_set:
            bg = chunk[chunk["unit_id"].isin(unit_set)]
            if not bg.empty:
                chunks.append(bg)
        if flag_cols:
            flagged = chunk[chunk[flag_cols].astype(bool).any(axis=1)]
            if not flagged.empty:
                flag_chunks.append(flagged)
        if summary_usecols:
            summary_chunks.append(chunk[summary_usecols])
    if not chunks and not summary_chunks and not flag_chunks:
        return pd.DataFrame()
    work = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    if flag_chunks:
        work.attrs["_flag_rows"] = pd.concat(flag_chunks, ignore_index=True).drop_duplicates()
    if summary_chunks and score_cols:
        summary_src = pd.concat(summary_chunks, ignore_index=True)
        work.attrs["_precomputed_summary"] = {
            score_col: _aggregate_by_candidate_u(summary_src, score_col) for score_col in score_cols
        }
    return work


def _load_umax_candidates_csv_for_plot(
    path: str,
    *,
    max_background_cases: int = 150,
    sampled_units: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    usecols = _umax_csv_usecols(path, _CANDIDATES_PLOT_COLS)
    if usecols is None:
        return pd.DataFrame()
    if not sampled_units:
        sampled_units = _sample_umax_unit_ids_chunked(path, max_units=max_background_cases, seed=44)
    unit_set = set(sampled_units)
    component_cols = [
        c
        for c in (
            "aspiration_dose_term",
            "aspiration_pauc_term",
            "aspiration_lr_term",
            "aspiration_pathogen_log_term",
            "LR_shortfall_norm",
            "PAUC_shortfall_norm",
            "pathogen_violation_norm",
            "dose_burden_norm",
        )
        if c in usecols
    ]
    summary_usecols = [c for c in ("candidate_u_max", *component_cols) if c in usecols]
    chunks: List[pd.DataFrame] = []
    summary_chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=400_000):
        if "model" in chunk.columns:
            chunk = chunk[chunk["model"] == "TAR"]
        if chunk.empty:
            continue
        chunk = _ensure_umax_unit_id(chunk)
        if unit_set:
            bg = chunk[chunk["unit_id"].isin(unit_set)]
            if not bg.empty:
                chunks.append(bg)
        if summary_usecols:
            summary_chunks.append(chunk[summary_usecols])
    if not chunks and not summary_chunks:
        return pd.DataFrame()
    work = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    if summary_chunks and component_cols:
        summary_src = pd.concat(summary_chunks, ignore_index=True)
        work.attrs["_precomputed_component_summary"] = {
            col: _aggregate_by_candidate_u(summary_src, col) for col in component_cols
        }
    return work


def plot_umax_response_landscape(
    landscape_df: pd.DataFrame,
    outpath: str,
    *,
    score_col: Optional[str] = None,
    faint_line_alpha: float = 0.08,
    max_background_cases: int = 30,
    selected_umax_csv: Optional[str] = None,
) -> None:
    """Fig. 5A: median composite-penalty landscape with IQR band and selected-Umax marker."""
    apply_matplotlib_style()
    work = _ensure_umax_unit_id(landscape_df.copy())
    if score_col is None or score_col not in work.columns:
        score_col = _resolve_landscape_score_col(work)
    if work.empty or not score_col:
        return

    group_col = "unit_id" if "unit_id" in work.columns else None
    flag_rows = work.attrs.get("_flag_rows")
    rug_source = flag_rows if isinstance(flag_rows, pd.DataFrame) and not flag_rows.empty else work

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    precomputed = work.attrs.get("_precomputed_summary", {})
    if group_col:
        unit_ids = work[group_col].drop_duplicates().tolist()
        if len(unit_ids) > max_background_cases:
            rng = np.random.default_rng(42)
            unit_ids = rng.choice(unit_ids, size=max_background_cases, replace=False).tolist()
        bg = work[work[group_col].isin(unit_ids)] if unit_ids else work.iloc[0:0]
        for uid in unit_ids:
            sub = bg[bg[group_col] == uid].sort_values("candidate_u_max")
            if sub.empty:
                continue
            ax.plot(
                sub["candidate_u_max"], sub[score_col],
                color=PALETTE_BLUE_LIGHT, alpha=faint_line_alpha, linewidth=0.7, zorder=1,
            )

    if score_col in precomputed:
        summary = precomputed[score_col]
        if "q25" not in summary.columns or "q75" not in summary.columns:
            summary = _aggregate_penalty_landscape_summary(work, score_col)
    else:
        summary = _aggregate_penalty_landscape_summary(work, score_col)

    x = summary["candidate_u_max"].to_numpy(dtype=float)
    ax.plot(x, summary["median"], color=PALETTE_BLUE_MID, linewidth=2.8, zorder=4, label="Median penalty")
    ax.fill_between(
        x, summary["q25"], summary["q75"],
        color=PALETTE_BLUE_LIGHT, alpha=0.28, zorder=3, label="IQR",
    )
    if {"q10", "q90"}.issubset(summary.columns):
        ax.fill_between(
            x, summary["q10"], summary["q90"],
            color=PALETTE_BLUE_LIGHT, alpha=0.10, zorder=2,
        )

    selected_u = _extract_selected_umax_values(rug_source if isinstance(rug_source, pd.DataFrame) else work)
    if selected_u.size:
        median_selected = float(np.median(selected_u))
        ax.axvline(
            median_selected, color=PALETTE_RED_MID, linestyle="--", linewidth=1.6,
            zorder=5, alpha=0.85, label="Selected Umax",
        )
        y0, y1 = ax.get_ylim()
        yspan = y1 - y0 + 1e-9
        rug_y = y0 + 0.015 * yspan
        ax.scatter(
            selected_u, np.full(selected_u.size, rug_y),
            marker="|", s=36, color=PALETTE_RED_MID, alpha=0.30, linewidths=0.8, zorder=6,
        )
        ax.text(
            0.02, 0.05, f"median selected $U_{{max}}$ = {median_selected:.1f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_ANNOT, color=PALETTE_RED_MID,
        )
        if selected_umax_csv and isinstance(rug_source, pd.DataFrame):
            sel_flag = None
            for flag_col in ("selected_by_optimizer", "is_minimum_score"):
                if flag_col in rug_source.columns:
                    sel_flag = flag_col
                    break
            if sel_flag:
                sel_df = rug_source[rug_source[sel_flag].astype(bool)]
                export_cols = [
                    c for c in [group_col, "repeat_id", "case_index", "candidate_u_max", score_col]
                    if c and c in sel_df.columns
                ]
                if export_cols and not sel_df.empty:
                    sel_df[export_cols].drop_duplicates().to_csv(selected_umax_csv, index=False)

    ax.set_title("Umax-response landscape", fontsize=FS_TRAJECTORY_TITLE, pad=6)
    ax.set_xlabel(r"$U_{max}$", fontsize=FS_BASE + 1)
    ax.set_ylabel("Composite penalty", fontsize=FS_BASE + 1)
    ax.tick_params(axis="both", labelsize=FS_TICK + 1)
    ax.set_xlim(left=0)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=FS_LEGEND + 1, frameon=False)
    finalize_figure_layout(fig)
    save_figure(fig, outpath)
    plt.close(fig)


def plot_umax_constraint_feasibility(
    landscape_df: pd.DataFrame,
    outpath: str,
    *,
    selected_umax_values: Optional[np.ndarray] = None,
) -> None:
    """Fig. 5B: fraction of cases satisfying each constraint across candidate Umax."""
    apply_matplotlib_style()
    work = _filter_tar_optimized_landscape(_ensure_umax_unit_id(landscape_df.copy()))
    if work.empty or "candidate_u_max" not in work.columns:
        return

    summary = _constraint_feasibility_by_u(work)
    if summary.empty:
        return

    if selected_umax_values is None:
        selected_umax_values = _extract_selected_umax_values(work)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = summary["candidate_u_max"].to_numpy(dtype=float)
    line_specs = [
        ("lr_fraction", "LR", PALETTE_BLUE_MID),
        ("pauc_fraction", r"$P_{AUC}$", PALETTE_GREEN_MID),
        ("pathogen_fraction", "pathogen", PALETTE_BLUE_LIGHT),
        ("all_fraction", "all constraints", PALETTE_RED_MID),
    ]
    for col, label, color in line_specs:
        if col in summary.columns:
            ax.plot(x, summary[col], linewidth=2.2, color=color, label=label, zorder=3)

    if selected_umax_values is not None and selected_umax_values.size:
        median_selected = float(np.median(selected_umax_values))
        ax.axvline(
            median_selected, color=PALETTE_RED_MID, linestyle="--", linewidth=1.6, alpha=0.85, zorder=4,
        )
        q25_sel = float(np.quantile(selected_umax_values, 0.25))
        q75_sel = float(np.quantile(selected_umax_values, 0.75))
        if q75_sel > q25_sel:
            ax.axvspan(q25_sel, q75_sel, color=PALETTE_RED_MID, alpha=0.08, zorder=1)

    ax.set_title("Constraint feasibility across Umax", fontsize=FS_TRAJECTORY_TITLE, pad=6)
    ax.set_xlabel(r"$U_{max}$", fontsize=FS_BASE + 1)
    ax.set_ylabel("Fraction of cases satisfying constraint", fontsize=FS_BASE + 1)
    ax.tick_params(axis="both", labelsize=FS_TICK + 1)
    ax.set_xlim(left=0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=FS_LEGEND + 1, frameon=False)
    finalize_figure_layout(fig)
    save_figure(fig, outpath)
    plt.close(fig)


def plot_umax_score_landscape(
    curve_df: pd.DataFrame,
    outpath: str,
    *,
    score_col: str = "aspiration_abs_distance_to_initial",
    faint_line_alpha: float = 0.06,
    max_background_cases: int = 150,
    selected_umax_csv: Optional[str] = None,
) -> None:
    """Fig. 5A: median aspiration-distance landscape with selection rugs."""
    apply_matplotlib_style()
    work = _ensure_umax_unit_id(curve_df.copy())
    if score_col not in work.columns and "optimization_penalty_score" in work.columns:
        score_col = "optimization_penalty_score"
    if work.empty or score_col not in work.columns:
        return
    group_col = "unit_id" if "unit_id" in work.columns else None
    if group_col is None and {"repeat_id", "case_index"}.issubset(work.columns):
        work["unit_id"] = (
            "repeat_" + work["repeat_id"].astype(int).astype(str).str.zfill(3)
            + "_case_" + work["case_index"].astype(int).astype(str).str.zfill(4)
        )
        group_col = "unit_id"
    flag_rows = work.attrs.get("_flag_rows")
    rug_source = flag_rows if isinstance(flag_rows, pd.DataFrame) and not flag_rows.empty else work

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    precomputed = work.attrs.get("_precomputed_summary", {})
    if group_col:
        unit_ids = work[group_col].drop_duplicates().tolist()
        if len(unit_ids) > max_background_cases:
            rng = np.random.default_rng(42)
            unit_ids = rng.choice(unit_ids, size=max_background_cases, replace=False).tolist()
        bg = work[work[group_col].isin(unit_ids)] if unit_ids else work.iloc[0:0]
        for uid in unit_ids:
            sub = bg[bg[group_col] == uid].sort_values("candidate_u_max")
            if sub.empty:
                continue
            ax.plot(
                sub["candidate_u_max"], sub[score_col],
                color=PALETTE_BLUE_LIGHT, alpha=faint_line_alpha, linewidth=0.7, zorder=1,
            )

    if score_col in precomputed:
        summary = precomputed[score_col]
    else:
        summary = (
            work.groupby("candidate_u_max", as_index=False)[score_col]
            .agg(median="median", q10=lambda s: float(np.quantile(s, 0.10)), q90=lambda s: float(np.quantile(s, 0.90)))
            .sort_values("candidate_u_max")
        )
    x = summary["candidate_u_max"].to_numpy(dtype=float)
    ylabel = (
        "Aspiration absolute distance"
        if score_col == "aspiration_abs_distance_to_initial"
        else "Composite penalty score"
    )
    ax.plot(x, summary["median"], color=PALETTE_BLUE_MID, linewidth=2.8, zorder=4, label="Median distance")
    ax.fill_between(x, summary["q10"], summary["q90"], color=PALETTE_BLUE_LIGHT, alpha=0.22, zorder=3)

    def _rug_points(flag_col: str) -> pd.Series:
        if flag_col not in rug_source.columns or group_col is None:
            return pd.Series(dtype=float)
        flagged = rug_source[rug_source[flag_col].astype(bool)]
        if flagged.empty:
            return pd.Series(dtype=float)
        return flagged.groupby(group_col, sort=False)["candidate_u_max"].first().dropna().astype(float)

    rug_specs = [
        ("is_initial_aspiration_dominator", PALETTE_BLUE_MID, "Initial aspiration dominators"),
        ("is_closest_to_initial_aspiration", "#6b8e23", "Closest to initial aspiration"),
        ("is_minimum_score", PALETTE_RED_MID, "Final selected Umax"),
    ]
    rug_y_base = None
    for idx, (flag_col, color, label) in enumerate(rug_specs):
        pts = _rug_points(flag_col)
        if pts.empty:
            continue
        if rug_y_base is None:
            rug_y_base = ax.get_ylim()[0]
        y_off = rug_y_base + idx * 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0] + 1e-9)
        ax.scatter(
            pts.to_numpy(dtype=float),
            np.full(len(pts), y_off),
            marker="|",
            s=40,
            color=color,
            alpha=0.45,
            linewidths=0.9,
            zorder=6 + idx,
            label=label,
        )

    selected_u = _rug_points("is_minimum_score")
    if selected_u.empty and flag_rows is not None and isinstance(flag_rows, pd.DataFrame):
        if "is_minimum_score" in flag_rows.columns:
            selected_u = flag_rows[flag_rows["is_minimum_score"].astype(bool)]["candidate_u_max"].dropna().astype(float)
    if selected_u.empty and "is_minimum_score" in work.columns:
        selected_u = work[work["is_minimum_score"].astype(bool)]["candidate_u_max"].dropna().astype(float)
    if not selected_u.empty:
        median_selected = float(selected_u.median())
        ax.axvline(median_selected, color=PALETTE_RED_MID, linestyle="--", linewidth=1.6, zorder=5, alpha=0.85)
        if selected_umax_csv and flag_rows is not None and isinstance(flag_rows, pd.DataFrame):
            if "is_minimum_score" in flag_rows.columns:
                sel_df = flag_rows[flag_rows["is_minimum_score"].astype(bool)]
            else:
                sel_df = pd.DataFrame()
            export_cols = [
                c for c in [group_col, "repeat_id", "case_index", "candidate_u_max", score_col]
                if c in sel_df.columns
            ]
            if export_cols and not sel_df.empty:
                sel_df[export_cols].drop_duplicates().to_csv(selected_umax_csv, index=False)

    ax.set_xlabel(r"$U_{max}$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=FS_LEGEND, frameon=False)
    finalize_figure_layout(fig)
    save_figure(fig, outpath)
    plt.close(fig)


def plot_umax_score_landscape_full_spaghetti(
    curve_df: pd.DataFrame,
    outpath: str,
    *,
    score_col: str = "optimization_penalty_score",
    faint_line_alpha: float = 0.12,
) -> None:
    """Diagnostic full spaghetti plot for Fig. 5A."""
    apply_matplotlib_style()
    if curve_df.empty or score_col not in curve_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    group_col = "unit_id" if "unit_id" in curve_df.columns else None
    plot_df = curve_df
    if group_col:
        unit_ids = plot_df[group_col].drop_duplicates().tolist()
        if len(unit_ids) > 200:
            unit_ids = np.random.default_rng(45).choice(unit_ids, size=200, replace=False).tolist()
            plot_df = plot_df[plot_df[group_col].isin(unit_ids)]
        for _, group in plot_df.groupby(group_col, sort=False):
            sub = group.sort_values("candidate_u_max")
            ax.plot(
                sub["candidate_u_max"], sub[score_col],
                color=PALETTE_BLUE_LIGHT, alpha=faint_line_alpha, linewidth=0.7, zorder=1,
            )
            min_idx = int(np.argmin(sub[score_col].to_numpy(dtype=float)))
            mrow = sub.iloc[min_idx]
            ax.scatter(
                mrow["candidate_u_max"], mrow[score_col],
                s=14, color=PALETTE_RED_MID, alpha=0.35, edgecolors="none", zorder=2,
            )
    precomputed = curve_df.attrs.get("_precomputed_summary", {})
    if score_col in precomputed:
        summary = precomputed[score_col]
    else:
        summary = (
            curve_df.groupby("candidate_u_max", as_index=False)[score_col]
            .agg(median="median", q10=lambda s: float(np.quantile(s, 0.10)), q90=lambda s: float(np.quantile(s, 0.90)))
            .sort_values("candidate_u_max")
        )
    x = summary["candidate_u_max"].to_numpy(dtype=float)
    ax.plot(x, summary["median"], color=PALETTE_BLUE_MID, linewidth=2.4, zorder=4)
    ax.fill_between(x, summary["q10"], summary["q90"], color=PALETTE_BLUE_LIGHT, alpha=0.18, zorder=3)
    ax.set_xlabel(r"$U_{max}$")
    ax.set_ylabel("Composite penalty score")
    ax.set_xlim(left=0)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, outpath)
    plt.close(fig)


def plot_umax_score_components_landscape(
    candidates_df: pd.DataFrame,
    outpath: str,
    *,
    fixed_policy_df: Optional[pd.DataFrame] = None,
    optimized_cases_df: Optional[pd.DataFrame] = None,
    max_background_cases: int = 150,
    faint_line_alpha: float = 0.06,
) -> None:
    """Fig. 5A debug: component shortfall norms vs Umax."""
    apply_matplotlib_style()
    if candidates_df.empty:
        return
    work = _ensure_umax_unit_id(candidates_df.copy())
    if "model" in work.columns:
        work = work[work["model"] == "TAR"].copy()
    component_specs = [
        ("aspiration_dose_term", "Dose absolute normalized term"),
        ("aspiration_pauc_term", "P_AUC absolute normalized term"),
        ("aspiration_lr_term", "LR absolute normalized term"),
        ("aspiration_pathogen_log_term", "Pathogen log absolute normalized term"),
    ]
    if not all(col in work.columns for col, _ in component_specs):
        component_specs = [
            ("LR_shortfall_norm", "LR shortfall norm"),
            ("PAUC_shortfall_norm", "P_AUC shortfall norm"),
            ("pathogen_violation_norm", "Pathogen violation norm"),
            ("dose_burden_norm", "Dose burden norm"),
        ]
    if not all(col in work.columns for col, _ in component_specs):
        return

    if "unit_id" not in work.columns and {"repeat_id", "case_index"}.issubset(work.columns):
        work["unit_id"] = (
            "repeat_" + work["repeat_id"].astype(int).astype(str).str.zfill(3)
            + "_case_" + work["case_index"].astype(int).astype(str).str.zfill(4)
        )
    group_col = "unit_id" if "unit_id" in work.columns else None

    vlines: List[Tuple[float, str]] = []
    if fixed_policy_df is not None and not fixed_policy_df.empty:
        med_u = fixed_policy_df.get("training_median_soft_u_max")
        tuned_u = fixed_policy_df.get("training_tuned_global_u_max")
        if med_u is not None and med_u.notna().any():
            vlines.append((float(med_u.median()), "training-median"))
        if tuned_u is not None and tuned_u.notna().any():
            vlines.append((float(tuned_u.median()), "training-tuned global"))
    if optimized_cases_df is not None and not optimized_cases_df.empty:
        opt = optimized_cases_df
        if "ablation_condition" in opt.columns:
            opt = opt[opt["ablation_condition"] == "TAR_optimized"]
        ucol = "selected_u_max" if "selected_u_max" in opt.columns else "optimized_u_max"
        if ucol in opt.columns and opt[ucol].notna().any():
            vlines.append((float(opt[ucol].median()), "median optimized"))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True)
    axes_flat = axes.ravel()
    rng = np.random.default_rng(44)
    unit_ids: List = []
    if group_col:
        unit_ids = work[group_col].drop_duplicates().tolist()
        if len(unit_ids) > max_background_cases:
            unit_ids = rng.choice(unit_ids, size=max_background_cases, replace=False).tolist()

    for ax, (col, ylabel) in zip(axes_flat, component_specs):
        precomputed = work.attrs.get("_precomputed_component_summary", {})
        if group_col and unit_ids:
            bg = work[work[group_col].isin(unit_ids)]
            for uid in unit_ids:
                sub = bg[bg[group_col] == uid].sort_values("candidate_u_max")
                if sub.empty:
                    continue
                ax.plot(
                    sub["candidate_u_max"],
                    sub[col],
                    color=PALETTE_BLUE_LIGHT,
                    alpha=faint_line_alpha,
                    linewidth=0.7,
                    zorder=1,
                )
        if col in precomputed:
            summary = precomputed[col]
        else:
            summary = (
                work.groupby("candidate_u_max", as_index=False)[col]
                .agg(median="median", q10=lambda s: float(np.quantile(s, 0.10)), q90=lambda s: float(np.quantile(s, 0.90)))
                .sort_values("candidate_u_max")
            )
        x = summary["candidate_u_max"].to_numpy(dtype=float)
        ax.plot(x, summary["median"], color=PALETTE_BLUE_MID, linewidth=2.2, zorder=4)
        ax.fill_between(x, summary["q10"], summary["q90"], color=PALETTE_BLUE_LIGHT, alpha=0.22, zorder=3)
        for u_val, _label in vlines:
            ax.axvline(u_val, color=PALETTE_RED_MID, linestyle="--", linewidth=1.2, alpha=0.75, zorder=5)
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        ax.grid(axis="both", linestyle="--", alpha=0.3)
    for ax in axes_flat[2:]:
        ax.set_xlabel(r"$U_{max}$")
    if vlines:
        fig.legend(
            [f"{label} Umax={u:.1f}" for u, label in vlines],
            loc="upper center",
            ncol=min(len(vlines), 3),
            fontsize=FS_LEGEND,
            frameon=False,
            bbox_to_anchor=(0.5, 0.98),
        )
        finalize_figure_layout(fig, rect=(0.0, 0.04, 1.0, 0.92))
    else:
        finalize_figure_layout(fig)
    save_figure(fig, outpath)
    plt.close(fig)



def plot_umax_optima_alignment(
    align_df: pd.DataFrame,
    outpath: str,
    *,
    landscape_summary_df: Optional[pd.DataFrame] = None,
    landscape_curve_df: Optional[pd.DataFrame] = None,
    u_grid: Optional[Sequence[float]] = None,
    faint_line_alpha: float = 0.04,
    max_background_cases: int = 150,
) -> None:
    del landscape_summary_df, landscape_curve_df
    apply_matplotlib_style()
    try:
        from closed_loop_eval import (
            U_OBJECTIVE_CATEGORY_LABELS,
            U_OBJECTIVE_PREFERRED_COLUMNS,
            normalize_umax_optima_alignment_df,
        )
    except ImportError:
        U_OBJECTIVE_PREFERRED_COLUMNS = (
            "U_dose_plateau",
            "U_P_AUC_constraint_limit",
            "U_LR_feasibility",
            "U_pathogen_feasibility",
            "U_composite_selected",
        )
        U_OBJECTIVE_CATEGORY_LABELS = {
            "U_dose_plateau": "Dose plateau Umax",
            "U_P_AUC_constraint_limit": "P_AUC constraint limit",
            "U_LR_feasibility": "LR feasibility threshold",
            "U_pathogen_feasibility": "Pathogen feasibility threshold",
            "U_composite_selected": "Composite selected Umax",
        }
        normalize_umax_optima_alignment_df = lambda df: df  # noqa: E731

    align_df = normalize_umax_optima_alignment_df(align_df)
    cols = [col for col in U_OBJECTIVE_PREFERRED_COLUMNS if col in align_df.columns]
    if not cols or align_df.empty:
        return

    categories = [(col, U_OBJECTIVE_CATEGORY_LABELS.get(col, col)) for col in cols]
    x = np.arange(len(categories))
    group_col = "unit_id" if "unit_id" in align_df.columns else None
    if group_col is None and {"repeat_id", "case_index"}.issubset(align_df.columns):
        align_df = align_df.copy()
        align_df["unit_id"] = (
            "repeat_" + align_df["repeat_id"].astype(int).astype(str).str.zfill(3)
            + "_case_" + align_df["case_index"].astype(int).astype(str).str.zfill(4)
        )
        group_col = "unit_id"

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    if group_col:
        unit_ids = align_df[group_col].drop_duplicates().tolist()
        rng = np.random.default_rng(43)
        if len(unit_ids) > max_background_cases:
            unit_ids = rng.choice(unit_ids, size=max_background_cases, replace=False).tolist()
        for uid in unit_ids:
            row = align_df[align_df[group_col] == uid].iloc[0]
            ys = [float(row[col]) for col in cols]
            if not any(np.isfinite(v) for v in ys):
                continue
            ax.plot(x, ys, color=PALETTE_BLUE_LIGHT, alpha=faint_line_alpha, linewidth=0.7, zorder=1)

    data = [align_df[col].dropna().astype(float).to_numpy() for col in cols]
    parts = ax.violinplot(data, positions=x, widths=0.55, showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(PALETTE_BLUE_LIGHT)
        body.set_edgecolor(PALETTE_BLUE_MID)
        body.set_alpha(0.35)
        body.set_zorder(2)
    for pos, col in enumerate(cols):
        vals = align_df[col].dropna().astype(float)
        if vals.empty:
            continue
        jitter = np.random.default_rng(44 + pos).normal(0.0, 0.04, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals.to_numpy(), s=8, color=PALETTE_BLUE_MID, alpha=0.25, zorder=2)

    medians = [float(align_df[col].median()) for col in cols]
    q25 = [float(align_df[col].quantile(0.25)) for col in cols]
    q75 = [float(align_df[col].quantile(0.75)) for col in cols]
    yerr = np.vstack([np.array(medians) - np.array(q25), np.array(q75) - np.array(medians)])
    ax.errorbar(
        x, medians, yerr=yerr, fmt="none", ecolor=PALETTE_RED_MID,
        elinewidth=1.6, capsize=5, alpha=0.9, zorder=4,
    )
    ax.scatter(x, medians, color=PALETTE_RED_MID, s=70, zorder=5, edgecolors="white", linewidths=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in categories], rotation=18, ha="right")
    ax.set_ylabel(r"Preferred $U_{max}$")
    if u_grid is not None and len(u_grid) > 0:
        ax.set_ylim(float(np.min(u_grid)), float(np.max(u_grid)))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_figure(fig, outpath)
    plt.close(fig)


def plot_umax_optima_alignment_full_spaghetti(
    align_df: pd.DataFrame,
    outpath: str,
    *,
    u_grid: Optional[Sequence[float]] = None,
    faint_line_alpha: float = 0.16,
) -> None:
    """Diagnostic full connected-line plot for Fig. 5B."""
    apply_matplotlib_style()
    try:
        from closed_loop_eval import (
            U_OBJECTIVE_CATEGORY_LABELS,
            U_OBJECTIVE_PREFERRED_COLUMNS,
            normalize_umax_optima_alignment_df,
        )
    except ImportError:
        U_OBJECTIVE_PREFERRED_COLUMNS = (
            "U_dose_plateau",
            "U_P_AUC_constraint_limit",
            "U_LR_feasibility",
            "U_pathogen_feasibility",
            "U_composite_selected",
        )
        U_OBJECTIVE_CATEGORY_LABELS = {
            "U_dose_plateau": "Dose plateau Umax",
            "U_P_AUC_constraint_limit": "P_AUC constraint limit",
            "U_LR_feasibility": "LR feasibility threshold",
            "U_pathogen_feasibility": "Pathogen feasibility threshold",
            "U_composite_selected": "Composite selected Umax",
        }
        normalize_umax_optima_alignment_df = lambda df: df  # noqa: E731

    align_df = normalize_umax_optima_alignment_df(align_df)
    cols = [col for col in U_OBJECTIVE_PREFERRED_COLUMNS if col in align_df.columns]
    if not cols or align_df.empty:
        return
    categories = [(col, U_OBJECTIVE_CATEGORY_LABELS.get(col, col)) for col in cols]
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for _, row in align_df.iterrows():
        ys = [float(row[col]) for col in cols]
        if not any(np.isfinite(v) for v in ys):
            continue
        ax.plot(x, ys, color=PALETTE_BLUE_LIGHT, alpha=faint_line_alpha, linewidth=0.7, zorder=1)
    medians = [float(align_df[col].median()) for col in cols]
    ax.scatter(x, medians, color=PALETTE_RED_MID, s=58, zorder=3, edgecolors="white", linewidths=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in categories], rotation=18, ha="right")
    ax.set_ylabel(r"Preferred $U_{max}$")
    if u_grid is not None and len(u_grid) > 0:
        ax.set_ylim(float(np.min(u_grid)), float(np.max(u_grid)))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    save_figure(fig, outpath)
    plt.close(fig)


def _umax_ablation_display_label(condition: str) -> str:
    try:
        from closed_loop_eval import FIG5_ABLATION_DISPLAY_LABELS
        return FIG5_ABLATION_DISPLAY_LABELS.get(condition, condition)
    except ImportError:
        return condition


def plot_umax_ode_ablation(
    trajectories_df: pd.DataFrame,
    condition_labels: Sequence[str],
    t_thr_by_condition: Dict[str, np.ndarray],
    outpath: str,
    *,
    display_labels: Optional[Sequence[str]] = None,
    policy_umax_by_condition: Optional[Dict[str, float]] = None,
    total_dosage_by_condition: Optional[Dict[str, float]] = None,
) -> None:
    """Fig. 5C: TAR training-median / training-tuned global / optimized illustrative trajectories."""
    try:
        from closed_loop_eval import FIG5_REPRESENTATIVE_CONDITIONS
        order = [c for c in FIG5_REPRESENTATIVE_CONDITIONS if c in condition_labels]
        required = list(FIG5_REPRESENTATIVE_CONDITIONS)
    except ImportError:
        order = list(condition_labels)[:3]
        required = order
    missing = [c for c in required if c not in condition_labels]
    if missing:
        raise RuntimeError(
            f"umax_ode_ablation.png requires all representative conditions {required}; missing {missing}."
        )
    apply_matplotlib_style()
    n_cols = len(order)
    fig, axes = plt.subplots(3, n_cols, figsize=(5.5 * n_cols, 8.4), sharex=True)
    if n_cols == 1:
        axes = np.array(axes).reshape(3, 1)
    pathogen_colors = TRAJECTORY_PATHOGENS[:N_ODE_STRAINS]
    legend_handles = None
    legend_labels = None
    cond_col = "ablation_condition" if "ablation_condition" in trajectories_df.columns else "model"
    all_times: List[np.ndarray] = []

    for col, condition in enumerate(order):
        sub = _first_trajectory_segment(trajectories_df[trajectories_df[cond_col] == condition])
        if sub.empty:
            raise RuntimeError(
                f"umax_ode_ablation.png: no trajectory rows for condition '{condition}'."
            )
        times = sub["time_h"].to_numpy(dtype=float)
        all_times.append(times)
        t_thr = np.asarray(t_thr_by_condition[condition], dtype=float)
        if display_labels is not None and col < len(display_labels):
            title = display_labels[col]
        else:
            policy_label = FIG5_ODE_POLICY_SHORT_LABELS.get(condition, _umax_ablation_display_label(condition))
            title_parts = [policy_label]
            umax_val = (policy_umax_by_condition or {}).get(condition)
            dose_val = (total_dosage_by_condition or {}).get(condition)
            if umax_val is not None and np.isfinite(umax_val):
                title_parts.append(f"$U_{{max}}$={umax_val:.1f}")
            if dose_val is not None and np.isfinite(dose_val):
                title_parts.append(f"dosage={dose_val:.0f}")
            title = "\n".join(title_parts)
        axes[0, col].plot(times, sub["C_ug_per_mL"], linewidth=1.8, color=TRAJECTORY_AMP)
        axes[0, col].set_title(title, fontsize=FS_PANEL, pad=10)
        if col == 0:
            axes[0, col].set_ylabel(r"AMP ($\mu$g/mL)")
        axes[1, col].plot(times, sub["P_total_CFU_per_mL"] / 1e6, linewidth=1.8, color=TRAJECTORY_PROBIOTIC)
        if col == 0:
            axes[1, col].set_ylabel(r"Probiotic ($\times10^6$ CFU/mL)")
        for i in range(N_ODE_STRAINS):
            color_i = pathogen_colors[i]
            b_col = f"B_total_{i + 1}_CFU_per_mL"
            axes[2, col].plot(
                times, sub[b_col].to_numpy(dtype=float) / 1e6, linewidth=1.6, color=color_i,
                label=f"B{i + 1}" if col == 0 else "_nolegend_",
            )
            axes[2, col].hlines(
                t_thr[i] / 1e6, times[0], times[-1], colors=color_i, linestyles="dashed", linewidth=0.9, alpha=0.45
            )
        axes[2, col].set_xlabel("Time (h)")
        if col == 0:
            axes[2, col].set_ylabel(r"Pathogen burden ($\times10^6$ CFU/mL)")
            legend_handles, legend_labels = axes[2, col].get_legend_handles_labels()
    if legend_handles and legend_labels:
        pass  # placed after shared y-limits below
    if all_times:
        x_min = float(min(t[0] for t in all_times))
        x_max = float(max(t[-1] for t in all_times))
        for row in range(3):
            for c in range(n_cols):
                axes[row, c].set_xlim(x_min, x_max)
    for row in range(3):
        ylims = [axes[row, c].get_ylim() for c in range(n_cols)]
        ymin = min(0.0, *(y[0] for y in ylims))
        ymax = max(y[1] for y in ylims)
        for c in range(n_cols):
            axes[row, c].set_ylim(ymin, ymax * 1.03)
    if legend_handles and legend_labels:
        place_figure_legend_below(fig, legend_handles, legend_labels, ncol=5, y=-0.01, bottom_rect=0.10)
    else:
        finalize_figure_layout(fig, rect=(0.0, 0.06, 1.0, 0.98))
    fig.text(
        0.99, 0.99, "representative case",
        ha="right", va="top", fontsize=FS_NOTE, alpha=0.55, style="italic",
        transform=fig.transFigure,
    )
    save_figure(fig, outpath)
    plt.close(fig)


def plot_umax_summary_ablation(
    stats_df: pd.DataFrame,
    outpath: str,
    *,
    n_repeats: int = 1,
    significance_pairs: Optional[object] = None,
    outdir: Optional[str] = None,
    conditions_order: Optional[Sequence[str]] = None,
    metric_specs: Optional[Sequence[Tuple[str, str]]] = None,
    use_short_labels: bool = False,
    horizontal_bars: bool = False,
) -> None:
    """Fig. 5D: repeated ablation summary (TAR policy comparison)."""
    try:
        from closed_loop_eval import (
            FIG5_ABLATION_CONDITIONS,
            FIG5_SIGNIFICANCE_REFERENCE,
            validate_fig5_main_ablation_conditions,
        )
        order = list(conditions_order or FIG5_ABLATION_CONDITIONS)
        if conditions_order is None or list(conditions_order) == list(FIG5_ABLATION_CONDITIONS):
            counts_path = None
            if outdir:
                for fname in ("umax_policy_ablation_condition_counts.csv", "umax_ablation_cases.csv"):
                    candidate = os.path.join(outdir, fname)
                    if os.path.isfile(candidate):
                        counts_path = candidate
                        break
            if counts_path and counts_path.endswith("umax_ablation_cases.csv"):
                cases_df = pd.read_csv(counts_path)
                validate_fig5_main_ablation_conditions(cases_df, context="plot_umax_summary_ablation")
            elif counts_path:
                counts_df = pd.read_csv(counts_path)
                missing = [c for c in FIG5_ABLATION_CONDITIONS if c not in counts_df["condition"].tolist()]
                if missing:
                    raise RuntimeError(
                        f"plot_umax_summary_ablation: missing conditions {missing}; "
                        f"observed {counts_df['condition'].tolist()}"
                    )
        if metric_specs is None:
            metric_specs = list(FIG5_MAIN_SUMMARY_METRICS) if use_short_labels else [
                ("mean_total_dosage", "Total dosage ($\\mu$g mL$^{-1}$)"),
                ("mean_P_AUC", "Mean $P_{AUC}$"),
                ("mean_LR", "Mean LR"),
                ("mean_terminal_pathogen", "Terminal pathogen (CFU mL$^{-1}$)"),
                ("mean_composite_score", "Composite penalty score"),
            ]
        ref = FIG5_SIGNIFICANCE_REFERENCE
    except ImportError:
        order = stats_df["model"].tolist()
        metric_specs = list(metric_specs or [("mean_composite_score", "Composite penalty score")])
        ref = order[0] if order else TAR_MODEL
    apply_matplotlib_style()
    plot_df = _reorder_bar_plot_df(stats_df.copy(), order)
    available = set(plot_df["model"].tolist())
    missing_bars = [c for c in order if c not in available]
    if missing_bars:
        raise RuntimeError(
            f"plot_umax_summary_ablation: missing bar data for conditions {missing_bars}; "
            f"observed models {sorted(available)}"
        )
    label_fn = (
        (lambda m: FIG5_ABLATION_SHORT_LABELS.get(m, m))
        if use_short_labels
        else _umax_ablation_display_label
    )
    bar_colors = (
        [FIG5_ABLATION_BAR_COLORS.get(m, color_for_model(m)) for m in plot_df["model"].tolist()]
        if use_short_labels
        else None
    )
    n_panels = sum(1 for col, _ in metric_specs if col in plot_df.columns)
    if horizontal_bars:
        n_bars = len(plot_df)
        # Keep bar thickness readable: short height for few conditions, avoid tall empty space.
        fig_h = max(2.6, 0.62 * n_bars + 1.15) if n_panels <= 1 else max(3.2, 1.15 * n_bars + 1.0)
        fig_w = 6.6 if n_panels <= 1 else 3.6 * max(n_panels, 1)
        fig, axes = plt.subplots(1, max(n_panels, 1), figsize=(fig_w, fig_h))
    else:
        fig, axes = plt.subplots(1, max(n_panels, 1), figsize=(3.4 * max(n_panels, 1), 5.2))
    if n_panels == 1:
        axes = np.array([axes])
    significance_pairs_by_metric = significance_pairs if isinstance(significance_pairs, dict) else {}
    panel_idx = 0
    for col, ylabel in metric_specs:
        if col not in plot_df.columns:
            continue
        ax = axes[panel_idx]
        panel_idx += 1
        pairs = significance_pairs_by_metric.get(col)
        _plot_metric_panel_with_ci(
            ax, plot_df, col, ylabel,
            significance_pairs=pairs,
            focus_ylim=True,
            baseline_focus_bars=(col == "mean_P_AUC"),
            significance_bracket_step_scale=1.35,
            display_labels=[label_fn(m) for m in plot_df["model"].tolist()],
            bar_colors=bar_colors,
            horizontal=horizontal_bars,
            horizontal_bar_height=0.72 if horizontal_bars and n_panels <= 1 else 0.92,
        )
        if horizontal_bars:
            ax.tick_params(axis="y", labelsize=10)
            for tick in ax.get_yticklabels():
                tick.set_fontsize(10)
        if not horizontal_bars:
            ax.set_xticklabels(
                [label_fn(m) for m in plot_df["model"]],
                rotation=25, ha="right", fontsize=FS_TICK_SM,
            )
            ax.tick_params(axis="x", pad=5)
    for ax in axes[panel_idx:]:
        ax.set_visible(False)
    if horizontal_bars:
        finalize_figure_layout(fig, rect=(0.02, 0.06, 0.98, 0.98))
    else:
        finalize_figure_layout(fig, rect=(0.0, 0.14, 1.0, 1.0))
    save_figure(fig, outpath)
    plt.close(fig)


def build_fig5_plot_manifest(
    *,
    primary_outputs: Dict[str, str],
    n_repeats: int,
    significance_for_manuscript: bool,
    prediction_sources: Optional[Sequence[str]] = None,
) -> dict:
    sig_mode = (
        "TAR_optimized_vs_TAR_training_median_and_training_tuned_global_n_repeats_ge_10"
        if significance_for_manuscript
        else "exploratory_no_formal_stars_n_repeats_lt_10"
    )
    panel_mapping = [
        _fig_panel_record(
            panel, filename, desc,
            role="primary_manuscript_figure",
            input_sources={
                "umax_score_landscape.png": [
                    "umax_optimization/umax_response_landscape.csv",
                    "umax_optimization/umax_selected_umax_distribution.csv",
                    "umax_optimization/umax_score_landscape_curves.csv",
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
            }.get(filename, []),
            n_repeats=n_repeats,
            umax_setting={
                "umax_score_landscape.png": "tar_optimized_response_landscape",
                "umax_constraint_feasibility.png": "constraint_feasibility_fraction_by_u",
                "umax_ode_ablation.png": "tar_policy_representative_trajectories",
                "umax_summary_ablation.png": "tar_only_policy_ablation_three_conditions",
            }.get(filename, "optimized_and_fixed_ablation"),
            significance_mode="none" if filename != "umax_summary_ablation.png" else sig_mode,
        )
        for panel, filename, desc in FIG5_PANEL_ORDER
    ]
    for entry in panel_mapping:
        mapped = MANUSCRIPT_SOURCE_MAPPING.get(entry.get("filename", ""), {})
        if mapped.get("description"):
            entry["description"] = mapped["description"]
        if mapped.get("source_group"):
            entry["source_group"] = mapped["source_group"]
    return {
        "figure": "Fig. 5",
        "figure_title": "Umax optimization analysis",
        "validation_section": "Umax optimization analysis",
        "prediction_input_sources": list(prediction_sources or []),
        "primary_outputs": {k: primary_outputs[k] for k in FIG5_PRIMARY_FIGURES if k in primary_outputs},
        "figure_panel_mapping": panel_mapping,
        "manuscript_source_mapping": {
            k: dict(v)
            for k, v in MANUSCRIPT_SOURCE_MAPPING.items()
            if v.get("source_group") == "umax_optimization"
        },
        "fig5_primary_figures": list(FIG5_PRIMARY_FIGURES),
        "optimizer_score_rule": (
            "One-sided optimization_penalty_score (>= 0); signed_relative_rms is diagnostic only."
        ),
        "n_repeats": int(n_repeats),
        "significance_for_manuscript": bool(significance_for_manuscript),
        "exploratory": bool(n_repeats < 10),
        "manuscript_safe": bool(n_repeats >= 100 and significance_for_manuscript),
        "illustrative_only_ablation_trajectories": True,
    }


build_umax_optimization_plot_manifest = build_fig5_plot_manifest


def generate_umax_optimization_plots(outdir: str) -> Dict[str, str]:
    """Regenerate Fig. 5 Umax optimization justification panels."""
    figure_dir = _ensure_figure_dir(outdir)
    manifest_path = os.path.join(outdir, UMAX_OPTIMIZATION_MANIFEST_JSON)
    cl_manifest: dict = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            cl_manifest = json.load(fh)
    n_repeats = int(cl_manifest.get("n_repeats", 1))
    outputs: Dict[str, str] = {}
    primary_outputs: Dict[str, str] = {}
    sampled_units = _sample_umax_unit_ids_from_outdir(outdir, max_units=30, seed=42)

    landscape_df, landscape_source = _load_umax_landscape_for_fig5(
        outdir, max_background_cases=30, sampled_units=sampled_units,
    )
    if not landscape_df.empty:
        print(
            f"[Fig.5] Plotting umax_score_landscape from {landscape_source} "
            f"({len(landscape_df)} background rows) ...",
            flush=True,
        )
        landscape_png = os.path.join(figure_dir, "umax_score_landscape.png")
        selected_csv = os.path.join(outdir, "umax_selected_umax_distribution.csv")
        plot_umax_response_landscape(
            landscape_df, landscape_png, selected_umax_csv=selected_csv, max_background_cases=30,
        )
        primary_outputs["umax_score_landscape.png"] = landscape_png
        outputs["umax_score_landscape.png"] = landscape_png
        if landscape_source == "curves":
            full_landscape_png = os.path.join(figure_dir, "umax_score_landscape_full_spaghetti.png")
            plot_umax_score_landscape_full_spaghetti(landscape_df, full_landscape_png)
            outputs["umax_score_landscape_full_spaghetti.png"] = full_landscape_png

    feasibility_df = _load_umax_feasibility_df_full(outdir, landscape_source)
    if not feasibility_df.empty:
        print(f"[Fig.5] Plotting umax_constraint_feasibility ({len(feasibility_df)} rows) ...", flush=True)
        selected_u = _extract_selected_umax_values(feasibility_df)
        constraint_png = os.path.join(figure_dir, "umax_constraint_feasibility.png")
        plot_umax_constraint_feasibility(
            feasibility_df, constraint_png, selected_umax_values=selected_u,
        )
        primary_outputs["umax_constraint_feasibility.png"] = constraint_png
        outputs["umax_constraint_feasibility.png"] = constraint_png

    candidates_path = os.path.join(outdir, "umax_optimization_u_candidates.csv")
    if os.path.exists(candidates_path):
        print(f"[Fig.5] Loading U candidates from {candidates_path} ...", flush=True)
        candidates_df = _load_umax_candidates_csv_for_plot(candidates_path, sampled_units=sampled_units)
        print(f"[Fig.5] Plotting score components ({len(candidates_df)} background rows) ...", flush=True)
        policy_path = os.path.join(outdir, "fixed_umax_policy_by_repeat.csv")
        fixed_policy_df = pd.read_csv(policy_path) if os.path.exists(policy_path) else None
        cases_path = os.path.join(outdir, "umax_policy_ablation_cases.csv")
        if not os.path.exists(cases_path):
            cases_path = os.path.join(outdir, "umax_ablation_cases.csv")
        optimized_cases_df = pd.read_csv(cases_path) if os.path.exists(cases_path) else None
        components_png = os.path.join(figure_dir, "umax_score_components_landscape.png")
        plot_umax_score_components_landscape(
            candidates_df,
            components_png,
            fixed_policy_df=fixed_policy_df,
            optimized_cases_df=optimized_cases_df,
        )
        outputs["umax_score_components_landscape.png"] = components_png

    align_path = os.path.join(outdir, "umax_objective_alignment.csv")
    if os.path.exists(align_path):
        print("[Fig.5] Plotting umax_objective_alignment (supplementary) ...", flush=True)
        align_df = pd.read_csv(align_path)
        align_supp_png = os.path.join(figure_dir, "umax_objective_alignment_supplementary.png")
        plot_umax_optima_alignment(
            align_df, align_supp_png, u_grid=cl_manifest.get("u_grid"), faint_line_alpha=0.04,
        )
        outputs["umax_objective_alignment_supplementary.png"] = align_supp_png
        align_png = os.path.join(figure_dir, "umax_objective_alignment.png")
        plot_umax_optima_alignment(
            align_df, align_png, u_grid=cl_manifest.get("u_grid"), faint_line_alpha=0.04,
        )
        outputs["umax_objective_alignment.png"] = align_png
        full_align_png = os.path.join(figure_dir, "umax_objective_alignment_full_spaghetti.png")
        plot_umax_optima_alignment_full_spaghetti(align_df, full_align_png, u_grid=cl_manifest.get("u_grid"))
        outputs["umax_objective_alignment_full_spaghetti.png"] = full_align_png

    traj_path = os.path.join(outdir, "umax_ablation_representative_trajectories.csv")
    plot_manifest_path = os.path.join(outdir, FIG5_PLOT_MANIFEST_JSON)
    if os.path.exists(traj_path):
        print("[Fig.5] Plotting umax_ode_ablation ...", flush=True)
        traj_df = pd.read_csv(traj_path)
        plot_manifest = {}
        if os.path.exists(plot_manifest_path):
            with open(plot_manifest_path, encoding="utf-8") as fh:
                plot_manifest = json.load(fh)
        conditions = plot_manifest.get("plot_conditions") or list(traj_df.get("ablation_condition", traj_df.get("model", [])).unique())
        t_thr_raw = plot_manifest.get("t_thr_by_condition") or {}
        t_thr_by_condition = {k: np.asarray(v, dtype=float) for k, v in t_thr_raw.items()}
        policy_umax = plot_manifest.get("policy_umax_by_condition") or {}
        total_dosage_by_condition: Dict[str, float] = {}
        cases_path = os.path.join(outdir, "umax_ablation_cases.csv")
        if os.path.exists(cases_path):
            cases_df = pd.read_csv(cases_path)
            rep_repeat = plot_manifest.get("selected_repeat_id")
            rep_case = plot_manifest.get("selected_case_index")
            cond_col = "ablation_condition" if "ablation_condition" in cases_df.columns else "condition"
            dose_col = "total_dosage_ug_per_mL" if "total_dosage_ug_per_mL" in cases_df.columns else "total_dosage"
            if rep_repeat is not None and rep_case is not None and cond_col in cases_df.columns:
                rep_cases = cases_df[
                    (cases_df["repeat_id"] == rep_repeat) & (cases_df["case_index"] == rep_case)
                ]
                for _, row in rep_cases.iterrows():
                    cond = str(row[cond_col])
                    if dose_col in row.index and pd.notna(row[dose_col]):
                        total_dosage_by_condition[cond] = float(row[dose_col])
        ablation_png = os.path.join(figure_dir, "umax_ode_ablation.png")
        plot_umax_ode_ablation(
            traj_df, conditions, t_thr_by_condition, ablation_png,
            display_labels=plot_manifest.get("plot_display_labels"),
            policy_umax_by_condition=policy_umax,
            total_dosage_by_condition=total_dosage_by_condition or None,
        )
        primary_outputs["umax_ode_ablation.png"] = ablation_png
        outputs["umax_ode_ablation.png"] = ablation_png

    stats_path = os.path.join(outdir, "umax_ablation_repeated_plot_stats.csv")
    if os.path.exists(stats_path):
        print("[Fig.5] Plotting umax_summary_ablation ...", flush=True)
        stats_df = pd.read_csv(stats_path)
        annotations_df = _read_optional_csv(outdir, "umax_ablation_significance_annotations.csv")
        significance_for_manuscript = bool(
            n_repeats >= 10 and cl_manifest.get("significance_for_manuscript", n_repeats >= 10)
        )
        main_metric_names = [m for m, _ in FIG5_MAIN_SUMMARY_METRICS]
        try:
            from closed_loop_eval import FIG5_SIGNIFICANCE_REFERENCE
            sig_pairs: Dict[str, List[Tuple[str, str, str]]] = {}
            if significance_for_manuscript:
                for metric in main_metric_names:
                    pairs = closed_loop_significance_pairs_for_plot(
                        annotations_df, FIG5_SIGNIFICANCE_REFERENCE, metric=metric
                    )
                    if pairs:
                        sig_pairs[metric] = pairs
        except ImportError:
            sig_pairs = {}
        summary_png = os.path.join(figure_dir, "umax_summary_ablation.png")
        plot_umax_summary_ablation(
            stats_df, summary_png, n_repeats=n_repeats,
            significance_pairs=sig_pairs if sig_pairs else None, outdir=outdir,
            use_short_labels=True,
        )
        primary_outputs["umax_summary_ablation.png"] = summary_png
        outputs["umax_summary_ablation.png"] = summary_png
        composite_supp_png = os.path.join(figure_dir, "umax_summary_ablation_composite_supplementary.png")
        if "mean_composite_score" in stats_df.columns:
            plot_umax_summary_ablation(
                stats_df, composite_supp_png, n_repeats=n_repeats,
                significance_pairs=None, outdir=outdir,
                metric_specs=[("mean_composite_score", "Composite penalty score")],
                use_short_labels=True,
                horizontal_bars=True,
            )
            outputs["umax_summary_ablation_composite_supplementary.png"] = composite_supp_png
        try:
            from closed_loop_eval import FIG5_SECONDARY_SUMMARY_CONDITIONS
            rf_summary_png = os.path.join(figure_dir, "umax_summary_ablation_with_rf.png")
            plot_umax_summary_ablation(
                stats_df, rf_summary_png, n_repeats=n_repeats,
                significance_pairs=None, outdir=outdir,
                conditions_order=list(FIG5_SECONDARY_SUMMARY_CONDITIONS),
            )
            outputs["umax_summary_ablation_with_rf.png"] = rf_summary_png
        except ImportError:
            pass

    plot_body = build_fig5_plot_manifest(
        primary_outputs=primary_outputs,
        n_repeats=n_repeats,
        significance_for_manuscript=bool(
            n_repeats >= 10 and cl_manifest.get("significance_for_manuscript", n_repeats >= 10)
        ),
        prediction_sources=cl_manifest.get("prediction_sources") or cl_manifest.get("prediction_input_sources"),
    )
    plot_manifest = {}
    if os.path.exists(plot_manifest_path):
        with open(plot_manifest_path, encoding="utf-8") as fh:
            plot_manifest = json.load(fh)
    plot_manifest.update(plot_body)
    plot_manifest["generated_pngs"] = list(outputs.keys())
    write_json_manifest(outdir, FIG5_PLOT_MANIFEST_JSON, plot_manifest)
    if cl_manifest:
        cl_manifest.update(plot_body)
        cl_manifest["generated_pngs"] = list(outputs.keys())
        write_json_manifest(outdir, UMAX_OPTIMIZATION_MANIFEST_JSON, cl_manifest)
    return outputs


def write_json_manifest(outdir: str, filename: str, manifest: dict) -> str:
    path = os.path.join(outdir, filename)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    def _json_default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=_json_default)
    return path


def _fig_panel_record(
    panel: str,
    filename: Optional[str],
    description: str,
    *,
    role: str,
    input_sources: Sequence[str],
    n_repeats: int,
    umax_setting: str,
    significance_mode: str,
) -> dict:
    entry = {
        "panel": panel,
        "description": description,
        "role": role,
        "input_sources": list(input_sources),
        "n_repeats": int(n_repeats),
        "umax_setting": umax_setting,
        "significance_mode": significance_mode,
    }
    if filename:
        entry["filename"] = filename
    return entry


def build_fig3_plot_manifest(
    *,
    primary_outputs: Dict[str, str],
    supplementary_outputs: Dict[str, str],
    n_repeats: int,
    split_mode: str,
    significance_for_manuscript: bool,
    single_split_exploratory: bool,
    model_order: Sequence[str],
) -> dict:
    manuscript_safe = bool(n_repeats >= 100 and split_mode == "group" and significance_for_manuscript)
    sig_mode = (
        "TAR_vs_RF_UniformTreeMean_BestTree_bidirectional_stars_n_repeats_ge_10"
        if significance_for_manuscript
        else "exploratory_no_formal_stars"
    )
    panel_mapping = []
    for panel, filename, description in FIG3_PANEL_ORDER:
        if filename:
            mapped = MANUSCRIPT_SOURCE_MAPPING.get(filename, {})
            panel_description = mapped.get("description", description)
            source_group = mapped.get("source_group")
            input_sources = {
                "model_compare_r2.png": [
                    "tree_srl_benchmark/model_compare_summary.csv",
                    "tree_srl_benchmark/parameter_pairwise_significance.csv",
                ],
                "prediction_error_heatmap.png": [
                    "tree_srl_benchmark/model_compare_summary.csv",
                    "tree_srl_benchmark/model_compare_per_target.csv",
                    "tree_srl_benchmark/parameter_pairwise_significance.csv",
                ],
                "ode_back_outcome_heatmap.png": [
                    f"ode_back_validation/{ODE_BACK_PER_OUTCOME_CSV}",
                    f"ode_back_validation/{ODE_BACK_SUMMARY_CSV}",
                    f"ode_back_validation/{ODE_BACK_MANIFEST_JSON}",
                ],
                "uncertainty_decomposition.png": [
                    "tree_srl_benchmark/uncertainty_decomposition.csv",
                    "tree_srl_benchmark/uncertainty_summary.csv",
                ],
            }.get(filename, [])
            panel_mapping.append(
                _fig_panel_record(
                    panel, filename, panel_description,
                    role="primary_manuscript_figure",
                    input_sources=input_sources,
                    n_repeats=n_repeats,
                    umax_setting="not_applicable",
                    significance_mode=sig_mode if filename in {"model_compare_r2.png", "prediction_error_heatmap.png"} else "none",
                )
            )
            if source_group:
                panel_mapping[-1]["source_group"] = source_group
        else:
            panel_mapping.append(
                _fig_panel_record(
                    panel, None, description,
                    role="manual_schematic",
                    input_sources=["tree_srl_benchmark/model_compare_manifest.json"],
                    n_repeats=n_repeats,
                    umax_setting="not_applicable",
                    significance_mode="none",
                )
            )
            panel_mapping[-1]["manual_schematic_required"] = True
            panel_mapping[-1]["architecture_text"] = FIG3_ARCHITECTURE_TEXT
    return {
        "figure": "Fig. 3",
        "figure_title": "TAR benchmark",
        "primary_outputs": {k: v for k, v in primary_outputs.items() if k in FIG3_PRIMARY_FIGURES},
        "supplementary_outputs": supplementary_outputs,
        "figure_panel_mapping": panel_mapping,
        "manuscript_source_mapping": {
            k: dict(v)
            for k, v in MANUSCRIPT_SOURCE_MAPPING.items()
            if v.get("source_group") in {"tree_srl_benchmark", "ode_back_validation"}
            or k in supplementary_outputs
            or k in primary_outputs
        },
        "model_order": list(model_order),
        "display_labels": {m: model_display_label(m) for m in model_order},
        "prediction_input_source": "tree_srl_benchmark/repeats/repeat_XXX/predictions.csv",
        "statistics_used": {
            "model_compare_r2.png": (
                "threshold benchmark panel: mean target-wise R² original scale, mean ± 95% CI when n_repeats > 1; "
                "bidirectional TAR-vs-control stars (↑ TAR better, ↓ control better)"
            ),
            "prediction_error_heatmap.png": "RMSE original scale per target",
            "ode_back_outcome_heatmap.png": (
                "ODE-back functional outcome R²: predicted Tthr fed to ODE vs reference ODE outcomes; "
                "fixed non-optimized Umax per sample"
            ),
            "ode_back_r2_barplot.png": (
                "repeated ODE-back functional R2: mean outcome R² ± 95% CI; "
                "bidirectional TAR-vs-control stars (↑ TAR better, ↓ control better)"
            ),
            "target_weight_heatmap.png": "signed target-wise stacking coefficients (ridge/convex-aware colormap)",
            "uncertainty_decomposition.png": "aleatoric vs epistemic std (4×3 layout; pooled and per Tthr); supplementary diagnostic",
            "uncertainty_decomposition_by_target.png": "auxiliary uncertainty fraction by target",
        },
        "fig3_panel_semantics": {
            "Fig3B": "Direct Tthr prediction metrics (tree_srl_benchmark).",
            "Fig3C": "Direct Tthr prediction error heatmap.",
            "Fig3D": "ODE-back functional validation (ode_back_validation.py).",
        },
        "ode_back_validation_notes": (
            "ODE-back validation uses fixed/non-optimized Umax to isolate the effect of predicted Tthr. "
            "Fig. 3D heatmap and ode_back_r2_barplot live under results/ode_back_validation/figure/ "
            "(not copied into tree_srl_benchmark/figure/)."
        ),
        "significance_rule": SIGNIFICANCE_RULE_TEXT,
        "n_repeats": int(n_repeats),
        "significance_for_manuscript": bool(significance_for_manuscript),
        "exploratory": bool(n_repeats < 10 or single_split_exploratory),
        "manuscript_safe": manuscript_safe,
        "fig3_architecture_text": FIG3_ARCHITECTURE_TEXT,
        "fig3_manual_schematic_note": (
            "Panel A schematic is not auto-rendered; prepare manually for manuscript layout."
        ),
    }


def _constraint_success_figure_redundant(stats_df: pd.DataFrame) -> Tuple[bool, str]:
    """Return (skip, reason) when optional Fig. 4C would add no visual information."""
    cols = [
        "pathogen_constraint_success_rate",
        "probiotic_constraint_success_rate",
        "constraint_success_rate",
    ]
    sub = stats_df[stats_df["model"].isin(FIG4_FIXED_UMAX_MODELS)].copy()
    if sub.empty:
        return True, "no_fig4_models"
    available = [c for c in cols if c in sub.columns]
    if not available:
        return True, "missing_constraint_columns"
    if all(sub[c].nunique(dropna=True) <= 1 for c in available):
        return True, "identical_across_models"
    return False, ""


def plot_fixed_umax_constraint_success(
    stats_df: pd.DataFrame,
    outpath: str,
    *,
    model_order: Optional[List[str]] = None,
) -> None:
    """Optional Fig. 4C: per-constraint success rates (TAR leftmost)."""
    apply_matplotlib_style()
    order = model_order or list(FIG4_FIXED_UMAX_MODELS)
    plot_df = _reorder_bar_plot_df(stats_df.copy(), order)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    panel_specs = [
        ("pathogen_constraint_success_rate", "Pathogen"),
        ("probiotic_constraint_success_rate", "Probiotic"),
        ("constraint_success_rate", "All constraints"),
    ]
    x = np.arange(len(plot_df))
    colors = colors_for_models(plot_df["model"].tolist())
    labels = [_closed_loop_display_label(m) for m in plot_df["model"]]
    for ax, (col, subtitle) in zip(axes, panel_specs):
        if col not in plot_df.columns:
            ax.set_visible(False)
            continue
        y = plot_df[col].to_numpy(dtype=float)
        lo_col, hi_col = f"{col}_ci_low", f"{col}_ci_high"
        ax.bar(x, y, color=colors, edgecolor="white", linewidth=1.0, width=0.72)
        if lo_col in plot_df.columns and hi_col in plot_df.columns:
            lo = plot_df[lo_col].to_numpy(dtype=float)
            hi = plot_df[hi_col].to_numpy(dtype=float)
            yerr = np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])
            centers = x
            ax.errorbar(centers, y, yerr=yerr, fmt="none", capsize=4, linewidth=1.0, color="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=FS_TICK_SM)
        ax.tick_params(axis="x", pad=5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Success rate")
        ax.set_title(subtitle)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    finalize_figure_layout(fig, rect=(0.0, 0.16, 1.0, 1.0))
    save_figure(fig, outpath)
    plt.close(fig)


def build_fig4_plot_manifest(
    *,
    primary_outputs: Dict[str, str],
    optional_outputs: Dict[str, str],
    n_repeats: int,
    significance_for_manuscript: bool,
    illustrative_only: bool = True,
    skipped_optional: Optional[Dict[str, str]] = None,
    prediction_sources: Optional[Sequence[str]] = None,
) -> dict:
    sig_mode = (
        "TAR_vs_BestTree_and_UniformTreeMean_stars_when_TAR_better_n_repeats_ge_10"
        if significance_for_manuscript
        else "exploratory_no_formal_stars_n_repeats_lt_10"
    )
    panel_mapping = [
        _fig_panel_record(
            panel, filename, description,
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
            n_repeats=n_repeats,
            umax_setting="fixed_paper_figure_profile",
            significance_mode="none" if filename == "fixed_umax_representative.png" else sig_mode,
        )
        for panel, filename, description in FIG4_PANEL_ORDER
    ]
    for entry in panel_mapping:
        mapped = MANUSCRIPT_SOURCE_MAPPING.get(entry.get("filename", ""), {})
        if mapped.get("description"):
            entry["description"] = mapped["description"]
        if mapped.get("source_group"):
            entry["source_group"] = mapped["source_group"]
    primary_names = [p["filename"] for p in panel_mapping if p["filename"] in primary_outputs]
    return {
        "figure": "Fig. 4",
        "figure_title": "fixed-Umax validation",
        "validation_section": "fixed-Umax validation",
        "validation_mode": "fixed_umax_comparative",
        "prediction_input_sources": list(prediction_sources or []),
        "primary_outputs": {k: primary_outputs[k] for k in primary_names if k in primary_outputs},
        "optional_outputs": optional_outputs,
        "skipped_optional_figures": skipped_optional or {},
        "figure_panel_mapping": panel_mapping,
        "manuscript_source_mapping": {
            k: dict(v)
            for k, v in MANUSCRIPT_SOURCE_MAPPING.items()
            if v.get("source_group") == "fixed_umax_validation"
        },
        "fig4_models": list(FIG4_FIXED_UMAX_MODELS),
        "fig4_significance_comparisons": list(FIG4_SIGNIFICANCE_CONTROLS),
        "model_order": list(FIG4_FIXED_UMAX_MODELS),
        "display_labels": {m: model_display_label(m) for m in FIG4_FIXED_UMAX_MODELS},
        "statistics_used": {
            "fixed_umax_representative.png": "deterministic representative trajectories at shared fixed Umax",
            "fixed_umax_summary.png": (
                "total dosage, mean P_AUC, mean LR, terminal pathogen; single forward ODE point estimates"
            ),
            "fixed_umax_constraint_success.png": (
                "pathogen, probiotic, and all-constraint success rates; mean ± 95% CI (optional)"
            ),
        },
        "fig4b_metrics": [
            "mean_total_dosage",
            "mean_P_AUC",
            "mean_LR",
            "mean_terminal_pathogen",
        ],
        "significance_rule": FIG4_SIGNIFICANCE_RULE_TEXT,
        "n_repeats": int(n_repeats),
        "significance_for_manuscript": bool(significance_for_manuscript),
        "exploratory": bool(n_repeats < 10),
        "manuscript_safe": bool(n_repeats >= 100 and significance_for_manuscript),
        "illustrative_only_representative": illustrative_only,
    }


build_closed_loop_plot_manifest = build_fig4_plot_manifest


def _ode_back_validation_dir(base: str) -> str:
    return os.path.join(base, ODE_BACK_VALIDATION_SUBDIR)


def generate_ode_back_plots(outdir: str, config: Optional[dict] = None) -> Dict[str, str]:
    """Regenerate ODE-back validation figures from CSV artifacts."""
    figure_dir = os.path.join(outdir, "figure")
    os.makedirs(figure_dir, exist_ok=True)
    summary_path = os.path.join(outdir, ODE_BACK_SUMMARY_CSV)
    per_outcome_path = os.path.join(outdir, ODE_BACK_PER_OUTCOME_CSV)
    case_path = os.path.join(outdir, ODE_BACK_CASE_CSV)
    if not os.path.isfile(summary_path) or not os.path.isfile(per_outcome_path):
        return {}

    summary_df = _normalize_ode_back_summary_df(pd.read_csv(summary_path))
    per_outcome_df = pd.read_csv(per_outcome_path)
    case_df = _read_optional_csv(outdir, ODE_BACK_CASE_CSV)
    if "model" in summary_df.columns:
        summary_df["model"] = summary_df["model"].map(normalize_model_name)
    if "model" in per_outcome_df.columns:
        per_outcome_df["model"] = per_outcome_df["model"].map(normalize_model_name)
    if not case_df.empty and "model" in case_df.columns:
        case_df["model"] = case_df["model"].map(normalize_model_name)

    model_order = [m for m in ODE_BACK_BAR_MODEL_ORDER if m in set(summary_df["model"])]
    outputs: Dict[str, str] = {}

    pairwise_df = _read_optional_csv(outdir, ODE_BACK_PAIRWISE_CSV)
    if not pairwise_df.empty:
        if "srl_model" in pairwise_df.columns:
            pairwise_df = pairwise_df.copy()
            pairwise_df["srl_model"] = pairwise_df["srl_model"].map(normalize_model_name)
            pairwise_df["control_model"] = pairwise_df["control_model"].map(normalize_model_name)
    n_repeats = int(summary_df["n_repeats"].max()) if "n_repeats" in summary_df.columns else 0
    if n_repeats <= 0 and not pairwise_df.empty and "n_repeats" in pairwise_df.columns:
        n_repeats = int(pairwise_df["n_repeats"].max())
    main_controls = [m for m in model_order if m != TAR_MODEL]
    significance_for_manuscript = n_repeats >= 10
    sig_pairs = significance_pairs_for_plot(
        pairwise_df,
        TAR_MODEL,
        main_controls,
        metric=ODE_BACK_BAR_METRIC,
        significance_for_manuscript=significance_for_manuscript,
        bidirectional=True,
    )

    bar_path = os.path.join(figure_dir, "ode_back_r2_barplot.png")
    plot_ode_back_r2_barplot(
        summary_df,
        bar_path,
        model_order=model_order,
        significance_pairs=sig_pairs if sig_pairs else None,
        n_repeats=n_repeats,
    )
    outputs["ode_back_r2_barplot.png"] = bar_path

    if ODE_BACK_TRAJECTORY_R2_METRIC in summary_df.columns:
        traj_pairwise_df = _read_optional_csv(outdir, ODE_BACK_TRAJECTORY_PAIRWISE_CSV)
        if not traj_pairwise_df.empty and "srl_model" in traj_pairwise_df.columns:
            traj_pairwise_df = traj_pairwise_df.copy()
            traj_pairwise_df["srl_model"] = traj_pairwise_df["srl_model"].map(normalize_model_name)
            traj_pairwise_df["control_model"] = traj_pairwise_df["control_model"].map(normalize_model_name)
        traj_sig_pairs = significance_pairs_for_plot(
            traj_pairwise_df,
            TAR_MODEL,
            main_controls,
            metric=ODE_BACK_TRAJECTORY_R2_METRIC,
            significance_for_manuscript=significance_for_manuscript,
            bidirectional=True,
        )
        traj_bar_path = os.path.join(figure_dir, "ode_back_trajectory_r2_barplot.png")
        plot_ode_back_trajectory_r2_barplot(
            summary_df,
            traj_bar_path,
            model_order=model_order,
            significance_pairs=traj_sig_pairs if traj_sig_pairs else None,
            n_repeats=n_repeats,
        )
        outputs["ode_back_trajectory_r2_barplot.png"] = traj_bar_path

    heatmap_path = os.path.join(figure_dir, "ode_back_outcome_heatmap.png")
    plot_ode_back_outcome_heatmap(per_outcome_df, heatmap_path, model_order=model_order)
    outputs["ode_back_outcome_heatmap.png"] = heatmap_path

    if not case_df.empty:
        scatter_path = os.path.join(figure_dir, "ode_back_pred_vs_ref_scatter.png")
        plot_ode_back_pred_vs_ref_scatter(case_df, scatter_path, model_order=model_order)
        outputs["ode_back_pred_vs_ref_scatter.png"] = scatter_path

    manifest_path = os.path.join(outdir, ODE_BACK_MANIFEST_JSON)
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            ob_manifest = json.load(fh)
        ob_manifest["significance_rule"] = SIGNIFICANCE_RULE_TEXT
        ob_manifest["statistics_used"] = {
            **dict(ob_manifest.get("statistics_used") or {}),
            "ode_back_r2_barplot.png": (
                "repeated ODE-back functional R2; bidirectional TAR-vs-control stars "
                "(↑ TAR better, ↓ control better); CSV p-values/diffs unchanged"
            ),
        }
        ob_manifest["manuscript_source_mapping"] = {
            "ode_back_r2_barplot.png": dict(MANUSCRIPT_SOURCE_MAPPING["ode_back_r2_barplot.png"]),
        }
        write_json_manifest(outdir, ODE_BACK_MANIFEST_JSON, ob_manifest)

    return outputs


def write_fig3_manifest(outdir: str, manifest: dict) -> str:
    return write_json_manifest(outdir, "model_compare_manifest.json", manifest)


def _read_optional_csv(outdir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(outdir, filename)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_stacker_type(outdir: str) -> str:
    manifest_path = os.path.join(outdir, "model_compare_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        rows = manifest.get("repeat_metadata", [])
        if rows:
            return str(rows[-1].get("chosen_stacker_type", "ridge")) or "ridge"
    return "ridge"


def _ensure_figure_dir(parent_dir: str) -> str:
    """PNG outputs for benchmark / closed-loop steps live under parent_dir/figure/."""
    figure_dir = os.path.join(parent_dir, "figure")
    os.makedirs(figure_dir, exist_ok=True)
    return figure_dir


def generate_benchmark_plots(outdir: str, config: Optional[dict] = None) -> Dict[str, str]:
    """Regenerate all benchmark PNGs from CSV artifacts in outdir."""
    figure_dir = _ensure_figure_dir(outdir)
    manifest_path = os.path.join(outdir, "model_compare_manifest.json")
    if config is None:
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing {manifest_path}; run tree_srl_benchmark first.")
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        config = dict(manifest.get("run_config", {}))
        config["feature_schemas"] = manifest.get("feature_schemas", {})
        config["n_repeats"] = manifest.get("n_repeats", config.get("n_repeats", 1))
        repeat_metadata_rows = manifest.get("repeat_metadata", [])
    else:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh) if os.path.exists(manifest_path) else {}
        repeat_metadata_rows = manifest.get("repeat_metadata", [])

    n_repeats = int(config.get("n_repeats", 1))
    single_split_exploratory = bool(config.get("single_split_exploratory", n_repeats == 1))
    split_mode = str(config.get("split_mode", "group"))
    significance_for_manuscript = bool(
        manifest.get("significance_for_manuscript")
        if manifest.get("significance_for_manuscript") is not None
        else (n_repeats >= 10 and split_mode == "group" and not config.get("not_for_manuscript", False))
    )

    summary_df = pd.read_csv(os.path.join(outdir, "model_compare_summary.csv"))
    summary_df = summary_df.copy()
    summary_df["model"] = summary_df["model"].map(normalize_model_name)
    per_target_df = _read_optional_csv(outdir, "model_compare_per_target.csv")
    if not per_target_df.empty and "model" in per_target_df.columns:
        per_target_df = per_target_df.copy()
        per_target_df["model"] = per_target_df["model"].map(normalize_model_name)
    weights_df = _read_optional_csv(outdir, "target_weight_table.csv")
    pairwise_df = _read_optional_csv(outdir, "parameter_pairwise_significance.csv")
    if not pairwise_df.empty and "srl_model" in pairwise_df.columns:
        pairwise_df = pairwise_df.copy()
        pairwise_df["srl_model"] = pairwise_df["srl_model"].map(normalize_model_name)
        pairwise_df["control_model"] = pairwise_df["control_model"].map(normalize_model_name)
    stacker_type = _load_stacker_type(outdir)

    available = set(summary_df["model"].tolist())
    outputs: Dict[str, str] = {}

    main_plot_order = list(MANUSCRIPT_BAR_MODEL_ORDER)
    main_models = select_main_controls_plot(available, main_order=main_plot_order)
    main_controls = [m for m in main_models if m != TAR_MODEL]
    main_pairs = significance_pairs_for_plot(
        pairwise_df,
        TAR_MODEL,
        main_controls,
        significance_for_manuscript=significance_for_manuscript,
        bidirectional=True,
    )
    rmse_pairs = significance_pairs_for_plot(
        pairwise_df,
        TAR_MODEL,
        main_controls,
        metric="mean_RMSE_original",
        significance_for_manuscript=significance_for_manuscript,
        bidirectional=True,
    )

    main_path = os.path.join(figure_dir, "model_compare_r2.png")
    plot_model_metric_combined_panel(
        summary_df,
        per_target_df,
        main_path,
        summary_metric_col="mean_R2_original",
        summary_ylabel="Mean target-wise $R^2$ (original scale)",
        summary_title="Mean target-wise $R^2$",
        heatmap_value_col="R2_original",
        heatmap_cbar_label="$R^2$ (original scale)",
        model_order=main_models,
        significance_pairs=main_pairs if significance_for_manuscript else None,
    )
    primary_outputs: Dict[str, str] = {}
    primary_outputs["model_compare_r2.png"] = main_path
    outputs["model_compare_r2.png"] = main_path

    tw_path = os.path.join(figure_dir, "target_weight_heatmap.png")
    plot_target_weight_heatmap(weights_df, tw_path, stacker_type=stacker_type)
    supplementary_outputs: Dict[str, str] = {"target_weight_heatmap.png": tw_path}
    outputs["target_weight_heatmap.png"] = tw_path

    pe_path = os.path.join(figure_dir, "prediction_error_heatmap.png")
    plot_model_metric_combined_panel(
        summary_df,
        per_target_df,
        pe_path,
        summary_metric_col="mean_RMSE_original",
        summary_ylabel="Mean target-wise RMSE (original scale)",
        summary_title="Mean target-wise RMSE",
        heatmap_value_col="RMSE_original",
        heatmap_cbar_label="RMSE (original scale)",
        model_order=main_models,
        significance_pairs=rmse_pairs if significance_for_manuscript else None,
    )
    primary_outputs["prediction_error_heatmap.png"] = pe_path
    outputs["prediction_error_heatmap.png"] = pe_path

    uncertainty_case_df = _read_optional_csv(outdir, "uncertainty_decomposition.csv")
    uncertainty_summary_df = _read_optional_csv(outdir, "uncertainty_decomposition_summary.csv")
    if uncertainty_summary_df.empty:
        uncertainty_summary_df = _read_optional_csv(outdir, "uncertainty_summary.csv")
    if not uncertainty_case_df.empty:
        unc_path = os.path.join(figure_dir, "uncertainty_decomposition.png")
        plot_uncertainty_decomposition(
            uncertainty_case_df,
            uncertainty_summary_df,
            unc_path,
            method=str(config.get("uncertainty_method", "mc_dropout")),
        )
        # Fig. 3D is ODE-back; uncertainty stays supplementary even with --show_uncertainty_main.
        supplementary_outputs["uncertainty_decomposition.png"] = unc_path
        outputs["uncertainty_decomposition.png"] = unc_path
        by_target_path = os.path.join(figure_dir, "uncertainty_decomposition_by_target.png")
        if os.path.exists(by_target_path):
            supplementary_outputs["uncertainty_decomposition_by_target.png"] = by_target_path
            outputs["uncertainty_decomposition_by_target.png"] = by_target_path

    ode_back_dir = config.get("ode_back_outdir")
    if not ode_back_dir:
        parent = os.path.dirname(os.path.abspath(outdir))
        candidate = os.path.join(parent, ODE_BACK_VALIDATION_SUBDIR)
        if os.path.isfile(os.path.join(candidate, ODE_BACK_PER_OUTCOME_CSV)):
            ode_back_dir = candidate
    if ode_back_dir and os.path.isdir(ode_back_dir):
        ode_back_outputs = generate_ode_back_plots(ode_back_dir)
        heatmap_src = ode_back_outputs.get("ode_back_outcome_heatmap.png")
        if heatmap_src and os.path.isfile(heatmap_src):
            primary_outputs["ode_back_outcome_heatmap.png"] = heatmap_src
            outputs["ode_back_outcome_heatmap.png"] = heatmap_src
        for name, src_path in ode_back_outputs.items():
            if name == "ode_back_outcome_heatmap.png":
                continue
            supplementary_outputs[name] = src_path
            outputs[name] = src_path

    for csv_name in [
        "model_compare_summary.csv",
        "model_compare_per_target.csv",
        "repeated_parameter_metrics.csv",
        "parameter_pairwise_significance.csv",
        "tree_expert_table.csv",
        "target_weight_table.csv",
        "model_compare_manifest.json",
        "predictions_manifest.json",
        "uncertainty_decomposition.csv",
        "uncertainty_summary.csv",
        "uncertainty_manifest.json",
    ]:
        csv_path = os.path.join(outdir, csv_name)
        if os.path.exists(csv_path):
            outputs[csv_name] = csv_path
    for pred_path in sorted(glob.glob(os.path.join(outdir, "predictions_repeat_*.csv"))):
        outputs[os.path.basename(pred_path)] = pred_path
    if os.path.exists(os.path.join(outdir, "predictions_all_models.csv")):
        outputs["predictions_all_models.csv"] = os.path.join(outdir, "predictions_all_models.csv")

    plot_manifest = build_fig3_plot_manifest(
        primary_outputs=primary_outputs,
        supplementary_outputs=supplementary_outputs,
        n_repeats=n_repeats,
        split_mode=split_mode,
        significance_for_manuscript=significance_for_manuscript,
        single_split_exploratory=single_split_exploratory,
        model_order=main_models,
    )
    write_json_manifest(outdir, "benchmark_plot_manifest.json", plot_manifest)

    manifest_update = {
        **{k: v for k, v in manifest.items() if k != "outputs"},
        "outputs": outputs,
        "final_model_name": TAR_MODEL,
        "final_architecture": "single-tree experts + target-wise OOF stack",
        "cycle_consistency_used": False,
        "residual_used_in_final_model": bool(manifest.get("residual_used_in_final_model", False)),
        "ET_used_as_control": bool(config.get("include_et_debug", False)),
        "manuscript_baselines": ["RF", "BestTree", "UniformTreeMean"],
        "row_split_manuscript_safe": False,
        "n_repeats": n_repeats,
        "significance_for_manuscript": significance_for_manuscript,
        "single_split_exploratory": single_split_exploratory,
        "main_plot_models": main_models,
        "generated_primary_png": list(primary_outputs.keys()),
        "benchmark_plot_manifest": "benchmark_plot_manifest.json",
        "deprecated_outputs_removed": True,
        "csv_data_sources": {
            "model_compare_r2.png": {
                "description": MANUSCRIPT_SOURCE_MAPPING["model_compare_r2.png"]["description"],
                "source_group": "tree_srl_benchmark",
                "files": ["model_compare_summary.csv", "parameter_pairwise_significance.csv"],
            },
            "target_weight_heatmap.png": {
                "description": MANUSCRIPT_SOURCE_MAPPING["target_weight_heatmap.png"]["description"],
                "source_group": "tree_srl_benchmark",
                "files": ["target_weight_table.csv"],
            },
            "prediction_error_heatmap.png": {
                "description": "Per-target RMSE heatmap",
                "source_group": "tree_srl_benchmark",
                "files": ["model_compare_per_target.csv"],
            },
            "uncertainty_decomposition.png": {
                "description": "aleatoric vs epistemic uncertainty scatters",
                "source_group": "tree_srl_benchmark",
                "files": ["uncertainty_decomposition.csv", "uncertainty_summary.csv"],
            },
            "uncertainty_decomposition_by_target.png": {
                "description": MANUSCRIPT_SOURCE_MAPPING["uncertainty_decomposition_by_target.png"]["description"],
                "source_group": "tree_srl_benchmark",
                "files": ["uncertainty_decomposition.csv", "uncertainty_summary.csv"],
            },
            "ode_back_r2_barplot.png": {
                "description": MANUSCRIPT_SOURCE_MAPPING["ode_back_r2_barplot.png"]["description"],
                "source_group": "ode_back_validation",
                "files": [ODE_BACK_SUMMARY_CSV, ODE_BACK_PAIRWISE_CSV],
            },
        },
        "manuscript_source_mapping": dict(MANUSCRIPT_SOURCE_MAPPING),
        "significance_rule": SIGNIFICANCE_RULE_TEXT,
    }
    write_fig3_manifest(outdir, manifest_update)
    outputs["model_compare_manifest.json"] = manifest_path
    return outputs


def _ode_dir(base: str) -> str:
    return os.path.join(base, "ode")


def generate_ode_plots(outdir: str) -> Dict[str, str]:
    """Regenerate representative ODE trajectory PNGs from CSV artifacts in outdir."""
    manifest_path = os.path.join(outdir, "ode_simulation_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing {manifest_path}; run multi_pathogen_simulator --mode representative first.")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    outputs: Dict[str, str] = {}
    history_name = manifest.get("trajectory_history_csv", "representative_trajectory_history.csv")
    history_path = os.path.join(outdir, history_name)
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Missing trajectory history CSV: {history_path}")
    history_df = pd.read_csv(history_path)
    t_thr = read_t_thr_from_manifest(manifest)
    png_name = manifest.get("figure_png", f"representative_{manifest.get('profile', 'paper_figure')}.png")
    png_path = os.path.join(outdir, png_name)
    plot_representative_ode_trajectory(
        history_df,
        t_thr,
        png_path,
        total_dosage=float(manifest.get("total_dosage", 0.0)),
        probiotic_two_compartment=bool(manifest.get("probiotic_two_compartment", False)),
    )
    outputs[png_name] = png_path
    manifest["figures_note"] = f"Regenerated via: python figure_audit.py --mode generate_plots --groups ode --ode_outdir {outdir}"
    write_json_manifest(outdir, "ode_simulation_manifest.json", manifest)
    return outputs


def generate_fig4_plots(outdir: str) -> Dict[str, str]:
    """Regenerate Fig. 4 fixed-Umax validation panels."""
    figure_dir = _ensure_figure_dir(outdir)
    manifest_path = os.path.join(outdir, FIG4_MANIFEST_JSON)
    plot_manifest_path = os.path.join(outdir, FIG4_PLOT_MANIFEST_JSON)
    traj_path = os.path.join(outdir, FIXED_UMAX_TRAJECTORIES_CSV)
    if not os.path.exists(traj_path):
        raise FileNotFoundError(
            f"Missing {traj_path}; run fixed-Umax validation with closed_loop_eval.py first."
        )

    cl_manifest: dict = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            cl_manifest = json.load(fh)
    n_repeats = 1
    n_prediction_repeats = int(cl_manifest.get("n_prediction_repeats", cl_manifest.get("n_repeats", 1)))

    trajectories_df = pd.read_csv(traj_path)
    if os.path.exists(plot_manifest_path):
        with open(plot_manifest_path, encoding="utf-8") as fh:
            plot_manifest = json.load(fh)
    else:
        plot_manifest = {
            "plot_models": list(FIG4_FIXED_UMAX_MODELS),
            "t_thr_by_model": {},
            "metrics_by_model": {},
        }

    model_labels = [
        m
        for m in (plot_manifest.get("plot_models") or list(FIG4_FIXED_UMAX_MODELS))
        if m in FIG4_FIXED_UMAX_MODELS
    ]
    if not model_labels:
        model_labels = [m for m in FIG4_FIXED_UMAX_MODELS if m in trajectories_df["model"].unique()]
    display_labels = plot_manifest.get("plot_display_labels") or [_closed_loop_display_label(m) for m in model_labels]
    t_thr_by_model: Dict[str, np.ndarray] = {}
    raw_t_thr = plot_manifest.get("t_thr_by_model") or {}
    for model in model_labels:
        if model in raw_t_thr:
            t_thr_by_model[model] = np.asarray(raw_t_thr[model], dtype=float)
        else:
            t_thr_by_model[model] = read_t_thr_from_manifest(plot_manifest, model=model)

    outputs: Dict[str, str] = {}
    primary_outputs: Dict[str, str] = {}
    optional_outputs: Dict[str, str] = {}
    skipped_optional: Dict[str, str] = {}

    rep_path = os.path.join(figure_dir, "fixed_umax_representative.png")
    plot_fixed_umax_representative(
        trajectories_df,
        model_labels,
        t_thr_by_model,
        rep_path,
        display_labels=display_labels,
    )
    primary_outputs["fixed_umax_representative.png"] = rep_path
    outputs["fixed_umax_representative.png"] = rep_path

    stats_path = os.path.join(outdir, FIXED_UMAX_REPEATED_STATS_CSV)
    summary_fallback = os.path.join(outdir, "fixed_umax_validation_summary_by_model.csv")
    if os.path.exists(stats_path):
        stats_df = pd.read_csv(stats_path)
    elif os.path.exists(summary_fallback):
        stats_df = pd.read_csv(summary_fallback)
    else:
        raise FileNotFoundError(
            f"Missing {stats_path}; run fixed-Umax validation with closed_loop_eval.py first."
        )
    stats_df = stats_df[stats_df["model"].isin(FIG4_FIXED_UMAX_MODELS)].copy()

    annotations_df = _read_optional_csv(outdir, FIXED_UMAX_SIGNIFICANCE_CSV)
    significance_for_manuscript = False
    fig4b_metrics = (
        "mean_total_dosage",
        "mean_P_AUC",
        "mean_LR",
        "mean_terminal_pathogen",
    )
    sig_pairs_by_metric: Dict[str, List[Tuple[str, str, str]]] = {}
    if significance_for_manuscript:
        for metric in fig4b_metrics:
            pairs = closed_loop_significance_pairs_for_plot(annotations_df, TAR_MODEL, metric=metric)
            pairs = [
                (a, b, lbl)
                for a, b, lbl in pairs
                if b in FIG4_SIGNIFICANCE_CONTROLS
            ]
            if pairs:
                sig_pairs_by_metric[metric] = pairs

    summary_path = os.path.join(figure_dir, "fixed_umax_summary.png")
    plot_fixed_umax_summary(
        stats_df,
        summary_path,
        n_repeats=n_repeats,
        significance_pairs=sig_pairs_by_metric if sig_pairs_by_metric else None,
        outdir=outdir,
    )
    primary_outputs["fixed_umax_summary.png"] = summary_path
    outputs["fixed_umax_summary.png"] = summary_path

    skip_constraint, skip_reason = _constraint_success_figure_redundant(stats_df)
    if skip_constraint:
        skipped_optional["fixed_umax_constraint_success.png"] = skip_reason
    else:
        constraint_path = os.path.join(figure_dir, "fixed_umax_constraint_success.png")
        plot_fixed_umax_constraint_success(stats_df, constraint_path)
        optional_outputs["fixed_umax_constraint_success.png"] = constraint_path
        outputs["fixed_umax_constraint_success.png"] = constraint_path

    plot_manifest_body = build_fig4_plot_manifest(
        primary_outputs=primary_outputs,
        optional_outputs=optional_outputs,
        n_repeats=n_repeats,
        significance_for_manuscript=significance_for_manuscript,
        illustrative_only=True,
        skipped_optional=skipped_optional,
        prediction_sources=cl_manifest.get("prediction_sources") or [cl_manifest.get("prediction_source", "")],
    )
    plot_manifest.update(plot_manifest_body)
    plot_manifest["generated_pngs"] = list(outputs.keys())
    write_json_manifest(outdir, FIG4_PLOT_MANIFEST_JSON, plot_manifest)
    if cl_manifest:
        cl_manifest["generated_pngs"] = list(outputs.keys())
        cl_manifest.update(plot_manifest_body)
        write_json_manifest(outdir, FIG4_MANIFEST_JSON, cl_manifest)
    return outputs




def generate_heatmap_plots(outdir: str) -> Dict[str, str]:
    """Regenerate heatmap PNGs from saved correlation-matrix CSVs."""
    from heatmap import filter_heatmap_correlation, plot_heatmap, subset_title

    config_path = os.path.join(outdir, "heatmap_config.json")
    config: dict = {}
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as fh:
            config = json.load(fh)
    outputs: Dict[str, str] = {}
    figsize = tuple(config.get("figsize", [12.0, 10.0]))
    dpi = int(config.get("dpi", 300))
    clustered = bool(config.get("clustered", False))
    default_title = config.get("title") or subset_title(config.get("subset", "input"))

    csv_paths = sorted(glob.glob(os.path.join(outdir, "*_correlation_matrix.csv")))
    if not csv_paths and config.get("outputs"):
        for paths in config["outputs"].values():
            csv_path = paths.get("csv")
            if csv_path and os.path.exists(csv_path):
                csv_paths.append(csv_path)

    for csv_path in csv_paths:
        basename = os.path.basename(csv_path).replace("_correlation_matrix.csv", "")
        png_path = os.path.join(outdir, f"{basename}.png")
        if "feature_controller" in basename:
            title = subset_title("feature_controller")
        elif "_input" in basename or basename.startswith("heatmap_"):
            title = subset_title("input")
        else:
            title = default_title
        corr = filter_heatmap_correlation(pd.read_csv(csv_path, index_col=0))
        png_path, svg_path = plot_heatmap(corr, png_path, title, figsize, dpi, clustered)
        outputs[os.path.basename(png_path)] = png_path
        outputs[os.path.basename(svg_path)] = svg_path
    return outputs


def generate_figure_plots(
    *,
    groups: Sequence[str],
    results_root: str,
    ode_outdir: Optional[str] = None,
    heatmap_outdir: Optional[str] = None,
    benchmark_outdir: Optional[str] = None,
    closed_loop_outdir: Optional[str] = None,
    fixed_umax_outdir: Optional[str] = None,
    umax_optimization_outdir: Optional[str] = None,
    ode_back_outdir: Optional[str] = None,
) -> Dict[str, str]:
    """Generate PNG + SVG figures for one or more figure groups from saved CSV artifacts."""
    normalized = {g.strip().lower() for g in groups}
    if "all" in normalized:
        normalized = {"ode", "heatmap", "benchmark", "fixed_umax_validation", "umax_optimization", "ode_back_validation"}

    outputs: Dict[str, str] = {}
    if "ode" in normalized:
        ode_dir = ode_outdir or _ode_dir(results_root)
        outputs.update(generate_ode_plots(ode_dir))
    if "heatmap" in normalized:
        heatmap_dir = heatmap_outdir or _heatmap_dir(results_root)
        outputs.update(generate_heatmap_plots(heatmap_dir))
    if "benchmark" in normalized:
        bench_dir = benchmark_outdir or _benchmark_dir(results_root)
        outputs.update(generate_benchmark_plots(bench_dir))
    if normalized & {"ode_back_validation", "ode_back"}:
        ob_dir = ode_back_outdir or _ode_back_validation_dir(results_root)
        outputs.update(generate_ode_back_plots(ob_dir))
    if normalized & {"fixed_umax_validation", "fig4", "closed_loop"}:
        fig4_dir = (
            fixed_umax_outdir
            or closed_loop_outdir
            or _fixed_umax_validation_dir(results_root)
        )
        outputs.update(generate_fig4_plots(fig4_dir))
    if normalized & {"umax_optimization", "fig5"}:
        uo_dir = umax_optimization_outdir or _umax_optimization_dir(results_root)
        outputs.update(generate_umax_optimization_plots(uo_dir))
    return expand_figure_outputs(outputs)


# --- artifact audit ---

@dataclass(frozen=True)
class ExpectedArtifact:
    rel_path: str
    description: str
    suggest_command: str
    figure_group: str
    optional: bool = False


def _heatmap_dir(base: str) -> str:
    return os.path.join(base, "heatmap")


def _benchmark_dir(base: str) -> str:
    return os.path.join(base, "tree_srl_benchmark")


def _fixed_umax_validation_dir(base: str) -> str:
    return os.path.join(base, FIXED_UMAX_VALIDATION_SUBDIR)


_closed_loop_dir = _fixed_umax_validation_dir


def _umax_optimization_dir(base: str) -> str:
    return os.path.join(base, "umax_optimization")


def _ode_dir_from_root(base: str) -> str:
    return os.path.join(base, "ode")


def build_expected_artifacts(
    results_root: str,
    x_csv: str,
    y_csv: str,
    metadata_csv: Optional[str],
    enable_et_srl: bool = False,
) -> List[ExpectedArtifact]:
    heatmap = _heatmap_dir(results_root)
    benchmark = _benchmark_dir(results_root)
    benchmark_figures = os.path.join(benchmark, "figure")
    fixed_umax = _fixed_umax_validation_dir(results_root)
    fixed_umax_figures = os.path.join(fixed_umax, "figure")
    umax_optimization = _umax_optimization_dir(results_root)
    umax_optimization_figures = os.path.join(umax_optimization, "figure")
    ode = _ode_dir_from_root(results_root)
    meta_flag = f"  --metadata_csv {metadata_csv} `\n" if metadata_csv else ""
    weight_flag = "  --sample_weight_csv data/microbio_formal_dataset/sample_weights.csv `\n"

    ode_cmd = (
        f"python .\\multi_pathogen_simulator.py `\n"
        f"  --mode representative `\n"
        f"  --profile paper_figure `\n"
        f"  --outdir {ode}\n\n"
        f"python .\\figure_audit.py --mode generate_plots --groups ode --ode_outdir {ode}"
    )
    heatmap_input_cmd = (
        f"python .\\heatmap.py `\n"
        f"  --x_csv {x_csv} `\n"
        f"  --outdir {heatmap} `\n"
        f"  --method both `\n"
        f"  --subset input"
    )
    heatmap_fc_cmd = (
        f"python .\\heatmap.py `\n"
        f"  --x_csv {x_csv} `\n"
        f"  --y_csv {y_csv} `\n"
        f"  --outdir {heatmap} `\n"
        f"  --method both `\n"
        f"  --subset feature_controller"
    )
    smoke_cmd = (
        f"python .\\tree_srl_benchmark.py `\n"
        f"  --x_csv {x_csv} `\n"
        f"  --y_csv {y_csv} `\n"
        f"{meta_flag}"
        f"{weight_flag}"
        f"  --split_mode group `\n"
        f"  --outdir {benchmark} `\n"
        f"  --n_repeats 3 `\n"
        f"  --n_jobs 2 `\n"
        f"  --lgbm_device gpu `\n"
        f"  --bootstrap 0 `\n"
        f"  --expanded_tree_bank\n\n"
        f"python .\\figure_audit.py --mode generate_plots --groups benchmark --benchmark_outdir {benchmark}"
    )
    manuscript_cmd = (
        f"python .\\tree_srl_benchmark.py `\n"
        f"  --x_csv {x_csv} `\n"
        f"  --y_csv {y_csv} `\n"
        f"{meta_flag}"
        f"{weight_flag}"
        f"  --split_mode group `\n"
        f"  --outdir {benchmark} `\n"
        f"  --n_repeats 100 `\n"
        f"  --n_jobs 2 `\n"
        f"  --lgbm_device gpu `\n"
        f"  --bootstrap 500 `\n"
        f"  --expanded_tree_bank\n\n"
        f"python .\\figure_audit.py --mode generate_plots --groups benchmark --benchmark_outdir {benchmark}"
    )
    ode_back = _ode_back_validation_dir(results_root)
    ode_back_figures = os.path.join(ode_back, "figure")
    ode_back_cmd = (
        f"python .\\ode_back_validation.py `\n"
        f"  --predictions_dir {benchmark}\\repeats `\n"
        f"  --predictions_manifest {benchmark}\\predictions_manifest.json `\n"
        f"  --x_csv {x_csv} `\n"
        f"  --y_csv {y_csv} `\n"
        f"{meta_flag}"
        f"  --outdir {ode_back} `\n"
        f"  --reference_mode reference_tthr `\n"
        f"  --functional_umax_source metadata_soft_umax `\n"
        f"  --backend numba `\n"
        f"  --n_jobs -1 `\n"
        f"  --validate_fast_backend\n\n"
        f"python .\\figure_audit.py --mode generate_plots --groups benchmark --benchmark_outdir {benchmark}"
    )
    fixed_umax_cmd = (
        f"python .\\closed_loop_eval.py `\n"
        f"  --predictions_dir {benchmark}\\repeats `\n"
        f"  --predictions_manifest {benchmark}\\predictions_manifest.json `\n"
        f"  --x_csv {x_csv} `\n"
        f"  --metadata_csv {metadata_csv} `\n"
        f"  --outdir {fixed_umax} `\n"
        f"  --u_grid arange:0:100:0.5 `\n"
        f"  --target_total_dosage 2500 `\n"
        f"  --target_terminal_pathogen 4e7 `\n"
        f"  --weight_profile custom `\n"
        f"  --w_track 1.0 `\n"
        f"  --w_path 1.0 `\n"
        f"  --w_probiotic 1.0 `\n"
        f"  --w_dose 1.0\n\n"
        f"python .\\figure_audit.py --mode generate_plots --groups fixed_umax_validation --fixed_umax_outdir {fixed_umax}"
    )
    umax_optimization_cmd = (
        f"python .\\closed_loop_eval.py `\n"
        f"  --predictions_dir {benchmark}\\repeats `\n"
        f"  --predictions_manifest {benchmark}\\predictions_manifest.json `\n"
        f"  --x_csv {x_csv} `\n"
        f"  --metadata_csv {metadata_csv} `\n"
        f"  --outdir {umax_optimization} `\n"
        f"  --u_grid arange:0:101:1 `\n"
        f"  --target_total_dosage 2500 `\n"
        f"  --target_terminal_pathogen 4e7 `\n"
        f"  --weight_profile custom `\n"
        f"  --w_track 1.0 `\n"
        f"  --w_path 1.0 `\n"
        f"  --w_probiotic 1.0 `\n"
        f"  --w_dose 1.0 `\n"
        f"  --run_umax_optimization_study\n\n"
        f"python .\\figure_audit.py --mode generate_plots --groups umax_optimization --umax_optimization_outdir {umax_optimization}"
    )

    benchmark_artifacts = [
        ExpectedArtifact(
            os.path.join(benchmark_figures, "model_compare_r2.png"),
            MANUSCRIPT_SOURCE_MAPPING["model_compare_r2.png"]["description"],
            manuscript_cmd,
            "benchmark",
        ),
        ExpectedArtifact(
            os.path.join(benchmark_figures, "target_weight_heatmap.png"),
            MANUSCRIPT_SOURCE_MAPPING["target_weight_heatmap.png"]["description"],
            manuscript_cmd,
            "benchmark",
        ),
        ExpectedArtifact(
            os.path.join(benchmark_figures, "prediction_error_heatmap.png"),
            "Validation RMSE heatmap (core models)",
            manuscript_cmd,
            "benchmark",
        ),
        ExpectedArtifact(
            os.path.join(ode_back, "figure", "ode_back_outcome_heatmap.png"),
            "Fig. 3D — ODE-back functional outcome R² heatmap",
            ode_back_cmd,
            "ode_back_validation",
            optional=True,
        ),
        ExpectedArtifact(
            os.path.join(ode_back, ODE_BACK_SUMMARY_CSV),
            "ODE-back functional validation summary by model",
            ode_back_cmd,
            "ode_back_validation",
            optional=True,
        ),
        ExpectedArtifact(
            os.path.join(benchmark, "model_compare_summary.csv"),
            "Aggregated model metrics (+ CI when n_repeats>1)",
            manuscript_cmd,
            "benchmark",
        ),
        ExpectedArtifact(
            os.path.join(benchmark, "model_compare_per_target.csv"),
            "Per-target metrics per repeat",
            manuscript_cmd,
            "benchmark",
        ),
        ExpectedArtifact(
            os.path.join(benchmark, "repeated_parameter_metrics.csv"),
            "One row per repeat × model",
            manuscript_cmd,
            "benchmark",
        ),
        ExpectedArtifact(
            os.path.join(benchmark, "parameter_pairwise_significance.csv"),
            "Paired significance (TAR vs baselines)",
            manuscript_cmd,
            "benchmark",
        ),
        ExpectedArtifact(
            os.path.join(benchmark, "tree_expert_table.csv"),
            "Single-tree expert bank (not main bar plot)",
            manuscript_cmd,
            "benchmark",
        ),
        ExpectedArtifact(
            os.path.join(benchmark, "target_weight_table.csv"),
            "TAR stack weights",
            manuscript_cmd,
            "benchmark",
        ),
        ExpectedArtifact(
            os.path.join(benchmark, "model_compare_manifest.json"),
            "Benchmark run manifest",
            manuscript_cmd,
            "benchmark",
        ),
        ExpectedArtifact(
            os.path.join(benchmark, "predictions_manifest.json"),
            "Prediction file index",
            manuscript_cmd,
            "benchmark",
        ),
    ]

    return [
        ExpectedArtifact(
            os.path.join(ode, "representative_paper_figure.png"),
            "Representative forward ODE trajectory (Fig. 3 style)",
            ode_cmd,
            "ode",
        ),
        ExpectedArtifact(
            os.path.join(heatmap, "heatmap_pearson_input.png"),
            "Pearson heatmap (input features)",
            heatmap_input_cmd,
            "heatmap",
        ),
        ExpectedArtifact(
            os.path.join(heatmap, "heatmap_spearman_input.png"),
            "Spearman heatmap (input features)",
            heatmap_input_cmd,
            "heatmap",
        ),
        ExpectedArtifact(
            os.path.join(heatmap, "feature_controller_pearson.png"),
            "Pearson heatmap (features + Tthr targets)",
            heatmap_fc_cmd,
            "supplementary",
        ),
        ExpectedArtifact(
            os.path.join(heatmap, "feature_controller_spearman.png"),
            "Spearman heatmap (features + Tthr targets)",
            heatmap_fc_cmd,
            "supplementary",
        ),
        *benchmark_artifacts,
        ExpectedArtifact(
            os.path.join(fixed_umax_figures, "fixed_umax_representative.png"),
            MANUSCRIPT_SOURCE_MAPPING["fixed_umax_representative.png"]["description"],
            fixed_umax_cmd,
            "fixed_umax_validation",
            optional=True,
        ),
        ExpectedArtifact(
            os.path.join(fixed_umax_figures, "fixed_umax_summary.png"),
            "Fig. 4B — fixed-Umax forward ODE summary (point estimates per model)",
            fixed_umax_cmd,
            "fixed_umax_validation",
            optional=True,
        ),
        ExpectedArtifact(
            os.path.join(fixed_umax_figures, "fixed_umax_constraint_success.png"),
            "Fig. 4C (optional) — constraint success rates",
            fixed_umax_cmd,
            "fixed_umax_validation",
            optional=True,
        ),
        ExpectedArtifact(
            os.path.join(fixed_umax, FIXED_UMAX_SUMMARY_CSV),
            "Fixed-Umax validation summary by model",
            fixed_umax_cmd,
            "fixed_umax_validation",
            optional=True,
        ),
        ExpectedArtifact(
            os.path.join(fixed_umax, FIG4_MANIFEST_JSON),
            "Fig. 4 run manifest",
            fixed_umax_cmd,
            "fixed_umax_validation",
            optional=True,
        ),
        ExpectedArtifact(
            os.path.join(umax_optimization_figures, "umax_score_landscape.png"),
            MANUSCRIPT_SOURCE_MAPPING["umax_score_landscape.png"]["description"],
            umax_optimization_cmd,
            "umax_optimization",
            optional=True,
        ),
        ExpectedArtifact(
            os.path.join(umax_optimization_figures, "umax_constraint_feasibility.png"),
            MANUSCRIPT_SOURCE_MAPPING["umax_constraint_feasibility.png"]["description"],
            umax_optimization_cmd,
            "umax_optimization",
            optional=True,
        ),
        ExpectedArtifact(
            os.path.join(umax_optimization_figures, "umax_ode_ablation.png"),
            "Fig. 5C — illustrative ODE ablation trajectories",
            umax_optimization_cmd,
            "umax_optimization",
            optional=True,
        ),
        ExpectedArtifact(
            os.path.join(umax_optimization_figures, "umax_summary_ablation.png"),
            "Fig. 5D — repeated ablation summary (mean ± 95% CI)",
            umax_optimization_cmd,
            "umax_optimization",
            optional=True,
        ),
        ExpectedArtifact(
            os.path.join(umax_optimization, UMAX_OPTIMIZATION_MANIFEST_JSON),
            "Umax optimization study run manifest",
            umax_optimization_cmd,
            "umax_optimization",
            optional=True,
        ),
    ]


def audit_figures(
    results_root: str,
    x_csv: str,
    y_csv: str,
    metadata_csv: Optional[str],
    groups: Optional[List[str]] = None,
    enable_et_srl: bool = False,
) -> int:
    artifacts = build_expected_artifacts(
        results_root, x_csv, y_csv, metadata_csv, enable_et_srl=enable_et_srl
    )
    if groups:
        group_set = {g.lower() for g in groups}
        artifacts = [a for a in artifacts if a.figure_group.lower() in group_set]

    missing: List[ExpectedArtifact] = []
    present: List[str] = []
    for artifact in artifacts:
        if os.path.exists(artifact.rel_path):
            present.append(artifact.rel_path)
        elif not artifact.optional:
            missing.append(artifact)

    print(f"Figure audit root: {os.path.abspath(results_root)}")
    print(f"Checked {len(artifacts)} artifacts — {len(present)} present, {len(missing)} missing.\n")

    if present:
        print("Present:")
        for path in present:
            print(f"  [ok] {path}")
        print()

    manifest_path = os.path.join(_benchmark_dir(results_root), "model_compare_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        if not manifest.get("enable_et_srl", False):
            print("Note: ET-SRL was not enabled; ET-SRL is not required on the main plot.")
        elif not manifest.get("et_srl_valid", True):
            print(
                "Note: ET-SRL marked invalid in manifest — excluded from main plot. "
                f"Reason: {manifest.get('et_srl_invalid_reason', '(see manifest)')}"
            )
        elif manifest.get("et_srl_equals_et", False):
            print("Note: ET-SRL residual gate all disabled — ET-SRL equals ET anchor.")

    if not missing:
        print("All required figure artifacts are present.")
        return 0

    print("Missing:")
    unique_commands: dict[str, str] = {}
    for artifact in missing:
        print(f"  [missing] {artifact.rel_path}")
        print(f"            {artifact.description}")
        unique_commands[artifact.suggest_command] = artifact.suggest_command

    print("\nSuggested commands (deduplicated):")
    for idx, cmd in enumerate(unique_commands.values(), start=1):
        print(f"\n--- Command {idx} ---")
        print(cmd)

    report = {
        "results_root": os.path.abspath(results_root),
        "n_checked": len(artifacts),
        "n_present": len(present),
        "n_missing": len(missing),
        "present": present,
        "missing": [
            {
                "path": a.rel_path,
                "description": a.description,
                "figure_group": a.figure_group,
                "suggest_command": a.suggest_command,
            }
            for a in missing
        ],
    }
    report_path = os.path.join(results_root, "figure_audit_report.json")
    os.makedirs(results_root, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote audit report to {report_path}")
    return 1


def plot_training_parameter_screening_marginals(screening_root: str, outpath: str) -> None:
    """Marginal inner-CV R² vs each swept hyperparameter dimension (TAR / RF)."""
    train_path = os.path.join(screening_root, "training", "training_screening_results.csv")
    if not os.path.isfile(train_path):
        return
    tdf = pd.read_csv(train_path)
    metric = "mean_R2_original"
    if metric not in tdf.columns:
        return
    apply_matplotlib_style()
    tar = tdf[tdf["model"] == "TAR"].copy()
    rf = tdf[tdf["model"] == "RandomForest"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    if not tar.empty:
        tar["tree_bank"] = tar["expanded_tree_bank"].map({True: "expanded", False: "standard"})
        for ax, xcol, xlabel in (
            (axes[0, 0], "max_tree_experts", "max_tree_experts"),
            (axes[0, 1], "tree_bank", "tree_bank"),
        ):
            groups = sorted(tar[xcol].dropna().unique(), key=lambda v: (str(type(v)), v))
            vals = [tar.loc[tar[xcol] == g, metric].astype(float).values for g in groups]
            bp = ax.boxplot(vals, patch_artist=True)
            ax.set_xticks(range(1, len(groups) + 1))
            ax.set_xticklabels([str(g) for g in groups])
            for patch in bp["boxes"]:
                patch.set_facecolor(PALETTE_BLUE_LIGHT)
                patch.set_alpha(0.85)
            ax.scatter(
                np.repeat(np.arange(1, len(groups) + 1), [len(v) for v in vals]),
                np.concatenate(vals) if vals else [],
                color=PALETTE_BLUE_MID,
                s=28,
                alpha=0.9,
                zorder=3,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Inner CV mean $R^2$")
            ax.set_title(f"TAR — {xlabel}")
            ax.grid(axis="y", linestyle="--", alpha=0.35)
    else:
        for ax in axes[0]:
            ax.set_visible(False)

    if not rf.empty:
        for ax, xcol, xlabel in (
            (axes[1, 0], "n_estimators", "n_estimators"),
            (axes[1, 1], "min_samples_leaf", "min_samples_leaf"),
        ):
            groups = sorted(rf[xcol].dropna().unique())
            vals = [rf.loc[rf[xcol] == g, metric].astype(float).values for g in groups]
            bp = ax.boxplot(vals, patch_artist=True)
            ax.set_xticks(range(1, len(groups) + 1))
            ax.set_xticklabels([str(int(g)) for g in groups])
            for patch in bp["boxes"]:
                patch.set_facecolor(PALETTE_BLUE_LIGHT)
                patch.set_alpha(0.85)
            ax.scatter(
                np.repeat(np.arange(1, len(groups) + 1), [len(v) for v in vals]),
                np.concatenate(vals) if vals else [],
                color=PALETTE_BLUE_MID,
                s=28,
                alpha=0.9,
                zorder=3,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Inner CV mean $R^2$")
            ax.set_title(f"RandomForest — {xlabel}")
            ax.grid(axis="y", linestyle="--", alpha=0.35)
    else:
        for ax in axes[1]:
            ax.set_visible(False)

    fig.suptitle("Training screening — marginal hyperparameter distributions (inner CV)", fontsize=13)
    fig.tight_layout()
    save_figure(fig, outpath)
    plt.close(fig)


def plot_umax_weight_profile_case_distributions(
    case_df: pd.DataFrame,
    outpath: str,
    *,
    main_profile: str = "balanced",
) -> None:
    """Per-profile case-level distributions for screened Umax weight profiles."""
    apply_matplotlib_style()
    if case_df.empty:
        return
    profiles = [p for p in case_df["weight_profile"].dropna().unique()]
    if main_profile in profiles:
        profiles = [main_profile] + [p for p in profiles if p != main_profile]
    conditions = sorted(case_df["ablation_condition"].dropna().unique())
    n_profiles = len(profiles)
    n_conds = max(len(conditions), 1)
    fig, axes = plt.subplots(2, max(n_conds, 1), figsize=(max(4.5 * n_conds, 9), 8), squeeze=False)

    for col_idx, condition in enumerate(conditions):
        sub = case_df[case_df["ablation_condition"] == condition]
        for row_idx, (metric_col, ylabel, title_prefix) in enumerate(
            (
                ("composite_penalty", "Composite penalty", "Optimizer burden"),
                ("total_dosage", "Total dosage (µg/mL)", "Dose burden"),
            )
        ):
            ax = axes[row_idx, col_idx]
            if metric_col not in sub.columns:
                ax.set_visible(False)
                continue
            data = []
            positions = []
            colors = []
            for i, profile in enumerate(profiles):
                vals = sub.loc[sub["weight_profile"] == profile, metric_col].astype(float).dropna().values
                if len(vals) == 0:
                    continue
                data.append(vals)
                positions.append(i)
                colors.append(PALETTE_RED_MID if profile == main_profile else PALETTE_BLUE_MID)
            if not data:
                ax.set_visible(False)
                continue
            bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
            for pos, vals, color in zip(positions, data, colors):
                jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(pos + jitter, vals, color=color, s=10, alpha=0.35, zorder=2)
            ax.set_xticks(range(n_profiles))
            ax.set_xticklabels(profiles, rotation=20, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title_prefix} — {condition}")
            ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Umax weight profile screening — case-level outcome distributions", fontsize=13)
    fig.tight_layout()
    save_figure(fig, outpath)
    plt.close(fig)


def plot_umax_weight_profile_sensitivity(
    sens_df: pd.DataFrame,
    outpath: str,
    *,
    main_profile: str = "balanced",
    profile_weights: Optional[Dict[str, dict]] = None,
) -> None:
    """Grouped screening outcomes per Umax weight profile (development sensitivity only)."""
    apply_matplotlib_style()
    if sens_df.empty:
        return
    profiles = [p for p in sens_df["weight_profile"].dropna().unique()]
    if main_profile in profiles:
        profiles = [main_profile] + [p for p in profiles if p != main_profile]
    conditions = sorted(sens_df["ablation_condition"].dropna().unique())
    n_profiles = len(profiles)
    n_conds = max(len(conditions), 1)
    width = 0.8 / max(n_conds, 1)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 2.2 * n_profiles), 5.2))

    for ax, metric_col, ylabel, title in (
        (axes[0], "mean_composite_penalty", "Mean composite penalty", "Optimizer burden"),
        (axes[1], "mean_total_dosage", "Mean total dosage (µg/mL)", "Dose burden"),
    ):
        if metric_col not in sens_df.columns:
            ax.set_visible(False)
            continue
        x = np.arange(n_profiles)
        for j, condition in enumerate(conditions):
            vals = []
            for profile in profiles:
                sub = sens_df[(sens_df["weight_profile"] == profile) & (sens_df["ablation_condition"] == condition)]
                vals.append(float(sub[metric_col].mean()) if not sub.empty else np.nan)
            offset = (j - (n_conds - 1) / 2.0) * width
            colors = [
                PALETTE_RED_MID if (p == main_profile and j == 0) else PALETTE_BLUE_MID
                for p in profiles
            ]
            ax.bar(x + offset, vals, width=width, label=str(condition), color=colors, alpha=0.92, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(profiles, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(fontsize=9, loc="upper right")
    if profile_weights:
        note_lines = []
        for name in profiles:
            w = profile_weights.get(name, {})
            if w:
                note_lines.append(
                    f"{name}: track={w.get('w_track', 1)}, path={w.get('w_path', 1)}, "
                    f"prob={w.get('w_probiotic', 1)}, dose={w.get('w_dose', 0.25)}"
                )
        if note_lines:
            fig.text(0.01, 0.01, "\n".join(note_lines), fontsize=9, color="#444444", va="bottom")
    fig.suptitle("Umax weight profile sensitivity (development; not used to select main result)", fontsize=13)
    fig.tight_layout(rect=(0, 0.06 if profile_weights else 0, 1, 0.96))
    save_figure(fig, outpath)
    plt.close(fig)


def _load_umax_weight_profiles(screening_root: str) -> Dict[str, dict]:
    plan_path = os.path.join("analysis_plan", "parameter_screening_plan.yaml")
    if os.path.isfile(plan_path):
        try:
            import yaml

            with open(plan_path, encoding="utf-8") as fh:
                plan = yaml.safe_load(fh)
            profiles = dict(plan.get("umax_weights", {}).get("profiles", {}))
            if profiles:
                return profiles
        except Exception:
            pass
    locked_path = os.path.join(screening_root, "locked_final_config.json")
    if os.path.isfile(locked_path):
        with open(locked_path, encoding="utf-8") as fh:
            locked = json.load(fh)
        main = str(locked.get("umax_weight_profile", "balanced"))
        weights = dict(locked.get("umax_weights", {}))
        if weights:
            return {main: weights}
    return {}


def plot_umax_weight_profile_definitions(profiles: Dict[str, dict], outpath: str, *, main_profile: str = "balanced") -> None:
    """Heatmap of screened Umax weight coefficients (w_track, w_path, w_probiotic, w_dose)."""
    apply_matplotlib_style()
    if not profiles:
        return
    names = [main_profile] + [n for n in profiles if n != main_profile]
    weight_keys = ["w_track", "w_path", "w_probiotic", "w_dose"]
    mat = np.array([[float(profiles[n].get(k, np.nan)) for k in weight_keys] for n in names], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, max(3.5, 0.55 * len(names))))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(2.0, float(np.nanmax(mat))))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Weight")
    ax.set_xticks(range(len(weight_keys)))
    ax.set_xticklabels(weight_keys, rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} *" if n == main_profile else n for n in names])
    for i in range(len(names)):
        for j in range(len(weight_keys)):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:g}", ha="center", va="center", fontsize=11, color="black")
    ax.set_title("Screened Umax weight profiles (* = locked for manuscript)")
    fig.tight_layout()
    save_figure(fig, outpath)
    plt.close(fig)


def _tar_screening_label(row: pd.Series) -> str:
    bank = "expanded" if bool(row.get("expanded_tree_bank")) else "standard"
    return f"{bank}, experts={int(row['max_tree_experts'])}"


def _rf_screening_label(row: pd.Series) -> str:
    depth = row.get("max_depth")
    depth_s = "none" if pd.isna(depth) else str(int(depth))
    return f"n{int(row['n_estimators'])}/d{depth_s}/leaf{int(row['min_samples_leaf'])}"


def plot_training_parameter_screening(screening_root: str, outpath: str) -> None:
    """Full inner-CV grid for TAR / RF hyperparameter screening (training split only)."""
    train_path = os.path.join(screening_root, "training", "training_screening_results.csv")
    if not os.path.isfile(train_path):
        return
    tdf = pd.read_csv(train_path)
    metric = "mean_R2_original"
    if metric not in tdf.columns:
        return
    apply_matplotlib_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    tar = tdf[tdf["model"] == "TAR"].copy()
    if not tar.empty:
        tar["label"] = tar.apply(_tar_screening_label, axis=1)
        tar = tar.sort_values(metric, ascending=True)
        best_val = float(tar[metric].max())
        colors = [PALETTE_RED_MID if float(v) >= best_val - 1e-12 else PALETTE_BLUE_MID for v in tar[metric]]
        axes[0].barh(tar["label"], tar[metric], color=colors, edgecolor="white")
        axes[0].set_xlabel("Inner CV mean $R^2$ (original scale)")
        axes[0].set_title("TAR hyperparameter grid")
        axes[0].grid(axis="x", linestyle="--", alpha=0.35)

    rf = tdf[tdf["model"] == "RandomForest"].copy()
    if not rf.empty:
        rf["label"] = rf.apply(_rf_screening_label, axis=1)
        rf = rf.sort_values(metric, ascending=True)
        best_val = float(rf[metric].max())
        colors = [PALETTE_RED_MID if float(v) >= best_val - 1e-12 else PALETTE_BLUE_MID for v in rf[metric]]
        axes[1].barh(rf["label"], rf[metric], color=colors, edgecolor="white")
        axes[1].set_xlabel("Inner CV mean $R^2$ (original scale)")
        axes[1].set_title("RandomForest hyperparameter grid")
        axes[1].grid(axis="x", linestyle="--", alpha=0.35)

    fig.suptitle("Training parameter screening (development only; inner CV on training split)", fontsize=13)
    fig.tight_layout()
    save_figure(fig, outpath)
    plt.close(fig)


def generate_parameter_screening_table(screening_root: str) -> Tuple[str, str]:
    rows: List[dict] = []
    train_man = os.path.join(screening_root, "training", "training_screening_manifest.json")
    if os.path.isfile(train_man):
        with open(train_man, encoding="utf-8") as fh:
            tm = json.load(fh)
        rows.append(
            {
                "parameter_group": "training",
                "screened_values": "TAR/RF grids per plan",
                "selection_basis": str(tm.get("selected_metric", "mean_R2_original")),
                "selected_value": json.dumps(tm.get("selected_config_per_model", {})),
                "used_for_main_manuscript": True,
                "sensitivity_reported": False,
                "leakage_control": "inner_cv_training_only",
            }
        )
    umax_sens = os.path.join(screening_root, "umax_weights", "umax_weight_profile_sensitivity.csv")
    locked_path = os.path.join(screening_root, "locked_final_config.json")
    main_profile = "balanced"
    if os.path.isfile(locked_path):
        with open(locked_path, encoding="utf-8") as fh:
            main_profile = str(json.load(fh).get("umax_weight_profile", "balanced"))
    profiles = _load_umax_weight_profiles(screening_root)
    rows.append(
        {
            "parameter_group": "umax_weights",
            "screened_values": ",".join(profiles.keys()) if profiles else "balanced,efficacy,probiotic_sparing,dose_sparing",
            "selection_basis": "biologically_motivated_balanced",
            "selected_value": main_profile,
            "used_for_main_manuscript": True,
            "sensitivity_reported": os.path.isfile(umax_sens),
            "leakage_control": "sensitivity_not_selection",
        }
    )
    table_df = pd.DataFrame(rows)
    csv_path = os.path.join(screening_root, "parameter_screening_table.csv")
    tex_path = os.path.join(screening_root, "parameter_screening_table.tex")
    table_df.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(table_df.to_latex(index=False, escape=True))
    return csv_path, tex_path


def generate_parameter_screening_artifacts(screening_root: str) -> Dict[str, str]:
    os.makedirs(screening_root, exist_ok=True)
    outputs: Dict[str, str] = {}
    csv_path, tex_path = generate_parameter_screening_table(screening_root)
    outputs["parameter_screening_table.csv"] = csv_path
    outputs["parameter_screening_table.tex"] = tex_path

    train_png = os.path.join(screening_root, "training_parameter_screening.png")
    plot_training_parameter_screening(screening_root, train_png)
    outputs["training_parameter_screening.png"] = train_png

    train_marg_png = os.path.join(screening_root, "training_parameter_screening_marginals.png")
    plot_training_parameter_screening_marginals(screening_root, train_marg_png)
    outputs["training_parameter_screening_marginals.png"] = train_marg_png

    profiles = _load_umax_weight_profiles(screening_root)
    main_profile = "balanced"
    locked_path = os.path.join(screening_root, "locked_final_config.json")
    if os.path.isfile(locked_path):
        with open(locked_path, encoding="utf-8") as fh:
            main_profile = str(json.load(fh).get("umax_weight_profile", "balanced"))
    if profiles:
        def_png = os.path.join(screening_root, "umax_weight_profile_definitions.png")
        plot_umax_weight_profile_definitions(profiles, def_png, main_profile=main_profile)
        outputs["umax_weight_profile_definitions.png"] = def_png

    umax_csv = os.path.join(screening_root, "umax_weights", "umax_weight_profile_sensitivity.csv")
    if os.path.isfile(umax_csv):
        sens_df = pd.read_csv(umax_csv)
        umax_png = os.path.join(screening_root, "umax_weights", "umax_weight_profile_sensitivity.png")
        plot_umax_weight_profile_sensitivity(
            sens_df, umax_png, main_profile=main_profile, profile_weights=profiles or None
        )
        outputs["umax_weight_profile_sensitivity.png"] = umax_png

    umax_case_csv = os.path.join(screening_root, "umax_weights", "umax_weight_profile_case_sensitivity.csv")
    if os.path.isfile(umax_case_csv):
        case_df = pd.read_csv(umax_case_csv)
        case_png = os.path.join(screening_root, "umax_weights", "umax_weight_profile_case_distributions.png")
        plot_umax_weight_profile_case_distributions(case_df, case_png, main_profile=main_profile)
        outputs["umax_weight_profile_case_distributions.png"] = case_png

    return expand_figure_outputs(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and generate benchmark figure artifacts.")
    parser.add_argument(
        "--mode",
        choices=["audit_figures", "generate_plots"],
        default="audit_figures",
    )
    parser.add_argument("--results_root", default="results")
    parser.add_argument(
        "--benchmark_outdir",
        default=None,
        help="Benchmark output directory (default: results_root/tree_srl_benchmark).",
    )
    parser.add_argument(
        "--ode_outdir",
        default=None,
        help="ODE output directory for generate_plots (default: results_root/ode).",
    )
    parser.add_argument(
        "--heatmap_outdir",
        default=None,
        help="Heatmap output directory for generate_plots (default: results_root/heatmap).",
    )
    parser.add_argument(
        "--closed_loop_outdir",
        default=None,
        help="Deprecated alias for --fixed_umax_outdir.",
    )
    parser.add_argument(
        "--fixed_umax_outdir",
        default=None,
        help="Fixed-Umax validation output directory (default: results_root/fixed_umax_validation).",
    )
    parser.add_argument(
        "--umax_optimization_outdir",
        default=None,
        help="Umax optimization output directory for generate_plots (default: results_root/umax_optimization).",
    )
    parser.add_argument(
        "--ode_back_outdir",
        default=None,
        help="ODE-back validation output directory (default: results_root/ode_back_validation).",
    )
    from microbio_dataset import resolve_formal_dataset_paths

    formal_paths = resolve_formal_dataset_paths()
    parser.add_argument("--x_csv", default=formal_paths["x_csv"])
    parser.add_argument("--y_csv", default=formal_paths["y_csv"])
    parser.add_argument("--metadata_csv", default=formal_paths["metadata_csv"])
    parser.add_argument(
        "--groups",
        default=None,
        help="Comma-separated groups for audit or generate_plots: ode,heatmap,benchmark,ode_back_validation,fixed_umax_validation,fig4,umax_optimization,fig5,supplementary,all",
    )
    parser.add_argument(
        "--enable_et_srl",
        action="store_true",
        help="If set, notes ET-SRL-specific expectations when reading manifest.",
    )
    args = parser.parse_args()

    groups = [g.strip() for g in args.groups.split(",")] if args.groups else None
    if args.mode == "generate_plots":
        if groups is None and args.benchmark_outdir is not None:
            plot_groups = ["benchmark"]
        elif groups is None:
            plot_groups = ["all"]
        else:
            plot_groups = groups
        outputs = generate_figure_plots(
            groups=plot_groups,
            results_root=args.results_root,
            ode_outdir=args.ode_outdir,
            heatmap_outdir=args.heatmap_outdir,
            benchmark_outdir=args.benchmark_outdir,
            closed_loop_outdir=args.closed_loop_outdir,
            fixed_umax_outdir=args.fixed_umax_outdir,
            umax_optimization_outdir=args.umax_optimization_outdir,
            ode_back_outdir=args.ode_back_outdir,
        )
        png_count = sum(1 for p in outputs.values() if p.endswith(".png"))
        svg_count = sum(1 for p in outputs.values() if p.endswith(".svg"))
        print(f"\nGenerated {png_count} PNG + {svg_count} SVG figure(s) under {os.path.abspath(args.results_root)}")
        for name in sorted(k for k in outputs if k.endswith(".png")):
            print(f"  {name} -> {outputs[name]}")
            svg_name = name[:-4] + ".svg"
            if svg_name in outputs:
                print(f"  {svg_name} -> {outputs[svg_name]}")
        sys.exit(0)
    if args.mode == "audit_figures":
        code = audit_figures(
            args.results_root,
            args.x_csv,
            args.y_csv,
            args.metadata_csv,
            groups=groups,
            enable_et_srl=args.enable_et_srl,
        )
        sys.exit(code)
    sys.exit(2)


if __name__ == "__main__":
    main()
