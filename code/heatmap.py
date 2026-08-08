from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from figure_audit import (
    FS_BASE,
    FS_LEGEND,
    FS_TICK,
    FS_TITLE,
    apply_matplotlib_style,
    diverging_heatmap_kwargs,
    finalize_figure_layout,
    save_figure,
)

LR_COLS: List[str] = [f"LR{i}" for i in range(1, 6)]
DESIRED_LR_META_COLS: List[str] = [f"desired_LR{i}" for i in range(1, 6)]
DISPLAY_P_AUC = "P_AUC"
DISPLAY_MEAN_LR = "mean_LR"

BIO_COLS: List[str] = (
    [f"B0_{i}" for i in range(1, 6)]
    + [f"k_{i}" for i in range(1, 6)]
    + [f"g_{i}" for i in range(1, 6)]
    + [f"rho_{i}" for i in range(1, 6)]
    + [f"mu_{i}" for i in range(1, 6)]
)
# Formal biological inputs retained in correlation-matrix CSVs; omitted from default PNG/SVG panel.
PLOT_PANEL_EXCLUDE_COLS: Tuple[str, ...] = tuple(
    [f"rho_{i}" for i in range(1, 6)] + [f"mu_{i}" for i in range(1, 6)]
)
KPI_DISPLAY_COLS: List[str] = [DISPLAY_P_AUC, DISPLAY_MEAN_LR]
BIO_KPI_DISPLAY_COLS: List[str] = BIO_COLS + KPI_DISPLAY_COLS
Y_TTHR_COLS: List[str] = [f"Tthr_{i}" for i in range(1, 6)]

COLUMN_HUE_GROUPS: Tuple[str, ...] = ("Biological Inputs", "Control Thresholds", "Outcomes")
COLUMN_HUE_COLORS: Dict[str, str] = {
    "Biological Inputs": "#5F97D2",
    "Control Thresholds": "#D76364",
    "Outcomes": "#B1CE46",
}

SUBSETS_REQUIRING_Y = frozenset({"feature_controller", "controller_only", "all"})

TITLE_INPUT = "Input biological features and desired performance descriptors"
TITLE_FEATURE_CONTROLLER = (
    "Associations between input descriptors and predicted controller targets"
)
# Design note retained in audit only; not drawn on figures.
COLLAPSE_LR_DESIGN_NOTE = (
    "The five desired-LR schema columns were identical by construction and are "
    "displayed once as mean_LR. This duplication is a relabeling design choice, "
    "not an ODE-derived correlation."
)

EQUALITY_ATOL = 1e-12

try:
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_x_csv(x_csv: str, include_physics: bool) -> str:
    if not include_physics:
        return x_csv
    directory = os.path.dirname(os.path.abspath(x_csv))
    physics_path = os.path.join(directory, "X_features_physics.csv")
    if os.path.exists(physics_path):
        print(f"Using physics features: {physics_path}")
        return physics_path
    if "physics" in os.path.basename(x_csv).lower():
        return x_csv
    raise FileNotFoundError(
        f"--include_physics was set but {physics_path} was not found."
    )


def resolve_metadata_csv(x_csv: str, metadata_csv: Optional[str]) -> str:
    if metadata_csv:
        return metadata_csv
    directory = os.path.dirname(os.path.abspath(x_csv))
    candidate = os.path.join(directory, "sample_metadata.csv")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        "sample_metadata.csv is required. Pass --metadata_csv or place it next to X_features.csv."
    )


def load_y_frame(y_csv: str) -> pd.DataFrame:
    y_df = pd.read_csv(y_csv)
    if list(y_df.columns) != Y_TTHR_COLS and not set(Y_TTHR_COLS).issubset(y_df.columns):
        warnings.warn(
            "y_targets.csv columns are not exactly Tthr_1..Tthr_5; "
            "using available Tthr columns only (u_max is not expected in Tthr-only datasets).",
            stacklevel=2,
        )
    tthr_cols = [col for col in Y_TTHR_COLS if col in y_df.columns]
    if not tthr_cols:
        raise ValueError(
            f"No Tthr_* columns found in {y_csv}. "
            "Expected Tthr-only controller targets (Tthr_1..Tthr_5)."
        )
    return y_df[tthr_cols]


def _require_columns(df: pd.DataFrame, cols: Sequence[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def pairwise_max_abs_diff(frame: pd.DataFrame, cols: Sequence[str]) -> Dict[str, float]:
    values = frame.loc[:, list(cols)].to_numpy(dtype=float)
    out: Dict[str, float] = {}
    for i, left in enumerate(cols):
        for j in range(i + 1, len(cols)):
            right = cols[j]
            diff = np.abs(values[:, i] - values[:, j])
            out[f"{left}_vs_{right}"] = float(np.nanmax(diff)) if diff.size else 0.0
    return out


def validate_row_alignment(x_df: pd.DataFrame, meta_df: pd.DataFrame) -> Dict[str, object]:
    n_x = len(x_df)
    n_meta = len(meta_df)
    if n_x != n_meta:
        raise ValueError(
            f"Row count mismatch: X_features has {n_x} rows, sample_metadata has {n_meta} rows."
        )
    if "row_id" not in meta_df.columns:
        raise ValueError("sample_metadata.csv must contain row_id for alignment checks.")

    meta_row_id = meta_df["row_id"].to_numpy()
    if "row_id" in x_df.columns:
        x_row_id = x_df["row_id"].to_numpy()
        if not np.array_equal(x_row_id, meta_row_id):
            bad = np.where(x_row_id != meta_row_id)[0][:20]
            raise ValueError(
                "X_features.row_id and sample_metadata.row_id are not aligned. "
                f"First mismatched positional indices: {bad.tolist()}"
            )
        aligned_ids = x_row_id
        alignment_mode = "shared_row_id"
    else:
        expected = np.arange(n_x)
        if not np.array_equal(meta_row_id, expected):
            bad = np.where(meta_row_id != expected)[0][:20]
            raise ValueError(
                "sample_metadata.row_id must match positional index 0..n_rows-1 when "
                f"X_features has no row_id. First mismatched indices: {bad.tolist()}"
            )
        aligned_ids = meta_row_id
        alignment_mode = "positional_with_metadata_row_id"

    return {
        "n_rows": int(n_x),
        "alignment_mode": alignment_mode,
        "row_id_min": int(np.min(aligned_ids)) if len(aligned_ids) else None,
        "row_id_max": int(np.max(aligned_ids)) if len(aligned_ids) else None,
    }


def validate_x_vs_metadata_desired_lr(
    x_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> Dict[str, object]:
    _require_columns(x_df, LR_COLS, "X_features.csv")
    _require_columns(meta_df, DESIRED_LR_META_COLS + ["desired_P_AUC"], "sample_metadata.csv")

    per_column: Dict[str, object] = {}
    all_ok = True
    for x_col, meta_col in zip(LR_COLS, DESIRED_LR_META_COLS):
        x_vals = x_df[x_col].to_numpy(dtype=float)
        m_vals = meta_df[meta_col].to_numpy(dtype=float)
        ok = bool(np.allclose(x_vals, m_vals, equal_nan=True, rtol=0.0, atol=EQUALITY_ATOL))
        max_abs = float(np.nanmax(np.abs(x_vals - m_vals))) if len(x_vals) else 0.0
        per_column[f"{x_col}_vs_{meta_col}"] = {
            "allclose": ok,
            "max_abs_diff": max_abs,
            "rtol": 0.0,
            "atol": EQUALITY_ATOL,
            "equal_nan": True,
        }
        all_ok = all_ok and ok

    if "P_AUC" in x_df.columns:
        x_pauc = x_df["P_AUC"].to_numpy(dtype=float)
        m_pauc = meta_df["desired_P_AUC"].to_numpy(dtype=float)
        pauc_ok = bool(
            np.allclose(x_pauc, m_pauc, equal_nan=True, rtol=0.0, atol=EQUALITY_ATOL)
        )
        pauc_max = float(np.nanmax(np.abs(x_pauc - m_pauc))) if len(x_pauc) else 0.0
        per_column["P_AUC_vs_desired_P_AUC"] = {
            "allclose": pauc_ok,
            "max_abs_diff": pauc_max,
            "rtol": 0.0,
            "atol": EQUALITY_ATOL,
            "equal_nan": True,
        }
        all_ok = all_ok and pauc_ok

    if not all_ok:
        failing = [k for k, v in per_column.items() if not v["allclose"]]
        raise ValueError(
            "X_features desired descriptors do not match sample_metadata "
            f"(atol={EQUALITY_ATOL}). Failing comparisons: {failing}"
        )
    return {"passed": True, "per_column": per_column}


def assert_identical_desired_lr_rows(
    x_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> Dict[str, object]:
    """Require LR1=...=LR5 on every row. Never auto-merge on failure."""
    _require_columns(x_df, LR_COLS, "X_features.csv")
    lr = x_df.loc[:, LR_COLS].to_numpy(dtype=float)
    row_max = np.nanmax(lr, axis=1)
    row_min = np.nanmin(lr, axis=1)
    row_range = row_max - row_min
    inconsistent = np.where(~(row_range <= EQUALITY_ATOL) | ~np.isfinite(row_range))[0]

    if inconsistent.size:
        if "row_id" in meta_df.columns:
            bad_ids = meta_df.iloc[inconsistent]["row_id"].tolist()
        elif "row_id" in x_df.columns:
            bad_ids = x_df.iloc[inconsistent]["row_id"].tolist()
        else:
            bad_ids = inconsistent.tolist()
        preview = bad_ids[:50]
        raise ValueError(
            "Desired LR columns are not identical within rows; refusing to merge. "
            f"n_inconsistent={len(bad_ids)}; row_id preview={preview}"
        )

    pairwise = pairwise_max_abs_diff(x_df, LR_COLS)
    return {
        "passed": True,
        "tolerance": EQUALITY_ATOL,
        "max_abs_row_range": float(np.nanmax(row_range)) if len(row_range) else 0.0,
        "n_rows_checked": int(len(x_df)),
        "n_rows_inconsistent": 0,
        "columns_checked": list(LR_COLS),
        "pairwise_max_abs_diff": pairwise,
    }


def build_display_frame(
    x_df: pd.DataFrame,
    y_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Plot-only frame: collapse identical LRs to mean_LR. Does not write source CSVs."""
    _require_columns(x_df, LR_COLS + ["P_AUC"], "X_features.csv")
    display = x_df.drop(columns=LR_COLS).copy()
    display[DISPLAY_P_AUC] = x_df["P_AUC"].to_numpy(dtype=float)
    display[DISPLAY_MEAN_LR] = x_df.loc[:, LR_COLS].mean(axis=1).to_numpy(dtype=float)
    if DISPLAY_P_AUC != "P_AUC" and "P_AUC" in display.columns:
        display = display.drop(columns=["P_AUC"])
    if y_df is not None:
        if len(display) != len(y_df):
            raise ValueError(
                f"Row count mismatch: display frame has {len(display)} rows, "
                f"y has {len(y_df)} rows."
            )
        display = pd.concat(
            [display.reset_index(drop=True), y_df.reset_index(drop=True)],
            axis=1,
        )
    return display


def column_hue_group(name: str) -> Optional[str]:
    if name.startswith(("B0_", "k_", "g_", "rho_", "mu_")):
        return "Biological Inputs"
    if name.startswith("Tthr_"):
        return "Control Thresholds"
    if name in {DISPLAY_P_AUC, DISPLAY_MEAN_LR} or name.startswith("LR"):
        return "Outcomes"
    return None


def select_columns(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "input":
        cols = [c for c in df.columns if c not in Y_TTHR_COLS]
    elif subset == "bio_only":
        cols = [c for c in BIO_COLS if c in df.columns]
    elif subset == "bio_kpi":
        cols = [c for c in BIO_KPI_DISPLAY_COLS if c in df.columns]
    elif subset == "feature_controller":
        cols = [c for c in BIO_KPI_DISPLAY_COLS if c in df.columns] + [
            c for c in Y_TTHR_COLS if c in df.columns
        ]
    elif subset == "controller_only":
        cols = [c for c in Y_TTHR_COLS if c in df.columns]
    elif subset == "all":
        cols = list(df.columns)
    else:
        raise ValueError(f"Unknown subset: {subset}")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Subset '{subset}' requires missing columns: {missing}")
    if not cols:
        raise ValueError(f"Subset '{subset}' selected zero columns.")
    # rho_* / mu_* stay in the correlation frame; only the plot panel may omit them.
    return df[cols]


def parse_display_columns(
    raw: Optional[str],
    available: Sequence[str],
    *,
    plot_include_rho_mu: bool = False,
) -> Tuple[List[str], str, List[str]]:
    """Resolve PNG/SVG panel columns. Full correlation-matrix CSVs always keep ``available``."""
    if raw:
        token = raw.strip()
        if token.lower() == "full":
            return list(available), "full_correlation_columns", []
        requested = [c.strip() for c in token.split(",") if c.strip()]
        missing = [c for c in requested if c not in available]
        if missing:
            raise ValueError(
                f"--display_columns contains columns absent from the correlation frame: {missing}"
            )
        omitted = [c for c in available if c not in requested]
        return requested, "documented_panel_subset", omitted

    if plot_include_rho_mu:
        return list(available), "full_correlation_columns", []

    displayed = [c for c in available if c not in PLOT_PANEL_EXCLUDE_COLS]
    omitted = [c for c in available if c in PLOT_PANEL_EXCLUDE_COLS]
    return displayed, "documented_panel_without_rho_mu", omitted


def subset_title(subset: str) -> str:
    if subset in {"feature_controller", "controller_only", "all"}:
        return TITLE_FEATURE_CONTROLLER
    return TITLE_INPUT


def output_basename(output_prefix: str, subset: str, method: str) -> str:
    if subset in {"input", "bio_only", "bio_kpi"}:
        base = f"heatmap_{method}_{subset}"
    else:
        base = f"{subset}_{method}"
    if output_prefix:
        return f"{output_prefix}_{base}"
    return base


def cluster_order(corr: pd.DataFrame) -> List[str]:
    if not _SCIPY_AVAILABLE or corr.shape[0] < 2:
        return list(corr.columns)
    dist = 1.0 - corr.abs()
    np.fill_diagonal(dist.values, 0.0)
    condensed = squareform(dist.values, checks=False)
    linkage = hierarchy.linkage(condensed, method="average")
    order_idx = hierarchy.leaves_list(linkage)
    return [corr.columns[i] for i in order_idx]


def compute_correlation(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "pearson":
        return df.corr(method="pearson")
    if method == "spearman":
        return df.corr(method="spearman")
    raise ValueError(f"Unknown correlation method: {method}")


def _draw_hue_strip(ax, labels: Sequence[str], *, horizontal: bool) -> None:
    n = len(labels)
    if n == 0:
        ax.set_visible(False)
        return
    if horizontal:
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 1)
        for i, label in enumerate(labels):
            group = column_hue_group(str(label))
            if group is None:
                continue
            ax.add_patch(
                Rectangle(
                    (i - 0.5, 0),
                    1,
                    1,
                    facecolor=COLUMN_HUE_COLORS[group],
                    edgecolor="none",
                )
            )
    else:
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xlim(0, 1)
        for i, label in enumerate(labels):
            group = column_hue_group(str(label))
            if group is None:
                continue
            ax.add_patch(
                Rectangle(
                    (0, i - 0.5),
                    1,
                    1,
                    facecolor=COLUMN_HUE_COLORS[group],
                    edgecolor="none",
                )
            )
    ax.axis("off")


def plot_heatmap(
    corr: pd.DataFrame,
    outpath: str,
    title: str,
    figsize: Tuple[float, float],
    dpi: int,
    clustered: bool,
) -> Tuple[str, str]:
    plot_corr = corr
    if clustered:
        if not _SCIPY_AVAILABLE:
            print("scipy not available; skipping hierarchical clustering.")
        else:
            order = cluster_order(plot_corr)
            plot_corr = plot_corr.loc[order, order]

    apply_matplotlib_style()
    n = plot_corr.shape[0]
    fig_w, fig_h = figsize
    fig = plt.figure(figsize=(fig_w + 0.8, fig_h + 1.15))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[0.045, 1.0, 0.04],
        height_ratios=[0.05, 1.0],
        wspace=0.35,
        hspace=0.03,
        top=0.80,
        bottom=0.10,
        left=0.16,
        right=0.92,
    )
    ax_col_hue = fig.add_subplot(gs[0, 1])
    ax_row_hue = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    im = ax_main.imshow(
        plot_corr.values,
        aspect="auto",
        **diverging_heatmap_kwargs(plot_corr.values, fixed_limits=(-1.0, 1.0)),
    )
    fig.colorbar(im, cax=ax_cbar, label="Correlation")
    tick_fs = max(8, min(13, int(260 / max(n, 1))))
    ax_main.set_xticks(range(n))
    ax_main.set_xticklabels(plot_corr.columns, rotation=90, fontsize=tick_fs)
    ax_main.set_yticks(range(n))
    ax_main.set_yticklabels(plot_corr.columns, fontsize=tick_fs)
    # Keep y labels close to the heatmap so the left hue strip can sit further left.
    ax_main.tick_params(axis="x", labelsize=tick_fs, pad=2)
    ax_main.tick_params(axis="y", labelsize=tick_fs, pad=4)
    ax_cbar.tick_params(labelsize=FS_TICK + 1)
    ax_cbar.set_ylabel("Correlation", fontsize=FS_BASE + 1)

    _draw_hue_strip(ax_col_hue, list(plot_corr.columns), horizontal=True)
    _draw_hue_strip(ax_row_hue, list(plot_corr.index), horizontal=False)

    present_groups = {
        column_hue_group(str(label))
        for label in list(plot_corr.columns) + list(plot_corr.index)
        if column_hue_group(str(label)) is not None
    }
    legend_handles = [
        Patch(facecolor=COLUMN_HUE_COLORS[group], edgecolor="none", label=group)
        for group in COLUMN_HUE_GROUPS
        if group in present_groups
    ]
    fig.suptitle(title, y=0.98, fontsize=FS_TITLE + 1)
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.875),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=FS_LEGEND + 1,
        )

    finalize_figure_layout(fig, rect=(0.02, 0.06, 1.0, 0.88))
    # Place the left hue strip just left of the y-tick labels (mean_LR is the longest).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    main_bbox = ax_main.get_position()
    col_bbox = ax_col_hue.get_position()
    strip_width = max(ax_row_hue.get_position().width, 0.012)
    gap = 0.022
    label_left = main_bbox.x0
    for tick in ax_main.get_yticklabels():
        if not tick.get_visible():
            continue
        tick_bbox = tick.get_window_extent(renderer=renderer).transformed(
            fig.transFigure.inverted()
        )
        label_left = min(label_left, float(tick_bbox.x0))
    strip_x0 = label_left - gap - strip_width
    # If labels leave too little room on the left, shift the main panel right.
    min_left = 0.01
    if strip_x0 < min_left:
        shift = min_left - strip_x0
        ax_main.set_position(
            [main_bbox.x0 + shift, main_bbox.y0, main_bbox.width, main_bbox.height]
        )
        ax_col_hue.set_position(
            [col_bbox.x0 + shift, col_bbox.y0, col_bbox.width, col_bbox.height]
        )
        main_bbox = ax_main.get_position()
        label_left += shift
        strip_x0 = label_left - gap - strip_width
    ax_row_hue.set_position(
        [max(min_left, strip_x0), main_bbox.y0, strip_width, main_bbox.height]
    )
    png_path, svg_path = save_figure(fig, outpath, dpi=dpi)
    plt.close(fig)
    return png_path, svg_path


def parse_figsize(value: str) -> Tuple[float, float]:
    parts = [float(p.strip()) for p in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--figsize must be WIDTH,HEIGHT")
    return parts[0], parts[1]


def methods_to_run(method: str) -> List[str]:
    if method == "both":
        return ["pearson", "spearman"]
    return [method]


def load_validated_formal_inputs(
    x_csv: str,
    metadata_csv: str,
    include_physics: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, Dict[str, object]]:
    x_path = resolve_x_csv(x_csv, include_physics)
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Cannot find X CSV: {x_path}")
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Cannot find metadata CSV: {metadata_csv}")

    x_df = pd.read_csv(x_path)
    meta_df = pd.read_csv(metadata_csv)

    alignment = validate_row_alignment(x_df, meta_df)
    x_vs_meta = validate_x_vs_metadata_desired_lr(x_df, meta_df)
    identical_lr = assert_identical_desired_lr_rows(x_df, meta_df)

    audit = {
        "input_sha256": {
            "x_csv": sha256_file(x_path),
            "metadata_csv": sha256_file(metadata_csv),
        },
        "paths": {
            "x_csv": os.path.abspath(x_path),
            "metadata_csv": os.path.abspath(metadata_csv),
        },
        "n_rows": alignment["n_rows"],
        "row_alignment": alignment,
        "x_vs_metadata_equality": x_vs_meta,
        "desired_lr_within_row_identity": identical_lr,
        "pairwise_max_abs_diff_desired_lr": identical_lr["pairwise_max_abs_diff"],
        "collapse_lr_design_note": COLLAPSE_LR_DESIGN_NOTE,
    }
    return x_df, meta_df, x_path, audit


def main() -> None:
    from microbio_dataset import resolve_formal_dataset_paths

    default_paths = resolve_formal_dataset_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Generate Pearson/Spearman correlation heatmaps for formal biological inputs "
            "and desired performance descriptors (validated against sample_metadata)."
        )
    )
    parser.add_argument("--x_csv", default=default_paths["x_csv"])
    parser.add_argument("--y_csv", default=None)
    parser.add_argument(
        "--metadata_csv",
        default=default_paths.get("metadata_csv"),
        help="sample_metadata.csv (required; defaults beside formal X_features.csv).",
    )
    parser.add_argument("--outdir", default="results/heatmap")
    parser.add_argument(
        "--output_prefix",
        default="",
        help="Optional filename prefix (omit manuscript figure IDs; use pipeline.md for Fig labels).",
    )
    parser.add_argument(
        "--method",
        choices=["pearson", "spearman", "both"],
        default="both",
    )
    parser.add_argument("--include_y", action="store_true")
    parser.add_argument("--include_physics", action="store_true")
    parser.add_argument(
        "--subset",
        choices=["input", "bio_only", "bio_kpi", "feature_controller", "controller_only", "all"],
        default="input",
    )
    parser.add_argument(
        "--display_columns",
        default=None,
        help=(
            "PNG/SVG panel only: comma-separated columns, or 'full'. "
            "Default panel omits rho_*/mu_*. Correlation-matrix CSVs always keep the "
            "full selected column set including rho_* and mu_*."
        ),
    )
    parser.add_argument(
        "--plot_include_rho_mu",
        action="store_true",
        help="Include rho_* and mu_* in the PNG/SVG panel (default: omit from plot only).",
    )
    parser.add_argument("--figsize", type=parse_figsize, default=(12.0, 10.0))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--clustered",
        action="store_true",
        help="Hierarchically cluster rows/columns when scipy is available.",
    )
    args = parser.parse_args()

    metadata_csv = resolve_metadata_csv(args.x_csv, args.metadata_csv)

    if args.include_y and not args.y_csv:
        raise ValueError(
            "--include_y requires --y_csv so controller targets (Tthr_1..Tthr_5) can be merged."
        )
    if args.subset in SUBSETS_REQUIRING_Y and not args.y_csv:
        raise ValueError(
            f"--subset {args.subset} requires --y_csv with Tthr-only controller targets."
        )

    use_y = args.y_csv is not None and (
        args.include_y or args.subset in SUBSETS_REQUIRING_Y
    )

    x_df, _meta_df, x_path, audit = load_validated_formal_inputs(
        args.x_csv,
        metadata_csv,
        args.include_physics,
    )
    y_df = load_y_frame(args.y_csv) if use_y else None
    if y_df is not None:
        audit["input_sha256"]["y_csv"] = sha256_file(args.y_csv)
        audit["paths"]["y_csv"] = os.path.abspath(args.y_csv)

    display_df = build_display_frame(x_df, y_df)
    selected = select_columns(display_df, args.subset)
    correlation_columns = list(selected.columns)
    displayed_columns, display_mode, panel_omitted_columns = parse_display_columns(
        args.display_columns,
        correlation_columns,
        plot_include_rho_mu=args.plot_include_rho_mu,
    )
    title = subset_title(args.subset)

    audit["correlation_columns"] = correlation_columns
    audit["displayed_columns"] = displayed_columns
    audit["display_mode"] = display_mode
    audit["panel_omitted_columns"] = panel_omitted_columns
    audit["panel_omit_reason"] = (
        "Default heatmap PNG/SVG panel omits rho_* and mu_* for readability; "
        "they remain in figure-source and full correlation-matrix CSVs as formal biological inputs."
        if display_mode == "documented_panel_without_rho_mu"
        else None
    )
    audit["subset"] = args.subset
    audit["collapsed_desired_lr"] = {
        "source_columns": list(LR_COLS),
        "display_column": DISPLAY_MEAN_LR,
        "p_auc_display_column": DISPLAY_P_AUC,
        "source_csv_modified": False,
    }

    os.makedirs(args.outdir, exist_ok=True)

    figure_source_name = (
        f"heatmap_figure_source_{args.subset}.csv"
        if not args.output_prefix
        else f"{args.output_prefix}_heatmap_figure_source_{args.subset}.csv"
    )
    figure_source_path = os.path.join(args.outdir, figure_source_name)
    selected.to_csv(figure_source_path, index=False)
    print(f"Saved {figure_source_path}")

    outputs: Dict[str, Dict[str, str]] = {}
    for corr_method in methods_to_run(args.method):
        full_corr = compute_correlation(selected, corr_method)
        basename = output_basename(args.output_prefix, args.subset, corr_method)
        png_path = os.path.join(args.outdir, f"{basename}.png")
        csv_path = os.path.join(args.outdir, f"{basename}_correlation_matrix.csv")
        full_corr.to_csv(csv_path)

        plot_corr = full_corr.loc[displayed_columns, displayed_columns]
        png_path, svg_path = plot_heatmap(
            plot_corr,
            png_path,
            title,
            args.figsize,
            args.dpi,
            args.clustered,
        )
        outputs[corr_method] = {
            "png": png_path,
            "svg": svg_path,
            "correlation_matrix_csv": csv_path,
        }
        print(f"Saved {png_path}")
        print(f"Saved {svg_path}")
        print(f"Saved {csv_path}")

    audit_path = os.path.join(args.outdir, "heatmap_input_audit.json")
    with open(audit_path, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)
    print(f"Saved {audit_path}")

    manifest_path = os.path.join(args.outdir, "heatmap_manifest.json")
    prior: dict = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            prior = json.load(fh)
    outputs_by_subset = dict(prior.get("outputs_by_subset") or {})
    outputs_by_subset[args.subset] = {
        "title": title,
        "figure_source_csv": figure_source_path,
        "correlation_columns": correlation_columns,
        "displayed_columns": displayed_columns,
        "display_mode": display_mode,
        "panel_omitted_columns": panel_omitted_columns,
        "methods": outputs,
    }
    figure_sources = dict(prior.get("figure_source_csvs") or {})
    figure_sources[args.subset] = figure_source_path

    manifest = {
        "x_csv": x_path,
        "y_csv": args.y_csv,
        "metadata_csv": metadata_csv,
        "outdir": args.outdir,
        "output_prefix": args.output_prefix,
        "method": args.method,
        "include_y": bool(use_y),
        "include_physics": args.include_physics,
        "latest_subset": args.subset,
        "figsize": list(args.figsize),
        "dpi": args.dpi,
        "clustered": args.clustered,
        "scipy_available": _SCIPY_AVAILABLE,
        "collapse_lr_design_note": COLLAPSE_LR_DESIGN_NOTE,
        "n_rows": int(len(selected)),
        "audit_json": audit_path,
        "figure_source_csvs": figure_sources,
        "outputs_by_subset": outputs_by_subset,
        # Latest-run convenience mirrors (do not drop other subsets from outputs_by_subset).
        "subset": args.subset,
        "title": title,
        "correlation_columns": correlation_columns,
        "displayed_columns": displayed_columns,
        "display_mode": display_mode,
        "panel_omitted_columns": panel_omitted_columns,
        "figure_source_csv": figure_source_path,
        "outputs": outputs,
        "column_hue_groups": list(COLUMN_HUE_GROUPS),
        "silent_exclusions": [],
        "source_csv_modified": False,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Saved {manifest_path}")

    # Compatibility alias for earlier pipeline consumers.
    config_path = os.path.join(args.outdir, "heatmap_config.json")
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Saved {config_path}")


if __name__ == "__main__":
    main()
