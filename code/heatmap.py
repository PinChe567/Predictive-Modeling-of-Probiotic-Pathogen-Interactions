from __future__ import annotations

import argparse
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

HEATMAP_EXCLUDE_COLS: Tuple[str, ...] = tuple(
    [f"rho_{i}" for i in range(1, 6)] + [f"mu_{i}" for i in range(1, 6)]
)
BIO_COLS: List[str] = (
    [f"B0_{i}" for i in range(1, 6)]
    + [f"k_{i}" for i in range(1, 6)]
    + [f"g_{i}" for i in range(1, 6)]
)
COLUMN_HUE_GROUPS: Tuple[str, ...] = ("Biological Inputs", "Control Thresholds", "Outcomes")
COLUMN_HUE_COLORS: Dict[str, str] = {
    "Biological Inputs": "#5F97D2",
    "Control Thresholds": "#D76364",
    "Outcomes": "#B1CE46",
}
KPI_COLS: List[str] = ["P_AUC"] + [f"LR{i}" for i in range(1, 6)]
BIO_KPI_COLS: List[str] = BIO_COLS + KPI_COLS
Y_TTHR_COLS: List[str] = [f"Tthr_{i}" for i in range(1, 6)]

SUBSETS_REQUIRING_Y = frozenset({"feature_controller", "controller_only", "all"})

TITLE_INPUT = "Input biological features and desired performance descriptors"
TITLE_FEATURE_CONTROLLER = (
    "Associations between input descriptors and predicted controller targets"
)

try:
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


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


def build_merged_frame(
    x_csv: str,
    y_csv: Optional[str],
    include_physics: bool,
) -> pd.DataFrame:
    x_path = resolve_x_csv(x_csv, include_physics)
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Cannot find X CSV: {x_path}")
    x_df = pd.read_csv(x_path)
    if y_csv is None:
        return x_df
    y_df = load_y_frame(y_csv)
    if len(x_df) != len(y_df):
        raise ValueError(
            f"Row count mismatch: X has {len(x_df)} rows, y has {len(y_df)} rows."
        )
    return pd.concat([x_df.reset_index(drop=True), y_df.reset_index(drop=True)], axis=1)


def column_hue_group(name: str) -> Optional[str]:
    if name.startswith(("B0_", "k_", "g_")):
        return "Biological Inputs"
    if name.startswith("Tthr_"):
        return "Control Thresholds"
    if name == "P_AUC" or name.startswith("LR"):
        return "Outcomes"
    return None


def filter_heatmap_correlation(corr: pd.DataFrame) -> pd.DataFrame:
    """Drop rho/mu parameters from heatmap matrices (plot presentation only)."""
    drop = [c for c in HEATMAP_EXCLUDE_COLS if c in corr.columns or c in corr.index]
    if not drop:
        return corr
    return corr.drop(index=drop, columns=drop, errors="ignore")


def select_columns(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "input":
        cols = [c for c in df.columns if c not in Y_TTHR_COLS]
    elif subset == "bio_only":
        cols = [c for c in BIO_COLS if c in df.columns]
    elif subset == "bio_kpi":
        cols = [c for c in BIO_KPI_COLS if c in df.columns]
    elif subset == "feature_controller":
        cols = [c for c in BIO_KPI_COLS if c in df.columns] + [
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
    cols = [c for c in cols if c not in HEATMAP_EXCLUDE_COLS]
    return df[cols]


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
    plot_corr = filter_heatmap_correlation(corr)
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
        width_ratios=[0.05, 1.0, 0.04],
        height_ratios=[0.05, 1.0],
        wspace=0.20,
        hspace=0.03,
        top=0.80,
        bottom=0.10,
        left=0.11,
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
    tick_fs = max(9, min(13, int(260 / max(n, 1))))
    ax_main.set_xticks(range(n))
    ax_main.set_xticklabels(plot_corr.columns, rotation=90, fontsize=tick_fs)
    ax_main.set_yticks(range(n))
    ax_main.set_yticklabels(plot_corr.columns, fontsize=tick_fs)
    ax_main.tick_params(axis="both", labelsize=tick_fs, pad=2)
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

    finalize_figure_layout(fig, rect=(0.0, 0.06, 1.0, 0.88))
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


def main() -> None:
    from microbio_dataset import resolve_formal_dataset_paths

    default_paths = resolve_formal_dataset_paths()
    parser = argparse.ArgumentParser(
        description="Generate Pearson/Spearman correlation heatmaps for input and controller targets."
    )
    parser.add_argument("--x_csv", default=default_paths["x_csv"])
    parser.add_argument("--y_csv", default=None)
    parser.add_argument("--metadata_csv", default=None)
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
    parser.add_argument("--figsize", type=parse_figsize, default=(12.0, 10.0))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--clustered",
        action="store_true",
        help="Hierarchically cluster rows/columns when scipy is available.",
    )
    args = parser.parse_args()

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
    merged = build_merged_frame(
        args.x_csv,
        args.y_csv if use_y else None,
        args.include_physics,
    )
    selected = select_columns(merged, args.subset)
    title = subset_title(args.subset)

    os.makedirs(args.outdir, exist_ok=True)
    outputs: Dict[str, Dict[str, str]] = {}

    excluded = [c for c in HEATMAP_EXCLUDE_COLS if c in merged.columns]
    for corr_method in methods_to_run(args.method):
        corr = filter_heatmap_correlation(compute_correlation(selected, corr_method))
        basename = output_basename(args.output_prefix, args.subset, corr_method)
        png_path = os.path.join(args.outdir, f"{basename}.png")
        csv_path = os.path.join(args.outdir, f"{basename}_correlation_matrix.csv")
        png_path, svg_path = plot_heatmap(corr, png_path, title, args.figsize, args.dpi, args.clustered)
        corr.to_csv(csv_path)
        outputs[corr_method] = {"png": png_path, "svg": svg_path, "csv": csv_path}
        print(f"Saved {png_path}")
        print(f"Saved {svg_path}")
        print(f"Saved {csv_path}")

    config_path = os.path.join(args.outdir, "heatmap_config.json")
    prior: dict = {}
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as fh:
            prior = json.load(fh)
    merged_outputs = dict(prior.get("outputs") or {})
    merged_outputs.update(outputs)
    config = {
        "x_csv": args.x_csv,
        "y_csv": args.y_csv,
        "metadata_csv": args.metadata_csv,
        "outdir": args.outdir,
        "output_prefix": args.output_prefix,
        "method": args.method,
        "include_y": args.include_y,
        "include_physics": args.include_physics,
        "subset": args.subset,
        "figsize": list(args.figsize),
        "dpi": args.dpi,
        "clustered": args.clustered,
        "scipy_available": _SCIPY_AVAILABLE,
        "title": title,
        "columns": list(selected.columns),
        "excluded_columns": excluded,
        "column_hue_groups": list(COLUMN_HUE_GROUPS),
        "n_rows": int(len(selected)),
        "outputs": merged_outputs,
    }
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"Saved {config_path}")


if __name__ == "__main__":
    main()
