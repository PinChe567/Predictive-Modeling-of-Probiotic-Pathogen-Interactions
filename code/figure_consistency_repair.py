#!/usr/bin/env python
"""Figure-only consistency repair from locked CSV/JSON/SVG/PNG artifacts.

Does not rerun ODE simulation, dataset generation, relabeling, ML training,
100-repeat benchmarks, optimizer sweeps, or statistical tests.
Does not write or modify original CSVs or statistical tables.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

from figure_audit import (  # noqa: E402
    MAIN_COMPARE_ORDER,
    ODE_BACK_BAR_METRIC,
    ODE_BACK_BAR_MODEL_ORDER,
    SIGNIFICANCE_PLOT_SPECS,
    TAR_MODEL,
    plot_model_metric_combined_panel,
    plot_ode_back_r2_barplot,
    plot_umax_constraint_feasibility,
    plot_umax_response_landscape,
    plot_umax_summary_ablation,
    significance_pairs_for_plot,
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return f"unavailable ({exc})"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "not a git repository").strip()
        return f"none ({err})"
    return proc.stdout.strip()


def require(path: str, missing: List[str]) -> Optional[str]:
    if os.path.isfile(path):
        return path
    missing.append(path)
    return None


def parse_svg_path_points(d: str) -> np.ndarray:
    pts: List[Tuple[float, float]] = []
    for match in re.finditer(r"[ML]\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", d):
        pts.append((float(match.group(1)), float(match.group(2))))
    return np.asarray(pts, dtype=float) if pts else np.zeros((0, 2), dtype=float)


def extract_long_lines(svg_text: str, *, stroke: str, stroke_width: str, min_n: int = 50) -> List[np.ndarray]:
    lines: List[np.ndarray] = []
    for match in re.finditer(r'<path d="([^"]+)"([^>]*)/>', svg_text, flags=re.DOTALL):
        d, attrs = match.group(1), match.group(2)
        style_m = re.search(r'style="([^"]*)"', attrs)
        style = style_m.group(1) if style_m else attrs
        if f"stroke: {stroke}" not in style:
            continue
        if f"stroke-width: {stroke_width}" not in style:
            continue
        if "fill: none" not in style:
            continue
        pts = parse_svg_path_points(d)
        if len(pts) >= min_n:
            lines.append(pts)
    return lines


def extract_fill_bounds(svg_text: str, *, fill_opacity: str, fig_height: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    block_re = re.compile(
        r'<g id="FillBetweenPolyCollection_\d+">\s*<defs>\s*<path [^>]*d="([^"]+)"',
        flags=re.DOTALL,
    )
    use_re = re.compile(
        rf'FillBetweenPolyCollection_\d+.*?<use [^>]*y="([^"]+)"[^>]*fill-opacity: {re.escape(fill_opacity)}',
        flags=re.DOTALL,
    )
    blocks = list(block_re.finditer(svg_text))
    uses = list(re.finditer(
        rf'<g id="(FillBetweenPolyCollection_\d+)">.*?fill-opacity: {re.escape(fill_opacity)}',
        svg_text,
        flags=re.DOTALL,
    ))
    if not blocks or not uses:
        return None
    chosen = None
    for use in uses:
        cid = use.group(1)
        inner = re.search(
            rf'<g id="{re.escape(cid)}">\s*<defs>\s*<path [^>]*d="([^"]+)"',
            svg_text,
            flags=re.DOTALL,
        )
        use_y = re.search(
            rf'<g id="{re.escape(cid)}">.*?<use [^>]*y="([^"]+)"',
            svg_text,
            flags=re.DOTALL,
        )
        if inner is None or use_y is None:
            continue
        pts = parse_svg_path_points(inner.group(1))
        if len(pts) < 20:
            continue
        y_off = float(use_y.group(1))
        xs = pts[:, 0]
        turn = int(np.argmax(np.diff(xs) < -1e-6)) + 1 if np.any(np.diff(xs) < -1e-6) else len(pts) // 2
        fwd, back = pts[:turn], pts[turn:]
        if abs(fwd[0, 0] - fwd[1, 0]) < 1e-6:
            fwd = fwd[1:]
        if len(back) and abs(back[-1, 0] - back[-2, 0]) < 1e-6:
            back = back[:-1]
        back = back[::-1]
        n = min(len(fwd), len(back))
        if n < 20:
            continue
        chosen = (fwd[:n, 0], y_off + fwd[:n, 1], y_off + back[:n, 1])
        break
    return chosen


def svg_axes_box(svg_text: str) -> Tuple[float, float, float, float, float]:
    """Return (x_left, x_right_spine, y_bottom, y_top, fig_height) from matplotlib axes patch."""
    vh = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', svg_text)
    fig_h = float(vh.group(2)) if vh else 0.0
    m = re.search(
        r'<g id="patch_2">\s*<path d="M ([0-9.]+) ([0-9.]+) \s*L ([0-9.]+) \2 \s*L \3 ([0-9.]+) \s*L \1 \4',
        svg_text,
    )
    if m:
        x0, y_bottom, x1, y_top = map(float, m.groups())
        return x0, x1, y_bottom, y_top, fig_h
    m = re.search(
        r'<g id="patch_2">\s*<path d="M ([0-9.]+) ([0-9.]+) \s*L ([0-9.]+) \2 \s*L \3 ([0-9.]+)',
        svg_text,
    )
    if not m:
        raise RuntimeError("Could not parse axes box from locked SVG")
    x0, y_bottom, x1, y_top = map(float, m.groups())
    return x0, x1, y_bottom, y_top, fig_h


def invert_svg_y(display_y: np.ndarray, y_bottom: float, y_top: float, ylim: Tuple[float, float]) -> np.ndarray:
    frac = (y_bottom - display_y) / (y_bottom - y_top)
    return ylim[0] + frac * (ylim[1] - ylim[0])


def load_official_u_grid(manifest_path: str) -> np.ndarray:
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    grid = np.asarray(manifest["u_grid"], dtype=float)
    gmin = float(manifest.get("u_grid_min", np.min(grid)))
    gmax = float(manifest.get("u_grid_max", np.max(grid)))
    npts = int(manifest.get("n_u_grid_points", grid.size))
    if grid.size != npts or gmin != 0.0 or gmax != 101.0:
        raise RuntimeError(
            f"Locked u_grid is not 0–101 inclusive: min={gmin} max={gmax} n={grid.size} expected_n={npts}"
        )
    if float(np.min(grid)) != 0.0 or float(np.max(grid)) != 101.0:
        raise RuntimeError(f"u_grid values are not 0–101: {grid.min()}–{grid.max()}")
    return grid


def count_star_comments(svg_path: str) -> int:
    with open(svg_path, encoding="utf-8") as fh:
        text = fh.read()
    return len(re.findall(r"<!-- \*{1,4} -->", text))


def svg_xlim_spine_and_data_max(svg_path: str) -> Tuple[bool, Optional[float]]:
    with open(svg_path, encoding="utf-8") as fh:
        text = fh.read()
    has_101_tick = "<!-- 101 -->" in text
    has_100_tick = "<!-- 100 -->" in text
    return has_101_tick or not has_100_only_clip(text), 101.0 if has_101_tick else None


def has_100_only_clip(text: str) -> bool:
    return "<!-- 100 -->" in text and "<!-- 101 -->" not in text


def main() -> int:
    missing: List[str] = []
    inputs: Dict[str, List[str]] = {
        "fig3b": [
            os.path.join(ROOT, "results", "tree_srl_benchmark", "parameter_pairwise_significance.csv"),
            os.path.join(ROOT, "results", "tree_srl_benchmark", "model_compare_summary.csv"),
            os.path.join(ROOT, "results", "tree_srl_benchmark", "model_compare_per_target.csv"),
            os.path.join(ROOT, "results", "tree_srl_benchmark", "figure", "model_compare_r2.png"),
            os.path.join(ROOT, "results", "tree_srl_benchmark", "figure", "model_compare_r2.svg"),
        ],
        "fig4a": [
            os.path.join(ROOT, "results", "ode_back_validation", "ode_back_pairwise_significance.csv"),
            os.path.join(ROOT, "results", "ode_back_validation", "ode_back_summary_by_model.csv"),
            os.path.join(ROOT, "results", "ode_back_validation", "figure", "ode_back_r2_barplot.png"),
            os.path.join(ROOT, "results", "ode_back_validation", "figure", "ode_back_r2_barplot.svg"),
        ],
        "fig5d": [
            os.path.join(ROOT, "results", "umax_optimization", "umax_ablation_repeated_plot_stats.csv"),
            os.path.join(ROOT, "results", "umax_optimization", "umax_policy_ablation_condition_counts.csv"),
            os.path.join(ROOT, "results", "umax_optimization", "figure", "umax_summary_ablation_composite_supplementary.png"),
            os.path.join(ROOT, "results", "umax_optimization", "figure", "umax_summary_ablation_composite_supplementary.svg"),
        ],
        "fig5bc": [
            os.path.join(ROOT, "results", "umax_optimization", "repeats", "repeat_000", "umax_study", "umax_optimization_manifest.json"),
            os.path.join(ROOT, "results", "umax_optimization", "umax_feasible_region_summary.csv"),
            os.path.join(ROOT, "results", "umax_optimization", "figure", "umax_constraint_feasibility.png"),
            os.path.join(ROOT, "results", "umax_optimization", "figure", "umax_constraint_feasibility.svg"),
            os.path.join(ROOT, "results", "umax_optimization", "figure", "umax_score_landscape.png"),
            os.path.join(ROOT, "results", "umax_optimization", "figure", "umax_score_landscape.svg"),
        ],
    }
    absent_landscape_csv = [
        os.path.join(ROOT, "results", "umax_optimization", "umax_response_landscape.csv"),
        os.path.join(ROOT, "results", "umax_optimization", "umax_optimization_u_candidates.csv"),
        os.path.join(ROOT, "results", "umax_optimization", "umax_score_landscape_curves.csv"),
        os.path.join(ROOT, "results", "umax_optimization", "umax_selected_umax_distribution.csv"),
    ]
    for group, paths in inputs.items():
        for path in paths:
            require(path, missing)
    if missing:
        print("STOP: required locked artifacts are missing:")
        for path in missing:
            print(f"  {path}")
        return 1

    commit = git_commit()
    input_hashes: Dict[str, str] = {}
    for paths in inputs.values():
        for path in paths:
            rel = os.path.relpath(path, ROOT)
            input_hashes[rel] = sha256_file(path)
            print(f"INPUT {rel} {input_hashes[rel]}", flush=True)

    controls = [control for _, control in SIGNIFICANCE_PLOT_SPECS]
    audit_rows: List[str] = []
    output_hashes: Dict[str, str] = {}
    validation: List[str] = []

    # --- Fig. 3b ---
    pairwise_3 = pd.read_csv(inputs["fig3b"][0])
    summary_3 = pd.read_csv(inputs["fig3b"][1])
    per_target_3 = pd.read_csv(inputs["fig3b"][2])
    r2_pairs = significance_pairs_for_plot(
        pairwise_3, TAR_MODEL, controls, metric="mean_R2_original",
        significance_for_manuscript=True,
    )
    rmse_pairs = significance_pairs_for_plot(
        pairwise_3, TAR_MODEL, controls, metric="mean_RMSE_original",
        significance_for_manuscript=True,
    )
    if any("Best" in a or "Best" in b for a, b, _ in r2_pairs + rmse_pairs):
        raise RuntimeError("Fig. 3b still has an oracle BestTree/BestSingleTree pair")
    expected = {(TAR_MODEL, "RandomForest"), (TAR_MODEL, "UniformTreeMean")}
    got = {(a, b) for a, b, _ in r2_pairs}
    if got != expected:
        raise RuntimeError(f"Fig. 3b formal pairs mismatch: {got}")
    fig3b_out = os.path.join(ROOT, "results", "tree_srl_benchmark", "figure", "model_compare_r2.png")
    stars_3_before = count_star_comments(inputs["fig3b"][4])
    plot_model_metric_combined_panel(
        summary_3,
        per_target_3,
        fig3b_out,
        summary_metric_col="mean_R2_original",
        summary_ylabel="Mean target-wise R square",
        summary_title="Mean target-wise R square",
        heatmap_value_col="R2_original",
        heatmap_cbar_label="$R^2$ (original scale)",
        model_order=list(MAIN_COMPARE_ORDER),
        significance_pairs=r2_pairs,
        rmse_significance_pairs=rmse_pairs,
    )
    fig3b_svg = fig3b_out[:-4] + ".svg"
    stars_3_after = count_star_comments(fig3b_svg)
    validation.append(
        f"Fig. 3b pairs={r2_pairs} rmse_pairs={rmse_pairs} "
        f"star_comments {stars_3_before} -> {stars_3_after} (expect 4 = 2 R2 + 2 RMSE)"
    )
    if stars_3_after != 4:
        raise RuntimeError(f"Fig. 3b unexpected star count {stars_3_after}")

    # --- Fig. 4a ---
    pairwise_4 = pd.read_csv(inputs["fig4a"][0])
    summary_4 = pd.read_csv(inputs["fig4a"][1])
    ode_pairs = significance_pairs_for_plot(
        pairwise_4, TAR_MODEL, controls, metric=ODE_BACK_BAR_METRIC,
        significance_for_manuscript=True,
    )
    if any("Best" in a or "Best" in b for a, b, _ in ode_pairs):
        raise RuntimeError("Fig. 4a still has an oracle BestTree/BestSingleTree pair")
    got4 = {(a, b) for a, b, _ in ode_pairs}
    if got4 != expected:
        raise RuntimeError(f"Fig. 4a formal pairs mismatch: {got4}")
    fig4a_out = os.path.join(ROOT, "results", "ode_back_validation", "figure", "ode_back_r2_barplot.png")
    stars_4_before = count_star_comments(inputs["fig4a"][3])
    plot_ode_back_r2_barplot(
        summary_4,
        fig4a_out,
        model_order=list(ODE_BACK_BAR_MODEL_ORDER),
        significance_pairs=ode_pairs,
    )
    fig4a_svg = fig4a_out[:-4] + ".svg"
    stars_4_after = count_star_comments(fig4a_svg)
    validation.append(
        f"Fig. 4a pairs={ode_pairs} star_comments {stars_4_before} -> {stars_4_after} (expect 2)"
    )
    if stars_4_after != 2:
        raise RuntimeError(f"Fig. 4a unexpected star count {stars_4_after}")

    # --- Fig. 5d ---
    stats_5d = pd.read_csv(inputs["fig5d"][0])
    fig5d_out = os.path.join(
        ROOT, "results", "umax_optimization", "figure",
        "umax_summary_ablation_composite_supplementary.png",
    )
    stars_5d_before = count_star_comments(inputs["fig5d"][3])
    plot_umax_summary_ablation(
        stats_5d,
        fig5d_out,
        n_repeats=100,
        significance_pairs=None,
        outdir=os.path.join(ROOT, "results", "umax_optimization"),
        metric_specs=[("mean_composite_score", "Composite penalty score")],
        use_short_labels=True,
        horizontal_bars=True,
    )
    fig5d_svg = fig5d_out[:-4] + ".svg"
    stars_5d_after = count_star_comments(fig5d_svg)
    validation.append(
        f"Fig. 5d star_comments {stars_5d_before} -> {stars_5d_after} (expect 0); "
        "bars+95% CI retained from umax_ablation_repeated_plot_stats.csv"
    )
    if stars_5d_after != 0:
        raise RuntimeError(f"Fig. 5d still has inferential stars ({stars_5d_after})")

    # --- Fig. 5b/c: x from locked u_grid; y from locked SVG (row-level landscape CSV absent) ---
    u_grid = load_official_u_grid(inputs["fig5bc"][0])
    sel = pd.read_csv(
        inputs["fig5bc"][1],
        usecols=["model", "selected_u_max"],
    )
    selected_u = sel.loc[sel["model"].astype(str) == "TAR", "selected_u_max"].to_numpy(dtype=float)
    selected_u = selected_u[np.isfinite(selected_u)]
    if selected_u.size == 0:
        raise RuntimeError("No TAR selected_u_max values in locked umax_feasible_region_summary.csv")

    landscape_svg = inputs["fig5bc"][5]
    constraint_svg = inputs["fig5bc"][3]
    with open(landscape_svg, encoding="utf-8") as fh:
        land_text = fh.read()
    with open(constraint_svg, encoding="utf-8") as fh:
        cons_text = fh.read()

    med_lines = extract_long_lines(land_text, stroke="#5f97d2", stroke_width="2.8")
    if not med_lines:
        raise RuntimeError("Could not extract median-penalty polyline from locked landscape SVG")
    med_pts = max(med_lines, key=len)
    if len(med_pts) != len(u_grid):
        raise RuntimeError(
            f"Landscape median vertex count {len(med_pts)} != official u_grid n={len(u_grid)}"
        )
    x0, x1, yb, yt, fig_h = svg_axes_box(land_text)
    med_y = invert_svg_y(med_pts[:, 1], yb, yt, (0.0, 6.0))
    def map_fill_to_grid(fill, ylim=(0.0, 6.0)) -> Tuple[np.ndarray, np.ndarray]:
        """Map clipped SVG fill vertices onto official u_grid.

        Previous figures used xlim(0, 100), so fill polygons stop at u=100.
        u=101 is left NaN rather than extrapolated.
        """
        xs_disp, y_a_disp, y_b_disp = fill
        u_src = (xs_disp - x0) / (x1 - x0) * 100.0
        y_a = invert_svg_y(y_a_disp, yb, yt, ylim)
        y_b = invert_svg_y(y_b_disp, yb, yt, ylim)
        lo = np.full(u_grid.shape, np.nan, dtype=float)
        hi = np.full(u_grid.shape, np.nan, dtype=float)
        u_max = float(np.max(u_src))
        for i, ug in enumerate(u_grid):
            if ug > u_max + 0.51:
                continue
            j = int(np.argmin(np.abs(u_src - ug)))
            if abs(float(u_src[j]) - float(ug)) <= 0.51:
                lo[i] = min(y_a[j], y_b[j])
                hi[i] = max(y_a[j], y_b[j])
        return lo, hi

    iqr = extract_fill_bounds(land_text, fill_opacity="0.28", fig_height=fig_h)
    q1090 = extract_fill_bounds(land_text, fill_opacity="0.1", fig_height=fig_h)
    if iqr is None:
        raise RuntimeError("Could not extract IQR fill from locked landscape SVG")
    q25, q75 = map_fill_to_grid(iqr)
    if int(np.isfinite(q25).sum()) < 90:
        raise RuntimeError(f"IQR mapped too few grid points ({int(np.isfinite(q25).sum())})")
    summary = pd.DataFrame({
        "candidate_u_max": u_grid,
        "median": med_y,
        "q25": q25,
        "q75": q75,
    })
    if q1090 is not None:
        q10, q90 = map_fill_to_grid(q1090)
        if int(np.isfinite(q10).sum()) >= 90:
            summary["q10"] = q10
            summary["q90"] = q90
    land_df = pd.DataFrame({
        "candidate_u_max": u_grid,
        "composite_penalty": med_y,
    })
    land_df.attrs["_precomputed_summary"] = {"composite_penalty": summary}

    color_to_col = {
        "#5f97d2": "lr_fraction",
        "#b1ce46": "pauc_fraction",
        "#9dc3e7": "pathogen_fraction",
        "#d76364": "all_fraction",
    }
    cx0, cx1, cyb, cyt, _ = svg_axes_box(cons_text)
    feas = pd.DataFrame({"candidate_u_max": u_grid})
    for color, col in color_to_col.items():
        lines = extract_long_lines(cons_text, stroke=color, stroke_width="2.2")
        if not lines:
            raise RuntimeError(f"Could not extract {col} polyline ({color}) from locked constraint SVG")
        pts = max(lines, key=len)
        if len(pts) != len(u_grid):
            raise RuntimeError(f"{col} vertex count {len(pts)} != official u_grid n={len(u_grid)}")
        feas[col] = invert_svg_y(pts[:, 1], cyb, cyt, (0.0, 1.0))
        feas[col] = np.clip(feas[col], 0.0, 1.0)
    feas_df = pd.DataFrame({"candidate_u_max": u_grid, "composite_penalty": 0.0})
    feas_df.attrs["_precomputed_feasibility"] = feas

    fig5c_out = os.path.join(ROOT, "results", "umax_optimization", "figure", "umax_score_landscape.png")
    fig5b_out = os.path.join(ROOT, "results", "umax_optimization", "figure", "umax_constraint_feasibility.png")
    plot_umax_response_landscape(
        land_df,
        fig5c_out,
        score_col="composite_penalty",
        selected_umax_values=selected_u,
        selected_umax_csv=None,
        max_background_cases=0,
    )
    plot_umax_constraint_feasibility(
        feas_df,
        fig5b_out,
        selected_umax_values=selected_u,
    )
    fig5c_svg = fig5c_out[:-4] + ".svg"
    fig5b_svg = fig5b_out[:-4] + ".svg"
    with open(fig5c_svg, encoding="utf-8") as fh:
        new_land = fh.read()
    with open(fig5b_svg, encoding="utf-8") as fh:
        new_cons = fh.read()
    land_101 = "<!-- 101 -->" in new_land
    cons_101 = "<!-- 101 -->" in new_cons
    land_clip100 = has_100_only_clip(new_land)
    cons_clip100 = has_100_only_clip(new_cons)
    med_after = extract_long_lines(new_land, stroke="#5f97d2", stroke_width="2.8")
    if not med_after:
        raise RuntimeError("Replotted landscape is missing the median line")
    last_x = float(max(med_after, key=len)[-1, 0])
    x0n, x1n, _, _, _ = svg_axes_box(new_land)
    if last_x < x1n - 0.5:
        raise RuntimeError(f"Fig. 5c median last vertex still clipped: last_x={last_x} spine={x1n}")
    validation.append(
        f"Fig. 5b/c u_grid n={len(u_grid)} min={u_grid.min()} max={u_grid.max()}; "
        f"TAR selected_u_max n={selected_u.size} median={float(np.median(selected_u)):.1f}; "
        f"tick101 landscape={land_101} constraint={cons_101}; "
        f"100-only-clip landscape={land_clip100} constraint={cons_clip100}"
    )

    outputs = {
        "fig3b": [fig3b_out, fig3b_svg],
        "fig4a": [fig4a_out, fig4a_svg],
        "fig5d": [fig5d_out, fig5d_svg],
        "fig5b": [fig5b_out, fig5b_svg],
        "fig5c": [fig5c_out, fig5c_svg],
    }
    for paths in outputs.values():
        for path in paths:
            output_hashes[os.path.relpath(path, ROOT)] = sha256_file(path)

    label_hits = []
    for path in [fig3b_svg, fig4a_svg, fig5d_svg, fig5b_svg, fig5c_svg]:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        if re.search(r"individualized|multi-objective agreement", text, flags=re.I):
            label_hits.append(path)
    if label_hits:
        raise RuntimeError(f"Forbidden phrasing still present: {label_hits}")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    audit_path = os.path.join(ROOT, "figure_repair_audit.md")
    lines: List[str] = [
        "# Figure-only consistency repair audit",
        "",
        f"- Timestamp: {now}",
        f"- Current git commit: `{commit}`",
        "- Numeric values changed: **no** (bars/CIs/curves reused locked artifacts; no tests recomputed).",
        "- Original CSVs / statistical tables modified: **no**.",
        "- Full pipeline / ODE / ML / optimizer / 100-repeat runs: **not executed**.",
        "",
        "## Missing row-level landscape CSVs (not fabricated)",
        "",
        "These files are absent. Fig. 5b/c x-values were taken from locked `u_grid` "
        "(0–101 inclusive) in `repeats/repeat_000/umax_study/umax_optimization_manifest.json`; "
        "curve *y*-values were taken from the locked matplotlib SVG polylines (102 vertices), "
        "indexed onto that grid. Selected-Umax rugs/median used `selected_u_max` from "
        "`umax_feasible_region_summary.csv`.",
        "",
    ]
    for path in absent_landscape_csv:
        lines.append(f"- `{os.path.relpath(path, ROOT)}` — **missing**")
    lines += ["", "## Input artifact SHA256 (before overwrite)", ""]
    for rel, digest in input_hashes.items():
        lines.append(f"- `{rel}`: `{digest}`")
    lines += ["", "## Per-figure record", ""]

    def dump_group(title: str, in_paths: Sequence[str], out_paths: Sequence[str], notes: Sequence[str]) -> None:
        lines.append(f"### {title}")
        lines.append("")
        lines.append("**Numeric values changed:** no")
        lines.append("")
        lines.append("Input files + SHA256:")
        for path in in_paths:
            rel = os.path.relpath(path, ROOT)
            lines.append(f"- `{rel}`: `{input_hashes[rel]}`")
        lines.append("")
        lines.append("Output files + SHA256:")
        for path in out_paths:
            rel = os.path.relpath(path, ROOT)
            lines.append(f"- `{rel}`: `{output_hashes[rel]}`")
        lines.append("")
        lines.append("Annotations removed/modified:")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    dump_group(
        "Fig. 3b `model_compare_r2`",
        inputs["fig3b"],
        outputs["fig3b"],
        [
            "Removed TAR–BestTree/BestSingleTree significance brackets and stars on both R² and RMSE panels (oracle diagnostic; `is_oracle_diagnostic=True` in the saved table).",
            "Retained prespecified formal brackets only: TAR–RandomForest and TAR–UniformTreeMean, using saved Holm `significance_label` / `corrected_p_holm` from `parameter_pairwise_significance.csv` (no p-value recomputation).",
            "Bar heights and 95% CIs unchanged from `model_compare_summary.csv`.",
            "Wording `individualized` / `multi-objective agreement` was not present in this source figure.",
        ],
    )
    dump_group(
        "Fig. 4a `ode_back_r2_barplot`",
        inputs["fig4a"],
        outputs["fig4a"],
        [
            "Removed TAR–BestTree/BestSingleTree significance bracket and star (oracle diagnostic).",
            "Retained only non-oracle formal comparisons present as `is_formal_comparison=True` in `ode_back_pairwise_significance.csv`: TAR–RandomForest and TAR–UniformTreeMean.",
            "Bar heights and 95% CIs unchanged from `ode_back_summary_by_model.csv`.",
            "Wording `individualized` / `multi-objective agreement` was not present in this source figure.",
        ],
    )
    dump_group(
        "Fig. 5b `umax_constraint_feasibility`",
        inputs["fig5bc"],
        outputs["fig5b"],
        [
            "Removed display clipping `set_xlim(0, 100)` so the official 0–101 inclusive candidate grid is shown.",
            "No inferential brackets/stars (this panel never had them).",
            "Median selected Umax line/IQR span still from locked TAR `selected_u_max`.",
            "Wording `individualized` / `multi-objective agreement` was not present; no title rewrite.",
        ],
    )
    dump_group(
        "Fig. 5c `umax_score_landscape`",
        inputs["fig5bc"],
        outputs["fig5c"],
        [
            "Removed display clipping `set_xlim(0, 100)` so the official 0–101 inclusive candidate grid is shown.",
            "Median, IQR (and q10–q90 if present) taken from locked SVG vertex count matched to `u_grid`; not recomputed from ODE candidates.",
            "Wording `individualized` / `multi-objective agreement` was not present; no title rewrite.",
        ],
    )
    dump_group(
        "Fig. 5d `umax_summary_ablation_composite_supplementary`",
        inputs["fig5d"],
        outputs["fig5d"],
        [
            "Removed all inferential brackets/stars (`significance_pairs=None`). Saved `umax_ablation_significance_annotations.csv` was not used and was not modified.",
            "Retained bars and descriptive repeat-level 95% intervals from `umax_ablation_repeated_plot_stats.csv`.",
            "Did not recompute p-values.",
            "Wording `individualized` / `multi-objective agreement` was not present in this source figure (ylabel remains `Composite penalty score`).",
        ],
    )
    lines += [
        "## Label wording search",
        "",
        "Repo source figures / plot titles were searched for `individualized` and `multi-objective agreement`. "
        "**No matches.** No manuscript composite `Fig5.png` panel-A schematic is generated here. "
        "No title/label string replacement was applied.",
        "",
        "## Validation evidence",
        "",
    ]
    for item in validation:
        lines.append(f"- {item}")
    lines.append(f"- Fig. 5b/5c 100-only clip remaining: constraint={cons_clip100}, landscape={land_clip100} (must be False).")
    lines.append("- PNG+SVG exported via `save_figure` for every repaired panel.")
    lines.append("- `assert_figure_artists_inside_canvas` ran inside the 3b/4a/5b/5c/5d plot functions.")
    lines.append("")

    with open(audit_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {audit_path}")
    print("git commit:", commit)
    for item in validation:
        print(item)
    if cons_clip100 or land_clip100:
        raise RuntimeError("Fig. 5b/c still appear clipped at 100")
    return 0


if __name__ == "__main__":
    sys.exit(main())
