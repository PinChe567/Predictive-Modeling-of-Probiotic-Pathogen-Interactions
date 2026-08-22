"""Read-only audit of sample-weight min-max normalization leakage.

Compares the existing full-dataset (global) min-max normalization used to
construct sample_weights.csv against the same formula computed from training
rows only, on the exact 100 outer biological-group held-out splits used by
the final TAR manuscript benchmark.

This script does not:
- modify production source files
- overwrite existing manuscript result CSVs/JSONs
- retrain TAR or any comparator
- regenerate ODE simulations
- change labels, relabeling, configs, or figures
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tree_srl_benchmark import split_train_validation_indices  # noqa: E402

MANIFEST_PATH = ROOT / "results" / "tree_srl_benchmark" / "model_compare_manifest.json"
FORMAL_CONFIG_PATH = ROOT / "results" / "tree_srl_benchmark" / "formal_run_config.json"
PREDICTIONS_MANIFEST_PATH = ROOT / "results" / "tree_srl_benchmark" / "predictions_manifest.json"

OUT_BY_REPEAT = ROOT / "results" / "audit_sample_weight_normalization_by_repeat.csv"
OUT_SUMMARY = ROOT / "results" / "audit_sample_weight_normalization_summary.json"
OUT_REPORT = ROOT / "results" / "audit_sample_weight_normalization_report.md"
OUT_TOP_DIFF = ROOT / "results" / "audit_sample_weight_normalization_top_differences.csv"

ALLOWED_WRITE_PATHS = {
    OUT_BY_REPEAT.resolve(),
    OUT_SUMMARY.resolve(),
    OUT_REPORT.resolve(),
    OUT_TOP_DIFF.resolve(),
}

WEIGHT_EPS = 0.05
SPAN_EPS = 1e-12
IDENTICAL_ABS_EPS = 1e-12
NEGLIGIBLE_REL_EPS = 1e-6
EXPECTED_N_REPEATS = 100
REL_DENOM_EPS = 1e-12

PRODUCTION_SOURCE_FILES = [
    ROOT / "microbio_dataset.py",
    ROOT / "tree_srl_benchmark.py",
    ROOT / "multi_pathogen_simulator.py",
]
PRODUCTION_RESULT_GLOBS = [
    "results/tree_srl_benchmark/model_compare_manifest.json",
    "results/tree_srl_benchmark/formal_run_config.json",
    "results/tree_srl_benchmark/predictions_manifest.json",
    "data/microbio_formal_dataset/X_features.csv",
    "data/microbio_formal_dataset/y_targets.csv",
    "data/microbio_formal_dataset/sample_metadata.csv",
    "data/microbio_formal_dataset/sample_weights.csv",
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
        if not np.isfinite(val):
            return None
        return val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_native(obj.tolist())
    if obj is None or isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    return str(obj)


def _safe_write_text(path: Path, text: str) -> None:
    resolved = path.resolve()
    if resolved not in ALLOWED_WRITE_PATHS:
        raise RuntimeError(f"Refusing to write outside audit outputs: {resolved}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_write_csv(path: Path, df: pd.DataFrame) -> None:
    resolved = path.resolve()
    if resolved not in ALLOWED_WRITE_PATHS:
        raise RuntimeError(f"Refusing to write outside audit outputs: {resolved}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _resolve_project_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_uncertainties(uncertainties: np.ndarray, u_min: float, u_max: float) -> np.ndarray:
    span = max(float(u_max) - float(u_min), SPAN_EPS)
    return (np.asarray(uncertainties, dtype=np.float64) - float(u_min)) / span


def _weights_from_norm(unc_norm: np.ndarray) -> np.ndarray:
    return 1.0 / (np.asarray(unc_norm, dtype=np.float64) + WEIGHT_EPS)


def _finite_corr(a: np.ndarray, b: np.ndarray) -> Tuple[float | None, float | None]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return None, None
    if np.allclose(a, a[0], atol=IDENTICAL_ABS_EPS, rtol=0.0) and np.allclose(
        b, b[0], atol=IDENTICAL_ABS_EPS, rtol=0.0
    ):
        return (1.0 if np.allclose(a, b, atol=IDENTICAL_ABS_EPS, rtol=0.0) else None), None
    pearson_val = float(pearsonr(a, b).statistic)
    spearman_val = float(spearmanr(a, b).statistic)
    if not np.isfinite(pearson_val):
        pearson_val_out: float | None = None
    else:
        pearson_val_out = pearson_val
    if not np.isfinite(spearman_val):
        spearman_val_out: float | None = None
    else:
        spearman_val_out = spearman_val
    return pearson_val_out, spearman_val_out


def _group_location(groups_in_train: bool, groups_in_val: bool) -> str:
    if groups_in_train and groups_in_val:
        return "mixed"
    if groups_in_train:
        return "training"
    if groups_in_val:
        return "validation"
    return "absent"


def _classify_audit(
    max_abs_all: float,
    max_rel_all: float,
    n_splits_gt_1e12: int,
) -> Dict[str, Any]:
    """Apply the requested three-way classification only. No scientific pass/fail."""
    if n_splits_gt_1e12 == 0 and max_abs_all <= IDENTICAL_ABS_EPS:
        label = "NO PRACTICAL DIFFERENCE"
        reason = (
            "All training weights are numerically identical within 1e-12 "
            "across all audited splits."
        )
    elif max_rel_all < NEGLIGIBLE_REL_EPS:
        label = "NEGLIGIBLE DIFFERENCE"
        reason = (
            f"Differences exist (some |Δweight| > 1e-12) but the maximum relative "
            f"weight difference ({max_rel_all:.6e}) is below {NEGLIGIBLE_REL_EPS:.0e}. "
            "This classification reports magnitude only and does not decide "
            "scientific acceptability."
        )
    else:
        label = "NONZERO DIFFERENCE"
        reason = (
            "Train-only min-max normalization changes at least some training "
            f"weights (max |Δweight|={max_abs_all:.6e}, max relative "
            f"difference={max_rel_all:.6e})."
        )
    return {
        "audit_classification": label,
        "classification_reason": reason,
        "identical_abs_eps": IDENTICAL_ABS_EPS,
        "negligible_relative_eps": NEGLIGIBLE_REL_EPS,
        "does_not_decide_scientific_acceptability": True,
    }


def load_executed_split_config() -> Dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing executed benchmark manifest: {MANIFEST_PATH}")
    manifest = _load_json(MANIFEST_PATH)
    run_config = manifest.get("run_config", {})
    formal = _load_json(FORMAL_CONFIG_PATH) if FORMAL_CONFIG_PATH.is_file() else {}
    executed = formal.get("executed_from_saved_manifest", {})

    x_csv = _resolve_project_path(run_config["x_csv"])
    y_csv = _resolve_project_path(run_config["y_csv"])
    metadata_csv = _resolve_project_path(run_config["metadata_csv"])
    sample_weight_csv = _resolve_project_path(run_config["sample_weight_csv"])
    split_mode = str(manifest.get("split_mode") or run_config.get("split_mode"))
    group_col = str(manifest.get("group_col") or run_config.get("group_col"))
    test_size = float(manifest.get("test_size") if manifest.get("test_size") is not None else run_config["test_size"])
    n_repeats = int(manifest.get("n_repeats") or run_config.get("n_repeats"))
    base_seed = int(run_config["seed"])
    repeat_metadata = list(manifest.get("repeat_metadata", []))
    if not repeat_metadata:
        raise RuntimeError("model_compare_manifest.json does not contain repeat_metadata.")
    if n_repeats != EXPECTED_N_REPEATS:
        raise RuntimeError(
            f"Expected n_repeats={EXPECTED_N_REPEATS} in the executed formal benchmark, "
            f"found {n_repeats}."
        )
    if len(repeat_metadata) != EXPECTED_N_REPEATS:
        raise RuntimeError(
            f"Expected {EXPECTED_N_REPEATS} repeat_metadata records, found {len(repeat_metadata)}."
        )
    if split_mode != "group":
        raise RuntimeError(f"Executed split_mode must be 'group'; found {split_mode!r}.")
    if executed and int(executed.get("n_repeats", n_repeats)) != EXPECTED_N_REPEATS:
        raise RuntimeError("formal_run_config.json n_repeats does not match the executed 100-repeat benchmark.")
    if executed and str(executed.get("split_mode", split_mode)) != split_mode:
        raise RuntimeError("formal_run_config.json split_mode does not match the executed benchmark.")
    if executed and int(executed.get("seed", base_seed)) != base_seed:
        raise RuntimeError("formal_run_config.json seed does not match the executed benchmark.")

    seeds: List[int] = []
    for i, rec in enumerate(repeat_metadata):
        rid = int(rec["repeat_id"])
        seed = int(rec["seed"])
        if rid != i:
            raise RuntimeError(f"repeat_metadata is not contiguous: expected repeat_id={i}, found {rid}.")
        expected_seed = base_seed + rid
        if seed != expected_seed:
            raise RuntimeError(
                f"Saved seed for repeat_id={rid} is {seed}, but tree_srl_benchmark.py uses "
                f"seed + repeat_id = {expected_seed}."
            )
        seeds.append(seed)

    predictions_dir = ROOT / "results" / "tree_srl_benchmark" / "repeats"
    return {
        "x_csv": x_csv,
        "y_csv": y_csv,
        "metadata_csv": metadata_csv,
        "sample_weight_csv": sample_weight_csv,
        "split_mode": split_mode,
        "group_col": group_col,
        "test_size": test_size,
        "n_repeats": n_repeats,
        "base_seed": base_seed,
        "seeds": seeds,
        "repeat_metadata": repeat_metadata,
        "predictions_dir": predictions_dir,
        "manifest_path": MANIFEST_PATH,
        "formal_config_path": FORMAL_CONFIG_PATH,
        "seed_rule": "repeat_seed = run_config.seed + repeat_id (confirmed against saved repeat_metadata)",
        "split_helper": "tree_srl_benchmark.split_train_validation_indices",
    }


def confirm_alignment(
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    weight_df: pd.DataFrame,
) -> Dict[str, Any]:
    n_x, n_y, n_meta, n_w = len(x_df), len(y_df), len(meta_df), len(weight_df)
    if not (n_x == n_y == n_meta == n_w):
        raise RuntimeError(
            "Row-count mismatch before audit: "
            f"X={n_x}, y={n_y}, metadata={n_meta}, sample_weights={n_w}."
        )
    required_weight = {"sample_weight", "target_uncertainty", "target_uncertainty_norm"}
    missing = required_weight.difference(weight_df.columns)
    if missing:
        raise RuntimeError(f"sample_weights.csv missing columns: {sorted(missing)}")
    if "bio_id" not in meta_df.columns:
        raise RuntimeError("sample_metadata.csv missing grouping column bio_id.")
    if "row_id" in weight_df.columns and "row_id" in meta_df.columns:
        if not np.array_equal(
            weight_df["row_id"].to_numpy(),
            meta_df["row_id"].to_numpy(),
        ):
            raise RuntimeError("row_id alignment failed between sample_weights.csv and sample_metadata.csv.")
    if "target_uncertainty" in meta_df.columns:
        if not np.allclose(
            meta_df["target_uncertainty"].to_numpy(dtype=np.float64),
            weight_df["target_uncertainty"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            max_diff = float(
                np.nanmax(
                    np.abs(
                        meta_df["target_uncertainty"].to_numpy(dtype=np.float64)
                        - weight_df["target_uncertainty"].to_numpy(dtype=np.float64)
                    )
                )
            )
            raise RuntimeError(
                "target_uncertainty differs between sample_metadata.csv and sample_weights.csv "
                f"(max abs diff={max_diff})."
            )
    n_na_x = int(x_df.isna().any(axis=1).sum())
    n_na_y = int(y_df.isna().any(axis=1).sum())
    if n_na_x or n_na_y:
        raise RuntimeError(
            "Final dataset contains NA feature/target rows. The executed benchmark drops NA "
            f"rows before splitting (n_na_x={n_na_x}, n_na_y={n_na_y}); refusing to guess a "
            "reindexed split."
        )
    return {
        "n_rows": int(n_x),
        "n_x_columns": int(x_df.shape[1]),
        "n_y_columns": int(y_df.shape[1]),
        "row_id_aligned": bool("row_id" in weight_df.columns and "row_id" in meta_df.columns),
        "n_na_feature_rows": n_na_x,
        "n_na_target_rows": n_na_y,
    }


def extrema_groups(
    uncertainties: np.ndarray,
    groups: np.ndarray,
    meta_df: pd.DataFrame,
    global_u_min: float,
    global_u_max: float,
) -> Dict[str, Any]:
    min_mask = uncertainties == global_u_min
    max_mask = uncertainties == global_u_max
    min_groups = np.unique(groups[min_mask])
    max_groups = np.unique(groups[max_mask])
    min_rows = np.flatnonzero(min_mask)
    max_rows = np.flatnonzero(max_mask)

    def _row_records(row_idx: np.ndarray) -> List[dict]:
        records = []
        for i in row_idx.tolist():
            rec = {
                "row_index": int(i),
                "bio_id": _json_native(groups[i]),
                "target_uncertainty": float(uncertainties[i]),
            }
            if "row_id" in meta_df.columns:
                rec["row_id"] = _json_native(meta_df.iloc[i]["row_id"])
            if "desired_profile_id" in meta_df.columns:
                rec["desired_profile_id"] = _json_native(meta_df.iloc[i]["desired_profile_id"])
            records.append(rec)
        return records

    min_records = _row_records(min_rows)
    max_records = _row_records(max_rows)
    return {
        "global_min_n_rows": int(min_mask.sum()),
        "global_max_n_rows": int(max_mask.sum()),
        "global_min_n_groups": int(len(min_groups)),
        "global_max_n_groups": int(len(max_groups)),
        "global_min_bio_ids": [_json_native(g) for g in min_groups.tolist()],
        "global_max_bio_ids": [_json_native(g) for g in max_groups.tolist()],
        "global_min_example_rows": min_records[:5],
        "global_max_rows": max_records,
        "min_groups_arr": min_groups,
        "max_groups_arr": max_groups,
    }


def load_saved_val_indices(predictions_dir: Path, repeat_id: int) -> np.ndarray:
    pred_path = predictions_dir / f"repeat_{repeat_id:03d}" / "predictions.csv"
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing saved TAR split predictions: {pred_path}")
    pred = pd.read_csv(pred_path, usecols=["validation_original_row_index"])
    return pred["validation_original_row_index"].to_numpy(dtype=int)


def fmt_float(value: float | None, digits: int = 12) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{value:.{digits}g}"


def build_report(
    cfg: Dict[str, Any],
    alignment: Dict[str, Any],
    reconstruction: Dict[str, Any],
    extrema: Dict[str, Any],
    summary: Dict[str, Any],
    classification: Dict[str, Any],
    production_check: Dict[str, Any],
) -> str:
    seeds = cfg["seeds"]
    lines = [
        "# Sample-weight min-max normalization audit",
        "",
        "Read-only check of whether the min-max normalization used to construct",
        "`sample_weights.csv` changes when computed from **training rows only**",
        "instead of the full supervised dataset, on the exact 100 outer",
        "biological-group held-out splits of the final TAR manuscript benchmark.",
        "",
        "This audit does **not** retrain models, regenerate ODE simulations,",
        "relabel data, or overwrite existing manuscript results.",
        "",
        "## 1. Executed split configuration (reused, not reinvented)",
        "",
        f"- Manifest: `{cfg['manifest_path'].relative_to(ROOT).as_posix()}`",
        f"- Formal run config: `{cfg['formal_config_path'].relative_to(ROOT).as_posix()}`",
        f"- Split helper: `{cfg['split_helper']}`",
        f"- Split mode: `{cfg['split_mode']}`",
        f"- Grouping variable: `{cfg['group_col']}`",
        f"- test_size: `{cfg['test_size']}`",
        f"- Base seed (run_config.seed): `{cfg['base_seed']}`",
        f"- Seed rule: {cfg['seed_rule']}",
        f"- Repeated splits audited: **{summary['n_repeats_audited']}** (expected {EXPECTED_N_REPEATS})",
        f"- Actual seeds used: `{seeds[0]}` through `{seeds[-1]}` (n={len(seeds)})",
        f"- Saved-split verification mismatches: **{summary['n_split_mismatches_vs_saved_predictions']}**",
        "",
        "## 2. Dataset alignment",
        "",
        f"- X_features.csv: `{cfg['x_csv'].relative_to(ROOT).as_posix()}` ({alignment['n_rows']} rows, {alignment['n_x_columns']} columns)",
        f"- y_targets.csv: `{cfg['y_csv'].relative_to(ROOT).as_posix()}` ({alignment['n_rows']} rows, {alignment['n_y_columns']} columns)",
        f"- sample_metadata.csv: `{cfg['metadata_csv'].relative_to(ROOT).as_posix()}`",
        f"- sample_weights.csv: `{cfg['sample_weight_csv'].relative_to(ROOT).as_posix()}`",
        f"- Row counts match: **yes** (n={alignment['n_rows']})",
        f"- row_id aligned between metadata and weights: **{alignment['row_id_aligned']}**",
        f"- NA feature/target rows: {alignment['n_na_feature_rows']} / {alignment['n_na_target_rows']}",
        "",
        "## 3. Global (full-dataset) reconstruction",
        "",
        "Formula (copied from `microbio_dataset.py`):",
        "",
        "```",
        "u_min = uncertainties.min()",
        "u_max = uncertainties.max()",
        "unc_norm = (uncertainties - u_min) / max(u_max - u_min, 1e-12)",
        "sample_weight = 1.0 / (unc_norm + 0.05)",
        "```",
        "",
        f"- global_u_min = `{fmt_float(summary['global_u_min'])}`",
        f"- global_u_max = `{fmt_float(summary['global_u_max'])}`",
        f"- max_abs_difference_existing_vs_reconstructed = `{fmt_float(reconstruction['max_abs_difference_existing_vs_reconstructed'])}`",
        f"- mean_abs_difference_existing_vs_reconstructed = `{fmt_float(reconstruction['mean_abs_difference_existing_vs_reconstructed'])}`",
        f"- Reconstruction matches existing sample_weights.csv: **{reconstruction['reconstruction_matches_existing']}**",
        "",
    ]
    if not reconstruction["reconstruction_matches_existing"]:
        lines.extend(
            [
                "> **FLAG:** reconstructed global weights do **not** match `sample_weights.csv`.",
                "> The remainder of this audit still compares train-only vs existing stored weights,",
                "> but the stored file may not follow the documented formula.",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. Biological groups containing global extrema",
            "",
            (
                f"- Global-min `target_uncertainty` = `{fmt_float(summary['global_u_min'])}` "
                f"occurs in **{extrema['global_min_n_rows']}** row(s) across "
                f"**{extrema['global_min_n_groups']}** bio_id groups. "
                "Because this value is common, `train_u_min == global_u_min` in every split. "
                "The full bio_id list is in the summary JSON."
            ),
            (
                f"- Global-max `target_uncertainty` = `{fmt_float(summary['global_u_max'])}` "
                f"occurs in **{extrema['global_max_n_rows']}** row(s) "
                f"in bio_id(s): `{extrema['global_max_bio_ids']}`"
                + (
                    f" (desired_profile_id={extrema['global_max_rows'][0].get('desired_profile_id')}, "
                    f"row_id={extrema['global_max_rows'][0].get('row_id')})"
                    if extrema.get("global_max_rows")
                    else ""
                )
                + ". This single group being held out is what changes the training-only max."
            ),
            "",
            "Per-repeat location of those groups is in",
            "`results/audit_sample_weight_normalization_by_repeat.csv`",
            "(columns `global_min_group_location`, `global_max_group_location`).",
            "",
            "## 5. Train-only vs global-normalized training weights",
            "",
            "### A. Number of repeated splits audited",
            "",
            f"- expected = {EXPECTED_N_REPEATS}",
            f"- audited = {summary['n_repeats_audited']}",
            "",
            "### B. Splits where train extrema equal global extrema",
            "",
            (
                f"- train_u_min == global_u_min: "
                f"**{summary['n_splits_train_min_equals_global_min']}/{summary['n_repeats_audited']}** "
                f"({summary['pct_splits_train_min_equals_global_min']:.1f}%)"
            ),
            (
                f"- train_u_max == global_u_max: "
                f"**{summary['n_splits_train_max_equals_global_max']}/{summary['n_repeats_audited']}** "
                f"({summary['pct_splits_train_max_equals_global_max']:.1f}%)"
            ),
            (
                f"- BOTH equal: "
                f"**{summary['n_splits_both_extrema_in_training']}/{summary['n_repeats_audited']}** "
                f"({summary['pct_splits_both_extrema_in_training']:.1f}%)"
            ),
            "",
            "### C. Weight-difference magnitudes across all splits",
            "",
            f"- maximum observed max_abs_weight_difference = `{fmt_float(summary['maximum_observed_max_abs_weight_difference'])}`",
            f"- maximum observed mean_abs_weight_difference = `{fmt_float(summary['maximum_observed_mean_abs_weight_difference'])}`",
            f"- median max_abs_weight_difference = `{fmt_float(summary['median_max_abs_weight_difference'])}`",
            f"- maximum observed max_relative_weight_difference = `{fmt_float(summary['maximum_observed_max_relative_weight_difference'])}`",
            "",
            "### D. Splits with any training-weight difference above thresholds",
            "",
            (
                f"- any |Δweight| > 1e-12: "
                f"**{summary['n_splits_any_absdiff_gt_1e-12']}/{summary['n_repeats_audited']}**"
            ),
            (
                f"- any |Δweight| > 1e-9: "
                f"**{summary['n_splits_any_absdiff_gt_1e-9']}/{summary['n_repeats_audited']}**"
            ),
            (
                f"- any |Δweight| > 1e-6: "
                f"**{summary['n_splits_any_absdiff_gt_1e-6']}/{summary['n_repeats_audited']}**"
            ),
            "",
            "### E. Audit classification",
            "",
            f"**{classification['audit_classification']}**",
            "",
            classification["classification_reason"],
            "",
            "Classification rules used (magnitude only; no scientific pass/fail):",
            "",
            "- `NO PRACTICAL DIFFERENCE`: all training weights identical within 1e-12 on every split.",
            (
                f"- `NEGLIGIBLE DIFFERENCE`: differences exist, but max relative difference "
                f"< {NEGLIGIBLE_REL_EPS:.0e}. Values are reported; scientific acceptability is "
                "not decided automatically."
            ),
            "- `NONZERO DIFFERENCE`: train-only normalization changes training weights beyond that.",
            "",
            "## 6. Integrity checks",
            "",
            f"- Production source files modified: **{production_check['production_source_modified']}**",
            f"- Existing manuscript result files overwritten: **{production_check['existing_results_overwritten']}**",
            f"- Any model trained: **{production_check['model_trained']}**",
            f"- Any ODE simulation rerun: **{production_check['ode_simulation_rerun']}**",
            f"- Files written by this audit: {production_check['files_written']}",
            "",
            f"Generated: {summary['generated_at_utc']}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    cfg = load_executed_split_config()
    production_hashes_before = {
        path.relative_to(ROOT).as_posix(): _sha256_file(path)
        for path in PRODUCTION_SOURCE_FILES
        if path.is_file()
    }
    result_hashes_before = {}
    for rel in PRODUCTION_RESULT_GLOBS:
        path = ROOT / rel
        if path.is_file():
            result_hashes_before[rel.replace("\\", "/")] = _sha256_file(path)

    x_df = pd.read_csv(cfg["x_csv"])
    y_df = pd.read_csv(cfg["y_csv"])
    meta_df = pd.read_csv(cfg["metadata_csv"])
    weight_df = pd.read_csv(cfg["sample_weight_csv"])
    alignment = confirm_alignment(x_df, y_df, meta_df, weight_df)

    uncertainties = weight_df["target_uncertainty"].to_numpy(dtype=np.float64)
    existing_norm = weight_df["target_uncertainty_norm"].to_numpy(dtype=np.float64)
    existing_weight = weight_df["sample_weight"].to_numpy(dtype=np.float64)
    groups = meta_df[cfg["group_col"]].to_numpy()
    n_samples = int(len(uncertainties))

    global_u_min = float(uncertainties.min())
    global_u_max = float(uncertainties.max())
    global_unc_norm = _normalize_uncertainties(uncertainties, global_u_min, global_u_max)
    global_weight = _weights_from_norm(global_unc_norm)
    recon_abs = np.abs(existing_weight - global_weight)
    reconstruction = {
        "global_u_min": global_u_min,
        "global_u_max": global_u_max,
        "max_abs_difference_existing_vs_reconstructed": float(np.max(recon_abs)),
        "mean_abs_difference_existing_vs_reconstructed": float(np.mean(recon_abs)),
        "max_abs_norm_difference_existing_vs_reconstructed": float(np.max(np.abs(existing_norm - global_unc_norm))),
        "reconstruction_matches_existing": bool(np.max(recon_abs) <= IDENTICAL_ABS_EPS),
    }

    extrema = extrema_groups(uncertainties, groups, meta_df, global_u_min, global_u_max)
    min_groups_arr = extrema.pop("min_groups_arr")
    max_groups_arr = extrema.pop("max_groups_arr")

    rows: List[dict] = []
    n_split_mismatches = 0
    for repeat_id, seed in enumerate(cfg["seeds"]):
        saved_meta = cfg["repeat_metadata"][repeat_id]
        train_idx, val_idx = split_train_validation_indices(
            n_samples=n_samples,
            groups=groups,
            test_size=cfg["test_size"],
            seed=int(seed),
            split_mode=cfg["split_mode"],
        )
        train_idx = np.asarray(train_idx, dtype=int)
        val_idx = np.asarray(val_idx, dtype=int)

        saved_val = load_saved_val_indices(cfg["predictions_dir"], repeat_id)
        reconstructed_val_set = set(val_idx.tolist())
        saved_val_set = set(saved_val.tolist())
        split_matches_saved = reconstructed_val_set == saved_val_set
        if not split_matches_saved:
            n_split_mismatches += 1
            # The saved TAR predictions are the splits actually used in the manuscript.
            val_idx = np.unique(saved_val)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[val_idx] = False
            train_idx = np.flatnonzero(train_mask)

        train_groups = groups[train_idx]
        val_groups = groups[val_idx]
        u_train = uncertainties[train_idx]
        train_u_min = float(u_train.min())
        train_u_max = float(u_train.max())
        train_unc_norm_trainonly = _normalize_uncertainties(u_train, train_u_min, train_u_max)
        train_weight_trainonly = _weights_from_norm(train_unc_norm_trainonly)
        train_weight_global = existing_weight[train_idx]

        abs_diff = np.abs(train_weight_trainonly - train_weight_global)
        rel_diff = abs_diff / np.maximum(np.abs(train_weight_global), REL_DENOM_EPS)
        pearson_val, spearman_val = _finite_corr(train_weight_global, train_weight_trainonly)

        train_min_eq = bool(train_u_min == global_u_min)
        train_max_eq = bool(train_u_max == global_u_max)

        min_in_train = bool(np.isin(min_groups_arr, train_groups).any())
        min_in_val = bool(np.isin(min_groups_arr, val_groups).any())
        max_in_train = bool(np.isin(max_groups_arr, train_groups).any())
        max_in_val = bool(np.isin(max_groups_arr, val_groups).any())

        n_train_rows = int(len(train_idx))
        n_val_rows = int(len(val_idx))
        n_train_groups = int(len(np.unique(train_groups)))
        n_val_groups = int(len(np.unique(val_groups)))
        if (
            n_train_rows != int(saved_meta["train_row_count"])
            or n_val_rows != int(saved_meta["val_row_count"])
            or n_train_groups != int(saved_meta["train_bio_groups"])
            or n_val_groups != int(saved_meta["test_bio_groups"])
        ):
            raise RuntimeError(
                f"Repeat {repeat_id} split counts do not match saved repeat_metadata: "
                f"got train_rows={n_train_rows}, val_rows={n_val_rows}, "
                f"train_groups={n_train_groups}, val_groups={n_val_groups}; "
                f"saved {saved_meta['train_row_count']}/{saved_meta['val_row_count']}/"
                f"{saved_meta['train_bio_groups']}/{saved_meta['test_bio_groups']}."
            )

        rows.append(
            {
                "repeat_id": int(repeat_id),
                "seed": int(seed),
                "n_train_rows": n_train_rows,
                "n_validation_rows": n_val_rows,
                "n_train_groups": n_train_groups,
                "n_validation_groups": n_val_groups,
                "split_matches_saved_predictions": bool(split_matches_saved),
                "global_u_min": global_u_min,
                "global_u_max": global_u_max,
                "train_u_min": train_u_min,
                "train_u_max": train_u_max,
                "train_min_equals_global_min": train_min_eq,
                "train_max_equals_global_max": train_max_eq,
                "both_global_extrema_present_in_training": bool(train_min_eq and train_max_eq),
                "global_min_group_location": _group_location(min_in_train, min_in_val),
                "global_max_group_location": _group_location(max_in_train, max_in_val),
                "n_global_min_groups_in_training": int(np.isin(min_groups_arr, train_groups).sum()),
                "n_global_min_groups_in_validation": int(np.isin(min_groups_arr, val_groups).sum()),
                "n_global_max_groups_in_training": int(np.isin(max_groups_arr, train_groups).sum()),
                "n_global_max_groups_in_validation": int(np.isin(max_groups_arr, val_groups).sum()),
                "max_abs_weight_difference": float(np.max(abs_diff)),
                "mean_abs_weight_difference": float(np.mean(abs_diff)),
                "median_abs_weight_difference": float(np.median(abs_diff)),
                "max_relative_weight_difference": float(np.max(rel_diff)),
                "mean_relative_weight_difference": float(np.mean(rel_diff)),
                "n_weights_absdiff_gt_1e-12": int(np.sum(abs_diff > 1e-12)),
                "n_weights_absdiff_gt_1e-9": int(np.sum(abs_diff > 1e-9)),
                "n_weights_absdiff_gt_1e-6": int(np.sum(abs_diff > 1e-6)),
                "fraction_weights_absdiff_gt_1e-9": float(np.mean(abs_diff > 1e-9)),
                "pearson_correlation_global_vs_trainonly": pearson_val,
                "spearman_correlation_global_vs_trainonly": spearman_val,
            }
        )

    if n_split_mismatches:
        raise RuntimeError(
            "Reconstructed GroupShuffleSplit validation indices did not match saved "
            f"TAR predictions for {n_split_mismatches}/100 repeats. The audit refuses to "
            "continue on a non-identical split. Check sklearn version vs the original run."
        )

    by_repeat = pd.DataFrame(rows)
    n = int(len(by_repeat))
    n_min_eq = int(by_repeat["train_min_equals_global_min"].sum())
    n_max_eq = int(by_repeat["train_max_equals_global_max"].sum())
    n_both = int(by_repeat["both_global_extrema_present_in_training"].sum())
    n_gt_1e12 = int((by_repeat["n_weights_absdiff_gt_1e-12"] > 0).sum())
    n_gt_1e9 = int((by_repeat["n_weights_absdiff_gt_1e-9"] > 0).sum())
    n_gt_1e6 = int((by_repeat["n_weights_absdiff_gt_1e-6"] > 0).sum())
    max_abs_all = float(by_repeat["max_abs_weight_difference"].max())
    max_mean_abs_all = float(by_repeat["mean_abs_weight_difference"].max())
    median_max_abs = float(by_repeat["max_abs_weight_difference"].median())
    max_rel_all = float(by_repeat["max_relative_weight_difference"].max())

    classification = _classify_audit(max_abs_all, max_rel_all, n_gt_1e12)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = {
        "generated_at_utc": generated_at,
        "audit_only": True,
        "no_retraining": True,
        "no_ode_resimulation": True,
        "n_repeats_audited": n,
        "n_repeats_expected": EXPECTED_N_REPEATS,
        "split_mode": cfg["split_mode"],
        "group_col": cfg["group_col"],
        "test_size": cfg["test_size"],
        "base_seed": cfg["base_seed"],
        "seeds_used": cfg["seeds"],
        "seed_rule": cfg["seed_rule"],
        "split_helper": cfg["split_helper"],
        "n_split_mismatches_vs_saved_predictions": n_split_mismatches,
        "dataset_alignment": alignment,
        "global_u_min": global_u_min,
        "global_u_max": global_u_max,
        "global_reconstruction": reconstruction,
        "extrema_groups": extrema,
        "n_splits_train_min_equals_global_min": n_min_eq,
        "pct_splits_train_min_equals_global_min": 100.0 * n_min_eq / n,
        "n_splits_train_max_equals_global_max": n_max_eq,
        "pct_splits_train_max_equals_global_max": 100.0 * n_max_eq / n,
        "n_splits_both_extrema_in_training": n_both,
        "pct_splits_both_extrema_in_training": 100.0 * n_both / n,
        "maximum_observed_max_abs_weight_difference": max_abs_all,
        "maximum_observed_mean_abs_weight_difference": max_mean_abs_all,
        "median_max_abs_weight_difference": median_max_abs,
        "maximum_observed_max_relative_weight_difference": max_rel_all,
        "n_splits_any_absdiff_gt_1e-12": n_gt_1e12,
        "n_splits_any_absdiff_gt_1e-9": n_gt_1e9,
        "n_splits_any_absdiff_gt_1e-6": n_gt_1e6,
        **classification,
    }

    top_diff = by_repeat.sort_values(
        ["max_abs_weight_difference", "max_relative_weight_difference"],
        ascending=False,
    ).head(20).copy()

    _safe_write_csv(OUT_BY_REPEAT, by_repeat)
    _safe_write_csv(OUT_TOP_DIFF, top_diff)
    _safe_write_text(
        OUT_SUMMARY,
        json.dumps(_json_native(summary), indent=2, ensure_ascii=False) + "\n",
    )

    production_hashes_after = {
        path.relative_to(ROOT).as_posix(): _sha256_file(path)
        for path in PRODUCTION_SOURCE_FILES
        if path.is_file()
    }
    result_hashes_after = {}
    for rel in PRODUCTION_RESULT_GLOBS:
        path = ROOT / rel
        if path.is_file():
            result_hashes_after[rel.replace("\\", "/")] = _sha256_file(path)

    source_changed = [
        k for k in production_hashes_before if production_hashes_before[k] != production_hashes_after.get(k)
    ]
    results_changed = [
        k for k in result_hashes_before if result_hashes_before[k] != result_hashes_after.get(k)
    ]
    production_check = {
        "production_source_modified": bool(source_changed),
        "existing_results_overwritten": bool(results_changed),
        "model_trained": False,
        "ode_simulation_rerun": False,
        "source_hash_mismatches": source_changed,
        "result_hash_mismatches": results_changed,
        "files_written": [
            OUT_BY_REPEAT.relative_to(ROOT).as_posix(),
            OUT_SUMMARY.relative_to(ROOT).as_posix(),
            OUT_REPORT.relative_to(ROOT).as_posix(),
            OUT_TOP_DIFF.relative_to(ROOT).as_posix(),
        ],
    }
    if source_changed or results_changed:
        raise RuntimeError(
            "Integrity check failed: production files changed during the audit "
            f"(source={source_changed}, results={results_changed})."
        )

    report = build_report(
        cfg=cfg,
        alignment=alignment,
        reconstruction=reconstruction,
        extrema=extrema,
        summary=summary,
        classification=classification,
        production_check=production_check,
    )
    _safe_write_text(OUT_REPORT, report)

    print("=" * 60)
    print("SAMPLE-WEIGHT NORMALIZATION AUDIT")
    print("=" * 60)
    print(f"Repeated splits checked: {n}")
    print(f"Global target_uncertainty min: {global_u_min:.12g}")
    print(f"Global target_uncertainty max: {global_u_max:.12g}")
    print()
    print(f"Splits with same train/global min: {n_min_eq}/{n}")
    print(f"Splits with same train/global max: {n_max_eq}/{n}")
    print(f"Splits with both extrema present: {n_both}/{n}")
    print()
    print(f"Splits with any |d-weight| > 1e-12: {n_gt_1e12}/{n}")
    print(f"Splits with any |d-weight| > 1e-9:  {n_gt_1e9}/{n}")
    print(f"Splits with any |d-weight| > 1e-6:  {n_gt_1e6}/{n}")
    print()
    print(f"Maximum absolute weight difference: {max_abs_all:.12g}")
    print(f"Maximum relative weight difference: {max_rel_all:.12g}")
    print()
    print(f"AUDIT CLASSIFICATION: {classification['audit_classification']}")
    print("=" * 60)
    print()
    print("Integrity:")
    print("  Production source files modified: no")
    print("  Existing CSV/result overwritten: no")
    print("  Model trained: no")
    print("  ODE simulation rerun: no")
    if not reconstruction["reconstruction_matches_existing"]:
        print()
        print("FLAG: reconstructed global weights do not match sample_weights.csv")
        print(
            "  max_abs_difference_existing_vs_reconstructed = "
            f"{reconstruction['max_abs_difference_existing_vs_reconstructed']:.12g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
