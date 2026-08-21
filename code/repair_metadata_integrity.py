"""Metadata / manifest / provenance repair. Does not rerun ML, ODE, Morris, or tests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent
BEFORE_INVENTORY = ROOT / "results" / "metadata_correction" / "before_sha256_inventory.json"
AUDIT_JSON = ROOT / "results" / "metadata_correction" / "metadata_correction_audit.json"
AUDIT_MD = ROOT / "results" / "metadata_correction" / "metadata_correction_audit.md"
LOCKED_PATH = ROOT / "results" / "screening" / "locked_final_config.json"
LOCKED_AS_WRITTEN = ROOT / "results" / "screening" / "locked_final_config.as_written.json"
EXECUTED_CONFIG = ROOT / "results" / "screening" / "executed_final_config.json"

DT_DETECT_NOTE = (
    "Legacy nominal field (1/6 h); actual threshold evaluation once per 0.4-h Euler step"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def inventory_index(payload: dict) -> Dict[str, dict]:
    return {row["path"]: row for row in payload["files"]}


def audit_item(
    *,
    item_id: str,
    file_path: str,
    field: str,
    old_value,
    new_value,
    reason: str,
    evidence_files: List[str],
    sha_before: Optional[str],
    sha_after: Optional[str],
) -> dict:
    return {
        "id": item_id,
        "file": file_path,
        "field": field,
        "old_value": old_value,
        "new_value": new_value,
        "reason": reason,
        "evidence_files": evidence_files,
        "sha256_before": sha_before,
        "sha256_after": sha_after,
        "no_numerical_result_or_model_output_changed": True,
    }


def preserve_locked_config(before: Dict[str, dict], items: List[dict]) -> str:
    if not LOCKED_PATH.is_file():
        raise FileNotFoundError(LOCKED_PATH)
    sha_before = before[rel(LOCKED_PATH)]["sha256"]
    if not LOCKED_AS_WRITTEN.exists():
        shutil.copy2(LOCKED_PATH, LOCKED_AS_WRITTEN)
    if sha256_file(LOCKED_PATH) != sha_before:
        raise RuntimeError("locked_final_config.json was modified; refusing to continue.")
    if sha256_file(LOCKED_AS_WRITTEN) != sha_before:
        raise RuntimeError("locked_final_config.as_written.json does not match the original lock file.")
    items.append(
        audit_item(
            item_id="locked_final_config_preserved",
            file_path=rel(LOCKED_PATH),
            field="file_bytes",
            old_value="as_written planned lock (max_tree_experts=10, RF n_estimators=500, umax_selection_policy=aspiration_then_pareto)",
            new_value="unchanged; copy saved as results/screening/locked_final_config.as_written.json",
            reason=(
                "Do not silently overwrite the original lock or disguise executed settings as a "
                "pre-evaluation lock."
            ),
            evidence_files=[rel(LOCKED_PATH), rel(LOCKED_AS_WRITTEN)],
            sha_before=sha_before,
            sha_after=sha_before,
        )
    )
    return sha_before


def write_executed_config(lock_sha: str, items: List[dict]) -> None:
    from tree_srl_benchmark import make_baseline_control_factories

    compare = load_json(ROOT / "results" / "tree_srl_benchmark" / "model_compare_manifest.json")
    repeat_meta = compare.get("repeat_metadata") or []
    experts = sorted({int(row.get("n_experts_used", -1)) for row in repeat_meta})
    rf_n_estimators = int(make_baseline_control_factories(0)["RandomForest"]().n_estimators)
    planned = load_json(LOCKED_PATH)
    payload = {
        "config_role": "executed_configuration",
        "not_a_pre_evaluation_lock": True,
        "chosen_before_final_evaluation": False,
        "note": (
            "This file records the configuration that was actually executed. It is not a "
            "pre-evaluation lock and must not replace locked_final_config.json."
        ),
        "planned_locked_config_path": rel(LOCKED_PATH),
        "planned_locked_config_as_written_path": rel(LOCKED_AS_WRITTEN),
        "planned_locked_config_sha256": lock_sha,
        "planned_configuration": {
            "source": rel(LOCKED_PATH),
            "TAR_max_tree_experts": planned["model_training_params_per_model"]["TAR"]["max_tree_experts"],
            "RandomForest_n_estimators": planned["model_training_params_per_model"]["RandomForest"]["n_estimators"],
            "optimizer_selection_rule": planned.get("selection_rule"),
            "umax_selection_policy_as_written": planned.get("umax_selection_policy"),
            "umax_selection_policy_as_written_note": (
                "The as-written lock field umax_selection_policy=aspiration_then_pareto was a "
                "write_locked_final_config hardcode bug. The planned optimizer_selection_rule is "
                "constraint_first, whose primary policy is feasible_first."
            ),
        },
        "executed_configuration": {
            "source_manifest": "results/tree_srl_benchmark/model_compare_manifest.json",
            "source_code": "tree_srl_benchmark.make_baseline_control_factories",
            "max_tree_experts": compare["run_config"]["max_tree_experts"],
            "n_experts_used_per_repeat": experts,
            "n_repeats": compare.get("n_repeats"),
            "expanded_tree_bank": compare.get("expanded_tree_bank"),
            "RandomForest_n_estimators": rf_n_estimators,
            "umax_primary_selection_from_data": (
                "feasible_first / infeasible_min_composite_penalty fallback"
            ),
        },
    }
    dump_json(EXECUTED_CONFIG, payload)
    items.append(
        audit_item(
            item_id="executed_final_config_created",
            file_path=rel(EXECUTED_CONFIG),
            field="executed_configuration",
            old_value=None,
            new_value={
                "max_tree_experts": 0,
                "n_experts_used_per_repeat": experts,
                "RandomForest_n_estimators": rf_n_estimators,
            },
            reason=(
                "Planned lock had TAR max_tree_experts=10 and RF n_estimators=500; executed "
                "benchmark used max_tree_experts=0 (21 experts/repeat) and RF n_estimators=200."
            ),
            evidence_files=[
                rel(LOCKED_PATH),
                "results/tree_srl_benchmark/model_compare_manifest.json",
                "tree_srl_benchmark.py",
            ],
            sha_before=None,
            sha_after=sha256_file(EXECUTED_CONFIG),
        )
    )


def repair_generation_summary(before: Dict[str, dict], items: List[dict]) -> None:
    from microbio_dataset import microbio_generation_provenance_fields

    path = ROOT / "data" / "microbio_raw" / "microbio_generation_summary.json"
    old = load_json(path)
    sha_before = before[rel(path)]["sha256"]
    count_keys = [
        "n_bio",
        "n_biological_groups",
        "n_threshold_vectors",
        "u_grid",
        "raw_rows",
        "controller_candidates_per_group",
        "lr_definition",
    ]
    new = {k: old[k] for k in count_keys if k in old}
    new.update(microbio_generation_provenance_fields())
    dump_json(path, new)
    items.append(
        audit_item(
            item_id="microbio_generation_summary",
            file_path=rel(path),
            field="simulator_parameters_and_experimental_anchor_provenance",
            old_value={
                k: old.get(k)
                for k in (
                    "parameterization",
                    "ecoli_growth_rate_anchor_h_inv",
                    "llactis_growth_rate_h_inv",
                    "llactis_K_P_CFU_per_mL",
                    "gamma_P_preliminary_mL_ug_inv_h_inv",
                    "gamma_pathogen_preliminary_mL_ug_inv_h_inv",
                )
            },
            new_value={
                "parameterization": new.get("parameterization"),
                "P0_CFU_per_mL": new.get("P0_CFU_per_mL"),
                "K_P_CFU_per_mL": new.get("K_P_CFU_per_mL"),
                "k_P_h_inv": new.get("k_P_h_inv"),
                "experimental_anchor_provenance": new.get("experimental_anchor_provenance"),
            },
            reason=(
                "Keep original row/group counts, use current paper_figure simulator parameters, "
                "and move 0.241/0.2522/3.35e8/0.0015/0.0046 under experimental_anchor_provenance "
                "as not simulator parameters."
            ),
            evidence_files=[rel(path), "multi_pathogen_simulator.py"],
            sha_before=sha_before,
            sha_after=sha256_file(path),
        )
    )


def repair_soft_relabel_summary(before: Dict[str, dict], items: List[dict]) -> None:
    from microbio_dataset import ODE_PARAMETERIZATION, repo_relative_path

    path = ROOT / "data" / "microbio_formal_dataset" / "soft_relabel_summary.json"
    old = load_json(path)
    sha_before = before[rel(path)]["sha256"]
    new = dict(old)
    new["parameterization"] = ODE_PARAMETERIZATION
    new["raw_microbio_csv"] = repo_relative_path(str(old.get("raw_microbio_csv") or ""))
    dump_json(path, new)
    items.append(
        audit_item(
            item_id="soft_relabel_summary",
            file_path=rel(path),
            field="parameterization, raw_microbio_csv",
            old_value={
                "parameterization": old.get("parameterization"),
                "raw_microbio_csv": old.get("raw_microbio_csv"),
            },
            new_value={
                "parameterization": new.get("parameterization"),
                "raw_microbio_csv": new.get("raw_microbio_csv"),
            },
            reason=(
                "Correct parameterization wording to the paper_figure ODE benchmark profile and "
                "replace the absolute Windows path with a repo-relative path."
            ),
            evidence_files=[rel(path)],
            sha_before=sha_before,
            sha_after=sha256_file(path),
        )
    )


def repair_ode_profile_parameters_note(before: Dict[str, dict], items: List[dict]) -> None:
    path = ROOT / "data" / "microbio_raw" / "ode_profile_parameters.csv"
    sha_before = before[rel(path)]["sha256"]
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    lines = text.splitlines(keepends=True)
    old_note = None
    for i, line in enumerate(lines):
        if ",dt_detect," not in line:
            continue
        if line.endswith("\r\n"):
            ending = "\r\n"
        elif line.endswith("\n"):
            ending = "\n"
        else:
            ending = ""
        parts = line[: len(line) - len(ending)].split(",")
        # CSV has 8 columns; notes is last and has no internal commas in this file.
        if len(parts) < 8 or parts[1] != "dt_detect":
            continue
        old_note = parts[7]
        parts[7] = DT_DETECT_NOTE
        lines[i] = ",".join(parts) + ending
        break
    if old_note is None:
        raise RuntimeError("dt_detect row not found in ode_profile_parameters.csv")
    new_text = "".join(lines)
    path.write_bytes(new_text.encode("utf-8"))
    # Numeric fields must remain byte-identical except the notes cell.
    for old_line, new_line in zip(text.splitlines(), new_text.splitlines()):
        old_parts = old_line.split(",")
        new_parts = new_line.split(",")
        if old_parts[:7] != new_parts[:7]:
            raise RuntimeError("ode_profile_parameters.csv numeric/non-note fields changed; aborting.")
    items.append(
        audit_item(
            item_id="ode_profile_parameters_dt_detect_note",
            file_path=rel(path),
            field="notes[code_name=dt_detect]",
            old_value=old_note,
            new_value=DT_DETECT_NOTE,
            reason=(
                "1/6 h is a legacy nominal field; threshold evaluation is once per 0.4-h Euler step. "
                "Only the notes cell is rewritten in-place; numeric columns remain byte-identical."
            ),
            evidence_files=[rel(path)],
            sha_before=sha_before,
            sha_after=sha256_file(path),
        )
    )


def repair_uncertainty_manifest(before: Dict[str, dict], items: List[dict]) -> None:
    from tree_srl_benchmark import apply_uncertainty_torch_used_fields

    path = ROOT / "results" / "tree_srl_benchmark" / "uncertainty_manifest.json"
    old = load_json(path)
    sha_before = before[rel(path)]["sha256"]
    new = dict(old)
    old_torch = new.get("torch_used")
    apply_uncertainty_torch_used_fields(new, new.get("repeat_manifests") or [])
    dump_json(path, new)
    items.append(
        audit_item(
            item_id="uncertainty_torch_used",
            file_path=rel(path),
            field="torch_used / per_repeat_execution_status",
            old_value={"torch_used": old_torch},
            new_value={
                "torch_used": new.get("torch_used"),
                "n_repeats_mc_dropout_completed": new.get("n_repeats_mc_dropout_completed"),
                "n_repeats_skipped_no_torch": new.get("n_repeats_skipped_no_torch"),
            },
            reason=(
                "Per-repeat MC-dropout completed with uncertainty_skipped_no_torch=false and "
                "mc_inference_mode=batched_gpu; aggregate torch_used=false was misleading."
            ),
            evidence_files=[rel(path)],
            sha_before=sha_before,
            sha_after=sha256_file(path),
        )
    )


def repair_fig5_aggregate_manifest(before: Dict[str, dict], items: List[dict]) -> None:
    from closed_loop_eval import write_aggregate_fig5_plot_manifest

    path = ROOT / "results" / "umax_optimization" / "fig5_plot_manifest.json"
    old = load_json(path)
    sha_before = before[rel(path)]["sha256"]
    counts = pd.read_csv(ROOT / "results" / "umax_optimization" / "umax_policy_ablation_condition_counts.csv")
    n_repeats = int(counts["n_repeats"].max())
    last_study = ROOT / "results" / "umax_optimization" / "repeats" / "repeat_099" / "umax_study"
    write_aggregate_fig5_plot_manifest(
        str(path.parent),
        n_repeats=n_repeats,
        significance_for_manuscript=n_repeats >= 10,
        last_repeat_study_dir=str(last_study) if last_study.is_dir() else None,
        generated_pngs=old.get("generated_pngs"),
        primary_outputs=old.get("primary_outputs") or {},
    )
    new = load_json(path)
    items.append(
        audit_item(
            item_id="fig5_plot_manifest_reconstructed",
            file_path=rel(path),
            field="n_repeats, primary_selection, representative_trajectory_flags, aspiration_sensitivity",
            old_value={
                "n_repeats": old.get("n_repeats"),
                "manuscript_supplementary_composites_keys": list((old.get("manuscript_supplementary_composites") or {}).keys()),
                "panel_d_n_repeats": (old.get("figure_panel_mapping") or [{}])[-1].get("n_repeats") if old.get("figure_panel_mapping") else None,
            },
            new_value={
                "n_repeats": new.get("n_repeats"),
                "primary_selection_recorded_from_data": new.get("primary_selection_recorded_from_data"),
                "n_representative_cases": new.get("n_representative_cases"),
                "illustrative_only_ablation_trajectories": new.get("illustrative_only_ablation_trajectories"),
                "aspiration_selection_policy_sensitivity_executed": new.get(
                    "aspiration_selection_policy_sensitivity_executed"
                ),
                "manuscript_source_mapping_s4": {
                    k: new.get("manuscript_source_mapping", {}).get(k)
                    for k in ("umax_ode_ablation.png", "umax_summary_ablation.png")
                },
            },
            reason=(
                "Aggregate fig5_plot_manifest was a copy of the last repeat (n_repeats=1). "
                "Rebuild from aggregate artifacts: Fig. 5d and S4b n_repeats=100; representative "
                "trajectory n_representative_cases=1 / illustrative_only; primary selection from "
                "feasible_region_summary; do not claim aspiration sensitivity without a non-empty CSV."
            ),
            evidence_files=[
                rel(path),
                "results/umax_optimization/umax_policy_ablation_condition_counts.csv",
                "results/umax_optimization/umax_feasible_region_summary.csv",
            ],
            sha_before=sha_before,
            sha_after=sha256_file(path),
        )
    )


def repair_manuscript_source_mappings(before: Dict[str, dict], items: List[dict]) -> None:
    from figure_audit import MANUSCRIPT_COMPOSITE_PANEL_SOURCES, MANUSCRIPT_SOURCE_MAPPING

    mapping = {k: dict(v) for k, v in MANUSCRIPT_SOURCE_MAPPING.items()}

    compare_path = ROOT / "results" / "tree_srl_benchmark" / "model_compare_manifest.json"
    compare = load_json(compare_path)
    sha_before = before[rel(compare_path)]["sha256"]
    old_map = compare.get("manuscript_source_mapping")
    compare["manuscript_source_mapping"] = mapping
    dump_json(compare_path, compare)
    items.append(
        audit_item(
            item_id="model_compare_manuscript_source_mapping",
            file_path=rel(compare_path),
            field="manuscript_source_mapping",
            old_value=old_map,
            new_value=mapping,
            reason="S1=Morris, S2=uncertainty, S3=direct threshold, S4=Umax ablation.",
            evidence_files=[rel(compare_path), "figure_audit.py"],
            sha_before=sha_before,
            sha_after=sha256_file(compare_path),
        )
    )

    bench_path = ROOT / "results" / "tree_srl_benchmark" / "benchmark_plot_manifest.json"
    bench = load_json(bench_path)
    sha_before = before[rel(bench_path)]["sha256"]
    old_bench_map = bench.get("manuscript_source_mapping")
    bench["manuscript_source_mapping"] = {
        k: dict(v)
        for k, v in mapping.items()
        if v.get("source_group") in {"tree_srl_benchmark", "ode_back_validation"}
        and v.get("manuscript_composite") in {"Fig3.png", "Fig4.png", "supp_fig2.png", "supp_fig3.png"}
    }
    stats = dict(bench.get("statistics_used") or {})
    if "uncertainty_decomposition.png" in stats:
        stats["uncertainty_decomposition.png"] = (
            "supp_fig2.png: aleatoric vs epistemic std (4×3 layout; pooled and per Tthr)"
        )
    bench["statistics_used"] = stats
    dump_json(bench_path, bench)
    items.append(
        audit_item(
            item_id="benchmark_plot_manuscript_source_mapping",
            file_path=rel(bench_path),
            field="manuscript_source_mapping, statistics_used.uncertainty_decomposition.png",
            old_value=old_bench_map,
            new_value=bench["manuscript_source_mapping"],
            reason="Uncertainty scatter is S2 (supp_fig2.png), not S1.",
            evidence_files=[rel(bench_path)],
            sha_before=sha_before,
            sha_after=sha256_file(bench_path),
        )
    )

    morris_path = ROOT / "results" / "mu_sensitivity" / "mu_morris_manifest.json"
    morris = load_json(morris_path)
    sha_before = before[rel(morris_path)]["sha256"]
    old_composite = morris.get("manuscript_composite")
    morris["manuscript_composite"] = "supp_fig1.png"
    morris["manuscript_composite_panel_sources"] = [
        {
            "panel": panel,
            "filename": filename,
            "description": MANUSCRIPT_SOURCE_MAPPING[filename]["description"] if filename else description,
            "source_kind": "code_generated_source_image",
            "source_group": "mu_sensitivity",
            "manual_schematic_required": False,
        }
        for panel, filename, description in MANUSCRIPT_COMPOSITE_PANEL_SOURCES["supp_fig1.png"]
    ]
    morris["manuscript_source_mapping"] = {"mu_morris_summary.png": dict(mapping["mu_morris_summary.png"])}
    dump_json(morris_path, morris)
    items.append(
        audit_item(
            item_id="morris_manuscript_source_mapping",
            file_path=rel(morris_path),
            field="manuscript_composite",
            old_value=old_composite,
            new_value="supp_fig1.png",
            reason="S1 is Morris (mu_morris_summary.png), not Umax ablation.",
            evidence_files=[rel(morris_path)],
            sha_before=sha_before,
            sha_after=sha256_file(morris_path),
        )
    )

    ode_path = ROOT / "results" / "ode_back_validation" / "ode_back_validation_manifest.json"
    ode = load_json(ode_path)
    sha_before = before[rel(ode_path)]["sha256"]
    old_ode_map = ode.get("manuscript_source_mapping")
    ode["manuscript_source_mapping"] = {
        "ode_back_r2_barplot.png": dict(mapping["ode_back_r2_barplot.png"]),
        "direct_threshold_comparison.png": dict(mapping["direct_threshold_comparison.png"]),
    }
    dump_json(ode_path, ode)
    items.append(
        audit_item(
            item_id="ode_back_manuscript_source_mapping",
            file_path=rel(ode_path),
            field="manuscript_source_mapping",
            old_value=old_ode_map,
            new_value=ode["manuscript_source_mapping"],
            reason="S3 is the direct-threshold comparison; Fig4A remains ODE-back R2.",
            evidence_files=[rel(ode_path)],
            sha_before=sha_before,
            sha_after=sha256_file(ode_path),
        )
    )


def verify_result_csvs_unchanged(before: Dict[str, dict]) -> List[str]:
    mismatches: List[str] = []
    for path_str, row in before.items():
        if not path_str.startswith("results/") or not path_str.lower().endswith(".csv"):
            continue
        path = ROOT / path_str
        if not path.is_file():
            mismatches.append(f"MISSING {path_str}")
            continue
        current = sha256_file(path)
        if current != row["sha256"]:
            mismatches.append(f"{path_str}: before={row['sha256']} after={current}")
    return mismatches


def verify_png_svg_unchanged(before: Dict[str, dict]) -> List[str]:
    mismatches: List[str] = []
    for path_str, row in before.items():
        if not path_str.startswith("results/"):
            continue
        if not path_str.lower().endswith((".png", ".svg")):
            continue
        path = ROOT / path_str
        if not path.is_file():
            mismatches.append(f"MISSING {path_str}")
            continue
        current = sha256_file(path)
        if current != row["sha256"]:
            mismatches.append(f"{path_str}: before={row['sha256']} after={current}")
    return mismatches


def write_audit(items: List[dict], csv_ok: bool, figure_ok: bool) -> None:
    payload = {
        "title": "Metadata / manifest / provenance integrity correction",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "no_numerical_result_or_model_output_changed": True,
        "ml_ode_morris_bootstrap_permutation_rerun": False,
        "before_inventory": rel(BEFORE_INVENTORY),
        "result_csv_sha256_verified_identical": csv_ok,
        "result_png_svg_sha256_verified_identical": figure_ok,
        "items": items,
    }
    dump_json(AUDIT_JSON, payload)
    lines = [
        "# Metadata correction audit",
        "",
        "No numerical result or model output changed. ML, ODE, Morris, bootstrap, and permutation tests were not rerun.",
        "",
        f"- Result CSV SHA256 vs before inventory: **{'PASS' if csv_ok else 'FAIL'}**",
        f"- Result PNG/SVG SHA256 vs before inventory: **{'PASS' if figure_ok else 'FAIL'}**",
        "",
    ]
    for item in items:
        lines.extend(
            [
                f"## {item['id']}",
                "",
                f"- File: `{item['file']}`",
                f"- Field: `{item['field']}`",
                f"- Old value: `{json.dumps(item['old_value'], ensure_ascii=False)[:800]}`",
                f"- New value: `{json.dumps(item['new_value'], ensure_ascii=False)[:800]}`",
                f"- Reason: {item['reason']}",
                f"- Evidence: {', '.join(f'`{p}`' for p in item['evidence_files'])}",
                f"- SHA256 before: `{item['sha256_before']}`",
                f"- SHA256 after: `{item['sha256_after']}`",
                "- no numerical result or model output changed",
                "",
            ]
        )
    AUDIT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not BEFORE_INVENTORY.is_file():
        raise FileNotFoundError(f"Missing before inventory: {BEFORE_INVENTORY}")
    before_payload = load_json(BEFORE_INVENTORY)
    before = inventory_index(before_payload)
    items: List[dict] = []
    lock_sha = preserve_locked_config(before, items)
    write_executed_config(lock_sha, items)
    repair_generation_summary(before, items)
    repair_soft_relabel_summary(before, items)
    repair_ode_profile_parameters_note(before, items)
    repair_uncertainty_manifest(before, items)
    repair_fig5_aggregate_manifest(before, items)
    repair_manuscript_source_mappings(before, items)

    csv_mismatches = verify_result_csvs_unchanged(before)
    figure_mismatches = verify_png_svg_unchanged(before)
    csv_ok = not csv_mismatches
    figure_ok = not figure_mismatches
    write_audit(items, csv_ok, figure_ok)
    if csv_mismatches:
        print("FAIL: non-metadata result CSV SHA256 changed:")
        for line in csv_mismatches:
            print(" ", line)
        return 1
    if figure_mismatches:
        print("FAIL: result PNG/SVG SHA256 changed:")
        for line in figure_mismatches:
            print(" ", line)
        return 1
    print(f"PASS: {len(before_payload['files'])} inventoried files; all results/*.csv and PNG/SVG unchanged.")
    print(f"Audit: {rel(AUDIT_JSON)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
