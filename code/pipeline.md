# Pipeline — Manuscript Formal Workflow

Run commands from the **project root** in PowerShell, in order. Manuscript outputs live under `.\results\`.

---

## Environment setup (first run)

```powershell
pip install numpy pandas scipy scikit-learn matplotlib joblib pyyaml numba lightgbm xgboost SALib
```


| Package                                    | Role                                                              |
| ------------------------------------------ | ----------------------------------------------------------------- |
| `numpy`, `pandas`, `scipy`, `scikit-learn` | Data handling, models, statistical tests                          |
| `matplotlib`                               | All figures (`figure_audit.py`; exports **PNG + SVG**)            |
| `joblib`                                   | Parallel ODE-back / benchmark                                     |
| `pyyaml`                                   | Read `analysis_plan\parameter_screening_plan.yaml`                |
| `numba`                                    | Fast ODE (steps 6 / 8 / 10; `--backend auto`)                     |
| `lightgbm`                                 | TAR benchmark trees (step 5; `--lgbm_device gpu` needs GPU build) |
| `xgboost`                                  | Some `multi_pathogen_simulator.py` scripts                        |
| `SALib`                                    | Morris elementary effects (step 4b `--mode mu_sensitivity`)       |


**Optional (GPU relabel scoring):** `pip install torch` — only for step 2 with `--device gpu`; otherwise use `--device cpu`.

Verify versions:

```powershell
python -c "import sklearn, numpy, scipy, matplotlib, yaml, numba, lightgbm; print('ok', sklearn.__version__, numpy.__version__)"
```

---

## Execution overview

```
Phase A — Data preparation
 1. Generate dataset              → data\microbio_raw\MICROBIO.csv

Phase B — Relabel + hyperparameter selection (choose one; see step 2)
    Route A or B                   → data\microbio_formal_dataset\ (canonical formal dataset)
                                     + Route A also writes results\screening\

Phase C — Formal evaluation (Stage 1)
 3. Heatmap                        → results\heatmap\
 4. Representative ODE             → results\ode\
 4b. Morris μ sensitivity (opt.)   → results\mu_sensitivity\
 5. Benchmark + uncertainty        → results\tree_srl_benchmark\
 5b. Umax weight sensitivity (opt.) → results\screening\umax_weights\
 6. ODE-back validation             → results\ode_back_validation\
 7. Generate Fig. 3                 → results\tree_srl_benchmark\figure\
 8. Fixed-Umax validation           → results\fixed_umax_validation\
 9. Generate Fig. 4                 → results\fixed_umax_validation\figure\
10. Umax optimization              → results\umax_optimization\
11. Generate Fig. 5                → results\umax_optimization\figure\
```

**Step 2 is the fork:** if you run parameter screening (Route A), you do **not** need manual relabel (Route B).

---

## Core rules


| Item            | Rule                                                                   |
| --------------- | ---------------------------------------------------------------------- |
| **Final model** | **TAR** (single-tree experts + target-wise OOF stack)                  |
| **Baselines**   | **RF**, **BestTree**, **UniformTreeMean**                              |
| **Split**       | `--split_mode group`                                                   |
| **Repeats**     | Manuscript: `--n_repeats 100`                                          |
| **Figures**     | Always via `figure_audit.py` (each plot → `.png` + `.svg`)             |
| **Data**        | `data\microbio_formal_dataset\` (after step 2; both routes write here) |


### Python modules (9 files)


| File                             | Role                                                                                                                               |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `microbio_dataset.py`            | Steps 1–2: data generation, Tthr relabel bundle; **Stage-0 screening** (`run_parameter_screening_pipeline`)                        |
| `multi_pathogen_simulator.py`    | Step 4: representative ODE trajectories; **step 4b** Morris μ sensitivity (`--mode mu_sensitivity`)                               |
| `simulate_case_metrics_fast.py`  | Metrics-only ODE (internal fast path for steps 6 / 8 / 10)                                                                         |
| `heatmap.py`                     | Step 3: correlation heatmaps                                                                                                       |
| `tree_srl_benchmark.py`          | Step 5: TAR benchmark (candidate-score alignment, uncertainty decomposition); training screening                                   |
| `ode_back_validation.py`         | Step 6: ODE-back functional validation                                                                                             |
| `figure_audit.py`                | Steps 4 / 7 / 9 / 11: figure generation + audit                                                                                    |
| `closed_loop_eval.py`            | Steps 5b / 8 / 10b: Umax weight screening, closed-loop / Umax optimizer; CLI `--parameter_screening` / `--locked_final_evaluation` |
| `derive_optimizer_references.py` | Step 10a: dose reference, fixed Umax policies; **Fig. 5B** (`build_umax_objective_alignment`)                                      |


> Still **9** `.py` **files**. Removed duplicates: Fig. 4 plot wrapper (now `figure_audit`), Umax alignment + screening orchestration (merged into `derive_optimizer_references` / `microbio_dataset`). **Computation logic unchanged.**

### Verified environment


| Package      | Version |
| ------------ | ------- |
| Scikit-learn | 1.9.0   |
| NumPy        | 2.2.1   |
| SciPy        | 1.18.0  |


Manuscript numbers were validated in this environment:

```powershell
python -c "import sklearn, numpy, scipy; print(f'Scikit-learn: {sklearn.__version__}'); print(f'NumPy: {numpy.__version__}'); print(f'SciPy: {scipy.__version__}')"
```

---

## Prerequisites (prepare before running)


| File                                          | Notes                                                                                                                                                             |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `analysis_plan\parameter_screening_plan.yaml` | **Version-controlled** screening config (training CV grid, Umax weight profiles, lock paths). **Not** auto-generated; edit this file to change grids or profiles. |
| PyYAML                                        | `pip install pyyaml`                                                                                                                                              |


Stage 0 **produces** (not a prerequisite): everything under `results\screening\` (`locked_final_config.json`, `fairness_check.json`, `parameter_screening_manifest.json`).

---

## 1 — Generate dataset

```powershell
python .\microbio_dataset.py `
  --mode generate_dataset `
  --n_bio 500 `
  --n_threshold_vectors 16 `
  --outdir .\data\microbio_raw
```

**Outputs:** `data\microbio_raw\MICROBIO.csv`, `ode_profile_parameters.csv`, `ode_parameter_sources.csv`, `microbio_generation_summary.json`

**ODE profile:** `experimental_calibrated` (E. coli / L. lactis growth-curve calibration; pathogen `gamma_s=0.0046`, probiotic `gamma_P=0.0015` preliminary estimate).

**Note:** Stage 0 and all later steps depend on this raw library. If missing, parameter screening raises `FileNotFoundError`. If you generated the raw CSV with an old ODE profile, delete `data\microbio_raw\` and re-run step 1.

---

## 2 — Relabel dataset + parameter screening (choose one)

**Goal:** formal training dataset → `data\microbio_formal_dataset\` (steps 3+ read from here).

### Route A (recommended)

```powershell
python .\closed_loop_eval.py `
  --parameter_screening `
  --screening_stage all `
  --screening_plan .\analysis_plan\parameter_screening_plan.yaml `
  --microbio_csv .\data\microbio_raw\MICROBIO.csv
```

**Outputs:** `data\microbio_formal_dataset\` + `results\screening\` (locked config, training screening figures). Then proceed to step 3.

**Screened parameters:** training hyperparameter grid (locks TAR / RF settings for steps 5–6). Umax weight profiles are screened separately in **step 5b** (optional).

### Route B (quick run, skip screening)

```powershell
python .\microbio_dataset.py `
  --mode relabel_tthr_bundle `
  --microbio_csv .\data\microbio_raw\MICROBIO.csv `
  --bundle_outdir .\data\microbio_relabel_tthr_bundle `
  --fixed_top_k 8 `
  --acceptable_threshold 12.0 `
  --drop_infeasible_profiles `
  --device gpu `
  --chunk_size 512 `
  --lr_metric terminal_median `
  --pauc_quantiles 0.25,0.50,0.75 `
  --lr_quantiles 0.25,0.50,0.75 `
  --w_pauc 3.0 --w_dose 0.05 --w_threshold 0.01 --w_constraint 1.0 `
  --c_ref 25.0
```

Without GPU: `--device cpu --n_jobs 4`. Writes `data\microbio_formal_dataset\` (`fixed_k8` relabel only).

> Screening figure details: appendix **Screening figures**.

---

## Stage 1 — Formal evaluation (steps 3–11)

After Route A, run steps 3–11 in order. All commands use `data\microbio_formal_dataset\` (`heatmap.py` defaults to this path; `--x_csv` can be omitted).

Add `--locked_final_evaluation` to `closed_loop_eval.py` in steps 8 / 10 to force `results\screening\locked_final_config.json`.

## 3 — Heatmap

```powershell
python .\heatmap.py `
  --outdir .\results\heatmap `
  --method both `
  --subset input

python .\heatmap.py `
  --y_csv .\data\microbio_formal_dataset\y_targets.csv `
  --outdir .\results\heatmap `
  --method both `
  --subset feature_controller
```

**Outputs:** correlation CSV + **PNG + SVG** under `results\heatmap\`

---

## 4 — Representative ODE

```powershell
python .\multi_pathogen_simulator.py `
  --mode representative `
  --profile paper_figure `
  --outdir .\results\ode

python .\figure_audit.py --mode generate_plots --groups ode --ode_outdir .\results\ode
```

**Outputs:** trajectory CSV, manifest, `representative_paper_figure.png` + `.svg`

---

## 4b — Morris μ sensitivity (optional; development only)

**When:** after **step 4** (or anytime the ODE profile is available). Does **not** regenerate the supervised dataset and does **not** train any ML model. Not required for the manuscript mainline (steps 5–11).

**Goal:** SALib Morris elementary-effects analysis over the 31 raw-library factors (including μ₁–μ₅), using the formal raw-library parameter bounds and `simulate_case`.

```powershell
python .\multi_pathogen_simulator.py `
  --mode mu_sensitivity `
  --profile paper_figure `
  --outdir .\results\mu_sensitivity
```

**Outputs:** `mu_morris_all_factors.csv`, `mu_morris_indices.csv`, `mu_morris_summary.png` + `.svg`, `mu_morris_manifest.json` under `results\mu_sensitivity\`

**Settings:** 64 trajectories, 4 grid levels, seed `2026`.

---

## 5 — Benchmark TAR predictions

```powershell
python .\tree_srl_benchmark.py `
  --x_csv .\data\microbio_formal_dataset\X_features.csv `
  --y_csv .\data\microbio_formal_dataset\y_targets.csv `
  --metadata_csv .\data\microbio_formal_dataset\sample_metadata.csv `
  --sample_weight_csv .\data\microbio_formal_dataset\sample_weights.csv `
  --candidate_score_csv .\data\microbio_formal_dataset\candidate_score_table.csv `
  --outdir .\results\tree_srl_benchmark `
  --split_mode group `
  --group_col bio_id `
  --test_size 0.15 `
  --seed 42 `
  --n_repeats 100 `
  --n_jobs 10 `
  --target_transform log `
  --expanded_tree_bank `
  --max_tree_experts 0 `
  --enable_uncertainty_decomposition `
  --show_uncertainty_main `
  --lgbm_device gpu
```

**Outputs:** `model_compare_summary.csv`, `repeats\repeat_XXX\predictions.csv`, `predictions_manifest.json`; with uncertainty → `uncertainty_decomposition.csv`

**Notes:**

- Manuscript: `--n_repeats 100` (use `10` for smoke tests); formal significance needs `n_repeats >= 10`
- After Route A, align with `results\screening\locked_final_config.json` training settings (`expanded_tree_bank`, `max_tree_experts`, RF hyperparameters); commands above are manuscript defaults
- No GPU: `--lgbm_device cpu`
- Figures are step 7 (Fig. 3), not here

---

## 5b — Umax weight profile sensitivity (optional; development only)

**When:** after **step 5**, before **step 10** (formal Umax optimization). Requires benchmark predictions; does **not** change `locked_final_config.json` (manuscript still locks `balanced`).

**Goal:** compare four closed-loop weight profiles (`balanced`, `efficacy`, `probiotic_sparing`, `dose_sparing`) on penalty / dose outcomes. Skip this step if you only need the manuscript mainline (steps 6–11).

```powershell
python .\closed_loop_eval.py `
  --parameter_screening `
  --screening_stage umax_weights `
  --screening_plan .\analysis_plan\parameter_screening_plan.yaml `
  --run_umax_weight_sensitivity `
  --predictions_dir .\results\tree_srl_benchmark\repeats `
  --predictions_manifest .\results\tree_srl_benchmark\predictions_manifest.json `
  --x_csv .\data\microbio_formal_dataset\X_features.csv `
  --n_jobs 10

python .\closed_loop_eval.py --parameter_screening --screening_stage figures
```

**Outputs:** `results\screening\umax_weights\umax_weight_profile_sensitivity.csv`, case-level CSV, per-profile study subdirs; refreshed screening figures under `results\screening\` (see appendix **Screening figures**).

**Runtime (optimized):** Step 5b uses the `sensitivity` block in `parameter_screening_plan.yaml` — by default **10 evenly spaced repeats** (not 100), **coarser u_grid** (`step=2`), **TAR/RF optimized only** (skips fixed-policy ablations), **shared dose references** across four profiles, parallel repeat workers via `--n_jobs`, and **resume** via `skip_completed_repeats`. Typical wall time: **~30–90 min** with `--n_jobs 10` (vs many hours/days for the old full-study path). Step 10 still uses the full `u_grid` and 100 repeats.

If you started an old slow 5b run, stop it and delete `results\screening\umax_weights\` before re-running.

**Note:** Optional development sensitivity only. Steps 6–9 do not depend on 5b; step 10 uses locked `balanced` weights unless you override with `--allow_experimental_override`.

---

## 6 — ODE-back functional validation

```powershell
python .\ode_back_validation.py `
  --predictions_dir .\results\tree_srl_benchmark\repeats `
  --predictions_manifest .\results\tree_srl_benchmark\predictions_manifest.json `
  --x_csv .\data\microbio_formal_dataset\X_features.csv `
  --y_csv .\data\microbio_formal_dataset\y_targets.csv `
  --metadata_csv .\data\microbio_formal_dataset\sample_metadata.csv `
  --outdir .\results\ode_back_validation `
  --reference_mode reference_tthr `
  --functional_umax_source metadata_soft_umax `
  --models TAR,RandomForest,BestSingleTree,UniformTreeMean `
  --backend numba `
  --n_jobs 10 `
  --validate_fast_backend `
  --include_direct_threshold_baseline
```

**Outputs:** `ode_back_summary_by_model.csv`, `ode_back_per_outcome_metrics.csv`, `figure\ode_back_outcome_heatmap.png` + `.svg`; with `--include_direct_threshold_baseline` also `direct_threshold_case_results.csv`, `direct_threshold_summary.csv`, `direct_threshold_support_audit.csv`, `direct_threshold_comparison.png` + `.svg`

**Note:** No retraining; no Umax optimization; requires step 5. Direct-threshold baselines (`DirectRuleUnclipped` / `DirectRuleClipped`) use `Tthr_i = B0_i * 10**(-LR_i_target)` on the same held-out rows as TAR.

---

## 7 — Generate Fig. 3

```powershell
python .\figure_audit.py --mode generate_plots --groups benchmark --benchmark_outdir .\results\tree_srl_benchmark
```

**Outputs:** `results\tree_srl_benchmark\figure\` (PNG + SVG); Fig. 3D heatmap in `results\ode_back_validation\figure\`

**Note:** `n_repeats >= 10` for formal significance stars

---

## 8 — Fixed-Umax validation (Fig. 4 data)

```powershell
python .\closed_loop_eval.py `
  --predictions_dir .\results\tree_srl_benchmark\repeats `
  --predictions_manifest .\results\tree_srl_benchmark\predictions_manifest.json `
  --x_csv .\data\microbio_formal_dataset\X_features.csv `
  --metadata_csv .\data\microbio_formal_dataset\sample_metadata.csv `
  --outdir .\results\fixed_umax_validation `
  --u_grid arange:0:100:0.5 `
  --target_total_dosage 2500 `
  --target_terminal_pathogen 4e7 `
  --weight_profile custom `
  --w_track 1.0 `
  --w_path 1.0 `
  --w_probiotic 1.0 `
  --w_dose 1.0 `
  --verbose
```

**Outputs:** `fixed_umax_representative_trajectories.csv`, `fixed_umax_validation_manifest.json`, `*_repeated_stats.csv`

**Note:** Do **not** pass `--run_umax_optimization_study`. `fixed_umax_validation_cases.csv` must have exactly **3 rows** (TAR / BestTree / UniformTreeMean).

---

## 9 — Generate Fig. 4

```powershell
python .\figure_audit.py --mode generate_plots --groups fixed_umax_validation --fixed_umax_outdir .\results\fixed_umax_validation
```

**Outputs:** `results\fixed_umax_validation\figure\` (PNG + SVG)

**Note:** Requires step 8 and `fixed_umax_representative_trajectories.csv`. Fig. 4B shows **single forward ODE point estimates** (not 100-repeat CIs).

---

## 10 — Umax optimization (Fig. 5 data)

Run **10a** (training-only fixed Umax policies), then **10b** (policy ablation). **Route A (manuscript):** add `--locked_final_evaluation` to **10b** so `u_grid`, `balanced` weights, and `feasible_first` selection come from `results\screening\locked_final_config.json` — do **not** pass conflicting `--weight_profile`, `--w_*`, `--u_grid`, or `--umax_selection_policy` on the CLI. Optional **step 5b** (Umax weight sensitivity) should be finished before 10 if you want those development figures; it does not gate step 10.

**Module role:** TAR predicts **Tthr only** (step 5). **10b** is post-prediction **grid-based Umax inverse-design**: forward ODE reinsertion over the Umax grid, `feasible_first` primary selection (lowest-dose feasible candidate; else lowest composite-penalty fallback). `aspiration_then_pareto` is exported only in `umax_selection_policy_sensitivity.csv` for reviewer sensitivity.

### 10a — derive_optimizer_references

```powershell
python .\derive_optimizer_references.py `
  --predictions_dir .\results\tree_srl_benchmark\repeats `
  --predictions_manifest .\results\tree_srl_benchmark\predictions_manifest.json `
  --x_csv .\data\microbio_formal_dataset\X_features.csv `
  --y_csv .\data\microbio_formal_dataset\y_targets.csv `
  --metadata_csv .\data\microbio_formal_dataset\sample_metadata.csv `
  --outdir .\results\umax_optimization `
  --u_grid arange:0:101:1 `
  --dose_reference_quantile 0.90 `
  --fixed_representative_umax 18.0 `
  --backend auto `
  --n_jobs 10 `
  --skip_if_complete
```

**10a** writes `optimizer_reference_by_repeat.csv` and `fixed_umax_policy_by_repeat.csv`. **10b** reuses them when present (no duplicate derivation). `--skip_if_complete` skips 10a if both files already cover all repeats.

### 10b — closed_loop_eval (main Fig. 5 evaluation)

**Manuscript (Route A — recommended):**

```powershell
python .\closed_loop_eval.py `
  --predictions_dir .\results\tree_srl_benchmark\repeats `
  --predictions_manifest .\results\tree_srl_benchmark\predictions_manifest.json `
  --x_csv .\data\microbio_formal_dataset\X_features.csv `
  --y_csv .\data\microbio_formal_dataset\y_targets.csv `
  --metadata_csv .\data\microbio_formal_dataset\sample_metadata.csv `
  --outdir .\results\umax_optimization `
  --pauc_feasibility_fraction 0.90 `
  --pathogen_ceiling 4e7 `
  --dose_reference_source training_q90_reference_dosage `
  --locked_final_evaluation `
  --backend auto `
  --run_umax_optimization_study `
  --n_jobs 10 `
  --verbose
```

`--locked_final_evaluation` applies from `locked_final_config.json`: `u_grid arange:0:101:1`, `umax_weight_profile balanced` (`w_dose 0.25`), `umax_selection_policy feasible_first`. Omit `--weight_profile`, `--w_*`, `--u_grid`, and `--umax_selection_policy` here — they conflict with the lock and will raise `RuntimeError`.

**Exploratory override (not for manuscript):** add `--allow_experimental_override` and set CLI flags explicitly; outputs are marked not_for_manuscript.

**Re-run after policy / code changes:** add `--force_rerun` (otherwise completed repeats are skipped).

**Outputs:** `results\umax_optimization\` including:

- `umax_policy_ablation_cases.csv`, `umax_ablation_repeated_plot_stats.csv`
- `umax_response_landscape.csv` (canonical per-candidate ODE outcomes + selection metadata)
- `umax_feasible_region_summary.csv`, `umax_optimization_u_candidates.csv`
- `umax_score_landscape_curves.csv` (`optimization_penalty_score` aliases `composite_penalty`)
- `umax_selection_policy_sensitivity.csv` (feasible_first vs aspiration_then_pareto)
- `umax_objective_alignment.csv`, representative trajectories, `umax_optimization_manifest.json`

**Note:** Still the slowest manuscript step (100 repeats × full ablation). Do not mix outdir with Fig. 4. `--n_jobs` parallelizes repeats. Run **10a before 10b** so reference/policy derivation is not repeated inside 10b (or rely on existing 10a CSVs in the same outdir).

> Fig. 5 panel map, reference rules, and acceleration: appendix **Fig. 5 notes**.

---

## 11 — Generate Fig. 5 (step 10c)

```powershell
python .\figure_audit.py --mode generate_plots --groups umax_optimization --umax_optimization_outdir .\results\umax_optimization
```

**Outputs:** `results\umax_optimization\figure\` (PNG + SVG)

- `umax_score_landscape` — composite-penalty response landscape vs Umax (selected markers; aspiration distance available as diagnostic in CSV)
- `umax_score_components_landscape` — four normalized penalty / shortfall terms
- `umax_objective_alignment` — Dose reference limit / P_AUC constraint limit / LR feasibility / Pathogen feasibility / Final selected Umax
- `umax_ode_ablation` — three-column trajectories (raises if any column missing)
- `umax_summary_ablation` — Fig. 5D primary (three TAR policies)
- `umax_summary_ablation_with_rf` — secondary (TAR + RF optimized / tuned global)

---

## Appendix

### Audit and regenerate all figures

```powershell
python .\figure_audit.py --mode audit_figures --results_root .\results --groups ode,heatmap,benchmark,fixed_umax_validation,umax_optimization

python .\figure_audit.py --mode generate_plots --groups all --results_root .\results
```

Each generated figure is written as **both** `.png` (300 dpi) and `.svg` (vector) with the same basename.

### Fig. 4 behavior

- Reads per-repeat Tthr predictions from step 5; aggregates one median Tthr per model
- Runs paper_figure + fixed Umax 18 µg/mL forward ODE once per model (TAR / BestTree / UniformTreeMean) — **3 ODE runs total**
- Reading predictions does **not** run ODE per test row

```powershell
# Confirm case count
Import-Csv .\results\fixed_umax_validation\fixed_umax_validation_cases.csv | Measure-Object
```

If cases >> 3: wrong step (Fig. 5) or old logic — delete outdir and re-run with `--force_rerun`

### Screening figures (`results\screening\`)

Development-only records, **not** Fig. 3–5. Old `parameter_screening_summary.png` and its code are removed.


| Figure / table                                                     | Content                                                                  |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| `training_parameter_screening.png` / `.svg`                        | Full TAR / RF inner-CV grid (red = selected)                             |
| `training_parameter_screening_marginals.png` / `.svg`              | Marginal R² distributions per swept dimension                            |
| `umax_weight_profile_definitions.png` / `.svg`                     | Four weight-profile coefficient heatmap                                  |
| `umax_weights/umax_weight_profile_sensitivity.png` / `.svg`        | Per-profile penalty / dose means (needs `--run_umax_weight_sensitivity`) |
| `umax_weights/umax_weight_profile_case_distributions.png` / `.svg` | Case-level distributions (same)                                          |
| `parameter_screening_table.csv`                                    | Locked summary (training + umax weights)                                 |
| `locked_final_config.json`                                         | Locked params for Stage 2                                                |


**Parameters that affect downstream results:**

- **Training:** TAR `expanded_tree_bank` × `max_tree_experts`, RF `n_estimators` × `max_depth` × `min_samples_leaf` → steps 5–6 predictions, ODE-back
- **Umax weights:** `balanced` / `efficacy` / `probiotic_sparing` / `dose_sparing` — compared in **step 5b**; manuscript locks `balanced` for steps 8 / 10

### Fig. 5 notes


| Panel | Content                                                           |
| ----- | ----------------------------------------------------------------- |
| 5A    | Score landscape (composite-penalty response vs Umax)              |
| 5B    | Objective alignment (P_AUC uses constraint limit, not argmax)     |
| 5C    | Three-column ODE trajectories (median / tuned global / optimized) |
| 5D    | TAR three-policy comparison; secondary includes RF                |


**References:** P_AUC / LR use metadata `desired_*`; dose reference from training-only q90; fixed Umax policies from training rows only (per repeat; validation held out). **Primary optimizer (manuscript):** `feasible_first` — lowest-dosage feasible Umax on the grid, else lowest composite-penalty fallback. `aspiration_then_pareto` is sensitivity-only (`umax_selection_policy_sensitivity.csv`). Umax is **not** a supervised ML target. `--backend auto` uses Numba for u_grid; `--n_jobs` parallelizes repeats.

**Acceleration (same math / outputs):** run **10a → 10b** (10b reuses reference/policy CSVs); `--skip_if_complete` on 10a; `--verbose` on 10b; auto u_grid parallel inside repeat workers; preload features once per 10b run; resume per-repeat with default skip (use `--force_rerun` to refresh).

### Relabel

- Manuscript uses **fixed top_k = 8** only (no adaptive k)
- `relabel_tthr_bundle` writes `fixed_k8\` under the bundle outdir and publishes to `data\microbio_formal_dataset\`

