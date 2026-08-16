# Figure-only consistency repair audit

- Timestamp: 2026-08-16 07:51:34 UTC
- Current git commit: `none (fatal: not a git repository (or any of the parent directories): .git)`
- Numeric values changed: **no** (bars/CIs/curves reused locked artifacts; no tests recomputed).
- Original CSVs / statistical tables modified: **no**.
- Full pipeline / ODE / ML / optimizer / 100-repeat runs: **not executed**.
- Note: Fig. 3b/4a/5d PNG/SVG “input” hashes in the list below are from after the first successful replot in this session (CSV/JSON hashes are the locked originals). Fig. 5b/c figure hashes `82c05ab4…` / `00fe369b…` / `fc667c81…` / `8ac1557c…` are the pre-repair clipped figures.

## Missing row-level landscape CSVs (not fabricated)

These files are absent. Fig. 5b/c x-values were taken from locked `u_grid` (0–101 inclusive) in `repeats/repeat_000/umax_study/umax_optimization_manifest.json`; curve *y*-values were taken from the locked matplotlib SVG polylines (102 vertices), indexed onto that grid. Selected-Umax rugs/median used `selected_u_max` from `umax_feasible_region_summary.csv`.

- `results\umax_optimization\umax_response_landscape.csv` — **missing**
- `results\umax_optimization\umax_optimization_u_candidates.csv` — **missing**
- `results\umax_optimization\umax_score_landscape_curves.csv` — **missing**
- `results\umax_optimization\umax_selected_umax_distribution.csv` — **missing**

## Input artifact SHA256 (before overwrite)

- `results\tree_srl_benchmark\parameter_pairwise_significance.csv`: `3d3e9ad6d777a74f0f6189dee58c9581f0afc325e2ec614be42d9fd867043e0b`
- `results\tree_srl_benchmark\model_compare_summary.csv`: `f7b19a00a29d063e07815826308c0bc9cf24df9c174311044a06229f2af5a062`
- `results\tree_srl_benchmark\model_compare_per_target.csv`: `34157083975bf02d73b8d0288f5b2cf91f64d984c358024ffa2c59cc1b2953cd`
- `results\tree_srl_benchmark\figure\model_compare_r2.png`: `898fbd8f0059bf449df8871c1565cb3dabde3aaf76f283b045320fd9e4f8c88f`
- `results\tree_srl_benchmark\figure\model_compare_r2.svg`: `ea060a00db52242ec9077e0bd28fcf2c76fe960b03669de2b4b23419671b6c13`
- `results\ode_back_validation\ode_back_pairwise_significance.csv`: `cafb2b58f2e2830b2670335a2bd9b55ea403f92d4f81a5c63693e7c494860df0`
- `results\ode_back_validation\ode_back_summary_by_model.csv`: `ef2f90ddbe1cf6aff44c75c429c4a67068c232685f5bbd27ea02036fa11f6412`
- `results\ode_back_validation\figure\ode_back_r2_barplot.png`: `1282c8f99e8e924091c901ed0a355de794296c6790e489ac5b4e558c9f5bf37a`
- `results\ode_back_validation\figure\ode_back_r2_barplot.svg`: `8362145a613c3ad1d9803cb7058e119f6e423a77668065c471642fa6a06c0ce8`
- `results\umax_optimization\umax_ablation_repeated_plot_stats.csv`: `7a4f68b6aae19ac9c52c3347e513b377a87170e00e23eb703eb0e2e62d8fe845`
- `results\umax_optimization\umax_policy_ablation_condition_counts.csv`: `05e65cc914a6946348cad301144a991565236abc840dcca0692cea6daeb61779`
- `results\umax_optimization\figure\umax_summary_ablation_composite_supplementary.png`: `a089024106075d4d2dc8d247e68ca42c523026187f681be114ed05e9e5ba9b39`
- `results\umax_optimization\figure\umax_summary_ablation_composite_supplementary.svg`: `95980786516d877bb3d5aa8197cd1fba595f9e824efc622da9c005ddd5e6d361`
- `results\umax_optimization\repeats\repeat_000\umax_study\umax_optimization_manifest.json`: `3e37f5504b037c92883482ddd9673d901bd04556ea99627aeb3ab58fe0c674c6`
- `results\umax_optimization\umax_feasible_region_summary.csv`: `9dae2e6f3908265dd853a2e183573f6987baab566da049b830a3dfc103c463f2`
- `results\umax_optimization\figure\umax_constraint_feasibility.png`: `82c05ab405cf59919bac8182e1eebcf441781e3d1c2321db988f52a3ea90a1a6`
- `results\umax_optimization\figure\umax_constraint_feasibility.svg`: `00fe369b0fce5b639cc5dc28bfd45e12c5b6e25e51262a581183d6cdca36484d`
- `results\umax_optimization\figure\umax_score_landscape.png`: `fc667c81a49f2a8bb10c8ac7002e4a1669f5a112be36c26890ef8a1d0bb3c6ca`
- `results\umax_optimization\figure\umax_score_landscape.svg`: `8ac1557cd77fe274f018122eaecb6a56371bb8f4051a324a8213cc4c3f99aac9`

## Per-figure record

### Fig. 3b `model_compare_r2`

**Numeric values changed:** no

Input files + SHA256:
- `results\tree_srl_benchmark\parameter_pairwise_significance.csv`: `3d3e9ad6d777a74f0f6189dee58c9581f0afc325e2ec614be42d9fd867043e0b`
- `results\tree_srl_benchmark\model_compare_summary.csv`: `f7b19a00a29d063e07815826308c0bc9cf24df9c174311044a06229f2af5a062`
- `results\tree_srl_benchmark\model_compare_per_target.csv`: `34157083975bf02d73b8d0288f5b2cf91f64d984c358024ffa2c59cc1b2953cd`
- `results\tree_srl_benchmark\figure\model_compare_r2.png`: `898fbd8f0059bf449df8871c1565cb3dabde3aaf76f283b045320fd9e4f8c88f`
- `results\tree_srl_benchmark\figure\model_compare_r2.svg`: `ea060a00db52242ec9077e0bd28fcf2c76fe960b03669de2b4b23419671b6c13`

Output files + SHA256:
- `results\tree_srl_benchmark\figure\model_compare_r2.png`: `898fbd8f0059bf449df8871c1565cb3dabde3aaf76f283b045320fd9e4f8c88f`
- `results\tree_srl_benchmark\figure\model_compare_r2.svg`: `026a4098b222a825f8360f55cb6ff5b88ffaaab95a5533b3385dd0d1c89f9f9e`

Annotations removed/modified:
- Removed TAR–BestTree/BestSingleTree significance brackets and stars on both R² and RMSE panels (oracle diagnostic; `is_oracle_diagnostic=True` in the saved table).
- Retained prespecified formal brackets only: TAR–RandomForest and TAR–UniformTreeMean, using saved Holm `significance_label` / `corrected_p_holm` from `parameter_pairwise_significance.csv` (no p-value recomputation).
- Bar heights and 95% CIs unchanged from `model_compare_summary.csv`.
- Wording `individualized` / `multi-objective agreement` was not present in this source figure.

### Fig. 4a `ode_back_r2_barplot`

**Numeric values changed:** no

Input files + SHA256:
- `results\ode_back_validation\ode_back_pairwise_significance.csv`: `cafb2b58f2e2830b2670335a2bd9b55ea403f92d4f81a5c63693e7c494860df0`
- `results\ode_back_validation\ode_back_summary_by_model.csv`: `ef2f90ddbe1cf6aff44c75c429c4a67068c232685f5bbd27ea02036fa11f6412`
- `results\ode_back_validation\figure\ode_back_r2_barplot.png`: `1282c8f99e8e924091c901ed0a355de794296c6790e489ac5b4e558c9f5bf37a`
- `results\ode_back_validation\figure\ode_back_r2_barplot.svg`: `8362145a613c3ad1d9803cb7058e119f6e423a77668065c471642fa6a06c0ce8`

Output files + SHA256:
- `results\ode_back_validation\figure\ode_back_r2_barplot.png`: `1282c8f99e8e924091c901ed0a355de794296c6790e489ac5b4e558c9f5bf37a`
- `results\ode_back_validation\figure\ode_back_r2_barplot.svg`: `30474e1f1bf61eacefbb3e1be849ac0cbc75b8e1feb0af24dfff929825255862`

Annotations removed/modified:
- Removed TAR–BestTree/BestSingleTree significance bracket and star (oracle diagnostic).
- Retained only non-oracle formal comparisons present as `is_formal_comparison=True` in `ode_back_pairwise_significance.csv`: TAR–RandomForest and TAR–UniformTreeMean.
- Bar heights and 95% CIs unchanged from `ode_back_summary_by_model.csv`.
- Wording `individualized` / `multi-objective agreement` was not present in this source figure.

### Fig. 5b `umax_constraint_feasibility`

**Numeric values changed:** no

Input files + SHA256:
- `results\umax_optimization\repeats\repeat_000\umax_study\umax_optimization_manifest.json`: `3e37f5504b037c92883482ddd9673d901bd04556ea99627aeb3ab58fe0c674c6`
- `results\umax_optimization\umax_feasible_region_summary.csv`: `9dae2e6f3908265dd853a2e183573f6987baab566da049b830a3dfc103c463f2`
- `results\umax_optimization\figure\umax_constraint_feasibility.png`: `82c05ab405cf59919bac8182e1eebcf441781e3d1c2321db988f52a3ea90a1a6`
- `results\umax_optimization\figure\umax_constraint_feasibility.svg`: `00fe369b0fce5b639cc5dc28bfd45e12c5b6e25e51262a581183d6cdca36484d`
- `results\umax_optimization\figure\umax_score_landscape.png`: `fc667c81a49f2a8bb10c8ac7002e4a1669f5a112be36c26890ef8a1d0bb3c6ca`
- `results\umax_optimization\figure\umax_score_landscape.svg`: `8ac1557cd77fe274f018122eaecb6a56371bb8f4051a324a8213cc4c3f99aac9`

Output files + SHA256:
- `results\umax_optimization\figure\umax_constraint_feasibility.png`: `89291fb62c27e930b6ac2e9299064d5a829185ba476130d81b776a480c06d06b`
- `results\umax_optimization\figure\umax_constraint_feasibility.svg`: `44e81d999609bbcea871a66f6e2d732096e47e444e3b891da00fb834cf041bec`

Annotations removed/modified:
- Removed display clipping `set_xlim(0, 100)` so the official 0–101 inclusive candidate grid is shown.
- No inferential brackets/stars (this panel never had them).
- Median selected Umax line/IQR span still from locked TAR `selected_u_max`.
- Wording `individualized` / `multi-objective agreement` was not present; no title rewrite.

### Fig. 5c `umax_score_landscape`

**Numeric values changed:** no

Input files + SHA256:
- `results\umax_optimization\repeats\repeat_000\umax_study\umax_optimization_manifest.json`: `3e37f5504b037c92883482ddd9673d901bd04556ea99627aeb3ab58fe0c674c6`
- `results\umax_optimization\umax_feasible_region_summary.csv`: `9dae2e6f3908265dd853a2e183573f6987baab566da049b830a3dfc103c463f2`
- `results\umax_optimization\figure\umax_constraint_feasibility.png`: `82c05ab405cf59919bac8182e1eebcf441781e3d1c2321db988f52a3ea90a1a6`
- `results\umax_optimization\figure\umax_constraint_feasibility.svg`: `00fe369b0fce5b639cc5dc28bfd45e12c5b6e25e51262a581183d6cdca36484d`
- `results\umax_optimization\figure\umax_score_landscape.png`: `fc667c81a49f2a8bb10c8ac7002e4a1669f5a112be36c26890ef8a1d0bb3c6ca`
- `results\umax_optimization\figure\umax_score_landscape.svg`: `8ac1557cd77fe274f018122eaecb6a56371bb8f4051a324a8213cc4c3f99aac9`

Output files + SHA256:
- `results\umax_optimization\figure\umax_score_landscape.png`: `200fbae6bf03dad6d5ad848ba5c994270aa6da36ab26fb92e9f6e0d3f55bbef9`
- `results\umax_optimization\figure\umax_score_landscape.svg`: `246ceafc42233cc26dcdbd5173204ffa9365ae129a953362dbc01e1b8cea409e`

Annotations removed/modified:
- Removed display clipping `set_xlim(0, 100)` so the official 0–101 inclusive candidate grid is shown.
- Median, IQR (and q10–q90 if present) taken from locked SVG vertex count matched to `u_grid`; not recomputed from ODE candidates.
- Wording `individualized` / `multi-objective agreement` was not present; no title rewrite.

### Fig. 5d `umax_summary_ablation_composite_supplementary`

**Numeric values changed:** no

Input files + SHA256:
- `results\umax_optimization\umax_ablation_repeated_plot_stats.csv`: `7a4f68b6aae19ac9c52c3347e513b377a87170e00e23eb703eb0e2e62d8fe845`
- `results\umax_optimization\umax_policy_ablation_condition_counts.csv`: `05e65cc914a6946348cad301144a991565236abc840dcca0692cea6daeb61779`
- `results\umax_optimization\figure\umax_summary_ablation_composite_supplementary.png`: `a089024106075d4d2dc8d247e68ca42c523026187f681be114ed05e9e5ba9b39`
- `results\umax_optimization\figure\umax_summary_ablation_composite_supplementary.svg`: `95980786516d877bb3d5aa8197cd1fba595f9e824efc622da9c005ddd5e6d361`

Output files + SHA256:
- `results\umax_optimization\figure\umax_summary_ablation_composite_supplementary.png`: `a089024106075d4d2dc8d247e68ca42c523026187f681be114ed05e9e5ba9b39`
- `results\umax_optimization\figure\umax_summary_ablation_composite_supplementary.svg`: `50e60ebb4d7600cbf959be053edac8eaa2229ac57a005b161e41ebdf1bb4414b`

Annotations removed/modified:
- Removed all inferential brackets/stars (`significance_pairs=None`). Saved `umax_ablation_significance_annotations.csv` was not used and was not modified.
- Retained bars and descriptive repeat-level 95% intervals from `umax_ablation_repeated_plot_stats.csv`.
- Did not recompute p-values.
- Wording `individualized` / `multi-objective agreement` was not present in this source figure (ylabel remains `Composite penalty score`).

## Label wording search

Repo source figures / plot titles were searched for `individualized` and `multi-objective agreement`. **No matches.** No manuscript composite `Fig5.png` panel-A schematic is generated here. No title/label string replacement was applied.

## Validation evidence

- Fig. 3b pairs=[('TAR', 'RandomForest', '****'), ('TAR', 'UniformTreeMean', '****')] rmse_pairs=[('TAR', 'RandomForest', '****'), ('TAR', 'UniformTreeMean', '****')] star_comments 4 -> 4 (expect 4 = 2 R2 + 2 RMSE)
- Fig. 4a pairs=[('TAR', 'RandomForest', '****'), ('TAR', 'UniformTreeMean', '****')] star_comments 2 -> 2 (expect 2)
- Fig. 5d star_comments 0 -> 0 (expect 0); bars+95% CI retained from umax_ablation_repeated_plot_stats.csv
- Fig. 5b/c u_grid n=102 min=0.0 max=101.0; TAR selected_u_max n=270000 median=90.0; tick101 landscape=True constraint=True; 100-only-clip landscape=False constraint=False
- Fig. 5b/5c 100-only clip remaining: constraint=False, landscape=False (must be False).
- PNG+SVG exported via `save_figure` for every repaired panel.
- `assert_figure_artists_inside_canvas` ran inside the 3b/4a/5b/5c/5d plot functions.

