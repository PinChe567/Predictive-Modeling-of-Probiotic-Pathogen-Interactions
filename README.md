# Dataset: Predictive Modeling of Probiotic-Pathogen Interactions Under Threshold-Triggered Antimicrobial Peptide Control

This repository provides the **source code**, **parameter-analysis plan**, and **raw experimental data** used in the study, “Predictive Modeling of Probiotic–Pathogen Interactions Under Threshold-Triggered Antimicrobial Peptide Control.”

> **Repository scope:** GitHub intentionally contains only the code, parameter-analysis plan, and experimental data. The generated simulation database and final analysis results are archived separately on Zenodo: 10.5281/zenodo.21961676.

---
## 💻 Parameter Analysis

Author: Huang, P.-C.

* **`analysis_plan/parameter_screening_plan.yaml`**
  
---
## 💻 Code

Author: Huang, P.-C.

Download analysis_plan and code folders first.

The files correspond to the results presented in **Figures 2-5** of the manuscript. The computational workflow is organized into the following Python scripts:

| Script | Purpose |
|---|---|
| `microbio_dataset.py` | Generates the synthetic ODE simulation library and constructs the relabeled supervised dataset for Tthr prediction. |
| `multi_pathogen_simulator.py` | Runs the multi-pathogen threshold-triggered ODE simulator and representative ODE trajectories. |
| `simulate_case_metrics_fast.py` | Provides the fast metrics-only ODE backend used for ODE-back validation and Umax optimization. |
| `heatmap.py` | Generates correlation heatmaps for biological features, desired outcomes, and controller targets. |
| `tree_srl_benchmark.py` | Trains and evaluates the TAR threshold predictor and tree-based control models. |
| `ode_back_validation.py` | Re-inserts predicted Tthr values into the ODE simulator to evaluate functional closed-loop outcomes. |
| `closed_loop_eval.py` | Performs fixed-Umax validation, Umax weight screening, and Umax optimization studies. |
| `derive_optimizer_references.py` | Derives training-only dose reference scales and fixed Umax policies. |
| `figure_audit.py` | Regenerates manuscript figures from saved CSV outputs and exports PNG/SVG files. |
| `corrEcoli.py` | Performs the paired *E. coli* OD600–CFU linear regression and exports the calibration plot in PNG/SVG formats. |
| `corrLlactis.py` | Performs the paired *L. lactis* OD600–CFU linear regression and exports the calibration plot in PNG/SVG formats. |
| `correlation.py` | Verifies that the paired OD600–CFU regression results agree across the correlation scripts, simulator constants, and dataset provenance table. |

For the full execution order and command-line examples, please see:

```text
code/pipeline.md
```

---

## 📂 Data Overview

Author: Huang, P.-C.

The experimental data (locate in file: "experiment data") is provided in **Excel (.xlsx)** format. The files correspond to the results presented in **Figures 1** of the manuscript:

### Figure 1: Bacterial growth kinetics and preliminary AMP-response observations
Growth dynamics of pathogenic *E. coli* and probiotic *L. lactis* for ODE model fitting.
* **`E. coli growth curve OD.xlsx`**: Optical density (OD600) measurements for *E. coli*.
* **`E. coli growth curve CFU.xlsx`**: Colony Forming Units (CFU) counts for *E. coli*.
* **`L. lactis growth curve OD.xlsx`**: Standard growth curve (OD600) for *L. lactis*.
* **`L. lactis growth curve CFU.xlsx`**: Standard growth curve (CFU) for *L. lactis*.

Growth dynamics of E. coli and L. lactis used to define representative CFU-scale and growth-scale ranges for simulation anchoring.
Preliminary hBD-3 response assays used to define approximate AMP-response ranges. These experiments were not designed as statistically powered tolerance, resistance, or killing-coefficient calibration assays.
* **`AMP to L. lactis OD.xlsx`**: Preliminary *L. lactis* AMP-response observation under varying hBD-3 concentrations.
* **`AMP to L. lactis CFU.xlsx`**: Preliminary *L. lactis* AMP-response observation under varying hBD-3 concentrations.
* **`AMP to E. coli.xlsx`**: Preliminary *E. coli* AMP-response observation.
* **`AMP to E. coli second dose.xlsx`**: Preliminary repeated-exposure observation; not evidence of validated adaptive tolerance or stable resistance.
* **`Correlation.xlsx`**: Auxiliary workbook associated with the OD600–CFU correlation analysis.

---

## 📦 Generated Database and Results

The generated database and final analysis results are deposited on Zenodo as:

* **`formal_datasets.zip`**: The complete code-generated `data/` directory.
* **`predictions_and_results.zip`**: The complete code-generated `results/` directory, including saved predictions, numerical result tables, manifests, and CSV/PNG/SVG figure outputs.

Zenodo: 10.5281/zenodo.21961676
