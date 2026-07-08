# Dataset: Predictive Modeling of Probiotic-Pathogen Interactions Under Threshold-Triggered Antimicrobial Peptide Control

This repository contains the source code, raw experimental data, generated simulation datasets, and analysis outputs used in our study, **"Predictive Modeling of Probiotic-Pathogen Interactions Under Threshold-Triggered Antimicrobial Peptide Control"**.

The project combines experimental microbial characterization, a synthetic multi-pathogen ODE simulation engine, threshold inverse-design modeling, ODE-back validation, and Umax optimization for threshold-triggered antimicrobial peptide (AMP) control.

---

## 💻 Code

Author: Huang, P.-C. & Ciou, Z.-C.

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

For the full execution order and command-line examples, please see:

```text
pipeline.md
```

---

## 📂 Data Overview

Author: Huang, P.-C.

The experimental data (locate in file: "experiment data") is provided in **Excel (.xlsx)** format. The files correspond to the results presented in **Figures 1** of the manuscript:

### Figure 1: Bacterial Growth Kinetics & AMP Efficacy
Growth dynamics of pathogenic *E. coli* and probiotic *L. lactis* for ODE model fitting.
* **`E. coli growth curve OD.xlsx`**: Optical density (OD600) measurements for *E. coli*.
* **`E. coli growth curve CFU.xlsx`**: Colony Forming Units (CFU) counts for *E. coli*.
* **`L. lactis growth curve OD.xlsx`**: Standard growth curve (OD600) for *L. lactis*.
* **`L. lactis growth curve CFU.xlsx`**: Standard growth curve (CFU) for *L. lactis*.

Dose-response and resistance evolution assays using hBD-3 (AMP).
* **`AMP to L. lactis OD.xlsx`**: Probiotic tolerance test under varying AMP concentrations (OD).
* **`AMP to L. lactis CFU.xlsx`**: Probiotic tolerance test under varying AMP concentrations (CFU).
* **`AMP to E. coli.xlsx`**: *E. coli* survival rates and killing efficiency.
* **`AMP to E. coli second dose.xlsx`**: Adaptive resistance test (second-dose challenge).
