# Dataset: Predictive Modeling of Probiotic-Pathogen Interactions Under Threshold-Triggered Antimicrobial Peptide Control

This directory contains the source code, raw experimental data and genetic sequences used in our study, **"Predictive Modeling of Probiotic-Pathogen Interactions Under Threshold-Triggered Antimicrobial Peptide Control"**.

## 💻 Code

Author: Ciou, Z.-C.

Two core scripts drive the simulation and optimization framework:

* **`drug_delivery_optimizer.py`**: A Stacking Ensemble ML model (DCN, MLP, XGBoost, LightGBM) used to predict optimal drug delivery parameters.

* **`wound_environment_simulator.py`**: An ODE-based simulator modeling the interaction between E. coli (pathogen), L. lactis (probiotic), and AMP concentration over time.

Dependencies: Python 3.x, numpy, pandas, scikit-learn, tensorflow, xgboost, lightgbm.

---

## 📂 Data Overview

Author: Huang, P.-C.

The experimental data (locate in file: "experiment data") is provided in **Excel (.xlsx)** format. The files correspond to the results presented in **Figures 1–3** of the manuscript:

### Figure 1: Bacterial Growth Kinetics
Growth dynamics of pathogenic *E. coli* and probiotic *L. lactis* for ODE model fitting.
* **`E. coli growth curve OD.xlsx`**: Optical density (OD600) measurements for *E. coli*.
* **`E. coli growth curve CFU.xlsx`**: Colony Forming Units (CFU) counts for *E. coli*.
* **`L. lactis growth curve OD.xlsx`**: Standard growth curve (OD600) for *L. lactis*.
* **`L. lactis growth curve CFU.xlsx`**: Standard growth curve (CFU) for *L. lactis*.

### Figure 2: AMP Efficacy & Resistance
Dose-response and resistance evolution assays using hBD-3 (AMP).
* **`AMP to L. lactis OD.xlsx`**: Probiotic tolerance test under varying AMP concentrations (OD).
* **`AMP to L. lactis CFU.xlsx`**: Probiotic tolerance test under varying AMP concentrations (CFU).
* **`AMP to E. coli.xlsx`**: *E. coli* survival rates and killing efficiency.
* **`AMP to E. coli second dose.xlsx`**: Adaptive resistance test (second-dose challenge).
