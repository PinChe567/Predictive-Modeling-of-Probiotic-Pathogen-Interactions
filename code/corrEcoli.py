from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

OUT_DIR = Path("results") / "correlation"

# ==========================================
# 1. Data input — eight paired OD600 / CFU measurements (same times; no interpolation)
# ==========================================

# OD600 Data (0 ~ 11 hr)
od_time = np.array([
    0, 1, 2.166666667, 3.75, 4.25, 4.75, 5.25, 7.25
])
od_values = np.array([
    0.002, 0.022, 0.07, 0.4, 0.518, 0.646, 0.784, 1.36
])

# CFU Data Set 1 (paired with od_time / od_values)
cfu_data_1 = [
    (0, 1103333.333), (1, 2044166.667), (2.166666667, 6050000), (3.75, 29000000), (4.25, 44400000),
    (4.75, 64000000), (5.25, 125000000), (7.25, 285000000),

]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfu_time = np.array([t for t, _cfu in cfu_data_1], dtype=float)
    cfu_values = np.array([cfu for _t, cfu in cfu_data_1], dtype=float)

    if cfu_time.shape != od_time.shape or not np.allclose(cfu_time, od_time, rtol=0.0, atol=0.0):
        raise AssertionError("E. coli OD and CFU series must share identical paired time points")

    # ==========================================
    # 2. Linear regression on paired measurements (no interpolation)
    # ==========================================

    slope, intercept, r_value, p_value, std_err = linregress(od_values, cfu_values)
    r_squared = r_value**2

    # Manuscript / multi_pathogen_simulator provenance target
    assert abs(r_squared - 0.9016120870162722) <= 1e-15, r_squared

    # ==========================================
    # 3. Plotting and saving
    # ==========================================

    plt.figure(figsize=(8, 6))

    # Scatter plot
    plt.scatter(od_values, cfu_values, color='black', alpha=0.6, label='Experimental Data')

    # Regression line
    x_range = np.linspace(min(od_values), max(od_values), 100)
    y_pred = slope * x_range + intercept
    plt.plot(
        x_range,
        y_pred,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Linear Fit ($R^2={r_squared:.4f}$)',
    )

    # Equation annotation
    equation_text = f'y = {slope:.2e}x + {intercept:.2e}\n$R^2$ = {r_squared:.4f}'
    plt.text(
        0.05,
        0.85,
        equation_text,
        transform=plt.gca().transAxes,
        fontsize=16,
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'),
    )

    # Labels
    plt.title(
        r'$\bf{Correlation\ between\ OD_{600}\ and\ CFU/mL\ (}\boldsymbol{E.\ coli}\bf{)}$',
        fontsize=18,
    )
    plt.xlabel(r'$\bf{Optical\ Density\ (OD_{600})}$', fontsize=16)
    plt.ylabel('Viable Cell Count (CFU/mL)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')

    plt.tight_layout()

    # --- Save figures ---
    plt.savefig(OUT_DIR / "ecoli_correlation_plot.png", dpi=300)  # high-resolution PNG
    plt.savefig(OUT_DIR / "ecoli_correlation_plot.svg")           # vector SVG
    print(f"E. coli Correlation R-squared: {r_squared}")
    print(f"Figures saved to {OUT_DIR}/ecoli_correlation_plot.png and .svg")

    plt.show()


if __name__ == "__main__":
    main()
