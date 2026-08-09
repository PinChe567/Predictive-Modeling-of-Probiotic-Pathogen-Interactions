from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

OUT_DIR = Path("results") / "correlation"

# ==========================================
# 1. Data input (L. lactis)
# ==========================================

# time (hr), CFU/mL, OD600 — paired measurements at the same time points
data = [
    (0,           4154000,      0.034),
    (10,          31500000,     0.082),
    (13,          64400000,     0.182),
    (16,          105500000,    0.378),
    (19,          83933333.33,  0.664),
    (22,          163000000,    0.836),
    (25,          288000000,    1.054),
    (28.33333333, 300250000,    1.086),
    (30.5,        287000000,    1.112),
    (37,          434600000,    1.128),
    (40,          329166666.7,  1.108),
    (43,          269083333.3,  1.136),
    (46,          364250000,    1.132),
    (49,          288000000,    1.084),
    (73.25,       349400000,    1.082),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 2. Data processing
    # ==========================================

    time_np = np.array([x[0] for x in data])
    cfu_values_np = np.array([x[1] for x in data])
    od_values_np = np.array([x[2] for x in data])

    # ==========================================
    # 3. Linear regression (OD vs CFU; time points already aligned, no interpolation needed)
    # ==========================================

    slope, intercept, r_value, p_value, std_err = linregress(od_values_np, cfu_values_np)
    r_squared = r_value**2

    # ==========================================
    # 4. Plotting and saving
    # ==========================================

    plt.figure(figsize=(8, 6))

    # Scatter plot
    plt.scatter(od_values_np, cfu_values_np, color='black', alpha=0.6, label='Experimental Data')

    # Regression line
    x_range = np.linspace(min(od_values_np), max(od_values_np), 100)
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

    # Labels (bold via LaTeX)
    plt.title(
        r'$\bf{Correlation\ between\ OD_{600}\ and\ CFU/mL\ (}\boldsymbol{L.\ lactis}\bf{)}$',
        fontsize=18,
    )
    plt.xlabel(r'$\bf{Optical\ Density\ (OD_{600})}$', fontsize=16)
    plt.ylabel('Viable Cell Count (CFU/mL)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')

    plt.tight_layout()

    # --- Save figures ---
    plt.savefig(OUT_DIR / "lactis_correlation_plot.png", dpi=300)
    plt.savefig(OUT_DIR / "lactis_correlation_plot.svg")
    print(f"L. lactis Correlation R-squared: {r_squared}")
    print(f"Figures saved to {OUT_DIR}/lactis_correlation_plot.png and .svg")

    plt.show()


if __name__ == "__main__":
    main()
