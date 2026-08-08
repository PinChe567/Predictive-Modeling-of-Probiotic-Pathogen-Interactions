from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import linregress

OUT_DIR = Path("results") / "correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. Data input
# ==========================================

# OD600 Data (0 ~ 11 hr)
od_time = np.array([
    0, 1, 2.166666667, 3.75, 4.25, 4.75, 5.25, 7.25
])
od_values = np.array([
    0.002, 0.022, 0.07, 0.4, 0.518, 0.646, 0.784, 1.36
])

# CFU Data Set 1
cfu_data_1 = [
    (0, 1103333.333), (1, 2044166.667), (2.166666667, 6050000), (3.75, 29000000), (4.25, 44400000),
    (4.75, 64000000), (5.25, 125000000), (7.25, 285000000),

]

# Merge and filter data (keep only points within the OD time range)
all_cfu_time = []
all_cfu_values = []
max_od_time = max(od_time)

for t, cfu in cfu_data_1:
    if t <= max_od_time:
        all_cfu_time.append(t)
        all_cfu_values.append(cfu)

# ==========================================
# 2. Linear interpolation and regression
# ==========================================

# Build interpolation function (Time -> OD)
interp_func = interp1d(od_time, od_values, kind='linear', fill_value="extrapolate")
interpolated_od_values = interp_func(all_cfu_time)

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(interpolated_od_values, all_cfu_values)
r_squared = r_value**2

# ==========================================
# 3. Plotting and saving
# ==========================================

plt.figure(figsize=(8, 6))

# Scatter plot
plt.scatter(interpolated_od_values, all_cfu_values, color='black', alpha=0.6, label='Experimental Data')

# Regression line
x_range = np.linspace(min(interpolated_od_values), max(interpolated_od_values), 100)
y_pred = slope * x_range + intercept
plt.plot(x_range, y_pred, color='red', linestyle='--', linewidth=2, label=f'Linear Fit ($R^2={r_squared:.4f}$)')

# Equation annotation
equation_text = f'y = {slope:.2e}x + {intercept:.2e}\n$R^2$ = {r_squared:.4f}'
plt.text(0.05, 0.85, equation_text, transform=plt.gca().transAxes,
         fontsize=16, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

# Labels
plt.title(r'$\bf{Correlation\ between\ OD_{600}\ and\ CFU/mL\ (}\boldsymbol{E.\ coli}\bf{)}$', fontsize=18)
plt.xlabel(r'$\bf{Optical\ Density\ (OD_{600})}$', fontsize=16)
plt.ylabel('Viable Cell Count (CFU/mL)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')

plt.tight_layout()

# --- Save figures ---
plt.savefig(OUT_DIR / "ecoli_correlation_plot.png", dpi=300)  # high-resolution PNG
plt.savefig(OUT_DIR / "ecoli_correlation_plot.svg")           # vector SVG
print(f"Figures saved to {OUT_DIR}/ecoli_correlation_plot.png and .svg")

plt.show()
