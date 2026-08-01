import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import linregress

# ==========================================
# 1. 數據輸入
# ==========================================

# OD600 Data (0 ~ 11 hr)
od_time = np.array([
    0, 0.5, 1, 1.666666667, 2.166666667, 2.666666667, 3.166666667,
    3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25,
    8.75, 9.25, 9.75, 10.25, 10.75, 11
])
od_values = np.array([
    0.002, 0.012, 0.022, 0.038, 0.07, 0.116, 0.244,
    0.4, 0.518, 0.646, 0.784, 0.924, 1.074, 1.204, 1.36, 1.518, 1.638,
    1.712, 1.78, 1.854, 1.91, 1.96, 1.974
])

# CFU Data Set 1
cfu_data_1 = [
    (0, 1103333.333), (1, 2044166.667), (2.166666667, 6050000),
    (3.166666667, 106000000), (3.75, 29000000), (4.25, 44400000),
    (4.75, 64000000), (5.25, 125000000), (7.25, 285000000),
    (8.75, 746666666.7), (9.25, 666666666.7), (9.75, 486000000),
    (10.25, 417500000)
]

# CFU Data Set 2
cfu_data_2 = [
    (0, 1440833.333), (1, 2545000), (3, 3298000), (5, 83600000),
    (7, 260750000), (10, 590000000)
]

# 合併並過濾數據 (只取 OD 時間範圍內的點)
all_cfu_time = []
all_cfu_values = []
max_od_time = max(od_time)

for t, cfu in cfu_data_1 + cfu_data_2:
    if t <= max_od_time:
        all_cfu_time.append(t)
        all_cfu_values.append(cfu)

# ==========================================
# 2. 線性插值與回歸
# ==========================================

# 建立插值函數 (Time -> OD)
interp_func = interp1d(od_time, od_values, kind='linear', fill_value="extrapolate")
interpolated_od_values = interp_func(all_cfu_time)

# 線性回歸
slope, intercept, r_value, p_value, std_err = linregress(interpolated_od_values, all_cfu_values)
r_squared = r_value**2

# ==========================================
# 3. 畫圖與存檔
# ==========================================

plt.figure(figsize=(8, 6))

# 畫散佈點
plt.scatter(interpolated_od_values, all_cfu_values, color='black', alpha=0.6, label='Experimental Data')

# 畫回歸線
x_range = np.linspace(min(interpolated_od_values), max(interpolated_od_values), 100)
y_pred = slope * x_range + intercept
plt.plot(x_range, y_pred, color='red', linestyle='--', linewidth=2, label=f'Linear Fit ($R^2={r_squared:.4f}$)')

# 加上公式標籤
equation_text = f'y = {slope:.2e}x + {intercept:.2e}\n$R^2$ = {r_squared:.4f}'
plt.text(0.05, 0.85, equation_text, transform=plt.gca().transAxes,
         fontsize=16, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

# 標籤設定
plt.title(r'$\bf{Correlation\ between\ OD_{600}\ and\ CFU/mL\ (}\boldsymbol{E.\ coli}\bf{)}$', fontsize=18)
plt.xlabel(r'$\bf{Optical\ Density\ (OD_{600})}$', fontsize=16)
plt.ylabel('Viable Cell Count (CFU/mL)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')

plt.tight_layout()

# --- 存檔指令 ---
plt.savefig('correlation_plot.png', dpi=300)  # 存成高畫質 PNG
plt.savefig('correlation_plot.svg')           # 存成向量圖 SVG
print("圖片已儲存為 correlation_plot.png 和 correlation_plot.svg")

plt.show()