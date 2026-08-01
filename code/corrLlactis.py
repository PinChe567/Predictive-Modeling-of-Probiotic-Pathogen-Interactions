import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import linregress

# ==========================================
# 1. 數據輸入 (L. lactis 新數據)
# ==========================================

# OD Data Set 1 (來自 image_641372.png)
od_data_1 = [
    (0, 0), (10, 0.014), (13, 0.018), (16, 0.056), (19, 0.074), (22, 0.102),
    (25, 0.178), (28, 0.32), (31, 0.566), (34, 0.786), (37, 0.932), (40, 0.976),
    (43, 0.984), (46, 0.97), (49, 0.944), (52, 0.944), (55, 0.972), (66, 0.928),
    (76, 1.002), (88, 1.042)
]

# OD Data Set 2 (來自 image_64136d.png)
od_data_2 = [
    (0, 0.034), (10, 0.082), (13, 0.182), (16, 0.378), (19, 0.664), (22, 0.836),
    (25, 1.054), (28.33333333, 1.086), (30.5, 1.112), (34, 1.148), (37, 1.128),
    (40, 1.108), (43, 1.136), (46, 1.132), (49, 1.084), (61, 1.104),
    (73.25, 1.082), (121, 1.192), (130, 1.192), (178, 1.26), (202, 1.234), (226, 1.252)
]

# CFU Data (來自 image_641353.png)
cfu_data = [
    (0, 4154000), (10, 31500000), (13, 64400000), (16, 105500000),
    (19, 83933333.33), (22, 163000000), (25, 288000000), (28.33333333, 300250000),
    (30.5, 287000000), (37, 434600000), (40, 329166666.7), (43, 269083333.3),
    (46, 364250000), (49, 288000000), (73.25, 349400000)
]

# ==========================================
# 2. 數據處理 (合併 OD 並處理重複時間點)
# ==========================================

# 將兩組 OD 數據合併
raw_od_data = od_data_1 + od_data_2

# 根據時間排序
raw_od_data.sort(key=lambda x: x[0])

# 處理重複時間點：如果時間一樣，取 OD 平均值 (為了讓插值函數順利運作)
unique_od_dict = {}
for t, od in raw_od_data:
    if t in unique_od_dict:
        unique_od_dict[t].append(od)
    else:
        unique_od_dict[t] = [od]

final_od_time = []
final_od_values = []

for t in sorted(unique_od_dict.keys()):
    final_od_time.append(t)
    # 取該時間點所有 OD 測量值的平均
    final_od_values.append(np.mean(unique_od_dict[t]))

# 轉換為 numpy array
od_time_np = np.array(final_od_time)
od_values_np = np.array(final_od_values)

# 分離 CFU 數據
cfu_time_np = np.array([x[0] for x in cfu_data])
cfu_values_np = np.array([x[1] for x in cfu_data])

# ==========================================
# 3. 線性插值與回歸
# ==========================================

# 建立插值函數 (Time -> OD)
# fill_value="extrapolate" 允許推估稍微超出範圍的點，但建議 CFU 時間要在 OD 範圍內
interp_func = interp1d(od_time_np, od_values_np, kind='linear', fill_value="extrapolate")

# 算出 CFU 取樣時間點對應的 OD
interpolated_od_values = interp_func(cfu_time_np)

# 線性回歸
slope, intercept, r_value, p_value, std_err = linregress(interpolated_od_values, cfu_values_np)
r_squared = r_value**2

# ==========================================
# 4. 畫圖與存檔
# ==========================================

plt.figure(figsize=(8, 6))

# 畫散佈點
plt.scatter(interpolated_od_values, cfu_values_np, color='black', alpha=0.6, label='Experimental Data')

# 畫回歸線
x_range = np.linspace(min(interpolated_od_values), max(interpolated_od_values), 100)
y_pred = slope * x_range + intercept
plt.plot(x_range, y_pred, color='red', linestyle='--', linewidth=2, label=f'Linear Fit ($R^2={r_squared:.4f}$)')

# 加上公式標籤
equation_text = f'y = {slope:.2e}x + {intercept:.2e}\n$R^2$ = {r_squared:.4f}'
plt.text(0.05, 0.85, equation_text, transform=plt.gca().transAxes,
         fontsize=16, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

# 標籤設定 (使用 LaTeX 語法加粗)
# \bf 或 \mathbf 可以讓數學模式字體變粗
plt.title(r'$\bf{Correlation\ between\ OD_{600}\ and\ CFU/mL\ (}\boldsymbol{L.\ lactis}\bf{)}$', fontsize=18)
plt.xlabel(r'$\bf{Optical\ Density\ (OD_{600})}$', fontsize=16) # 這裡加粗了
plt.ylabel('Viable Cell Count (CFU/mL)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right')

plt.tight_layout()

# --- 存檔指令 ---
plt.savefig('lactis_correlation_plot.png', dpi=300)
plt.savefig('lactis_correlation_plot.svg')
print(f"L. lactis Correlation R-squared: {r_squared}")
print("圖片已儲存為 lactis_correlation_plot.png 和 lactis_correlation_plot.svg")

plt.show()