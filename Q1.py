import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

df = pd.read_csv('Temperature.csv')
data = df['mean']  
years = df['year']

print(f"平均值 (Mean): {data.mean():.3f}")
print(f"變異數 (Variance): {data.var():.3f}")
print(f"變異係數% (CV): {100*data.std()/data.mean():.3f}")
print(f"偏態 (Skewness): {skew(data):.3f}")
largest_3 = df.nlargest(3, 'mean')[['year', 'mean']]
print("最大三值與年份：",largest_3)
smallest_3 = df.nsmallest(3, 'mean')[['year', 'mean']]
print("最小三值與年份：",smallest_3)

# 圖1.Time Series Plot
plt.figure(figsize=(12,6))
plt.plot(years, data, marker='o', linestyle='-', color='tab:blue', markersize=4)
plt.title('Time Series of Annual Mean Temperature (1926-2025)')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.xticks(np.arange(1925, 2026, 5), rotation=45) 
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('I.Time series plot.png')
plt.show()

# 圖2.Histogram
plt.figure(figsize=(10, 6))
n = len(data)
k_sturges = int(np.ceil(np.log2(n)) + 1) #使用Sturges公式
plt.hist(data, bins=k_sturges, density=True, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f'Normalized Histogram of Annual Mean Temp (Sturges Rule, k={k_sturges})')
plt.xlabel('Temperature (°C)')
plt.ylabel('Density') 
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('I.Histogram.png')
plt.show()

# 圖3.Boxplot
plt.figure(figsize=(6, 8))
plt.boxplot(data, vert=True, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='black'),
            medianprops=dict(color='black', linewidth=2))
plt.title('Boxplot of Annual Mean Temperature')
plt.ylabel('Temperature (°C)')
plt.xticks([1], ['Annual Mean Temp'])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('I.Boxplot.png')
plt.show()