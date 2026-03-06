import numpy as np
from scipy.stats import lognorm, norm

mu_y = 3.14217   
sigma_y = 0.02453 
T = 20           

#解析法計算
z_upper = norm.ppf(0.95)
z_lower = norm.ppf(0.05) 
ana_upper = np.exp(mu_y + z_upper * sigma_y)
ana_lower = np.exp(mu_y + z_lower * sigma_y)
print("--- Analytical Results ---")
print(f"Upper 20-year Return Level: {ana_upper:.4f} °C")
print(f"Lower 20-year Return Level: {ana_lower:.4f} °C\n")

#蒙地卡羅法
n_samples = 1000000
np.random.seed(55688)
mc_samples = np.random.lognormal(mean=mu_y, sigma=sigma_y, size=n_samples)
#使用百分位數估計
mc_lower = np.percentile(mc_samples, 5)
mc_upper = np.percentile(mc_samples, 95)
print(f"--- Monte Carlo Results (N={n_samples}) ---")
print(f"Upper 20-year Return Level: {mc_upper:.4f} °C")
print(f"Lower 20-year Return Level: {mc_lower:.4f} °C\n")

#比較與誤差分析
err_upper = abs(ana_upper - mc_upper)
err_lower = abs(ana_lower - mc_lower)
print("--- Comparison ---")
print(f"Upper Tail Difference: {err_upper:.6f} °C")
print(f"Lower Tail Difference: {err_lower:.6f} °C")

#繪圖
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
count, bins, ignored = plt.hist(mc_samples, bins=100, density=True, alpha=0.5, color='gray', label='MC Samples')
x = np.linspace(min(mc_samples), max(mc_samples), 500)
pdf = lognorm.pdf(x, s=sigma_y, scale=np.exp(mu_y))
plt.plot(x, pdf, 'r-', lw=2, label='Theoretical Lognormal PDF')
plt.axvline(ana_upper, color='blue', linestyle='--', label=f'Analytical Upper 20yr: {ana_upper:.4f}°C')
plt.axvline(ana_lower, color='green', linestyle='--', label=f'Analytical Lower 20yr: {ana_lower:.4f}°C')
plt.title(f'20-Year Return Level Estimation (N={n_samples})')
plt.xlabel('Annual Mean Temperature (°C)')
plt.ylabel('Density')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig('VI. Return Level Estimation.png')
plt.show()