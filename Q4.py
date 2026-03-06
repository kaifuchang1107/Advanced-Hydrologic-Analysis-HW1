import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, probplot

df = pd.read_csv('Temperature.csv')
data = df['mean']
data_sorted = np.sort(data) #排序觀測值
n = len(data)
mu_n, sigma_n = 23.16100, 0.56884
mu_y, sigma_y = 3.14217, 0.02453
# Weibull 序位法 P=m/(n+1)
rank = np.arange(1, n + 1)
p_plotting = rank / (n + 1)
# 計算常態分佈理論值
theoretical_n = norm.ppf(p_plotting, loc=mu_n, scale=sigma_n)
# 計算對數常態分佈理論值
theoretical_ln = lognorm.ppf(p_plotting, s=sigma_y, scale=np.exp(mu_y))

#計算SSE
sse_n = np.sum((data_sorted - theoretical_n)**2)
sse_ln = np.sum((data_sorted - theoretical_ln)**2)
print(f"--- 誤差平方和 (SSE) 結果 ---")
print(f"Normal SSE   : {sse_n:.6f}")
print(f"Lognormal SSE: {sse_ln:.6f}")
print(f"最佳模型 (SSE較小者): {'Lognormal' if sse_ln < sse_n else 'Normal'}")


#Q-Q圖
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Normal 標準化
# 觀測值(x)標準化:(X-mu)/sigma
std_obs_n = (data_sorted - mu_n) / sigma_n
# 理論值(Y)標準化:標準常態分佈N(0,1)的分位數
std_theo_n = norm.ppf(p_plotting, loc=0, scale=1)

ax1.scatter(std_theo_n, std_obs_n, color='red', alpha=0.6, edgecolors='k', label='Standardized Data')
ax1.plot([-3, 3], [-3, 3], 'k--', lw=2, label='Reference Line (y=x)')
ax1.set_title(f'Standardized Normal Q-Q Plot\n(SSE = {sse_n:.4f})')
ax1.set_xlabel('Theoretical Quantiles (Z-score)')
ax1.set_ylabel('Observed Quantiles (Standardized)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])

#Lognormal標準化
std_obs_ln = (np.log(data_sorted) - mu_y) / sigma_y
#理論值同樣對應標準常態分佈N(0,1)
std_theo_ln = norm.ppf(p_plotting, loc=0, scale=1)

ax2.scatter(std_theo_ln, std_obs_ln, color='blue', alpha=0.6, edgecolors='k', label='Standardized Data')
ax2.plot([-3, 3], [-3, 3], 'k--', lw=2, label='Reference Line (y=x)')
ax2.set_title(f'Standardized Lognormal Q-Q Plot\n(SSE = {sse_ln:.4f})')
ax2.set_xlabel('Theoretical Quantiles (Z-score)')
ax2.set_ylabel('Observed Quantiles (Standardized)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-3, 3])
ax2.set_ylim([-3, 3])

plt.tight_layout()
plt.savefig('IV_Standardized_QQ_Plot.png')
plt.show()


print("\n--- 前三個數據點之詳細計算對照表 ---")
# 欄位定義：
# m: 序位, P: 機率位次, Obs(C): 觀測溫度, Obs(Z): 觀測標準化值
# Norm(C): 常態理論溫度, Norm(Z): 常態理論Z分數
# Logn(C): 對數常態理論溫度, Logn(Z): 對數常態理論Z分數

header = (f"{'Rank(m)':<7} | {'Prob(P)':<8} | {'Obs(°C)':<7} | {'Obs(Z)':<7} | "
          f"{'Norm(°C)':<8} | {'Norm(Z)':<7} | {'Logn(°C)':<8} | {'Logn(Z)':<7}")
print(header)
print("-" * 85)

for i in range(3):
    m = i + 1
    p = p_plotting[i]
    obs_c = data_sorted[i]
    
    # 1. 觀測值的 Z 分數 (以常態分佈為例，展示其在標準化空間的位置)
    obs_z = (obs_c - mu_n) / sigma_n
    
    # 2. 理論 Z 分數 (X座標，由機率位次決定)
    theo_z = norm.ppf(p)
    
    # 3. 常態理論值
    norm_c = mu_n + theo_z * sigma_n
    
    # 4. 對數常態理論值 (ln空間計算後取 exp)
    logy_z = theo_z # 理論 Z 是一樣的，因為橫軸都是理論分位數
    logn_c = np.exp(mu_y + logy_z * sigma_y)
    
    print(f"{m:<7d} | {p:<8.4f} | {obs_c:<7.2f} | {obs_z:<7.3f} | "
          f"{norm_c:<8.3f} | {theo_z:<7.3f} | {logn_c:<8.3f} | {theo_z:<7.3f}")

print("\n註：Normal Theo.(z) 與 Lognorm Theo.(z) 數值相同，係因兩者皆對應相同之理論分位數 (Theoretical Quantiles)。")