import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm
from scipy.optimize import minimize
df = pd.read_csv('Temperature.csv')
data = df['mean']

#1.MOM法(依照講義公式轉換)
x_bar=data.mean()
s=data.std()
cv=s / x_bar
mu_n_mom, sigma_n_mom = x_bar, s  #Normal MoMH0 參數
print("--- Method of Moments (MoM) ---")
print(f"Normal MoM 參數: mu={mu_n_mom:.5f}, sigma={sigma_n_mom:.5f}")
sigma_y_mom = np.sqrt(np.log(1 + cv**2)) #Lognormal MoM 參數
mu_y_mom = np.log(x_bar) - 0.5 * (sigma_y_mom**2)
print(f"Lognormal MoM 參數: mu={mu_y_mom:.5f}, sigma={sigma_y_mom:.5f}")

#2.MLE法
# Normal (常態分佈有公式解，直接使用scipy.stats.norm.fit)
mu_n_mle, sigma_n_mle = norm.fit(data)
print("--- Maximum Likelihood Estimation (MLE) ---")
print(f"MLE: mu={mu_n_mle:.5f}, sigma={sigma_n_mle:.5f}")

# Lognormal
def lognorm_nll(params, x):
    mu_y, sigma_y = params
    # PPT提示語法: s=sigma_y, scale=exp(mu_y)
    pdf_values = lognorm.pdf(x, s=sigma_y, scale=np.exp(mu_y))
    # 取對數求和並加負號 (將最大化問題轉為最小化)
    return -np.sum(np.log(np.maximum(pdf_values, 1e-10)))
initial_guess = [mu_y_mom, sigma_y_mom]
res_log = minimize(lognorm_nll, initial_guess, args=(data,), method='Nelder-Mead') #使用Nelder-Mead演算法
mu_y_mle, sigma_y_mle = res_log.x
print(f"MLE: mu_y={mu_y_mle:.5f}, sigma_y={sigma_y_mle:.5f}")

#直方圖 
plt.figure(figsize=(10, 6))
n = len(data)
k_sturges = int(np.ceil(np.log2(n)) + 1) 
plt.hist(data, bins=k_sturges, density=True, color='skyblue'\
         , edgecolor='black', alpha=0.7, label='Observations')

#疊加PDF曲線
x_axis = np.linspace(data.min()-0.5, data.max()+0.5, 200)
plt.plot(x_axis, norm.pdf(x_axis, mu_n_mle, sigma_n_mle), 'r-', lw=2, label='Normal')
plt.plot(x_axis, lognorm.pdf(x_axis, s=sigma_y_mle, scale=np.exp(mu_y_mle)), 'b--', lw=2, label='Lognormal')
plt.title('Probability Distribution Fitting for Annual Mean Temp')
plt.xlabel('Temperature (°C)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig('II.Probability Distribution Fitting.png')
plt.show()

#Q2比較圖

# # --- 繪圖部分：MoM 與 MLE 對照圖 ---
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
# x_axis = np.linspace(data.min()-0.5, data.max()+0.5, 200)

# # 1. 左圖：MoM 擬合結果
# ax1.hist(data, bins=k_sturges, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Observations')
# ax1.plot(x_axis, norm.pdf(x_axis, mu_n_mom, sigma_n_mom), 'r-', lw=2, label='Normal (MoM)')
# ax1.plot(x_axis, lognorm.pdf(x_axis, s=sigma_y_mom, scale=np.exp(mu_y_mom)), 'b--', lw=2, label='Lognormal (MoM)')
# ax1.set_title('Method of Moments (MoM) Fitting')
# ax1.set_xlabel('Temperature (°C)')
# ax1.set_ylabel('Probability Density')
# ax1.legend()
# ax1.grid(axis='y', alpha=0.3)

# # 2. 右圖：MLE 擬合結果
# ax2.hist(data, bins=k_sturges, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Observations')
# ax2.plot(x_axis, norm.pdf(x_axis, mu_n_mle, sigma_n_mle), 'r-', lw=2, label='Normal (MLE)')
# ax2.plot(x_axis, lognorm.pdf(x_axis, s=sigma_y_mle, scale=np.exp(mu_y_mle)), 'b--', lw=2, label='Lognormal (MLE)')
# ax2.set_title('Maximum Likelihood Estimation (MLE) Fitting')
# ax2.set_xlabel('Temperature (°C)')
# ax2.set_ylabel('Probability Density')
# ax2.legend()
# ax2.grid(axis='y', alpha=0.3)

# plt.tight_layout()
# plt.savefig('II.Probability_Distribution_Comparison.png')
# plt.show()