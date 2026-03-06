import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, chi2

df = pd.read_csv('Temperature.csv')
data = df['mean']
n = len(data)
print(f"樣本數 (n): {n}")

#Normal distbribution參數
mu_n = 23.16100
sigma_n = 0.56884
#lognormal distribution參數
mu_y = 3.14217
sigma_y = 0.02453

k = int(np.ceil(np.log2(n)) + 1) #分組數
counts, bin_edges = np.histogram(data, bins=k)

def perform_chi_square_test(dist_name, bin_edges, obs_freq, mu, sigma, is_lognorm=False):
    # 計算初始理論機率與次數
    if not is_lognorm:
        cdf_v = norm.cdf(bin_edges, mu, sigma)
    else:
        cdf_v = lognorm.cdf(bin_edges, s=sigma, scale=np.exp(mu))
    
    exp_probs = np.diff(cdf_v)
    exp_freq = n * exp_probs
    
    # 轉為 list 進行合併處理 (確保 Ei >= 5)
    obs_list = list(obs_freq)
    exp_list = list(exp_freq)
    
    # 合併邏輯
    i = 0
    while i < len(exp_list):
        if exp_list[i] < 5: #確保 Ei >= 5
            if i == len(exp_list) - 1 and i > 0: # 最後一組太小，併入前一組
                exp_list[i-1] += exp_list.pop(i)
                obs_list[i-1] += obs_list.pop(i)
            elif i < len(exp_list) - 1: # 非最後一組，併入後一組
                exp_list[i+1] += exp_list.pop(i)
                obs_list[i+1] += obs_list.pop(i)
            else:
                i += 1
        else:
            i += 1
            
    # 計算統計量
    obs_final = np.array(obs_list)
    exp_final = np.array(exp_list)
    chi_sq_stat = np.sum((obs_final - exp_final)**2 / exp_final)
    
    # 自由度計算: k_final - 1 - 參數個數(2)
    k_final = len(exp_final)
    dof = k_final - 1 - 2
    critical_v = chi2.ppf(1 - 0.05, dof)
    p_value = 1 - chi2.cdf(chi_sq_stat, dof)
    
    return {
        'name': dist_name,
        'obs': obs_final,
        'exp': exp_final,
        'chi_sq': chi_sq_stat,
        'dof': dof,
        'critical': critical_v,
        'p_val': p_value
    }

# 執行檢定
res_n = perform_chi_square_test("Normal", bin_edges, counts, mu_n, sigma_n)
res_ln = perform_chi_square_test("Lognormal", bin_edges, counts, mu_y, sigma_y, is_lognorm=True)

# 輸出結果報告
for res in [res_n, res_ln]:
    print(f"=== {res['name']} Distribution Chi-Square Test ===")
    print(f"合併後組數: {len(res['exp'])}")
    print(f"觀測次數 (O_i): {res['obs']}")
    print(f"理論次數 (E_i): {np.round(res['exp'], 2)}")
    print(f"自由度 (df): {res['dof']}")
    print(f"卡方統計量 (Chi-sq): {res['chi_sq']:.4f}")
    print(f"臨界值 (Critical Value): {res['critical']:.4f}")
    print(f"P-value: {res['p_val']:.4f}")
    
    if res['chi_sq'] < res['critical']:
        print(f"結論: 接受 H0 (擬合良好)\n")
    else:
        print(f"結論: 拒絕 H0 (擬合不佳)\n")

# 比較兩者
if res_ln['chi_sq'] < res_n['chi_sq']:
    print(f">>> 最終建議：Lognormal 分佈之卡方統計量較小，為本案例之最佳擬合模型。")
else:
    print(f">>> 最終建議：Normal 分佈之卡方統計量較小，為本案例之最佳擬合模型。")