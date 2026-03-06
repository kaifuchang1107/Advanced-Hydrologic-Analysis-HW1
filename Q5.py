import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, chi2
mu_y=3.14217    
sigma_y=0.02453 
n_sample=40     
n_sim=3     
np.random.seed(55688) #結果可重現

#產生合成資料並儲存成CSV
sim_datasets=[np.random.lognormal(mean=mu_y, sigma=sigma_y, size=n_sample) for _ in range(n_sim)]
df_csv=pd.DataFrame({f'Dataset_{i+1}': sim_datasets[i] for i in range(n_sim)})
df_csv.to_csv('synthetic_temperature_data.csv', index=False)

#卡方檢定
def perform_chi_square_test(data, mu, sigma, n_bins):
    n = len(data)
    counts, bin_edges = np.histogram(data, bins=n_bins)

    #計算理論累積機率與次數
    cdf_v = lognorm.cdf(bin_edges, s=sigma, scale=np.exp(mu))
    exp_probs = np.diff(cdf_v)
    exp_freq = n * exp_probs
    
    obs_list = list(counts)
    exp_list = list(exp_freq)
    
    #確保每一組Ei>=5 
    while True:
        #尋找第一個需要合併的索引
        low_idx = -1
        for i in range(len(exp_list)):
            if exp_list[i] < 5:
                low_idx = i
                break
        
        #若找不到或只剩一組則跳出
        if low_idx == -1 or len(exp_list) <= 1:
            break
            
        #若為最後一組則併入前一組，其餘併入後一組 
        if low_idx == len(exp_list) - 1:
            #向前合併
            exp_list[low_idx-1] += exp_list.pop(low_idx)
            obs_list[low_idx-1] += obs_list.pop(low_idx)
        else:
            #向後合併
            target_val = exp_list.pop(low_idx)
            exp_list[low_idx] += target_val
            
            target_obs = obs_list.pop(low_idx)
            obs_list[low_idx] += target_obs
            
    obs_final = np.array(obs_list)
    exp_final = np.array(exp_list)
    exp_final = (exp_final / exp_final.sum()) * n
    
    #計算統計量
    chi_sq_stat = np.sum((obs_final - exp_final)**2 / exp_final)
    
    #自由度計算:k_final-1-參數
    k_final = len(exp_final)
    dof = k_final - 1 - 2
    if dof <= 0: dof = 1 
    
    critical_v = chi2.ppf(1 - 0.05, dof)
    p_value = 1 - chi2.cdf(chi_sq_stat, dof)
    
    return {
        'obs': obs_final, 'exp': exp_final, 'chi_sq': chi_sq_stat,
        'dof': dof, 'critical': critical_v, 'p_val': p_value
    }

#繪圖
summary_list = []
k_bins = int(np.sqrt(n_sample)) + 1 # 根號40約7組

for i, data in enumerate(sim_datasets):
    m_val = np.mean(data)
    s_val = np.std(data, ddof=1)
    
    res = perform_chi_square_test(data, mu_y, sigma_y, k_bins)
    summary_list.append({
        'Dataset': f'Set {i+1}', 'Mean': m_val, 'Std': s_val,
        'Chi-sq': res['chi_sq'], 'P-value': res['p_val']
    })
    
    #PDF與CDF
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(data, bins=k_bins, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    x = np.linspace(data.min()*0.98, data.max()*1.02, 100)
    plt.plot(x, lognorm.pdf(x, s=sigma_y, scale=np.exp(mu_y)), 'r-')
    plt.title(f'Set {i+1}: Histogram & PDF')
    
    plt.subplot(1, 2, 2)
    sorted_x = np.sort(data)
    plt.step(sorted_x, np.arange(1, n_sample+1)/n_sample, where='post')
    plt.plot(sorted_x, lognorm.cdf(sorted_x, s=sigma_y, scale=np.exp(mu_y)), 'r--')
    plt.title(f'Set {i+1}: CDF Comparison')
    plt.tight_layout()
    plt.show()

#輸出摘要
df_summary = pd.DataFrame(summary_list)
print("\n--- Monte Carlo Simulation Summary Table ---")
print(df_summary.round(4))
