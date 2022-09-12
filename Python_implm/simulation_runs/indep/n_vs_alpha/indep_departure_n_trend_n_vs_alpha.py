import sys
sys.path.append("/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/modules") 

from tester_new import indepContiTester, indep_generator_nontrivial
import pandas as pd
import numpy as np
import torch

USE_CUDA = torch.cuda.is_available() 
print(f"cuda available: {USE_CUDA}")

device = torch.device('cuda:0' if USE_CUDA else 'cpu') 
print(f"code run on device:: {device}")




#n trend

#values_n = np.linspace(1000, 20000, n_n)
values_n = [39000,
52000,
65000,
78000,
91000,
104000,
117000,
130000,
143000,
156000,
169000,
182000,
195000,
208000,
221000]
n_n = len(values_n)
n_trend = pd.DataFrame({
    "n" : values_n,
    "power" : np.repeat(np.nan, n_n)
    })


for i in range(n_n):
    tester = indepContiTester(
        gamma = 0.05,
        cuda_device = device,
        seed = 0,
        kappa = 3)
    n_now = int(values_n[i])
    generator = indep_generator_nontrivial(
        cuda_device = device,
        n = n_now,
        d = 2,
        epsilon = 0.05)
    test_result = tester.estimate_power(
        data_generator = generator,
        alpha = 1.2,
        B = 300,
        n_test = 500).item()
    print(test_result)
    n_trend.loc[i,"power"] = (1/500)*test_result

n_trend.to_csv("/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/simulation_runs/indep/n_vs_alpha/indep_departure_n_trend_n_vs_alpha.csv")
