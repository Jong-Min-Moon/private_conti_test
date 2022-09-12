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



#alpha trend

values_alpha = [0.36,
0.48,
0.6,
0.72,
0.84,
0.96,
1.08,
1.2,
1.32,
1.44,
1.56,
1.68,
1.8,
1.92,
2.04]
n_alpha  = len(values_alpha)
alpha_trend = pd.DataFrame({
    "alpha" : values_alpha,
    "power" : np.repeat(np.nan, n_alpha)
    })

for i in range(n_alpha):
    tester = indepContiTester(
    gamma = 0.05,
    cuda_device = device,
    seed = 0,
    kappa = 3)
    alpha_now = int(values_alpha[i])
    generator = indep_generator_nontrivial(
        cuda_device = device,
        n = 130000,
        d = 2,
        epsilon = 0.05)
    test_result = tester.estimate_power(
        data_generator = generator,
        alpha = alpha_now,
        B = 300,
        n_test = 500).item()
    alpha_trend.loc[i,"power"] = (1/500)*test_result

alpha_trend.to_csv(
    "/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/simulation_runs/indep/n_vs_alpha/indep_departure_alpha_trend_n_vs_alpha.csv"
    )

