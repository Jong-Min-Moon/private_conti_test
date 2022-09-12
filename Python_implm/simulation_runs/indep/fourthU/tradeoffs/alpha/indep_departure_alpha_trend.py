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

n_alpha = 41
values_alpha = np.linspace(0.7, 1.8, 41)
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
    alpha_now = values_alpha[i]
    generator = indep_generator_nontrivial(
        cuda_device = device,
        n = 150000,
        d = 2,
        epsilon = 0.05
        )

    test_result = (1/500)*tester.estimate_power(
        data_generator = generator,
        alpha = alpha_now,
        B = 300,
        n_test = 500).item()
    alpha_trend.loc[i,"power"] = test_result

alpha_trend.to_csv(
    "/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/simulation_runs/indep/tradeoffs/alpha/indep_departure_alpha_trend.csv"
    )



