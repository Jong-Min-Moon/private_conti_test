import sys
sys.path.append("/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/modules") 

from tester_new import indepSplitContiTester, indep_generator_nontrivial
import pandas as pd
import numpy as np
import torch

USE_CUDA = torch.cuda.is_available() 
print(f"cuda available: {USE_CUDA}")

device = torch.device('cuda:0' if USE_CUDA else 'cpu') 
print(f"code run on device:: {device}")

n_n = 4
values_n = np.array([2,4,6,8])* 3000
n_trend = pd.DataFrame({
    "n" : values_n,
    "power" : np.repeat(np.nan, n_n)
    })

for i in range(n_n):
    tester = indepSplitContiTester(
        gamma = 0.05,
        cuda_device = device,
        seed = 0,
        kappa = 3)
    n_now = values_n[i]
    generator = indep_generator_nontrivial(
        cuda_device = device,
        n = n_now,
        d = 2,
        epsilon = 0.05
        )

    test_result = (1/500)*tester.estimate_power(
        data_generator = generator,
        alpha = 0.9,
        B = 300,
        n_test = 500)
    n_trend.loc[i,"power"] = test_result

n_trend.to_csv(
    "/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/simulation_runs/indep/split/tradeoffs/n/indep_split_n_trend.csv"
    )



