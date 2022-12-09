import sys
sys.path.append("/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/modules") 

from tester_new import twoSampleContiTester, two_sample_generator_mean_departure
import pandas as pd
import numpy as np
import torch

USE_CUDA = torch.cuda.is_available() 
print(f"cuda available: {USE_CUDA}")

device = torch.device('cuda:0' if USE_CUDA else 'cpu') 
print(f"code run on device:: {device}")



#alpha trend
n_alpha = 41
values_alpha = np.linspace(0.1, 1.2, 41)
alpha_trend = pd.DataFrame({
    "alpha" : values_alpha,
    "power" : np.repeat(np.nan, n_alpha)
    })

for i in range(n_alpha):
    tester = twoSampleContiTester(
        gamma = 0.05,
        cuda_device = device,
        seed = 0,
        kappa = 3)
    alpha_now = values_alpha[i]
    generator = two_sample_generator_mean_departure(
        cuda_device = device,
        n1 = 10000,
        n2 = 10000,
        d = 4)
    alpha_trend.loc[i,"power"] = (1/500)*tester.estimate_power(
        data_generator = generator,
        alpha = alpha_now,
        B = 300,
        n_test = 500)

alpha_trend.to_csv(
    "/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/simulation_runs/two_sample/mean_departure_alpha.csv"
    )

