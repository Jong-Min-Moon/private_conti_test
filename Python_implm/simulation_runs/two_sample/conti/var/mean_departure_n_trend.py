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




#n trend
n_n=20
values_n = np.linspace(1000, 20000, n_n)
n_trend = pd.DataFrame({
    "n" : values_n,
    "power" : np.repeat(np.nan, n_n)
    })


for i in range(n_n):
    tester = twoSampleContiTester(
        gamma = 0.05,
        cuda_device = device,
        seed = 0,
        kappa = 3)
    n_now = int(values_n[i])
    generator = two_sample_generator_mean_departure(
        cuda_device = device,
        n1 = n_now,
        n2 = n_now,
        d = 4)
    n_trend.loc[i,"power"] = (1/500)*tester.estimate_power(
        data_generator = generator,
        alpha = 0.8,
        B = 300,
        n_test = 500)

n_trend.to_csv("/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/simulation_runs/two_sample/conti/mean_departure_n_trend.csv")
