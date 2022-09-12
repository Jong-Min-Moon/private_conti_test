import sys
sys.path.append("/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/modules") 

from tester_new import twoSampleDiscTester, two_sample_generator_mean_departure
import pandas as pd
import numpy as np
import torch

USE_CUDA = torch.cuda.is_available() 
print(f"cuda available: {USE_CUDA}")

device = torch.device('cuda:0' if USE_CUDA else 'cpu') 
print(f"code run on device:: {device}")

values_d = [2,3,4,5,6,7,8]

d_trend = pd.DataFrame({
    "d" : values_d,
    "power" : np.repeat(np.nan, 7)
    })

for i in range(len(values_d)):
    tester = twoSampleDiscTester(
        gamma = 0.05,
        cuda_device = device,
        seed = 0,
        kappa = 3)
    d_now = values_d[i]
    generator = two_sample_generator_mean_departure(
        cuda_device = device,
        n1 = 10000,
        n2 = 10000,
        d = d_now)
    d_trend.loc[i,"power"] = (1/500)*tester.estimate_power(
        data_generator = generator,
        alpha = 0.8,
        B = 300,
        n_test = 500)

d_trend.to_csv("/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/simulation_runs/two_sample/disc/mean_departure_d_trend.csv")
