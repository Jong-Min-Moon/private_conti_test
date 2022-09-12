import sys
sys.path.append("/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/modules") 

from tester_new import twoSampleContiTester, twoSampleDiscTester, two_sample_generator_var_departure
import pandas as pd
import numpy as np
import torch

USE_CUDA = torch.cuda.is_available() 
print(f"cuda available: {USE_CUDA}")

device = torch.device('cuda:0' if USE_CUDA else 'cpu') 
print(f"code run on device:: {device}")

n_n=20
values_n = np.linspace(5000, 20000, n_n)
n_trend = pd.DataFrame({
    "n" : values_n,
    "power_conti" : np.repeat(np.nan, n_n),
    "power_disc" : np.repeat(np.nan, n_n)
    })


for i in range(n_n):
    tester_conti = twoSampleContiTester(
        gamma = 0.05,
        cuda_device = device,
        seed = 0,
        kappa = 5)
    
    tester_disc = twoSampleDiscTester(
        gamma = 0.05,
        cuda_device = device,
        seed = 0,
        kappa = 5)
    n_now = int(values_n[i])

    generator = two_sample_generator_var_departure(
        cuda_device = device,
        n1 = n_now,
        n2 = n_now,
        d = 2)

    test_result_conti = (1/500)*tester_conti.estimate_power(
        data_generator = generator,
        alpha = 8,
        B = 300,
        n_test = 500)

    test_result_disc = (1/500)*tester_disc.estimate_power(
        data_generator = generator,
        alpha = 8,
        B = 300,
        n_test = 500)

    print(test_result_conti)
    print(test_result_disc)

    n_trend.loc[i,"power_conti"] =test_result_conti
    n_trend.loc[i,"power_disc"] =test_result_disc
 

d_trend.to_csv("/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/simulation_runs/two_sample/contiVsDisc/contiVsDisc_highkappa.csv")
