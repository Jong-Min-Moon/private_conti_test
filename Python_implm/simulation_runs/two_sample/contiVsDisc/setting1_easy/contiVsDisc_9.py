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

n_test = 500
n_permute = 300
alpha = 9
start = 10
ntry = 30
values_param = np.array([start + i* 10 for i in range(ntry)])
n_params = values_param.shape[0]
param_trend = pd.DataFrame({
    "param" : values_param,
    "power_conti" : np.repeat(np.nan, n_params),
    "power_disc" : np.repeat(np.nan, n_params)
    })


for i in range(n_params):
    tester_conti = twoSampleContiTester(
        gamma = 0.05,
        cuda_device = device,
        seed = 0,
        kappa = 3)
    
    tester_disc = twoSampleDiscTester(
        gamma = 0.05,
        cuda_device = device,
        seed = 0,
        kappa = 3)
    param_now = int(values_param[i])

    generator = two_sample_generator_var_departure(
        cuda_device = device,
        n1 = param_now,
        n2 = param_now,
        d = 2)

    test_result_conti = (1/n_test)*tester_conti.estimate_power(
        data_generator = generator,
        alpha = alpha,
        B = n_permute,
        n_test = n_test)

    test_result_disc = (1/n_test)*tester_disc.estimate_power(
        data_generator = generator,
        alpha = alpha,
        B = n_permute,
        n_test = n_test)

    print(test_result_conti)
    print(test_result_disc)

    param_trend.loc[i,"power_conti"] =test_result_conti
    param_trend.loc[i,"power_disc"] =test_result_disc
 

param_trend.to_csv(
    f"/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/simulation_runs/two_sample/contiVsDisc/contiVsDisc_{alpha}.csv")
