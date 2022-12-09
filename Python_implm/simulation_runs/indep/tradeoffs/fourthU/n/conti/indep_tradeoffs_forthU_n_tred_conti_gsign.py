import sys
sys.path.append("/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/modules") 

from tester_new import indepContiTester, indep_generator_gsign
import pandas as pd
import numpy as np
import torch

USE_CUDA = torch.cuda.is_available() 
print(f"cuda available: {USE_CUDA}")

device = torch.device('cuda:0' if USE_CUDA else 'cpu') 
print(f"code run on device:: {device}")




#n trend
values_param = np.array([1,2,3,4,5,6,7,8])*100000
n_param = len(values_param)
param_trend = pd.DataFrame({
    "n" : values_param,
    "power" : np.repeat(np.nan, n_param)
    })


for i in range(n_param):
    tester = indepContiTester(
        gamma = 0.05,
        cuda_device = device,
        seed = 0,
        kappa = 3)

    param_now = int(values_param[i])
    generator = indep_generator_gsign(
        cuda_device = device,
        n = param_now,
        d = 1)
    test_result = tester.estimate_power(
        data_generator = generator,
        alpha = 0.9,
        B = 300,
        n_test = 500)
    print(test_result)
    param_trend.loc[i,"power"] = (1/500)*test_result

param_trend.to_csv("/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/simulation_runs/indep/tradeoffs/fourthU/n/conti/indep_tradeoffs_forthU_n_tred_conti_gsign.csv")
