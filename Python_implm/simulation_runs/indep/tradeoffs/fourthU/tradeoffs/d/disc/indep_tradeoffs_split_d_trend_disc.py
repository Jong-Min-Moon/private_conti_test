import sys
sys.path.append("/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/modules") 

from tester_new import indepSplitDiscTester, indep_generator_nontrivial
import pandas as pd
import numpy as np
import torch

USE_CUDA = torch.cuda.is_available() 
print(f"cuda available: {USE_CUDA}")

device = torch.device('cuda:0' if USE_CUDA else 'cpu') 
print(f"code run on device:: {device}")


values_param = np.array([1,2,3,4,5,6,7])
n_param = values_param.shape[0]
trend = pd.DataFrame({
    "param" : values_param,
    "power" : np.repeat(np.nan, n_param)
    })

for i in range(n_param):
    param_now = values_param[i]

    tester = indepSplitDiscTester(
        gamma = 0.05,
        cuda_device = device,
        seed = 0,
        kappa = 3)
    
    generator = indep_generator_nontrivial(
        cuda_device = device,
        n = 30000,
        d = param_now, ###varying d
        epsilon = 0.05
        )

    test_result = (1/500)*tester.estimate_power(
        data_generator = generator,
        alpha = 0.9,
        B = 300,
        n_test = 500)
    trend.loc[i,"power"] = test_result

trend.to_csv(
    "/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/simulation_runs/indep/tradeoffs/split/d/disc/indep_tradeoffs_split_d_trend_disc.csv"
    )



