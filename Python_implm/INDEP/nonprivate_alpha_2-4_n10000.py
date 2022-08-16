print("job started")
import time
import torch
import sys
import datetime
sys.path.append("/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/modules") 

# set seed number for reproducibility
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

from tester import LDPIndepTester

USE_CUDA = torch.cuda.is_available() 
print(f"cuda available: {USE_CUDA}")

device = torch.device('cuda:0' if USE_CUDA else 'cpu') 
print(f"code run on device:: {device}")


####CHANGE HERE#####
n = 10000
####################
kappa = 5 #number of bins
alpha = 0.8*3 #privacy level
gamma = 0.05 # significance level
nTests = 10 #number of tests for power estimation
B = 30 # number of permutations
d = 2


print(f"""
simulation started at = {datetime.datetime.now()} \n
n = {n},\n
kappa = {kappa}, alpha = {alpha},\n
gamma = {gamma}, nTests = {nTests},\n
B = {B}, d = {d}
""")

start_time = time.time()

multiplier = 1/2
copula_mean = multiplier * torch.ones(d).to(device)



sigma = (0.5 * torch.ones(d,d) + 0.5 * torch.eye(d)).to(device)

print("copula_mean")
print(copula_mean)



print("sigma")
print(sigma)



tester = LDPIndepTester(device)
generator_Y = torch.distributions.multivariate_normal.MultivariateNormal(
    loc = copula_mean, 
    covariance_matrix = sigma)

cdf_calculator = torch.distributions.normal.Normal(loc = 0.0, scale = 1.0)

test_results = torch.empty(nTests)
noise_vars_Y = torch.empty(nTests)
noise_vars_Z = torch.empty(nTests)

for rep in range(nTests):
    print(f"\n{rep+1}th run")
    
    y_og = generator_Y.sample((n,))
    data_y = cdf_calculator.cdf(y_og)   
    data_z = cdf_calculator.cdf(-y_og)  
    test_results[rep], noise_vars_Y[rep], noise_vars_Z[rep] = tester.run_test_conti_data(
        B,
        data_y,
        data_z,
        kappa, alpha, gamma,
        #################
        discrete_noise = False
        ###############
        )
    print(f"result: {test_results[rep]}")
    print(f"power_upto_now: { torch.sum(test_results[:(rep+1)])/(rep+1) }")

    print(f"noise variance for Y: {noise_vars_Y[rep]}")
    print(f"average noise variance for Y upto now: { torch.sum(noise_vars_Y[:(rep+1)])/(rep+1) }")
    
    print(f"noise variance for Z: {noise_vars_Z[rep]}")
    print(f"average noise variance for Z upto now: { torch.sum(noise_vars_Z[:(rep+1)])/(rep+1) }")
  
print( f"power estimate : { torch.sum(test_results)/nTests }" )
print( f"elapsed time: { time.time() - start_time }" )
print( f"simulation ended at {datetime.datetime.now()}" )



