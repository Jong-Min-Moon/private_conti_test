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

from tester import LDPTwoSampleTester

USE_CUDA = torch.cuda.is_available() 
print(f"cuda available: {USE_CUDA}")

device = torch.device('cuda:0' if USE_CUDA else 'cpu') 
print(f"code run on device:: {device}")

n1 = 4000
n2 = n1
kappa = 3 #number of bins
alpha = 0.8 #privacy level
gamma = 0.05 # significance level
nTests = 500 #number of tests for power estimation
B = 300 # number of permutations

d = 4

print(f"""
simulation started at = {datetime.datetime.now()} \n
n1 = {n1}, n2 = {n2},\n
kappa = {kappa}, alpha = {alpha},\n
gamma = {gamma}, nTests = {nTests},\n
B = {B}, d = {d}
""")

start_time = time.time()

multiplier = 1/2
copula_mean_1 = -multiplier * torch.ones(d).to(device)
copula_mean_2 =  multiplier * torch.ones(d).to(device)


sigma = (0.5 * torch.ones(d,d) + 0.5 * torch.eye(d)).to(device)

print("copula_mean_1")
print(copula_mean_1)

print("copula_mean_2")
print(copula_mean_2)

print("sigma")
print(sigma)

tester = LDPTwoSampleTester(device)
generator_X = torch.distributions.multivariate_normal.MultivariateNormal(
    loc = copula_mean_1, 
    covariance_matrix = sigma)
generator_Y = torch.distributions.multivariate_normal.MultivariateNormal(
    loc = copula_mean_2,
    covariance_matrix = sigma)
cdf_calculator = torch.distributions.normal.Normal(loc = 0.0, scale = 1.0)

test_results = torch.empty(nTests)
noise_vars = torch.empty(nTests)

for rep in range(nTests):
    print(f"\n{rep+1}th run")
    
    data_x = cdf_calculator.cdf(generator_X.sample((n1,)))
    data_y = cdf_calculator.cdf(generator_Y.sample((n2,)))
    
    
    test_results[rep], noise_vars[rep] = tester.run_test_conti_data(B, data_x, data_y,
                                             kappa, alpha, gamma,
                                             #################
                                             discrete = True
                                             ###############
                                            )
    print(f"result: {test_results[rep]}")
    print(f"rejected upto now: {torch.sum(test_results[:(rep+1)])}")
    print(f"number of test run upto now: {(rep+1) }")
    print(f"power_upto_now: { torch.sum(test_results.int()[:(rep+1)])/(rep+1) }")

    print(f"noise variance: {noise_vars[rep]}")
    print(f"average noise variance upto now: { torch.sum(noise_vars[:(rep+1)])/(rep+1) }")
  
print( f"power estimate : { torch.sum(test_results.int())/nTests }" )
print( f"elapsed time: { time.time() - start_time }" )
print( f"simulation ended at {datetime.datetime.now()}" )


