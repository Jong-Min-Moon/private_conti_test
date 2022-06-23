print("job started")
import time
import torch
import sys
sys.path.append("/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/modules") 

from tester import LDPTwoSampleTester

USE_CUDA = torch.cuda.is_available() 
print(f"cuda available: {USE_CUDA}")

device = torch.device('cuda:0' if USE_CUDA else 'cpu') 
print(f"code run on device:: {device}")

n1 = 20000
n2 = 20000

kappa = 3 #number of bins
####CHANGE HERE#####
alpha = 0.15 #privacy level
####################
gamma = 0.05 # significance level
nTests = 500 #number of tests for power estimation
B = 300 # number of permutations

print(f"""
n1 = {n1}, n2 = {n2},\n
kappa = {kappa}, alpha = {alpha},\n
gamma = {gamma}, nTests = {nTests},\n
B = {B}
""")

start_time = time.time()

copula_mean = torch.tensor([0.0, 0.0, 0.0]).to(device)
sigma_1 = torch.tensor([
    [1.0, 0.5, 0.5],
    [0.5, 1.0, 0.5], 
    [0.5, 0.5, 1.0]
    ]).to(device)
sigma_2 = torch.tensor([
    [5.0, 2.5, 2.5],
    [2.5, 5.0, 2.5], 
    [2.5, 2.5, 5.0],
    ]).to(device)

tester = LDPTwoSampleTester(device)
generator_X = torch.distributions.multivariate_normal.MultivariateNormal(
    loc = copula_mean, 
    covariance_matrix = sigma_1)
generator_Y = torch.distributions.multivariate_normal.MultivariateNormal(
    loc = copula_mean,
    covariance_matrix = sigma_2)
cdf_calculator = torch.distributions.normal.Normal(loc = 0.0, scale = 1.0)

test_results = torch.empty(nTests)
for rep in range(nTests):
    print(f"{rep+1}th run")
    
    data_x = cdf_calculator.cdf(generator_X.sample((n1,)))
    data_y = cdf_calculator.cdf(generator_Y.sample((n2,)))
    
    
    result_now = tester.run_test_conti_data(B, data_x, data_y,
                                             kappa, alpha, gamma, discrete = False
                                            )
    test_results[rep] = result_now
    print(f"result: {result_now}")
    print(f"power_upto_now: { torch.sum(test_results[:(rep+1)])/(rep+1) }")

  
print( f"power estimate : { torch.sum(test_results)/nTests }" )
print( f"elapsed time: { time.time() - start_time }" )


