print("job started")
import time
import torch
import sys
import matplotlib.pyplot as plt
sys.path.append("/mnt/nas/users/mjm/GitHub/private_conti_test/Python_implm/modules") 

from tester import LDPTwoSampleTester

USE_CUDA = torch.cuda.is_available() 
print(f"cuda available: {USE_CUDA}")

device = torch.device('cuda:0' if USE_CUDA else 'cpu') 
print(f"code run on device:: {device}")

n1 = 1000

n2 = n1
kappa = 5 #number of bins
####CHANGE HERE#####
alpha = 2.5 #privacy level
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

tester = LDPTwoSampleTester(device)

import data as datasets

sample_source = datasets.SSBlobs()




cdf_calculator = torch.distributions.normal.Normal(loc = 0.0, scale = 1.0)

test_results = torch.empty(nTests)
for rep in range(1):
    print(f"{rep+1}th run")

    tst_data = sample_source.sample(int(n1*0.8), rep)

    data_x = cdf_calculator.cdf(torch.tensor(tst_data.X).to(device))
    data_y = cdf_calculator.cdf(torch.tensor(tst_data.Y).to(device))
    
    plt.scatter(data_x[:,1], data_x[:,2])
    plt.savefig("plot.png")
    

