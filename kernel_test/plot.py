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

n1 = 15000

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
    x = torch.tensor(tst_data.X).to(device)
    y = torch.tensor(tst_data.Y).to(device)

    data_x = cdf_calculator.cdf(x)
    data_y = cdf_calculator.cdf(y)
    
    fig1, ax1 = plt.subplots()
    ax1.scatter(data_x[:,0], data_x[:,1], color = 'blue', s = 4, alpha = 0.4)
    ax1.scatter(data_y[:,0], data_y[:,1], color = 'red', s = 4, alpha = 0.4)
    fig1.savefig("/mnt/nas/users/mjm/GitHub/private_conti_test/kernel_test/plot_cdf.png")
    
    fig2, ax2 = plt.subplots()

    x_min = torch.min(x, 0).values
    x_max = torch.max(x, 0).values
    y_min = torch.min(y, 0).values
    y_max = torch.max(y, 0).values
    
    x_scaled = (x.sub(x_min)).div( torch.sub(x_max, x_min) )
    y_scaled = (y.sub(y_min)).div( torch.sub(y_max, y_min) )

    



    ax2.scatter(x_scaled[:,0], x_scaled[:,1], color = 'blue', s = 4, alpha = 0.4)
    ax2.scatter(y_scaled[:,0], y_scaled[:,1], color = 'red', s = 4, alpha = 0.4)
    fig2.savefig("/mnt/nas/users/mjm/GitHub/private_conti_test/kernel_test/plot_normed.png")
