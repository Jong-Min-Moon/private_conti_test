import datetime
import time

from abc import ABCMeta, abstractmethod
from math import sqrt

import torch
import numpy as np
import random

import scipy.stats as stats

import matplotlib.pyplot as plt

from scipy.linalg import block_diag, sqrtm, inv, svd


class tester(object):
    """Abstract class for two sample tests."""
    __metaclass__ = ABCMeta

    def __init__(self, gamma, cuda_device, seed):
        """
        gamma: significance level of the test
        """
        self.gamma = gamma
        self.cuda_device = cuda_device
        self.seed = seed
    
    @abstractmethod   
    def estimate_power(self):
        raise NotImplementedError()
    
    @abstractmethod
    def permu_test(self):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self):
        """Compute the test statistic"""
        raise NotImplementedError()
        
    @abstractmethod
    def privatize(self):
        raise NotImplementedError()

    def LapU(self, data, d, alpha, c):
        ''' Only for continuous data.
        for each dimension, transform the data in [0,1] into the interval index
        first interval = [0, x], the others = (y z]
        
        input arguments
            data: torch tensor object on GPU of multivariate data
            d: number of categories of multivariate data
            alpha: privacy level
            c: noise scale paramter
        output
            LDPView: \alpha-LDP view of the input multivariate data
        '''
 
        sigma = c / alpha
        oneHot = self.transform_onehot(data, d)
        laplaceSize = oneHot.size()
        laplaceNoise = self.generate_unit_laplace(laplaceSize)
        LDPView = torch.sqrt(torch.tensor(d)) * oneHot + sigma * laplaceNoise
        return(LDPView)
      
    def h_bin(self, data, kappa): 
        ''' Only for continuous data
        input arguments
            data: torch tensor of continuous data
            kappa: number of bin in each dimension
        output
            torch tensor of multivariate data
        '''
               
        # create designated number of intervals
        d = self.get_dimension(data)
     
        # 1. for each dimension, turn the continuous data into interval
        # each row now indicates a hypercube in [0,1]^d
        # the more the data is closer to 1, the larger the interval index.
        dataBinIndex = self.transform_bin_index(data = data, nIntervals = kappa)
        
        # 2. for each datapoint(row),
        #    turn the hypercube data into a multivariate data of (1, 2, ..., kappa^d)
        #    each row now becomes an integer.
        dataMultivariate = self.TransformMultivariate(dataBinIndex, kappa)
        
        return(dataMultivariate)
    
    def transform_bin_index(self, data, nIntervals):
        ''' Only for continuous data.
        for each dimension, transform the data in [0,1] into the interval index
        first interval = [0, x], the others = (y z]
        
        input arguments
            data: torch tensor object on GPU
            nIntervals: integer
        output
            dataIndices: torch tensor, dimension same as the input
        '''
        # create designated number of intervals
        d = self.get_dimension(data)
        breaks = torch.linspace(start = 0, end = 1, steps = nIntervals + 1).to(self.cuda_device) #floatTensor
        dataIndices = torch.bucketize(data, breaks, right = False) # ( ] form.
        dataIndices = dataIndices.add(
            dataIndices.eq(0)
        ) #move 0 values from the bin number 0 to the bin number 1        
        return(dataIndices)    

    def TransformMultivariate(self, dataBinIndex, nBin):
        """Only for continuous and multivariate data ."""
        d = self.get_dimension(dataBinIndex)
        
        if d == 1:
            return(dataInterval.sub(1))
        else:
            exponent = torch.linspace(start = (d-1), end = 0, steps = d, dtype = torch.long)
            vector = torch.tensor(nBin).pow(exponent)
            return( torch.matmul( dataBinIndex.sub(1).to(torch.float), vector.to(torch.float).to(self.cuda_device) ).to(torch.long) )   
    
    def generate_unit_laplace(self, size):
        '''
        input: torch.size object
        output: torch tensor of data from unit laplace distribution
        '''
     
        unit_laplace_generator = torch.distributions.laplace.Laplace(
            torch.tensor(0.0).to(self.cuda_device),
            torch.tensor(2**(-1/2)).to(self.cuda_device)
        )
        return unit_laplace_generator.sample(sample_shape = size)
        
    @staticmethod
    def transform_onehot(dataMultivariate, d):
        return(
            torch.nn.functional.one_hot(
                dataMultivariate,
                num_classes = d)
        )
 
    @staticmethod
    def get_dimension(data):
        if data.dim() == 1:
            return(1)
        elif data.dim() == 2:
            return( data.size(dim = 1) )
        else:
            return # we only use up to 2-dimensional tensor, i.e. matrix

    @staticmethod        
    def range_check(self, data):
        if (torch.sum(data.gt(1))).gt(0):
            print("check data range")
            return False
        elif (torch.sum(data.lt(0))).gt(0):
            print("check data range")
            return False
        else:
            return True

class data_generator(object):
    """Abstract class for two sample tests."""
    __metaclass__ = ABCMeta

    def __init__(self, cuda_device, seed):
        self.cuda_device = cuda_device
        self.seed = seed
        self.cdf_calculator = torch.distributions.normal.Normal(loc = 0.0, scale = 1.0)
    
   # @abstractmethod   
    #def generate_y(self):
        #raise NotImplementedError("implement generate_y")
        
    @abstractmethod   
    def generate_z(self):
        raise NotImplementedError("implement generate_z")
        
    def calculate_cdf(self, data):
        return self.cdf_calculator.cdf(data)
    

class twoSampleContiTester(tester):
    def __init__(self, gamma, cuda_device, seed, kappa):
        super(twoSampleContiTester, self).__init__(gamma, cuda_device, seed)
        self.kappa = kappa
    
    def estimate_power(self, data_generator, alpha, B, n_test):
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        start_time = time.time()
        print(f"""
        simulation started at = {datetime.datetime.now()} \n
        n1 = {data_generator.n1}, n2 = {data_generator.n2}, \n
        kappa = {self.kappa}, alpha = {alpha},\n
        gamma = {self.gamma}, nTests = {n_test},\n
        B = {B}, d = {data_generator.d}
        """)
        test_results = torch.empty(n_test)
        
        for rep in range(n_test):
            print(f"\n{rep+1}th run")
            tst_data_y = data_generator.generate_y()
            tst_data_z = data_generator.generate_z()
            test_results[rep] = self.permu_test(tst_data_y, tst_data_z, alpha, B)
            print(f"result: {test_results[rep]}")
            print(f"power_upto_now: { torch.sum(test_results[:(rep+1)])/(rep+1) }")
  
        print( f"power estimate : { torch.sum(test_results)/n_test }" )
        print( f"elapsed time: { time.time() - start_time }" )
        print( f"simulation ended at {datetime.datetime.now()}" )
        return(torch.sum(test_results)/n_test)
        
    def permu_test(self, tst_data_y, tst_data_z, alpha, B): 
        n_1 = tst_data_y.size(dim = 0)
        tst_data_priv = self.privatize(tst_data_y, tst_data_z, alpha)
        n = tst_data_priv.size(dim = 0)
        
        #original statistic
        ustatOriginal = self.compute_stat(tst_data_priv[:n_1,:], tst_data_priv[n_1:,:])
        print(f"original u-statistic:{ustatOriginal}")
        
        #permutation procedure
        permStats = torch.empty(B).to(self.cuda_device)
        
        for i in range(B):
            permutation = torch.randperm(n)
            perm_stat_now = self.compute_stat(
                tst_data_priv[permutation][:n_1,:],
                tst_data_priv[permutation][n_1:,:]
            ).to(self.cuda_device)
            permStats[i] = perm_stat_now

               
        p_value_proxy = (1 +
                         torch.sum(
                             torch.gt(input = permStats, other = ustatOriginal)
                         )
                        ) / (B + 1)
      
        print(f"p value proxy: {p_value_proxy}")
        return(p_value_proxy < self.gamma)#test result: TRUE = 1 = reject the null, FALSE = 0 = retain the null.    
 
    def compute_stat(self, tst_data_y_priv, tst_data_z_priv):
        n_1 = torch.tensor(tst_data_y_priv.size(dim = 0))
        n_2 = torch.tensor(tst_data_z_priv.size(dim = 0))
    
        y_row_sum = torch.sum(tst_data_y_priv, axis = 0)
        z_row_sum = torch.sum(tst_data_z_priv, axis = 0)
        phi_psi = torch.einsum('ji,jk->ik', tst_data_y_priv, tst_data_z_priv)


        one_Phi_one = torch.inner(y_row_sum, y_row_sum)
        one_Psi_one = torch.inner(z_row_sum, z_row_sum)

        tr_Phi = torch.sum(torch.square(tst_data_y_priv))
        tr_Psi = torch.sum(torch.square(tst_data_z_priv))

        one_Phi_tilde_one = one_Phi_one - tr_Phi
        one_Psi_tilde_one = one_Psi_one - tr_Psi

        onePhioneonePsione = one_Phi_tilde_one * one_Psi_tilde_one

        # y only part. log calculation in case of large n1
        sign_y = torch.sign(one_Phi_tilde_one)
        abs_u_y = torch.exp(torch.log(torch.abs(one_Phi_tilde_one)) - torch.log(n_1) - torch.log(n_1 - 1) )
        u_y = sign_y * abs_u_y


        # z only part. log calculation in case of large n2
        sign_z = torch.sign(one_Psi_tilde_one)

        abs_u_z = torch.exp(torch.log(torch.abs(one_Psi_tilde_one)) - torch.log(n_2) - torch.log(n_2 - 1) )
        u_z = sign_z * abs_u_z

        # cross part
        cross = torch.inner(y_row_sum, z_row_sum)
        sign_cross = torch.sign(cross)
        abs_cross = torch.exp(torch.log(torch.abs(cross)) +torch.log(torch.tensor(2))- torch.log(n_1) - torch.log(n_2) )
        u_cross = sign_cross * abs_cross

        return(u_y + u_z - u_cross)
    
        
    def privatize(self, tst_data_y, tst_data_z, alpha):
        d = self.kappa ** tst_data_y.size(dim = 1)
        c = torch.sqrt(torch.tensor(8 * d))
        tst_data_y_multi = self.h_bin(tst_data_y, self.kappa)
        tst_data_z_multi = self.h_bin(tst_data_z, self.kappa) 
        dataCombined = torch.cat([tst_data_y_multi, tst_data_z_multi], dim = 0)
        tst_data_priv = self.LapU(dataCombined, d, alpha, c)
        return(tst_data_priv)

    
class two_sample_generator_mean_departure(data_generator):
    def __init__(self, cuda_device, seed, n1, n2, d):
        super(two_sample_generator_mean_departure, self).__init__(cuda_device, seed)
        self.n1 = n1
        self.n2 = n2
        self.d = d

        copula_mean_y = -1/2 * torch.ones(d).to(self.cuda_device)
        copula_mean_z =  1/2 * torch.ones(d).to(self.cuda_device)

        sigma = (0.5 * torch.ones(d,d) + 0.5 * torch.eye(d)).to(self.cuda_device)


        print("copula_mean_y")
        print(copula_mean_y)

        print("copula_mean_z")
        print(copula_mean_z)

        print("sigma")
        print(sigma)

        self.generator_y = torch.distributions.multivariate_normal.MultivariateNormal(
            loc = copula_mean_y, 
            covariance_matrix = sigma)
        self.generator_z = torch.distributions.multivariate_normal.MultivariateNormal(
            loc = copula_mean_z,
            covariance_matrix = sigma)
        
    def generate_y(self):
            normalSample = self.generator_y.sample( (self.n1,) )
            return( self.calculate_cdf(normalSample) )  
        
    def generate_z(self):
            return(
                self.calculate_cdf(
                    self.generator_z.sample( (self.n2,) )
                )
            )