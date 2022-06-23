import torch
import itertools
from scipy.special import comb

class LDPIndepTester:
    def __init__(self, cuda_device):
        self.cuda_device = cuda_device


    def run_test_conti_data(self, B, data_X, data_Y, kappa, alpha, gamma, discrete = False):
        dataPrivatized, noise_var = self.preprocess_conti_data(data_X, data_Y, kappa, alpha, discrete)
        
        n_1 = data_X.size(dim = 0)
        
 
        
        ustatOriginal = self.u_stat_twosample(dataPrivatized, n_1)
        print(f"original u-statistic:{ustatOriginal}")
        
        #permutation procedure
        permStats = torch.empty(B).to(self.cuda_device)
        
        for i in range(B):
            perm_stat_now = self.u_stat_twosample(
                dataPrivatized[torch.randperm(dataPrivatized.size(dim=0))],
                n_1).to(self.cuda_device)
            permStats[i] = perm_stat_now
            #print(perm_stat_now)
         
        
        p_value_proxy = (1 +
                         torch.sum(
                             torch.gt(input = permStats, other = ustatOriginal)
                         )
                        ) / (B + 1)
        
        
        print(f"p value proxy: {p_value_proxy}")
        return(p_value_proxy < gamma, noise_var)#test result: TRUE = 1 = reject the null, FALSE = 0 = retain the null.
    
    def bin_separately(self, data_X, data_Y, kappa):
        return (
            self.bin(data_X, kappa),
            self.bin(data_Y, kappa)
            )

    def preprocess_conti_data(self, data_X, data_Y, kappa, alpha, discrete):
        data_X_binned, data_Y_binned = self.bin_separately(data_X, data_Y, kappa)
        
        dataCombined = torch.cat([data_X_binned, data_Y_binned], dim = 0)
        dataPrivatized, noise_var = self.privatize_twosample(dataCombined, alpha, discrete)
        return(dataPrivatized, noise_var)
        

    def indep_test(self, data_Y, data_Z, kappa, alpha):
        #1. bin
        data_Y_binned, data_Z_binned = self.bin_separately(data_Y, data_Z, kappa)

        #2. scaling factors for noises
        d_1 = data_Y.size(dim = 1)
        d_2 = data_Z.size(dim = 1)
        indicator_scale = torch.tensor(kappa)**{0.5*d_1 + 0.5*d_2}
        sigma_kappa = torch.tensor(
            (sqrt(32) * indicator_scale)/alpha
        )

        #3. generate laplaces
        data_Y_privatized = torch.add(
                input = generate_unit_laplace(data_Y_binned).reshape(n, -1),
                alpha = sigma_kappa,
                other = indicator_scale * data_Y
            ), 

        data_Z_privatized = torch.add(
                input = generate_unit_laplace(data_Z_binned).reshape(n, -1),
                alpha = sigma_kappa,
                other = indicator_scale * data_Z
            )
        
        #4 compute original u-stat
        ustatOriginal = self.u_stat_indep(data_Y_privatized, data_Z_privatized)

        print(f"original u-statistic:{ustatOriginal}")
        
        #permutation procedure
        permStats = torch.empty(B).to(self.cuda_device)
        
        for i in range(B):
            perm_stat_now = self.u_stat_indep(
                data_Y_privatized,
                data_Z_privatized[torch.randperm(dataPrivatized.size(dim=0))],
                n_1).to(self.cuda_device)

            permStats[i] = perm_stat_now
            #print(perm_stat_now)
         
        
        p_value_proxy = (1 +
                         torch.sum(
                             torch.gt(input = permStats, other = ustatOriginal)
                         )
                        ) / (B + 1)
        
        
        print(f"p value proxy: {p_value_proxy}")



    def generate_unit_laplace(self, data):
        n = data.size(dim = 0)
        d = data.size(dim = 1)
        unit_laplace_generator = torch.distributions.laplace.Laplace(
            torch.tensor(0.0).to(self.cuda_device),
            torch.tensor(2**(-1/2)).to(self.cuda_device)
        )
        return unit_laplace_generator.sample(sample_shape = torch.Size([n * d]))

  



##########################################################################################
################# For binning ############################################################
##########################################################################################
    def bin(self, data, kappa): 
        ''' Only for continuous data'''
        
        # create designated number of intervals
        d = self.get_dimension(data)
     
        # 1. for each dimension, turn the continuous data into interval
        # each row now indicates a hypercube in [0,1]^d
        # the more the data is closer to 1, the larger the interval index.
        dataInterval = self.transform_bin_index(data = data, nIntervals = kappa)
        
        # 2. for each datapoint(row),
        #    turn the hypercube data into a multivariate data of (1, 2, ..., kappa^d)
        #    each row now becomes an integer.
        dataMultivariate = self.TransformMultivariate(
            dataInterval = dataInterval,
            nBin = kappa,
        )
        # 3. turn the indices into one-hot vectors
        dataOnehot = self.TransformOnehot(dataMultivariate, kappa**d)
        return(dataOnehot)
    
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
    
    def TransformMultivariate(self, dataInterval, nBin):
        """Only for continuous and multivariate data ."""
        d = self.get_dimension(dataInterval)
        
        if d == 1:
            return(dataInterval.sub(1))
        else:
            exponent = torch.linspace(start = (d-1), end = 0, steps = d, dtype = torch.long)
            vector = torch.tensor(nBin).pow(exponent)
            return( torch.matmul( dataInterval.sub(1).to(torch.float), vector.to(torch.float).to(self.cuda_device) ).to(torch.long) )
    
    def TransformOnehot(self, dataMultivariate, newdim):
        return(
            torch.nn.functional.one_hot(
                dataMultivariate,
                num_classes = newdim)
        )
##########################################################################################
##########################################################################################


    
    
    def kernel_indep(self, fourchunk):
        mid = torch.matmul(torch.tensor([
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, -1.0, 0.0]
        ]) , fourchunk)

        return torch.dot(mid[0,], mid[1,])
 

    def u_stat_indep(self, data_X, data_Y):
        n = data_X.size(dim = 0)
        print(f"number of calculation = {2*scipy.special.comb(n,4) }")

        U_statistic = 0
        for comb in itertools.combinations(range(n), 4):
            U_statistic = U_statistic + (self.kernel_indep(data_X[comb,]) * self.kernel_indep(data_Y[comb,]))
        U_statistic = U_statistic / scipy.special.comb(n,4) 
        return(U_statistic)

