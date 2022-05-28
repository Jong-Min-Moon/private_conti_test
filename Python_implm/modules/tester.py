import torch

class LDPTwoSampleTester:
    def __init__(self, cuda_device):
        self.cuda_device = cuda_device

    def run_test_conti_data(self, B, data_X, data_Y, kappa, alpha, gamma, discrete = False):
        dataPrivatized = self.preprocess_conti_data(data_X, data_Y, kappa, alpha)
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
        return(p_value_proxy < gamma)#test result: TRUE = 1 = reject the null, FALSE = 0 = retain the null.
    
    def preprocess_conti_data(self, data_X, data_Y, kappa, alpha):
        data_X_binned = self.bin(data_X, kappa)
        data_Y_binned = self.bin(data_Y, kappa)
        
        dataCombined = torch.cat([data_X_binned, data_Y_binned], dim = 0)
        dataPrivatized = self.privatize_twosample(dataCombined, alpha)
        return(dataPrivatized)
        

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
    
    def privatize_twosample(self, data, alpha = float("inf"), discrete_noise = False):
        ## assume the data is discrete by nature or has already been dicretized.
        n = data.size(dim = 0)
        dim = data.size(dim = 1) #kappa^d if conti data, d if discrete data
        print(f"noise dimension : {dim}")
        scale = torch.tensor(dim**(1/2))
        
        if alpha == float("inf"): #non-private case
            return( torch.mul(scale, data) )
        else: # private case
            if discrete_noise:
                noise = self.noise_discrete(n = n, dim = dim, alpha = alpha)
            else:
                noise = self.noise_conti(n = n, dim = dim, alpha = alpha)
        return(
            
            torch.add(
                input = noise.reshape(n, -1),
                alpha = scale,
                other = data
            )
        )
    
    def noise_conti(self, n, dim, alpha):
        #dim = kappa^d for conti data, d for discrete data
        unit_laplace_generator = torch.distributions.laplace.Laplace(
            torch.tensor(0.0).to(self.cuda_device),
            torch.tensor(2**(-1/2)).to(self.cuda_device)
        )
        laplace_samples = unit_laplace_generator.sample(sample_shape = torch.Size([n * dim]))
        scale = (8**(1/2) / alpha) * (dim**(1/2))
        laplace_samples = scale*laplace_samples
        print(f"noise variance: {torch.var(laplace_samples)}")
        return(laplace_samples)
    
  
        
    
    def noise_discrete(self, n, dim, alpha):
        #dim = kappa^d for conti data, d for discrete data
        param_geom = 1 - torch.exp(torch.tensor(-alpha / (2* (dim**(1/2)) )))
        n_noise =  n * dim
        geometric_generator = torch.distributions.geometric.Geometric(param_geom.to(self.cuda_device))
        noise = geometric_generator.sample((n_noise,)) - geometric_generator.sample((n_noise,))
        return(noise)
    
    def u_stat_twosample(self, data, n_1):
        n_2 = data.size(dim = 0) - n_1
        
        data_x = data[ :n_1, ]
        data_y = data[n_1: , ]
        
        # x only part
        u_x = torch.matmul(data_x, torch.transpose(data_x, 0, 1))
        u_x.fill_diagonal_(0)
        u_x = torch.sum(u_x) / (n_1 * (n_1 - 1))
        
        # y only part
        u_y = torch.matmul(data_y, torch.transpose(data_y, 0, 1))
        u_y.fill_diagonal_(0)
        u_y = torch.sum(u_y) / (n_2 * (n_2 - 1))

        # x, y part
        u_xy = torch.matmul(data_x, torch.transpose(data_y, 0, 1))
        u_xy = torch.sum(u_xy) * ( 2 / (n_1 * n_2) )
        return(u_x + u_y - u_xy)
    
    def get_dimension(self, data):
        if data.dim() == 1:
            return(1)
        elif data.dim() == 2:
            return( data.size(dim = 1) )
        else:
            return # we only use up to 2-dimensional tensor, i.e. matrix
        