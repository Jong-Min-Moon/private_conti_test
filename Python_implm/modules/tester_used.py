import torch
import itertools
import scipy
from scipy.special import comb


class LDPTwoSampleTester:
    def __init__(self, cuda_device):
        self.cuda_device = cuda_device


    
    def preprocess_conti_data(self, data_X, data_Y, kappa, alpha, discrete):
        data_X_binned = self.bin(data_X, kappa)
        data_Y_binned = self.bin(data_Y, kappa)
        
        dataCombined = torch.cat([data_X_binned, data_Y_binned], dim = 0)
        dataPrivatized, noise_var = self.privatize_twosample(dataCombined, alpha, discrete)
        print(dataPrivatized.size())
        return(dataPrivatized, noise_var)
        


        # 3. turn the indices into one-hot vectors
        dataOnehot = self.TransformOnehot(dataMultivariate, kappa**d)
        return(dataOnehot)
    

    
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
                noise, noise_var = self.noise_discrete(n = n, dim = dim, alpha = alpha)
            else:
                noise, noise_var = self.noise_conti(n = n, dim = dim, alpha = alpha)
        return(
            
            torch.add(
                input = noise.reshape(n, -1),
                alpha = scale,
                other = data
            ), 
            noise_var
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
        print("noise type: conti")
        return(laplace_samples, torch.var(laplace_samples))
    
  
        
    
    def noise_discrete(self, n, dim, alpha):
        #dim = kappa^d for conti data, d for discrete data
        param_geom = 1 - torch.exp(torch.tensor(-alpha / (2* (dim**(1/2)) )))
        n_noise =  n * dim
        geometric_generator = torch.distributions.geometric.Geometric(param_geom.to(self.cuda_device))
        noise = geometric_generator.sample((n_noise,)) - geometric_generator.sample((n_noise,))
        print("noise type: discrete")

        return(noise, torch.var(noise))
    
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

class LDPIndepTester:
    def __init__(self, cuda_device):
        self.cuda_device = cuda_device
   
    def bin_separately(self, data_X, data_Y, kappa):
        return (
            self.bin(data_X, kappa),
            self.bin(data_Y, kappa)
            )
       


    def run_test_conti_data(self, B, data_Y, data_Z, kappa, alpha, gamma, discrete_noise = False):
 
    def privatize_indep(self, data_Y, data_Z, alpha = float("inf"), discrete_noise = False):
        ## assume the data is discrete by nature or has already been dicretized.
        n = data_Y.size(dim = 0) # Y and Z have the same sample size.
        kappa_d1 = data_Y.size(dim = 1) #kappa^d if conti data, d if discrete data
        kappa_d2 = data_Z.size(dim = 1) #kappa^d if conti data, d if discrete data

        print(f"noise dimension : {kappa_d1}, {kappa_d2}")
        
        scale_factor = torch.tensor( (kappa_d1 * kappa_d2)**(1/2) )
        sigma_kappa_alpha = 4 * (2 ** (1/2)) * scale_factor / alpha
        
        if alpha == float("inf"): #non-private case
            return( 
                    torch.mul(scale_factor, data_Y),
                    torch.mul(scale_factor, data_Z),
                    0, 0
                     )
        else:
            data_Y_priv, noise_var_Y = self.privatize_indep_separate(
                    data = data_Y,
                    scale_factor = scale_factor,
                    sigma_kappa_alpha = sigma_kappa_alpha,
                    discrete_noise = discrete_noise
            )
            data_Z_priv, noise_var_Z = self.privatize_indep_separate(
                    data = data_Z,
                    scale_factor = scale_factor,
                    sigma_kappa_alpha = sigma_kappa_alpha,
                    discrete_noise = discrete_noise
            )
        return(data_Y_priv, data_Z_priv, noise_var_Y, noise_var_Z)
        
    
    def privatize_indep_separate(self, data, scale_factor, sigma_kappa_alpha, discrete_noise):
        n = data.size(dim = 0)
        if discrete_noise:
            noise, noise_var = self.noise_conti(data, sigma_kappa_alpha) #fix here later
        else:
            noise, noise_var = self.noise_conti(data, sigma_kappa_alpha)
        return(   
            torch.add(
                input = noise.reshape(n, -1),
                alpha = scale_factor,
                other = data
            ), 
            noise_var
        )
                       
    def noise_conti(self, data, sigma_kappa_alpha):
        #dim = kappa^d for conti data, d for discrete data
        laplace_samples = self.generate_unit_laplace(data)
        laplace_samples = sigma_kappa_alpha * laplace_samples
        print("noise type: conti")
        return( laplace_samples, torch.var(laplace_samples) )
    




