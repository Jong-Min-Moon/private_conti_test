import torch
import itertools
import scipy
from scipy.special import comb


class LDPTwoSampleTester:
    def __init__(self, cuda_device):
        self.cuda_device = cuda_device

    def run_test_conti_data(self, B, data_X, data_Y, kappa, alpha, gamma, discrete = False):
        dataPrivatized, noise_var = self.preprocess_conti_data(data_X, data_Y, kappa, alpha, discrete)
        print(dataPrivatized)
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
            print(perm_stat_now)
         
        
        p_value_proxy = (1 +
                         torch.sum(
                             torch.gt(input = permStats, other = ustatOriginal)
                         )
                        ) / (B + 1)
        
        
        print(f"p value proxy: {p_value_proxy}")
        return(p_value_proxy < gamma, noise_var)#test result: TRUE = 1 = reject the null, FALSE = 0 = retain the null.
    
    def preprocess_conti_data(self, data_X, data_Y, kappa, alpha, discrete):
        data_X_binned = self.bin(data_X, kappa)
        data_Y_binned = self.bin(data_Y, kappa)
        
        dataCombined = torch.cat([data_X_binned, data_Y_binned], dim = 0)
        dataPrivatized, noise_var = self.privatize_twosample(dataCombined, alpha, discrete)
        print(dataPrivatized.size())
        return(dataPrivatized, noise_var)
        

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
    
    def u_stat_twosample_efficient(self, data, n_1_int):
        n_1 = torch.tensor(n_1_int)
        n_2 = data.size(dim = 0) - n_1
        data_X = data[ :n_1, ]
        data_Y = data[n_1: , ]

        X_row_sum = torch.sum(data_X, axis = 0)
        Y_row_sum = torch.sum(data_Y, axis = 0)
        phi_psi = torch.einsum('ji,jk->ik', data_X, data_Y)


        one_Phi_one = torch.inner(X_row_sum, X_row_sum)
        one_Psi_one = torch.inner(Y_row_sum, Y_row_sum)

        tr_Phi = torch.sum(torch.square(data_X))
        tr_Psi = torch.sum(torch.square(data_Y))

        one_Phi_tilde_one = one_Phi_one - tr_Phi
        one_Psi_tilde_one = one_Psi_one - tr_Psi

        onePhioneonePsione = one_Phi_tilde_one * one_Psi_tilde_one

        # x only part
        sign_x = torch.sign(one_Phi_tilde_one)
        abs_u_x = torch.exp(torch.log(torch.abs(one_Phi_tilde_one)) - torch.log(n_1) - torch.log(n_1 - 1) )
        u_x = sign_x * abs_u_x


        # y only part
        sign_y = torch.sign(one_Psi_tilde_one)

        abs_u_y = torch.exp(torch.log(torch.abs(one_Psi_tilde_one)) - torch.log(n_2) - torch.log(n_2 - 1) )
        u_y = sign_y * abs_u_y

        # x, y part
        cross = torch.inner(X_row_sum, Y_row_sum)
        sign_cross = torch.sign(cross)
        abs_cross = torch.exp(torch.log(torch.abs(cross)) +torch.log(torch.tensor(2))- torch.log(n_1) - torch.log(n_2) )
        u_cross = sign_cross * abs_cross

        return(u_x + u_y - u_cross)

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
       
    def range_check(self, data):
        if (torch.sum(data.gt(1))).gt(0):
            print("check data range")
            return False
        elif (torch.sum(data.lt(0))).gt(0):
            print("check data range")
            return False
        else:
            return True

    def run_test_conti_data(self, B, data_Y, data_Z, kappa, alpha, gamma, discrete_noise = False):
        #0. data range check
        
        if not self.range_check(data_Y):
            return
        if not self.range_check(data_Z):
            return
        
        #1. bin
        n = data_Y.size(dim = 0)
        data_Y_binned, data_Z_binned = self.bin_separately(data_Y, data_Z, kappa)

        #2. privatize
        data_Y_priv, data_Z_priv, noise_var_Y, noise_var_Z = self.privatize_indep(
            data_Y = data_Y_binned,
            data_Z = data_Z_binned,
            alpha = alpha,
            discrete_noise = discrete_noise
        )
        
        #4 compute original u-stat
        ustatOriginal = self.u_stat_indep_matrix_efficient(data_Y_priv, data_Z_priv)

        print(f"original u-statistic:{ustatOriginal}")
        
        #permutation procedure
        permStats = torch.empty(B).to(self.cuda_device)
        
        for i in range(B):
            perm_stat_now = self.u_stat_indep_matrix_efficient(
                data_Y_priv,
                data_Z_priv[
                    torch.randperm(data_Z_priv.size(dim=0))],
                ).to(self.cuda_device)

            permStats[i] = perm_stat_now
            print(f"perm_stat_now = {perm_stat_now}")
         
        
        p_value_proxy = (1 +
                         torch.sum(
                             torch.gt(input = permStats, other = ustatOriginal)
                         )
                        ) / (B + 1)
        
        
        #print(f"p value proxy: {p_value_proxy}")
        
        return(p_value_proxy < gamma, noise_var_Y, noise_var_Z)#test result: TRUE = 1 = reject the null, FALSE = 0 = retain the null.

        
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
    

    def generate_unit_laplace(self, data):
        n = data.size(dim = 0)
        d = data.size(dim = 1)
        unit_laplace_generator = torch.distributions.laplace.Laplace(
            torch.tensor(0.0).to(self.cuda_device),
            torch.tensor(2**(-1/2)).to(self.cuda_device)
        )
        return unit_laplace_generator.sample(sample_shape = torch.Size([n * d]))


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
    
    
    def get_dimension(self, data):
        if data.dim() == 1:
            return(1)
        elif data.dim() == 2:
            return( data.size(dim = 1) )
        else:
            return # we only use up to 2-dimensional tensor, i.e. matrix
##########################################################################################
##########################################################################################


    
    
    def kernel_indep(self, fourchunk):
        ip = torch.matmul(fourchunk, torch.transpose(fourchunk, 0, 1))
        return(ip[0,1] + ip[2,3] - ip[0,2] - ip[1,3])

     
    def u_stat_indep_matrix(self, data_X, data_Y):
        n = data_X.size(dim = 0)
        n_four = n * (n-1) * (n-2) * (n-3)



        Phi = torch.matmul(data_X, torch.transpose(data_X, 0, 1))
        Psi = torch.matmul(data_Y, torch.transpose(data_Y, 0, 1))
        Phi_tilde = Phi.fill_diagonal_(0.0)
        Psi_tilde = Psi.fill_diagonal_(0.0)

        one = torch.ones(n, 1).to(self.cuda_device)
        oneT = torch.transpose(one, 0, 1)

        PhiPsi = torch.matmul(Phi, Psi)
        trPhiPsi = torch.trace(PhiPsi)
        onePhiPsiOne = torch.matmul(oneT, torch.matmul(PhiPsi, one))

        onePhione = torch.matmul(oneT, torch.matmul(Phi, one))
        onePsione = torch.matmul(oneT, torch.matmul(Psi, one))
        onePhioneonePsione = torch.matmul(onePhione, onePsione)

        #Un = (
        #   4 * (onePhioneonePsione - 4 * onePhiPsiOne + 2 * trPhiPsi)
        # - 8 * (n-3) *(onePhiPsiOne - trPhiPsi)
        # + 4 * (n-3)*(n-2) * trPhiPsi
        # )

        Un = (
           4 * (onePhioneonePsione - 4 * onePhiPsiOne + 2 * trPhiPsi)
         - 8 * (n-3) *(onePhiPsiOne - trPhiPsi)
         - 8 * (n-3) *(onePhiPsiOne - trPhiPsi)


         + 4 * (n-3)*(n-2) * trPhiPsi
         )
        
        return(Un/n_four)

    def u_stat_indep_matrix_efficient(self, data_X, data_Y):
        #scalars
        n = data_X.size(dim = 0)
        
        log_n_four = (
        torch.log(torch.tensor(n))
        +  
        torch.log(torch.tensor(n-1))
        +
        torch.log(torch.tensor(n-2))
        +
        torch.log(torch.tensor(n-3))
        )

        #preliminary calculations
        X_row_sum = torch.sum(data_X, axis = 0)
        Y_row_sum = torch.sum(data_Y, axis = 0)
        phi_psi = torch.einsum('ji,jk->ik', data_X, data_Y)
        diag_Phi = torch.sum(torch.square(data_X), axis = 1)
        diag_Psi = torch.sum(torch.square(data_Y), axis = 1)
        rowsum_Phi = torch.einsum('i,ji -> j', X_row_sum, data_X)
        rowsum_Psi = torch.einsum('ij, j -> i', data_Y, Y_row_sum)

        #1. one term
        one_Phi_one = torch.inner(X_row_sum, X_row_sum)
        one_Psi_one = torch.inner(Y_row_sum, Y_row_sum)

        tr_Phi = torch.sum(torch.square(data_X))
        tr_Psi = torch.sum(torch.square(data_Y))

        one_Phi_tilde_one = one_Phi_one - tr_Phi
        one_Psi_tilde_one = one_Psi_one - tr_Psi

        onePhioneonePsione = one_Phi_tilde_one * one_Psi_tilde_one


        #2. one one term
        onePhiPsiOne = torch.matmul(
            torch.matmul(X_row_sum, phi_psi),
            Y_row_sum)  + torch.inner(diag_Phi, diag_Psi)-torch.inner(rowsum_Phi, diag_Psi)-torch.inner(diag_Phi, rowsum_Psi)


        #3. trace term
        trPhiPsi = torch.sum( torch.square(phi_psi) ) - torch.inner(
            torch.sum( torch.square(data_X), axis = 1),
            torch.sum( torch.square(data_Y), axis = 1)
        )
        
        sums = (4 * onePhioneonePsione - ( 8 * (n-1) ) * onePhiPsiOne + ( 4 * (n-1) * (n-2) ) * trPhiPsi )
        
        Un_sign = torch.sign(sums)
        abs_Un = torch.exp(torch.log(torch.abs(sums)) - log_n_four)
        Un = Un_sign * abs_Un

        return(Un)
    
    def u_stat_indep_original(self, data_X, data_Y):
        n = data_X.size(dim = 0)
        print(f"number of calculation = {2*scipy.special.comb(n,4) }")
        n_four = n * (n-1) * (n-2) * (n-3)
        U_statistic = 0
        for i in range(n):
            set_j = set(range(n)) - {i}
            for j in set_j:
                set_k = set_j - {j}
                for k in set_k:
                    set_r = set_k - {k}
                    for r in set_r:
                        comb = [i,j,k,r]
                        U_statistic = U_statistic + (
                            self.kernel_indep(data_X[comb,]) * self.kernel_indep(data_Y[comb,])
                        )/n_four
        return(U_statistic)


