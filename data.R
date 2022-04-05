#The structure and part of this code was adapted from Wittawat Jitkrittum
#Linear-time interpretable nonparametric two-sample test
#__author__ = 'leon'

#from abc import ABCMeta, abstractmethod
#import math

#import matplotlib.pyplot as plt
#import scipy.stats as stats
#import numpy as np

#import private_me.util as util

library(TruncatedNormal)

SameGauss <- function(n, d, seed){
  ### Two same standard Gaussians for P, Q.
  ### The null hypothesis H0: P=Q is true.
  set.seed(seed)
  mean.Y <- rep(0, d)
  mean.y[1] <- first.diff
  X <- rtmvnorm(n, mean = rep(0, d))
  Y <- rtmvnorm(n, mean = rep(0, d)) 
  
  return(list("X" = X, "Y" = Y))
}


GaussMeanDiff <- function(n1, n2, d, diff=1/4, seed=1){
  ### Toy dataset one in Chwialkovski et al., 2015. 
  ### P = N(0, I), Q = N( (my,0,0, 000), I).
  ### Only the first dimension of the means differ.
  set.seed(seed)
  #mean.y <- rep(0, d)
  #mean.y[1] <- first.diff
  X <- rtmvnorm(n1, mu = rep(1/5, d), sigma = diag(d)/1000, lb=rep(0,d), ub=rep(1,d))
  Y <- rtmvnorm(n2, mu = rep(1/5, d), sigma = diag(d)/1000, lb=rep(0,d), ub=rep(1,d)) + rep(diff, d)
  
  return(list("X" = X, "Y" = Y))
}

GaussVarDiff <- function(n, d, seed){
  ### Toy dataset two in Chwialkovski et al., 2015. 
  ### P = N(0, I), Q = N(0, diag((2, 1, 1, ...))).
  ### Only the variances of the first dimension differ.
  set.seed(seed)
  std.Y <- diag(1,d)
  std.Y[1] <- 2.0
  X <- rmvnorm(n, mean = rep(0, d))
  Y <- rmvnorm(n, mean = rep(0, d))%*%std.Y
  
  return(list("X" = X, "Y" = Y)) 
}


# 
# class SSBlobs(SampleSource):
#   """Mixture of 2d Gaussians arranged in a 2d grid. This dataset is used 
#     in Chwialkovski et al., 2015 as well as Gretton et al., 2012. 
#     Part of the code taken from Dino Sejdinovic and Kacper Chwialkovski's code."""
# 
# def __init__(self, blob_distance=5, num_blobs=4, stretch=2, angle=math.pi/4.0):
#   self.blob_distance = blob_distance
# self.num_blobs = num_blobs
# self.stretch = stretch
# self.angle = angle
# 
# def dim(self):
#   return 2
# 
# def sample(self, n, seed):
#   rstate = np.random.get_state()
# np.random.seed(seed)
# 
# x = gen_blobs(stretch=1, angle=0, blob_distance=self.blob_distance,
#               num_blobs=self.num_blobs, num_samples=n)
# 
# y = gen_blobs(stretch=self.stretch, angle=self.angle,
#               blob_distance=self.blob_distance, num_blobs=self.num_blobs,
#               num_samples=n)
# 
# np.random.set_state(rstate)
# return TSTData(x, y, label='blobs_s%d'%seed)
# 
# 
# gen_blobs<- function(stretch, angle, blob_distance, num_blobs, num_samples){
# ## Generate 2d blobs dataset
# 
# # rotation matrix
# r <- matrix(c( cos(angle), -sin(angle), sin(angle), cos(angle) ), nrow=2, byrow=TRUE)
# eigenvalues <- diag( c(sqrt(stretch), 1))
# mod_matix <- r%*%eigenvalues
# mean <- as.numeric(blob_distance * (num_blobs-1)) / 2
# k <- num_blobs - 1
# n <-  num_samples * 2
# blob.membership <- sample(0:k, n, replace=T)
# mu <- matrix(blob.membership, nrow=num_samples) * blob_distance - mean
# 
# return(rmvnorm(num_samples, mean = c(0,0))%*%mod_matrix + mu)
# }
# 
# library(mvtnorm)





