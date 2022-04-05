library(TruncatedNormal)

source("/Users/mac/Documents/GitHub/private_conti_test/data.R")
source("/Users/mac/Documents/GitHub/private_conti_test/test.R")
n1 = 100
n2 = 120
kappa = 5
alpha = 0.1
d = 2
B = 100


# visualize how difficult the testing is
start_time <- Sys.time()
#GMD <- GaussMeanDiff(n1 = n1, n2 = n2, d = d, first.diff=2/5, seed=11)
end_time <- Sys.time()

X <- rtmvnorm(n1, mu = rep(1/5, d), sigma = diag(d)/30, lb=rep(0,d), ub=rep(1,d))
Y <- rtmvnorm(n2, mu = rep(1/5, d), sigma = diag(d)/30, lb=rep(0,d), ub=rep(1,d)) + rep(3/5, d)
par(mfrow=c(1,1))
plot(X, col = "blue", xlim=c(0,1), ylim = c(0,1))
points(Y, col = "red")

X <- rtmvnorm(n1, mu = rep(1/2, d), sigma = diag(d)/30, lb=rep(0,d), ub=rep(1,d))
Y <- rtmvnorm(n2, mu = rep(1/2, d), sigma = diag(d)/30, lb=rep(0,d), ub=rep(1,d))
par(mfrow=c(1,1))
plot(X, col = "blue", xlim=c(0,1), ylim = c(0,1))
points(Y, col = "red")


#gamma.discrete <- permutation.test(20, GMD$X, GMD$Y, kappa, alpha, discrete = TRUE)
#gamma.conti <- permutation.test(20, GMD$X, GMD$Y, kappa, alpha)
start_time <- Sys.time()
result.conti.h1 <- rep(0,100)
for (rep in 1:100){
  if (rep%%10 == 0){cat(rep, "th run\n")}
  X <- rtmvnorm(n1, mu = rep(1/5, d), sigma = diag(d)/30, lb=rep(0,d), ub=rep(1,d))
  Y <- rtmvnorm(n2, mu = rep(1/5, d), sigma = diag(d)/30, lb=rep(0,d), ub=rep(1,d)) + rep(3/5, d)
  
  result.conti.h1[rep] <- permutation.test(B, X, Y, kappa, alpha)
  
}
end_time <- Sys.time()
sum(result.conti.h1<0.05)/100


result.conti.h0 <- rep(0,100)
start_time <- Sys.time()

for (rep in 1:100){
  if (rep%%10 == 0){cat(rep, "th run\n")}
  
  X <- rtmvnorm(n1, mu = rep(1/2, d), sigma = diag(d)/30, lb=rep(0,d), ub=rep(1,d))
  Y <- rtmvnorm(n2, mu = rep(1/2, d), sigma = diag(d)/30, lb=rep(0,d), ub=rep(1,d))
  
  result.conti.h0[rep] <- permutation.test(B, X, Y, kappa, alpha)
}
end_time <- Sys.time()

sum(result.conti.h0<0.05)/100
