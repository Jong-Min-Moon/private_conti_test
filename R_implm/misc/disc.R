source("/Users/mac/Documents/GitHub/private_conti_test/data.R")
source("/Users/mac/Documents/GitHub/private_conti_test/test.R")
n1 = 100
n2 = 120
kappa = 4
alpha = 1/2
d = 2


# visualize how difficult the testing is
GMD <- GaussMeanDiff(n1 = n1, n2 = n2, d = d, first.diff=1/4, seed=11)
#par(mfrow=c(1,1))
#plot(GMD$X, col = "blue")
#points(GMD$Y, col = "red")


#gamma.discrete <- permutation.test(20, GMD$X, GMD$Y, kappa, alpha, discrete = TRUE)
#gamma.conti <- permutation.test(20, GMD$X, GMD$Y, kappa, alpha)

result.discrete <- rep(0,100)
result.conti <- rep(0,100)
for (rep in 1:100){
  cat(rep, "th run")
  start_time <- Sys.time()
  result.discrete[rep] <- permutation.test(20, GMD$X, GMD$Y, kappa, alpha, discrete = TRUE)
  result.conti[rep] <- permutation.test(20, GMD$X, GMD$Y, kappa, alpha)
  
  end_time <- Sys.time()
  print(end_time - start_time)
}