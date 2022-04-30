source("~/GitHub/private_conti_test/test.R")
n1 <- n2 <- 1000
kappa = 5
alpha = 0.5
gamma = 0.05
d = 1
B = 100
###################
param_beta_1 <- 10
param_beta_2 <- 10
###################
set.seed(1)

# visualize how difficult the testing is
start_time <- Sys.time()
#GMD <- GaussMeanDiff(n1 = n1, n2 = n2, d = d, first.diff=2/5, seed=11)




X <- runif(n1)
Y <- rbeta(n2, param_beta_1, param_beta_2)

result.conti.h1 <- rep(0, 100)
for (rep in 1:100) {
  
    cat(rep, "th run\n")

  X <- matrix(runif(n1))
  Y <- matrix(rbeta(n2, param_beta_1, param_beta_2))
  result.now <- PrivatePermutationTwoSampleTest(B, X, Y, kappa, alpha, gamma, discrete = FALSE)
  result.conti.h1[rep] <- result.now
  print(result.now)
  }
end_time <- Sys.time()

filename <- paste0("~/GitHub/private_conti_test/L2_trend_unif/L2_trend_unif", param_beta_1, ".txt") 
sink(filename)
cat("kappa = ", kappa, "\nn1 = ", n1, "\nn2 = ", n2, "\nalpha = ", alpha, "\ngamma = ", gamma, "\nB = ", B, "\n")
cat("power = ", sum(result.conti.h1) / 100)
cat("time elapsed = ", end_time - start_time)

