source("~/GitHub/private_conti_test/test.R")
n1 <- n2 <- 1000
kappa = 5
###################
alpha = 0.3
###################
gamma = 0.05
d = 1
B = 100
beta <- 30

set.seed(1)

# visualize how difficult the testing is
start_time <- Sys.time()

result.conti.h1 <- rep(0, 100)
for (rep in 1:100) {
  
    cat(rep, "th run\n")

  X <- matrix(rbeta(n1, 3, beta))
  Y <- matrix(rbeta(n2, beta, 3))
  result.now <- PrivatePermutationTwoSampleTest(B, X, Y, kappa, alpha, gamma, discrete = FALSE)
  result.conti.h1[rep] <- result.now
  print(result.now)
  }
end_time <- Sys.time()

filename <- paste0("~/GitHub/private_conti_test/alpha_trend/alpha_trend_", alpha, ".txt") 
sink(filename)
cat("kappa = ", kappa, "\nn1 = ", n1, "\nn2 = ", n2, "\nalpha = ", alpha, "\ngamma = ", gamma, "\nB = ", B, "\n")
cat("power = ", sum(result.conti.h1) / 100)
cat("time elapsed = ", end_time - start_time)

