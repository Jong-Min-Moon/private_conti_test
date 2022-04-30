source("~/GitHub/private_conti_test/test.R")
n1 <- n2 <- 1000
##########
kappa = 80
##########
alpha = 0.5
gamma = 0.05
d = 1
B = 100


# visualize how difficult the testing is
start_time <- Sys.time()
#GMD <- GaussMeanDiff(n1 = n1, n2 = n2, d = d, first.diff=2/5, seed=11)
end_time <- Sys.time()

param_beta_1 <- 20
param_beta_2 <- 20

X <- runif(n1)
Y <- rbeta(n2, param_beta_1, param_beta_2)

# check the distribution shapes
curve(dbeta(x, param_beta_1, param_beta_2), col = "red")
curve(dunif(x, 0, 1), col = "blue", add = TRUE)


start_time <- Sys.time()
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

filename <- paste0("~/GitHub/private_conti_test/kappa_trend/kappa_trend_",kappa, ".txt") 
sink(filename)
cat("kappa = ", kappa, "n1 = ", n1, "n2 = ", n2, "alpha = ", alpha, "gamma = ", gamma, "B = ", B, "\n")
cat("power = ", sum(result.conti.h1) / 100)

