source("~/GitHub/private_conti_test/R_implm/test.R")
source("~/GitHub/private_conti_test/R_implm/data_generator.R")

####CHANGE HERE#####
n1 <- n2 <- 10000
####################
kappa = 5 #number of bins
alpha = 0.3 #privacy level
gamma = 0.05 # significance level
n.tests = 100 #number of tests for power estimation
B = 100 # number of permutations

set.seed(2022)

start_time <- Sys.time()


copula.mean.1 <- c(-1, -1)
copula.mean.2 <- c( 1,  1)
sigma = matrix(c(1, .5, .5, 1 ), nrow = 2)

result.conti.h1 <- rep(0, 100)
for (rep in 1:100) {
  
    cat(rep, "th run\n")

  X <- CopulaNonuniform(n1, copula.mean.1, sigma)
  Y <- CopulaNonuniform(n2, copula.mean.2, sigma)

  result.now <- PrivatePermutationTwoSampleTest(B, X, Y, kappa, alpha, gamma, FALSE)
  result.conti.h1[rep] <- result.now
  print(result.now)

  power_now <- sum(result.conti.h1[1:rep]) / rep
  cat("power up to now :", power_now, "\n")
  }
end_time <- Sys.time()

filename <- paste0("~/GitHub/private_conti_test/R_implm/simulations/copula_2d/n_trend/n_trend_", n1, ".txt") 
sink(filename)
cat("copula test\n")
cat("kappa = ", kappa, "\nn1 = ", n1, "\nn2 = ", n2, "\nalpha = ", alpha, "\ngamma = ", gamma, "\nB = ", B, "\n")
cat("power = ", sum(result.conti.h1) / 100)
cat("\ntime elapsed = ")
print(end_time - start_time)
