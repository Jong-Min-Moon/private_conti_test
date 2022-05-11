source("~/GitHub/private_conti_test/test.R")
library(rBeta2009, lib.loc = "~/GitHub/private_conti_test/required_packages")


####CHANGE HERE#####
n1 <- n2 <- 5000
####################
kappa = 5 #number of bins
alpha = 0.5 #higher privacy level
gamma = 0.05 # significance level
B = 100 # number of permutations

set.seed(1)

start_time <- Sys.time()

param.dirichlet.1 <- c(2,1,1,1,1)*20
param.dirichlet.2 <- c(1,1,2,1,1)*20

result.conti.h1 <- rep(0, 100)
for (rep in 1:100) {
  
    cat(rep, "th run\n")

  X <- as.matrix(rdirichlet(n1, param.dirichlet.1))
  Y <- as.matrix(rdirichlet(n2, param.dirichlet.2))
  result.now <- PrivatePermutationTwoSampleTest(B, X, Y, kappa, alpha, gamma, FALSE)
  result.conti.h1[rep] <- result.now
  print(result.now)
  }
end_time <- Sys.time()

filename <- paste0("~/GitHub/private_conti_test/simul_3d_dirichlet/d_trend/d_5.txt")
sink(filename)
cat("3d dirichlet test\n")
cat("kappa = ", kappa, "\nn1 = ", n1, "\nn2 = ", n2, "\nalpha = ", alpha, "\ngamma = ", gamma, "\nB = ", B, "\n")
cat("power = ", sum(result.conti.h1) / 100)
cat("\ntime elapsed = ")
print(end_time - start_time)
