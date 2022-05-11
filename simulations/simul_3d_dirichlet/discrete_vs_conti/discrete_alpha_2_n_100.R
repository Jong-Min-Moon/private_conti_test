source("~/GitHub/private_conti_test/test.R")
library(rBeta2009, lib.loc = "~/GitHub/private_conti_test/required_packages")

dir.name <- "discrete_vs_conti"
file.name <- "discrete_alpha_2_n_100"

###############################
n1 <- n2 <- 100
###############################

kappa = 5 #number of bins

###############################
alpha = 2 #higher privacy level
###############################

gamma = 0.05 # significance level
B = 200 # number of permutations
n.test <- 200
set.seed(1)

start_time <- Sys.time()

param.dirichlet.1 <- c(2,1,1)*20
param.dirichlet.2 <- c(1,2,1)*20

result.conti.h1 <- rep(0, n.test)
for (rep in 1:n.test) {
  
    cat(rep, "th run\n")

  X <- as.matrix(rdirichlet(n1, param.dirichlet.1))
  Y <- as.matrix(rdirichlet(n2, param.dirichlet.2))
  ######### discrete = TRUE ##############
  result.now <- PrivatePermutationTwoSampleTest(B, X, Y, kappa, alpha, gamma, TRUE)
  result.conti.h1[rep] <- result.now
  print(result.now)
  
  power_now <- sum(result.conti.h1[1:rep]) / rep
  cat("power up to now :", power_now, "\n")
  }
end_time <- Sys.time()

filename <- paste0("~/GitHub/private_conti_test/simul_3d_dirichlet/", dir.name, "/", file.name, ".txt") 
sink(filename)
cat("3d dirichlet test\n")
cat("kappa = ", kappa, "\nn1 = ", n1, "\nn2 = ", n2, "\nalpha = ", alpha, "\ngamma = ", gamma, "\nB = ", B, "\n")
cat("power = ", sum(result.conti.h1) / n.test)
cat("\ntime elapsed = ")
print(end_time - start_time)
