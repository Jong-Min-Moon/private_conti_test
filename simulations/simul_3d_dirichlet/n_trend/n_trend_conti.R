source("~/GitHub/private_conti_test/test.R")
library(rBeta2009, lib.loc = "~/GitHub/private_conti_test/required_packages")

n_trend_conti <- function(n1, n2){
####CHANGE HERE#####
n1 <- n2 <- 1000
####################
kappa = 5 #number of bins
alpha = 0.5 #privacy level
gamma = 0.05 # significance level
B = 200 # number of permutations

set.seed(2022)

start_time <- Sys.time()


param.dirichlet.1 <- c(2,1,1)*20
param.dirichlet.2 <- c(1,2,1)*20

result.conti.h1 <- rep(0, 100)
for (rep in 1:100) {
  
    cat(rep, "th run\n")

  X <- as.matrix(rdirichlet(n1, param.dirichlet.1))
  Y <- as.matrix(rdirichlet(n2, param.dirichlet.2))
  result.now <- PrivatePermutationTwoSampleTest(B, X, Y, kappa, alpha, gamma, FALSE)
  result.conti.h1[rep] <- result.now
  print(result.now)

  power_now <- sum(result.conti.h1[1:rep]) / rep
  cat("power up to now :", power_now, "\n")
  }
end_time <- Sys.time()

filename <- paste0("~/GitHub/private_conti_test/simul_3d_dirichlet/n_trend/n_trend_", n1, ".txt") 
sink(filename)
cat("3d dirichlet test\n")
cat("kappa = ", kappa, "\nn1 = ", n1, "\nn2 = ", n2, "\nalpha = ", alpha, "\ngamma = ", gamma, "\nB = ", B, "\n")
cat("power = ", sum(result.conti.h1) / 100)
cat("\ntime elapsed = ")
print(end_time - start_time)
}
