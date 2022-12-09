source("~/GitHub/private_conti_test/test.R")
library(rBeta2009)

n1 <- n2 <- 500
kappa = 20
alpha = 0.5
gamma = 0.05
d = 1
B = 5


# visualize how difficult the testing is
start_time <- Sys.time()


end_time <- Sys.time()

param.dirichlet.1 <- c(2,1,1)*20
param.dirichlet.2 <- c(1,2,1)*20

start_time <- Sys.time()
result.conti.h1 <- rep(0, 100)
for (rep in 1:100) {
  
    cat(rep, "th run\n")

  X <- matrix(rdirichlet(n1, param.dirichlet.1))
  Y <- matrix(rdirichlet(n2, param.dirichlet.2))
  result.now <- PrivatePermutationTwoSampleTest(B, X, Y, kappa, alpha, gamma, discrete = FALSE)
  result.conti.h1[rep] <- result.now
  print(result.now)
  }

end_time <- Sys.time()

sink("")
cat("power = ", sum(result.conti.h1) / 100)

