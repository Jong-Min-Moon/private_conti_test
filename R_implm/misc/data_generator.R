library(mvtnorm, lib.loc = "~/GitHub/private_conti_test/R_implm/required_packages")

CopulaNonuniform <- function(n, mu, sigma){
  x <- rmvnorm(n, mean = mu, sigma = sigma)
  return(as.matrix(pnorm(x)))
}
