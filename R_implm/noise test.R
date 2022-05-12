source("test.R")
n = 100
kappa = 4
d = 2
data <- cbind(runif(n), runif(n), runif(n), runif(n))

data.binned <- Bin(data, kappa)
head(data.binned)

par(mfrow=c(1,2))
#hist(data.binned)

data.private.conti <-PrivatizeTwoSample(data.binned, alpha = 0.5)
data.private.discrete <-PrivatizeTwoSample(data.binned, alpha = 0.5, discrete.noise = TRUE)

#hist(PrivatizeTwoSample(data.multivaraite, alpha = 0.5))

range(data.private)
range(data.private.discrete)

