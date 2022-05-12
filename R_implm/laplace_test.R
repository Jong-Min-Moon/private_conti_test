library(rmutil)

b <- 1 / sqrt(2)
n = 1000000


sample.rlap <- rlaplace(n, m = 0, s = b)



sample.rexp <- rexp(n, 1/b) - rexp(n, 1/b)


par(mfrow = c(1,2))
hist(sample.rlap, breaks = 1000, xlim= c(-6, 6))
hist(sample.rexp, breaks = 1000, xlim= c(-6, 6))
var(sample.rlap)
var(sample.rexp)


