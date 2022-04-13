source("test.R")
library(ggplot2)

# 1. does my functions properly bin uniform distribution?
n = 1000000
kappa = 20
d = 4
data <- cbind(runif(n), runif(n), runif(n), runif(n))

data.interval <- .TransformIntervalIndex(data, kappa)
table(data.interval)

data.multivaraite <- .TransformMultivariate(data.interval, kappa, d)
plot(data.multivaraite)
hist(data.multivaraite, breaks = 1000)

# 2. does my functions properly bin bivaraite beta distribution?
n = 10000
kappa = 30
d = 2

data.beta.upper <- cbind(rbeta(n, 5, 1), rbeta(n, 5, 1))
head(data.beta.upper)
plot(data.beta.upper, pch = ".")

data.beta.upper.interval <- .TransformIntervalIndex(data.beta.upper, kappa)
data.beta.upper.interval.df <- data.frame(data.beta.upper.interval)
colnames(data.beta.upper.interval.df) <- c("d1", "d2")

ggplot(data.beta.upper.interval.df, aes(x=d1, y=d2)) + geom_point(alpha = 0.03)

data.beta.upper.interval <- .TransformIntervalIndex(data.beta.upper, kappa) # matrix form
data.beta.upper.multivaraite <- .TransformMultivariate(data.beta.upper.interval, kappa, d)

hist(data.beta.upper.multivaraite, breaks = 100)


# 3. does my functions properly bin bivaraite beta distribution?
n = 10000
kappa = 30
d = 2

data.beta.center <- cbind(rbeta(n, 10, 10), rbeta(n, 10, 10))
head(data.beta.center)
plot(data.beta.center, pch = ".")

data.beta.center.interval <- .TransformIntervalIndex(data.beta.center, kappa)
data.beta.center.interval.df <- data.frame(data.beta.center.interval)
colnames(data.beta.center.interval.df) <- c("d1", "d2")

ggplot(data.beta.center.interval.df, aes(x=d1, y=d2)) + geom_point(alpha = 0.03)

data.beta.center.interval <- .TransformIntervalIndex(data.beta.center, kappa) # matrix form
data.beta.center.multivaraite <- .TransformMultivariate(data.beta.center.interval, kappa, d)

hist(data.beta.center.multivaraite, breaks = 100)
