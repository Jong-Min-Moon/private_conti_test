library(mvtnorm)



#1. two independent standard normals
sigma = matrix(c(1, 0, 0, 1 ), nrow = 2)
x <- rmvnorm(5000, mean = rep(0,2), sigma = sigma)
par(mfrow=c(2,2))
plot(x, pch = 20, cex = .1)
x1_tilde <- pnorm(x[,1])
x2_tilde <- pnorm(x[,2])
plot(x1_tilde, x2_tilde, pch = 20, cex = .1)

hist(x1_tilde, breaks = 20)
hist(x2_tilde, breaks = 20)



#2. half correlation
sigma = matrix(c(1, .5, .5, 1 ), nrow = 2)
x <- rmvnorm(5000, mean = rep(0,2), sigma = sigma)
par(mfrow=c(2,2))
plot(x, pch = 20, cex = .1)
x1_tilde <- pnorm(x[,1])
x2_tilde <- pnorm(x[,2])
plot(x1_tilde, x2_tilde, pch = 20, cex = .1)

hist(x1_tilde, breaks = 20)
hist(x2_tilde, breaks = 20)



#3. full correlation
sigma = matrix(c(1, 1, 1, 1 ), nrow = 2)
x <- rmvnorm(5000, mean = rep(0,2), sigma = sigma)
par(mfrow=c(2,2))
plot(x, pch = 20, cex = .1)
x1_tilde <- pnorm(x[,1])
x2_tilde <- pnorm(x[,2])
plot(x1_tilde, x2_tilde, pch = 20, cex = .1)

hist(x1_tilde, breaks = 20)
hist(x2_tilde, breaks = 20)


#4. different mean
sigma = matrix(c(1, .5, .5, 1 ), nrow = 2)

par(mfrow=c(2,2))

x <- rmvnorm(5000, mean = c(-1, -1), sigma = sigma)
x1_tilde <- pnorm(x[,1])
x2_tilde <- pnorm(x[,2])
plot(x1_tilde, x2_tilde, pch = 20, cex = .1)

x <- rmvnorm(5000, mean = c(-0.5, -0.5), sigma = sigma)
x1_tilde <- pnorm(x[,1])
x2_tilde <- pnorm(x[,2])
plot(x1_tilde, x2_tilde, pch = 20, cex = .1)


x <- rmvnorm(5000, mean = c(.5, .5), sigma = sigma)
x1_tilde <- pnorm(x[,1])
x2_tilde <- pnorm(x[,2])
plot(x1_tilde, x2_tilde, pch = 20, cex = .1)

x <- rmvnorm(5000, mean = c(1, 1), sigma = sigma)
x1_tilde <- pnorm(x[,1])
x2_tilde <- pnorm(x[,2])
plot(x1_tilde, x2_tilde, pch = 20, cex = .1)




########################################################
#5. two sample
# 2d plots of samples from two different distributions(blue and red)

sigma = matrix(c(1, .5, .5, 1 ), nrow = 2) # slightly correlated

PlotTwosampleCopula <- function(mean_x, mean_y, sigma){
  x <- rmvnorm(5000, mean = mean_x, sigma = sigma)
  x1_tilde <- pnorm(x[,1])
  x2_tilde <- pnorm(x[,2])
  y <- rmvnorm(5000, mean = mean_y, sigma = sigma)
  y1_tilde <- pnorm(y[,1])
  y2_tilde <- pnorm(y[,2])
  
  plot(x1_tilde, x2_tilde, pch = 20, cex = .1, 
       col = adjustcolor("blue",alpha = 0.25), 
       main = paste(mean_x, " vs. ", mean_y),
       xlab = "", ylab = ""
       )
  points(y1_tilde, y2_tilde, pch = 20, cex = .1, col = adjustcolor("red",alpha = 0.25))
}

#plot1
par(mfrow=c(2,2))
PlotTwosampleCopula(c(-2, -2), c(2, 2), sigma)
PlotTwosampleCopula(c(-1, -1), c(1, 1), sigma)
PlotTwosampleCopula(c(-.5, -.5), c(.5, .5), sigma)
PlotTwosampleCopula(c(-.1, -.1), c(.1, .1), sigma)

#plot2
par(mfrow=c(2,2))
PlotTwosampleCopula(c(-4/5, -4/5), c(4/5, 4/5), sigma)
PlotTwosampleCopula(c(-3/4, -3/4), c(3/4, 3/4), sigma)
PlotTwosampleCopula(c(-1/2, -1/2), c(1/2, 1/2), sigma)



# plot 3
x <- rmvnorm(5000, mean = c(-.5, -.5), sigma = sigma)
x1_tilde <- pnorm(x[,1])
x2_tilde <- pnorm(x[,2])
y <- rmvnorm(5000, mean = c(.5, .5), sigma = sigma)
y1_tilde <- pnorm(y[,1])
y2_tilde <- pnorm(y[,2])

plot(x1_tilde, x2_tilde, pch = 20, cex = .1, col = "blue")
points(y1_tilde, y2_tilde, pch = 20, cex = .1, col = "red")

# plot 4
x <- rmvnorm(5000, mean = c(-.2, -.2), sigma = sigma)
x1_tilde <- pnorm(x[,1])
x2_tilde <- pnorm(x[,2])
y <- rmvnorm(5000, mean = c(.2, .2), sigma = sigma)
y1_tilde <- pnorm(y[,1])
y2_tilde <- pnorm(y[,2])

plot(x1_tilde, x2_tilde, pch = 20, cex = .1, col = "blue")
points(y1_tilde, y2_tilde, pch = 20, cex = .1, col = "red")
