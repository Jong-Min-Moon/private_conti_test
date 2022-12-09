curve( 8 * x^2 - 2 * exp(x/2) / ( exp(x) -1 )^2, from=0, to = 100)
curve(2 * exp(x/2) / ( exp(x) -1 )^2, from=0, to = 100)

var(noise.conti(100000,25, 0.5))

sd(nc)
mean(nd)
sd(nd)
mean(nd)
noise.vairance.theoretic.discrete(dim = 20^2, alpha = 2)
noise.vairance.theoretic.conti(dim = 20^2, alpha = 2)
var(noise.discrete(1000,20^4, 2))
var(noise.conti(1000,20^4, 2))


curve(noise.vairance.theoretic.conti(dim = 5^2, alpha = x), col = "red")
curve(noise.vairance.theoretic.discrete(dim = 5^2, alpha = x), col = "blue", add = TRUE)



curve(noise.vairance.theoretic.conti(dim = 10^2, alpha = x), col = "red", from = 0, to = 10)
curve(noise.vairance.theoretic.discrete(dim = 10^2, alpha = x), col = "blue", add = TRUE, from = 0, to = 10)

curve(noise.vairance.theoretic.conti(dim = 2, alpha = x)-noise.vairance.theoretic.discrete(dim = 10^2, alpha = x), col = "red", from = 0, to = 1)
