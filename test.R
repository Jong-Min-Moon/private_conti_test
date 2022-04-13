library(rmutil)
library(combinat) # for permutation


multivariate.binning <- function(data.in, kappa) {
  data <- data.in
  
  # 1. for each dimension, turn the continuous data into interval format
  data.interval <- function(data.raw = data, n.intervals = kappa)
  
  hypercube.index <- rep(1, n)
  for (col in 1:d) {
    hypercube.index <-
      hypercube.index + (data[, col] - 1) * kappa ^ (d - col)
  }
  
  # 3. turn the indices into one-hot vectors
  data.onehot <- .TransformOnehot(
    vector.indices = hypercube.index,
    n.dim = kappa ^ d
    )
  return(data.onehot)
}

.TransformIntervalIndex <- function(data.raw, n.intervals) {
  # for each dimension, transform the data in [0,1] into the interval index
  # first interval = [0, x], the others = (y z]
  
  # create designated number of intervals
  if (is.vector(data.raw)){ d <- 1 }
  else{ d <- ncol(data.raw) }
  
  breaks <- seq(from = 0, to = 1, length = n.intervals + 1)
  
  # for each dimension.
  data.indices <- cut(x = data.raw,
                      breaks = breaks,
                      include.lowest = TRUE)
  data.indices <- as.numeric(data.indices)
  data.indices <- matrix(data.indices,
                         ncol = d,
                         byrow = FALSE # since cut function collapsed a matrix column-wise.
                         )
  return( data.indices ) 
}

.TransformOnehot <- function(vector.indices, n.dim) {
  data.onehot <- matrix(0, nrow = n, ncol = n.dim) #initialize with 0
  for (row.num in 1:n) {
    bin.num <- vector.indices[row.num]
    data.onehot[row.num, bin.num] <- 1
  }
  return(data.onehot)
}






noise.conti <- function(n, d, kappa, alpha) {
  kappa.d <- kappa ^ d
  scale <- 2 * sqrt(2) * sqrt(kappa.d) / alpha
  noise <- rlaplace(n = n * kappa.d,
                    m = 0,
                    s = 1 / sqrt(2))
  noise <- scale * noise
  return(noise)
}

noise.discrete <- function(n, d, kappa, alpha) {
  kappa.d <- kappa ^ d
  t <- 2 * sqrt(kappa.d) / alpha
  param.geom <- 1 - exp(-1 / t)
  noise <- rgeom(n = n * kappa.d, param.geom) + 1
  return(noise)
}
#
# #### variance checking
# var.discrete <- rep(0,1000)
# var.conti <- rep(0,1000)
# for (i in 1:1000){
#   var.discrete[i] <- var(noise.discrete(n = 10000, d = d,  kappa = kappa, alpha = alpha))
#   var.conti[i] <-var(noise.conti(10000,  d = d,  kappa = kappa, alpha = alpha))
# }
# par(mfrow=c(1,2))
# hist(var.discrete)
# hist(var.conti)
#
# var.expected <- 8 * kappa^d / alpha^2

u.stat.two.sample <- function(sample.x, sample.y) {
  n1 <- nrow(sample.x)
  n2 <- nrow(sample.y)
  
  # x only part
  u.x <- sample.x %*% t(sample.x)
  diag(u.x) <- 0
  u.x <- sum(u.x) / (n1 * (n1 - 1))
  
  # y only part
  u.y <- sample.y %*% t(sample.y)
  diag(u.y) <- 0
  u.y <- sum(u.y) / (n2 * (n2 - 1))
  
  # x, y part
  u.xy <- sample.x %*% t(sample.y)
  u.xy <- sum(u.xy) * 2 / (n1 * n2)
  
  return(u.x + u.y - u.xy)
}

permutation.test <-
  function(B, data.x, data.y, kappa, alpha, discrete = FALSE) {
    n1 <- nrow(data.x)
    n2 <- nrow(data.y)
    d <- ncol(data.x)
    
    data.x <- multivariate.binning(data.x, kappa)
    data.y <- multivariate.binning(data.y, kappa)
    
    if (discrete) {
      noise.x <-
        noise.discrete(
          n = n1,
          d = d,
          kappa = kappa,
          alpha = alpha
        )
      
      noise.y <-
        noise.discrete(
          n = n2,
          d = d,
          kappa = kappa,
          alpha = alpha
        )
      
    } else{
      noise.x <- noise.conti(
        n = n1,
        d = d,
        kappa = kappa,
        alpha = alpha
      )
      noise.y <-
        noise.conti(
          n = n2,
          d = d,
          kappa = kappa,
          alpha = alpha
        )
    }
    #print(mean(noise.x))
    #print(mean(noise.y))
    data.x <- kappa ^ {
      d / 2
    } * data.x + noise.x
    data.y <- kappa ^ {
      d / 2
    } * data.y + noise.y
    
    u.n1.n2 <- u.stat.two.sample(data.x, data.y)
    
    #permutation procedure
    
    data.combined <- rbind(data.x, data.y)
    permutation.stats <- rep(0, B)
    for (rep in 1:B) {
      #cat(rep, "th permutation\n")
      perm <-
        sample(1:(n1 + n2)) # Question: same permutation can appear.
      data.permuted <- data.combined[perm, ]
      data.permuted.x <- data.permuted[1:n1, ]
      data.permuted.y <- data.permuted[(n1 + 1):(n1 + n2), ]
      permutation.stats[rep] <-
        u.stat.two.sample(data.permuted.x, data.permuted.y)
    }
    return(sum(permutation.stats > u.n1.n2) / (B + 1))
  }
