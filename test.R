
Bin <- function(data, kappa) {
  
  d <- ncol(data)
  # 1. for each dimension, turn the continuous data into interval
  #   each row now indicates a hypercube in [0,1]^d
  #  the more the data is closer to 1, the larger the interval index.
  data.interval <- .TransformIntervalIndex(
    data = data, 
    n.intervals = kappa
    )
  
  # 2. for each datapoint(row),
  #    turn the hypercube data into a multivariate data of (1, 2, ..., kappa^d)
  #    each row now becomes an integer.
  data.multivariate <- .TransformMultivariate(
    data.interval = data.interval,
    n.bin = kappa,
    dim = d
    )
  
  # 3. turn the indices into one-hot vectors
  data.onehot <- .TransformOnehot(data.multivariate, kappa^d)
  
  return(data.onehot)
}

.TransformIntervalIndex <- function(data, n.intervals) {
  # for each dimension, transform the data in [0,1] into the interval index
  # first interval = [0, x], the others = (y z]
  
  # create designated number of intervals
  if (is.vector(data)){ d <- 1 }
  else{ d <- ncol(data) }
  
  breaks <- seq(from = 0, to = 1, length = n.intervals + 1)
  
  # for each dimension.
  data.indices <- cut(x = data,
                      breaks = breaks,
                      labels = FALSE,
                      include.lowest = TRUE)
  data.indices <- matrix(data.indices,
                         ncol = d,
                         byrow = FALSE # since cut function collapsed a matrix column-wise.
                         )
  return( data.indices ) 
}

.TransformMultivariate <- function(data.interval, n.bin, dim){
  return(1 + (data.interval - 1) %*% n.bin^( (dim-1) : 0 ))
}

.TransformOnehot <- function(data.multivariate, dim) {
  n <- nrow(data.multivariate)
  data.onehot <- matrix(0, nrow = n, ncol = dim) #initialize with 0
  for (row.num in 1:n) {
    bin.num <- data.multivariate[row.num]
    data.onehot[row.num, bin.num] <- 1
  }
  return(data.onehot)
}


PrivatizeTwoSample <-
  function(data,
           alpha = Inf,
           discrete.noise = FALSE) {
    ## assume the data is discrete by nature or has already been dicretized.
    n <- nrow(data)
    dim <- ncol(data) #kappa^d if conti data, d if discrete data
    scale <- sqrt(dim)
      
    if (alpha == Inf) {
      #non-private case
      return(scale * data)
    }
    else{
      #private case
        if (discrete.noise) {
          noise <- noise.discrete(
            n = n,
            dim = dim,
            alpha = alpha
          )
         
        } else{
          noise <- noise.conti(
            n = n,
            dim = dim,
            alpha = alpha
          )
        
        }
      return(scale * data + noise)
    }
  } #end of function PrivatizeTwoSample


      
noise.conti <- function(n, dim, alpha) {
  #dim = kappa^d for conti data, d for discrete data
  scale <- (sqrt(8) / alpha) * sqrt(dim)
  n.noise <- n * dim
  unit.laplace <- rexp(n.noise, sqrt(2)) - rexp(n.noise, sqrt(2))
  noise <- scale * unit.laplace
  return(noise)
}

noise.discrete <- function(n, dim, alpha) {
  #dim = kappa^d for conti data, d for discrete data
  t <- 2 * sqrt(dim) / alpha
  param.geom <- 1 - exp(-1 / t)
  n.noise <-  n * dim
  noise <- rgeom(n.noise, param.geom) - rgeom(n.noise, param.geom)
  return(noise)
}

UstatTwoSample <- function(data, n.1) {
  n.2 <- nrow(data) - n.1
  
  data.x <- data[1:n.1,]
  data.y <- data[(n.1 + 1):(n.1 + n.2), ]
  # x only part
  u.x <- data.x %*% t(data.x)
  diag(u.x) <- 0
  u.x <- sum(u.x) / (n.1 * (n.1 - 1))
  
  # y only part
  u.y <- data.y %*% t(data.y)
  diag(u.y) <- 0
  u.y <- sum(u.y) / (n.2 * (n.2 - 1))
  
  # x, y part
  u.xy <- data.x %*% t(data.y)
  u.xy <- sum(u.xy) * ( 2 / (n.1 * n.2) )
  
  return(u.x + u.y - u.xy)
}

PrivatePermutationTwoSampleTest <-
  function(B, data.x, data.y, kappa, alpha, gamma, discrete = FALSE) {
    n.1 <- nrow(data.x)
    n.2 <- nrow(data.y)
    d <- ncol(data.x)
    
    data.x.binned <- Bin(data.x, kappa)
    data.y.binned <- Bin(data.y, kappa)
    
    data.combined <- rbind(data.x.binned, data.y.binned)
    data.privatized <- PrivatizeTwoSample(data.combined, alpha)
    ustat.original <- UstatTwoSample(data.privatized, n.1)
    
    #permutation procedure
    perm.stats <- rep(0, B)
    for (rep in 1:B) {
      perm <- sample(1:(n.1 + n.2)) 
      perm.stats[rep] <- UstatTwoSample(data.privatized[perm, ], n.1)
    }
    p.value.proxy <- (1 + sum(ustat.original < perm.stats)) / (B + 1)
    
    #test result: TRUE = 1 = reject the null, FALSE = 0 = retain the null.
    
    return(p.value.proxy < gamma)
  }
