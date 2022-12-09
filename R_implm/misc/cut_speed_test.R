library(tictoc)

data.raw <- runif(10000)
for (i in 1:100){
  data.raw <- cbind(data.raw, runif(10000))
}


tic("cut the matrix")
data.indices <- cut(x = data.raw,
                    breaks = seq(from = 0, to = 1, length = 100),
                    include.lowest = TRUE)

data.indices <- as.numeric(data.indices)
data.indices <- matrix(data.indices, ncol = d, byrow = FALSE)
toc()

tic("for loop by columns")
for (j in 1:ncol(data.raw)){
  data.raw[,j] <- cut(x = data.raw[,j],
                      breaks = seq(from = 0, to = 1, length = 100),
                      include.lowest = TRUE)
}
data.indices <- as.numeric(data.indices)
toc()