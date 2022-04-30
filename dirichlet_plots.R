library(rBeta2009)
library(plotly)

# visualize how difficult the testing is
start_time <- Sys.time()


plot_3d <- function(data){
  p <- plot_ly(
    data,
    x = data[ ,1],
    y = data[ ,2],
    z = data[ ,3],
  ) %>% add_markers(marker=list(size=1))
  return(p)
}

plot_3d_dirichlet <- function(n, params){
  plot_3d(data.frame(rdirichlet(n, params)))  
}

plot_3d_dirichlet_compare <- function(n, params1, params2){
  blue <- rdirichlet(n, params1)
  red <- rdirichlet(n, params2)
  label <- c(rep(1,n), rep(0,n))
  data <- rbind(blue, red)
  data <- cbind(data, label)
  data <- data.frame(data)
  
  p <- plot_ly(
    data,
    x = data[ ,1],
    y = data[ ,2],
    z = data[ ,3],
    color = data[,4], colors = c('#BF382A', '#0C4B8E')
  ) %>% add_markers(marker=list(size=1))
  return(p)
}

plot_3d_dirichlet(10000, c(1, 1, 1) * 1)
plot_3d_dirichlet(10000, c(1, 1, 1) * 5)
plot_3d_dirichlet(10000, c(1, 1, 1) * 10)
plot_3d_dirichlet(10000, c(1, 1, 1) * 20)

plot_3d_dirichlet(10000, c(10, 1, 1) * 1)
plot_3d_dirichlet(10000, c(10, 1, 1) * 5)
plot_3d_dirichlet(10000, c(10, 1, 1) * 10)
plot_3d_dirichlet(10000, c(10, 1, 1) * 20)

plot_3d_dirichlet(10000, c(10, 5, 1) * 1)
plot_3d_dirichlet(10000, c(10, 5, 1) * 5)
plot_3d_dirichlet(10000, c(10, 5, 1) * 10)
plot_3d_dirichlet(10000, c(10, 5, 1) * 20)

plot_3d_dirichlet_compare(10000, c(5, 1, 1) * 5, c(1, 5, 1) * 5)
plot_3d_dirichlet_compare(10000, c(5, 1, 1) * 10, c(1, 5, 1) * 10)
plot_3d_dirichlet_compare(10000, c(5, 1, 1) * 20, c(1, 5, 1) * 20)

plot_3d_dirichlet_compare(10000, c(2, 1, 1) * 1, c(1, 2, 1) * 1)
plot_3d_dirichlet_compare(10000, c(2, 1, 1) * 5, c(1, 2, 1) * 5)
plot_3d_dirichlet_compare(10000, c(2, 1, 1) * 10, c(1, 2, 1) * 10)
plot_3d_dirichlet_compare(10000, c(2, 1, 1) * 20, c(1, 2, 1) * 20)