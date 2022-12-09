# install dependencies (uncomment and do only once)
#install.packages("plot3D"); install.packages("ggplot2"); install.packages("MCMCpack"); install.packages('akima'); install.packages('rgl')
# load dependencies
require(plot3D); require(ggplot2); require(MCMCpack); require(akima); require(rgl)

# Dirichlet parameters (Customize these!)
alpha_params = c(2,2,2)

# Get a grid of points and normalize them to be on the simplex
granularity = 20
draws <- matrix(ncol=3,nrow=(granularity*granularity*granularity)-1)
# lots of points on the edge
draws[1,]
i=0
for (x in 1:granularity){
  for (y in 1:granularity){
    for (z in 1:granularity){
      draws[i,] <- c(x,y,z) # point on grid
      draws[i,] = draws[i,] / sum(draws[i,]) # normalize
      i = i+1
    }
  }
}
x <- draws[,1]
y <- draws[,2]
z <- draws[,3]
density <- ddirichlet(draws, alpha_params)
# transform the simplex to euclidean space (eliminating one dimension)
x <- .5 * (2*x+y)
y <- .5 * y * sqrt(3)
# interpolate a fine (100x100) grid from our points
grid <- interp.new(x,y,density,duplicate="strip",linear=FALSE,
                   xo=seq(min(x),max(x),length=100),
                   yo=seq(min(y),max(y),length=100),
)

# PLOT #1: a heatmap 
image2D(x=grid$x, y=grid$y, z=grid$z)

# PLOT #2: an interactive 3d plot
surface3d(x=grid$x,y=grid$y,z=grid$z,color=terrain.colors(2),back="lines")

# PLOT #3: a 3D image full colored image
grid <- interp.new(x,y,density,duplicate="strip",linear=FALSE) # more coarse grid
persp3D(grid$x, grid$y, grid$z, colvar = grid$z, 
        col = ramp.col(col = c("violet", "pink"), n = 100),
        colkey = FALSE, shade = 0.5, alpha = 0.3, expand = 1.2,
        box = FALSE, phi = 35, border = "black", theta = 70, 
        lighting = TRUE, ltheta = 560, lphi = -100)