
### adapted from exerciseflat.m matlab code

library(ggplot2)
library(gganimate)
library(reshape2)
library(jsonlite)
library(dplyr)

config <- read_json('config.json')

# System discretisation
DeltaX <- config$DeltaX # (m)
DeltaY <- config$DeltaY # (m)

# Diffusion constants for plants, soil water and surface water (see remark above)
DifP <- config$DifP   # (m2.d-1)
DifW <- config$DifW   # (m2.d-1)
DifO <- config$DifO   # (m2.d-1)

# Initial fraction of grid cells with bare ground
frac <- config$frac  # (-)

# Parameter values
R       <-   config$R      # Rainfall (mm.d-1)
alpha   <-   config$alpha  # Proportion of surface water available for infiltration (d-1)
W0      <-   config$W0     # Bare soil infiltration (-)
beta    <-   config$beta   # Plant loss rate due to grazing (d-1)
rw      <-   config$rw     # Soil water loss rate due to seepage and evaporation (d-1)
c       <-   config$c      # Plant uptake constant (g.mm-1.m-2)
gmax    <-   config$gmax   # Plant growth constant (mm.g-1.m-2.d-1)
d       <-   config$d      # Plant senescence rate (d-1)
k1      <-   config$k1     # Half saturation constant for plant uptake and growth (mm)
k2      <-   config$k2     # Half saturation constant for water infiltration (g.m-2)

# Number of grid cells
m <- config$m
NX <- m
NY <- m

# Timesteps
dT <- 1     # timestep
Time <- 1      # begin time
EndTime <- 5000    # end time
PlotStep <- 10 # (d)
PlotTime <- PlotStep # (d)

# Initialisation
popP <- matrix(0,m,m);
popW <- matrix(0,m,m);
popO <- matrix(0,m,m);
dP <- matrix(0,m,m);
dO <- matrix(0,m,m);
dW <- matrix(0,m,m);
NetP <- matrix(0,m,m);
NetW <- matrix(0,m,m);
NetO <- matrix(0,m,m);


#Boundary conditions
FYP <- matrix(0,NY+1,NX);   	# bound.con. no flow in/out to Y-direction
FXP <- matrix(0,NY,NX+1);		# bound.con. no flow in/out to X-direction
FYW <- matrix(0,NY+1,NX);   	# bound.con. no flow in/out to Y-direction
FXW <- matrix(0,NY,NX+1);		# bound.con. no flow in/out to X-direction
FYO <- matrix(0,NY+1,NX);   	# bound.con. no flow in/out to Y-direction
FXO <- matrix(0,NY,NX+1);		# bound.con. no flow in/out to X-direction

# Initial state
for (i in 1:m) {
  for (j in 1:m) {
    if (runif(1) > frac) {
      popO[i,j] <- R/(alpha*W0); # Homogeneous equilibrium surface water in absence of plants
      popW[i,j]<-R/rw; # Homogeneous equilibrium soil water in absence of plants
      popP[i,j]<-90; # Initial plant biomass
    } else {
      popO[i,j]<-R/(alpha*W0); # Homogeneous equilibrium surface water in absence of plants
      popW[i,j]<-R/rw; # Homogeneous equilibrium soil water in absence of plants
      popP[i,j]<-0 # Initial plant biomass
    }
  }
}


numStepsToAnimate <- 50



animationDataFrame <- data.frame(Var1=numeric(numStepsToAnimate*m*m),
                                 Var2=numeric(numStepsToAnimate*m*m),
                                 value=numeric(numStepsToAnimate*m*m),
                                 timeStep=numeric(numStepsToAnimate*m*m))


rowOffset <- 0

## Timesteps
while (Time <= EndTime) {

# Reaction
    drO <-(R-alpha*(popP+k2*W0) /(popP+k2)*popO)
    drW<-(alpha*(popP+k2*W0) /(popP+k2)*popO-gmax*popW/(popW+k1)*popP-rw*popW);
    drP<-(c*gmax*popW /(popW+k1)*popP -(d+beta)*popP);

# Diffusion

# calculate Flow in x-direction : Flow <- -D * dpopP/dx;
    FXP[1:NY,2:NX] <- -DifP* (popP[1:m,2:NX]-popP[1:m,1:NX-1]) *DeltaY/ DeltaX;
    FXW[1:NY,2:NX] <- -DifW* (popW[1:m,2:NX]-popW[1:m,1:NX-1]) *DeltaY/ DeltaX;
    FXO[1:NY,2:NX] <- -DifO* (popO[1:m,2:NX]-popO[1:m,1:NX-1]) *DeltaY/ DeltaX;

    # calculate Flow in y-direction: Flow <- -D * dpopP/dy;
    FYP[2:NY,1:NX] <- -DifP* (popP[2:NY,1:m]-popP[1:(NY-1),1:m]) *DeltaX/ DeltaY;
    FYW[2:NY,1:NX] <- -DifW* (popW[2:NY,1:m]-popW[1:(NY-1),1:m]) *DeltaX/ DeltaY;
    FYO[2:NY,1:NX] <- -DifO* (popO[2:NY,1:m]-popO[1:(NY-1),1:m]) *DeltaX/ DeltaY;

# calculate netflow
    NetP <- (FXP[,1:NX] - FXP[,2:(NX+1)]) + (FYP[1:NY,] - FYP[2:(NY+1),]);
    NetW <- (FXW[,1:NX] - FXW[,2:(NX+1)]) + (FYW[1:NY,] - FYW[2:(NY+1),]);
    NetO <- (FXO[,1:NX] - FXO[,2:(NX+1)]) + (FYO[1:NY,] - FYO[2:(NY+1),]);
#%NewO(1:NY,1:NX)<-0;
# Update
    popW <- popW+(drW+(NetW/(DeltaX*DeltaY)))*dT;
    popO <- popO+(drO+(NetO/(DeltaX*DeltaY)))*dT;
    popP <- popP+(drP+(NetP/(DeltaX*DeltaY)))*dT;

    Time<-Time+dT;

    if (Time %% (EndTime / numStepsToAnimate) == 0) {
        print(paste("time", Time))
        popP_melted <- melt(popP)
        popFrame <- cbind(popP_melted, Time)
        animationDataFrame[(rowOffset*m*m):((rowOffset+1)*m*m-1),] <- popFrame
        rowOffset <- rowOffset + 1
    }

    PlotTime <- PlotTime-dT;
    if (PlotTime <= 0) {
        popP_melted <- melt(popP)

        plot <- ggplot(data=popP_melted, aes(x=Var1,y=Var2,fill=value)) + geom_tile()

        PlotTime<-PlotStep;

    }

}


animation <- ggplot(data=animationDataFrame,
                    aes(x=Var1,y=Var2,fill=value)) +
    geom_tile() + scale_fill_gradientn(colours=terrain.colors(10),
                                       limits=c(3,5.5)) +
    transition_time(timeStep) + labs(title = "Time: {frame_time}") + view_follow()
