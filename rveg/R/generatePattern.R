
### adapted from exerciseflat.m matlab code

#library(ggplot2)
#library(gganimate)
#library(reshape2)
#library(jsonlite)
#library(dplyr)

#' Load configuration from a JSON file
#' @param configFile Name of the JSON file containing configuration parameters
#' @return config an environment containing the configured parameters.
#' @export
loadConfig <- function(configFile=file.path("..","testdata","patternConfig.json")) {
    config <- jsonlite::read_json(configFile)
    return(config)
}

#' Given a 2D array of floats, return a 2D array (same dimensions) of 1s and 0s.
#' By default, take the threshold to be halfway between min and max values.
#' @param pattern Matrix of floats representing biomass in each square of a grid
#' @param threshold Optionally specify a threshold above which we will put a 1 in the output matrix.
binarizePattern <- function(pattern, threshold=NULL) {
    if (is.null(threshold)) {
        threshold = (max(pattern) + min(pattern)) / 2
    }
    binaryPattern = (pattern > threshold)*1
    return(binaryPattern)

}


#' Load a CSV file containing a 2D array of 1s and zeros, representing a starting pattern.
#' If no file is given, start with all zeros.
#' @param patternFile filename of CSV file containing starting pattern of 1s and 0s.
#' @param m length of each side of the matrix.
#' @return pattern m*m matrix of 1s and 0s.
#' @export
getStartingPattern <- function(patternFile=NULL,m=50) {
    pattern <- matrix(0,m,m);
    if (! is.null(patternFile)) {
        pattern <- as.matrix(read.csv(patternFile, header=FALSE))
        pattern <- pattern[1:m, 1:m]
    }
    return(pattern)
}

#' Use the model to evolve the starting pattern through several timesteps
#' keeping track of surface and soil water.  Return the final pattern.
#' @param configFile JSON configuration file
#' @param patternFile CSV file containing initial pattern (if not provided, will be all 0s).
#' @return popP Final pattern - a 2D array.
#' @export
generatePattern <- function(configFile=file.path("..","testdata","patternGenConfig.json"),
                            startingPatternFilename=NULL) {
    print(getwd())
    config <- loadConfig(configFile)
    ## System discretisation
    DeltaX <- config$DeltaX # (m)
    DeltaY <- config$DeltaY # (m)
    ## Diffusion constants for plants, soil water and surface water (see remark above)
    DifP <- config$DifP   # (m2.d-1)
    DifW <- config$DifW   # (m2.d-1)
    DifO <- config$DifO   # (m2.d-1)
    ## Initial fraction of grid cells with bare ground
    frac <- config$frac  # (-)

    ## Parameter values
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

    ## Number of grid cells
    m <- config$m
    NX <- m
    NY <- m

    ## Timesteps
    dT <- 1     # timestep
    Time <- 1      # begin time
    EndTime <- 5000    # end time
    PlotStep <- 10 # (d)
    PlotTime <- PlotStep # (d)

    ## Initialisation
    popP <- getStartingPattern(startingPatternFilename, m)
    popW <- matrix(0,m,m);
    popO <- matrix(0,m,m);
    dP <- matrix(0,m,m);
    dO <- matrix(0,m,m);
    dW <- matrix(0,m,m);
    NetP <- matrix(0,m,m);
    NetW <- matrix(0,m,m);
    NetO <- matrix(0,m,m);


    ## Boundary conditions
    FYP <- matrix(0,NY+1,NX);   	# bound.con. no flow in/out to Y-direction
    FXP <- matrix(0,NY,NX+1);		# bound.con. no flow in/out to X-direction
    FYW <- matrix(0,NY+1,NX);   	# bound.con. no flow in/out to Y-direction
    FXW <- matrix(0,NY,NX+1);		# bound.con. no flow in/out to X-direction
    FYO <- matrix(0,NY+1,NX);   	# bound.con. no flow in/out to Y-direction
    FXO <- matrix(0,NY,NX+1);		# bound.con. no flow in/out to X-direction

    ## Initial state
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


    rowOffset <- 0

    ## Timesteps
    while (Time <= EndTime) {

        ## Reaction
        drO <-(R-alpha*(popP+k2*W0) /(popP+k2)*popO)
        drW<-(alpha*(popP+k2*W0) /(popP+k2)*popO-gmax*popW/(popW+k1)*popP-rw*popW);
        drP<-(c*gmax*popW /(popW+k1)*popP -(d+beta)*popP);

        ## Diffusion

        ## calculate Flow in x-direction : Flow <- -D * dpopP/dx;
        FXP[1:NY,2:NX] <- -DifP* (popP[1:m,2:NX]-popP[1:m,1:NX-1]) *DeltaY/ DeltaX;
        FXW[1:NY,2:NX] <- -DifW* (popW[1:m,2:NX]-popW[1:m,1:NX-1]) *DeltaY/ DeltaX;
        FXO[1:NY,2:NX] <- -DifO* (popO[1:m,2:NX]-popO[1:m,1:NX-1]) *DeltaY/ DeltaX;

        ## calculate Flow in y-direction: Flow <- -D * dpopP/dy;
        FYP[2:NY,1:NX] <- -DifP* (popP[2:NY,1:m]-popP[1:(NY-1),1:m]) *DeltaX/ DeltaY;
        FYW[2:NY,1:NX] <- -DifW* (popW[2:NY,1:m]-popW[1:(NY-1),1:m]) *DeltaX/ DeltaY;
        FYO[2:NY,1:NX] <- -DifO* (popO[2:NY,1:m]-popO[1:(NY-1),1:m]) *DeltaX/ DeltaY;

        ## calculate netflow
        NetP <- (FXP[,1:NX] - FXP[,2:(NX+1)]) + (FYP[1:NY,] - FYP[2:(NY+1),]);
        NetW <- (FXW[,1:NX] - FXW[,2:(NX+1)]) + (FYW[1:NY,] - FYW[2:(NY+1),]);
        NetO <- (FXO[,1:NX] - FXO[,2:(NX+1)]) + (FYO[1:NY,] - FYO[2:(NY+1),]);
        ## Update
        popW <- popW+(drW+(NetW/(DeltaX*DeltaY)))*dT;
        popO <- popO+(drO+(NetO/(DeltaX*DeltaY)))*dT;
        popP <- popP+(drP+(NetP/(DeltaX*DeltaY)))*dT;

        Time<-Time+dT;
    }

    return(popP)
}
