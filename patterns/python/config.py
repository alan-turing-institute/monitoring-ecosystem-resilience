# System discretisation (see remark above)
DeltaX=101 #(m)
DeltaY=101 #(m)

#Diffusion constants for plants, soil water and surface water (see remark above)
DifP=100   #(m2.d-1)
DifW=10     #(m2.d-1)
DifO=1000  #(m2.d-1)

#Initial fraction of grid cells with bare ground
frac=0.90  #(-)

#Parameter values
alpha   =   0.1  #Proportion of surface water available for infiltration (d-1)
W0      =  0.15  #Bare soil infiltration (-)
beta    =  0.3   #Plant loss rate due to grazing (d-1)
rw      =   0.1  #Soil water loss rate due to seepage and evaporation (d-1)
c       =    10  #Plant uptake constant (g.mm-1.m-2)
gmax    =  0.05  #Plant growth constant (mm.g-1.m-2.d-1)
d       =   0.1  #Plant senescence rate (d-1)
k1      =     3  #Half saturation constant for plant uptake and growth (mm)
k2      =     5  #Half saturation constant for water infiltration (g.m-2)

#Number of grid cells
m=50
NX=m
NY=m
