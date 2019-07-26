from config import *
import matplotlib.pyplot as plt
import numpy as np
import random


def patterns():
    # Initialisation
    popP = np.zeros((m, m))
    popW = np.zeros((m, m))
    popO = np.zeros((m, m))
    dP = np.zeros((m, m))
    dO = np.zeros((m, m))
    dW = np.zeros((m, m))
    NetP = np.zeros((m, m))
    NetW = np.zeros((m, m))
    NetO = np.zeros((m, m))

    # Boundary conditions
    FYP = np.zeros((NY + 1, NX)) # bound.con.no flow in / outo Y - direction
    FXP = np.zeros((NY, NX + 1)) # bound.con.no fow in / out to X - direction
    FYW = np.zeros((NY + 1, NX)) # bound.con.no flow in / out to Y - direction
    FXW = np.zeros((NY, NX + 1)) # bound.con.no flow in / out to X - direction
    FYO = np.zeros((NY + 1, NX)) # bound.con.no flow in / out to Y - direction
    FXO = np.zeros((NY, NX + 1)) # bound.con.no flow in / out to X - direction

    # Initial state
    for i in range(1,m):
        for j in range(1,m):
            if random.random() > frac:
                popO[i, j] = R / (alpha * W0)
                popW[i, j] = R / rw # Homogeneous equilibrium soil water in absence of plants
                popP[i, j] = 90 # Initial plant biomass
            else:
                popO[i, j] = R / (alpha * W0) # Homogeneousequilibriumsurfacewater in absenceof plants
                popW[i, j] = R / rw # Homogeneousequilibriumsoil water in absenceofplants
                popP[i, j] = 0 # Initialplant biomass

    # Timesteps
    dT = 1  # timestep
    Time = 1  # begin time
    EndTime = 5000  # end time
    PlotStep = 10  # (d)
    PlotTime = PlotStep  #(d)
    #  Timesteps


    while Time <= EndTime:

        # Reaction
        drO = (R - np.divide(alpha * (popP + k2 * W0), (popP + k2))* popO)
        drW = (alpha * np.divide((popP + k2 * W0), (popP + k2)) * popO - gmax * np.divide(popW, (popW + k1))* popP - rw * popW)
        drP = (c * gmax * np.divide(popW,(popW + k1)) * popP - (d + beta)* popP)

        # Diffusion
        # calculate Flow in x - direction: Flow = -D * dpopP / dx;
        FXP[0:NY, 1:NX] = -DifP * (popP[:, 1:NX] - popP[:, 0:(NX - 1)]) *DeltaY / DeltaX
        FXW[0:NY, 1:NX] = -DifW * (popW[:, 1:NX] - popW[:, 0:(NX - 1)]) *DeltaY / DeltaX
        FXO[0:NY, 1:NX] = -DifO * (popO[:, 1:NX] - popO[:, 0:(NX - 1)]) *DeltaY / DeltaX

        # calculate Flow in y - direction: Flow = -D * dpopP / dy;
        FYP[1:NY, 0:NX] = -DifP * (popP[1:NY,:] - popP[0:(NY - 1),:]) *DeltaX / DeltaY
        FYW[1:NY, 0:NX] = -DifW * (popW[1:NY,:] - popW[0:(NY - 1),:]) *DeltaX / DeltaY
        FYO[1:NY, 0:NX] = -DifO * (popO[1:NY,:] - popO[0:(NY - 1),:]) *DeltaX / DeltaY

        # calculate netflow
        NetP = (FXP[:, 0:NX] - FXP[:, 1:(NX + 1)]) + (FYP[0:NY,:] - FYP[1:NY + 1,:])
        NetW = (FXW[:, 0:NX] - FXW[:, 1:(NX + 1)]) + (FYW[0:NY,:] - FYW[1:NY + 1,:])
        NetO = (FXO[:, 0:NX] - FXO[:, 1:(NX + 1)]) + (FYO[0:NY,:] - FYO[1:NY + 1,:])
        # NewO(1:NY, 1:NX)=0;
        # Update
        popW = popW + (drW + (NetW / (DeltaX * DeltaY))) * dT
        popO = popO + (drO + (NetO / (DeltaX * DeltaY))) * dT
        popP = popP + (drP + (NetP / (DeltaX * DeltaY))) * dT

        Time = Time + dT

        PlotTime = PlotTime - dT
        if PlotTime <= 0:
            plt.imshow(popP)
            plt.colorbar()
            plt.show()
patterns()