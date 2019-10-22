
"""
Translation of Matlab code to model patterned vegetation in semi-arid landscapes.
"""
import sys
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# import a set of constants from configuration file
from config import *

def plot_image(value_array):
    im = plt.imshow(value_array)
    plt.show()


def make_binary(value_array, threshold=None):
    """
    if not given a threshold to use,  look at the (max+min)/2 value
    - for anything below, set to zero, for anything above, set to 1
    """
    if not threshold:
        threshold = (value_array.max() + value_array.min()) / 2.
    new_list_x = []
    for row in value_array:
        new_list_y = np.array([float(val > threshold) for val in row])
        new_list_x.append(new_list_y)
    return np.array(new_list_x)

def generate_pattern(rainfall):  # rainfall in mm
    """
    Run the code to converge on a vegetation pattern
    """
    print("Generating pattern with rainfall {}mm".format(rainfall))
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
    FXP = np.zeros((NY, NX + 1)) # bound.con.no flow in / out to X - direction
    FYW = np.zeros((NY + 1, NX)) # bound.con.no flow in / out to Y - direction
    FXW = np.zeros((NY, NX + 1)) # bound.con.no flow in / out to X - direction
    FYO = np.zeros((NY + 1, NX)) # bound.con.no flow in / out to Y - direction
    FXO = np.zeros((NY, NX + 1)) # bound.con.no flow in / out to X - direction

    # Initial state
    for i in range(1,m):
        for j in range(1,m):
            if random.random() > frac:
                popO[i, j] = rainfall / (alpha * W0)
                popW[i, j] = rainfall / rw # Homogeneous equilibrium soil water in absence of plants
                popP[i, j] = 90 # Initial plant biomass
            else:
                popO[i, j] = rainfall / (alpha * W0) # Homogeneous equilibrium surface water in absenceof plants
                popW[i, j] = rainfall / rw # Homogeneous equilibriums oil water in absence of plants
                popP[i, j] = 0 # Initial plant biomass

    # Timesteps
    dT = 1  # timestep
    Time = 1  # begin time
    EndTime = 10000  # end time
    PlotStep = 10  # (d)
    PlotTime = PlotStep  #(d)
    #  Timesteps

    snapshots = []
    while Time <= EndTime:

        # Reaction
        drO = (rainfall - np.divide(alpha * (popP + k2 * W0), (popP + k2))* popO)
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
        axes = plt.gca()

        if PlotTime <= 0:
            snapshots.append(popP)

    # create figure
    fig = plt.figure()

    # plot final figure
    im = plt.imshow(snapshots[-1], vmax=5.5,vmin=4.5)
    plt.colorbar()

    # animation function
    def animate_func(i):
        if i % fps == 0:
            print('.', end='')
        im.set_array(snapshots[i])
        return [im]

    fps = len(snapshots)

    # run animation
    anim = animation.FuncAnimation(
        fig,
        animate_func,
    )

    try:
        anim.save('test_anim.html', fps=fps*1000, extra_args=['-vcodec', 'libx264'])
    except:
        pass
    print('Done!')
    return snapshots

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate vegetation patterns")
    parser.add_argument("--rainfall", help="rainfall in mm",type=float, default=1.4)
    args = parser.parse_args()
    snapshots = generate_pattern(args.rainfall)


    binary_pattern = make_binary(snapshots[-1])
    plot_image(binary_pattern)
