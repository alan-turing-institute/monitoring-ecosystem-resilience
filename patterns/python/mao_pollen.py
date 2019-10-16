"""
Python version of mao_pollen.m matlab code to look at connectedness of
pixels on a binary image
"""

import numpy as np
from scipy import spatial
from casadi import *


def read_file(input_filename):
    """
    Do a bunch of random stuff
    """
    input_image = np.loadtxt(input_filename, dtype='i', delimiter=',')
    return input_image



def mao_pollen(input_array, threshold=255, neighbour_threshold = 2):
    """
    placeholder
    """

    input_array = input_array[0:9,0:9]
    input_flat = input_array.flatten()
    white_x_y = np.where(input_array>=threshold)

    # convert ([x1, x2, ...], [y1,y2,...]) to ([x1,y1],...)
    white_coords = [(white_x_y[0][i], white_x_y[1][i]) \
                    for i in range(len(white_x_y[0]))]

    distances = spatial.distance.pdist(white_coords)
    dist_square = spatial.distance.squareform(distances)

    W = np.zeros(dist_square.shape)
    T = np.zeros(dist_square.shape)

    neighbour_x_y = np.where((dist_square > 0) & (dist_square <neighbour_threshold))

    for i in range(len(neighbour_x_y[0])):
        if (input_flat[neighbour_x_y[0][i]] == input_flat[neighbour_x_y[1][i]]):
            W[neighbour_x_y[0][i]][neighbour_x_y[1][i]] = 1
        else:
            W[neighbour_x_y[0][i]][neighbour_x_y[1][i]] = 0.01

        T[neighbour_x_y[0][i]][neighbour_x_y[1][i]] = 1

    # diagonlizing the dense matrix
    # subgraph centrality


    from scipy.sparse import isspmatrix

    W_lambda, W_phi = np.linalg.eig(W)


    phi2_explamba = np.dot(W_phi * W_phi, np.exp(W_lambda))


    Ind = np.argsort(phi2_explamba)[::-1]
    sorted_subgraph_centrality = sorted(phi2_explamba,reverse=True)


    n1 = max(T.shape)

    for j in range(n1):
        print ('something')
        T[j][j] = 1


    # this is 20-dim. e.g. use range(0,2,100) for 50-dim
    input = 0
    end = 100
    steps = 5
    x = [i for i in range(input,end+1,steps)]
    g = x;
    g[0] = 0;

    for i in range(1,len(g)):

        print (i)

        t = round(x[i] * n1 / 100)

        sub = Ind[0:t]

        S = DM(T[sub, sub])

        nb, rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock = S.sparsity().btf()

        p = rowperm
        q = colperm
        r = np.array(rowblock)
        s = colblock

        print (r)
        print (s)
    #
        g[i] = len(r) - 1 # maybe 0 ?
        r2 = np.array(r[0:(len(r) - 1)]) # used for kicking out too small component
        k = np.where((r[1:end]-r2) < 1); # value 1  can be changed to other values
        g[i] = g[i] - len(k)


    g = g[1:end]

    print (g)


if __name__ == "__main__":

    image_matrix = read_file("binary_image.txt")

    mao_pollen(image_matrix)