"""
Python version of mao_pollen.m matlab code to look at connectedness of
pixels on a binary image
"""

import numpy as np
from scipy import spatial



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

    W_phi, W_lambda = np.linalg.eig(W)

    print (W_phi)


if __name__ == "__main__":

    image_matrix = read_file("binary_image.txt")

    mao_pollen(image_matrix)