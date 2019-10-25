"""
Python version of mao_pollen.m matlab code to look at connectedness of
pixels on a binary image
"""

import numpy as np
from scipy import spatial
import casadi


def read_file(input_filename):
    """
    Do a bunch of random stuff
    """
    input_image = np.loadtxt(input_filename, dtype='i', delimiter=',')
    return input_image



def calc_adjacency_matrix(input_array,
                          threshold=255,
                          neighbour_threshold=2,
                          upper_threshold=True):
    """
    Given an input image as a 2D array of pixel values, return a
    symmetric matrix of (n-pixels-over-threshold)x(n-pixel-over-threshold)
    where each element ij is 0 or 1 depending on whether the distance between
    pixel i and pixel j is < or > neighbour_threshold
    """

    # flatten the input image for later use
    input_flat = input_array.flatten()
    # find all the "white" pixels
    white_x_y = np.where(input_array>=threshold)

    # convert ([x1, x2, ...], [y1,y2,...]) to ([x1,y1],...)
    white_coords = [(white_x_y[0][i], white_x_y[1][i]) \
                    for i in range(len(white_x_y[0]))]
    # find all the distances between pairs of white pixels
    # (if there are N white pixels, this will be a
    # vector of length N(N-1)/2 )
    distances = spatial.distance.pdist(white_coords)
    # convert this into an NxN distance matrix, with each element ij
    # as the distance between pixel i and pixel j
    dist_square = spatial.distance.squareform(distances)

    # prepare to make the adjacency matrix
    W = np.zeros(dist_square.shape)
    T = np.zeros(dist_square.shape)
    # get 2xM array where M is the number of elements of
    # the distance matrix that satisfy the "neighbour threshold" criterion.
    # i.e. the x,y coordinates of all "neighbours" in the distance matrix.
    neighbour_x_y = np.where(dist_square>0 and dist_square < neighbour_threshold)

    # loop over coordinates of neighbours in distance matrix
    for i in range(len(neighbour_x_y[0])):
        # ===> NOTE!!! we don't really understand this step.
        # Looks like for 2 consecutive neighbours, one gets weighted down to 0.01
        if (input_flat[neighbour_x_y[0][i]] == input_flat[neighbour_x_y[1][i]]):
            W[neighbour_x_y[0][i]][neighbour_x_y[1][i]] = 1
        else:
            W[neighbour_x_y[0][i]][neighbour_x_y[1][i]] = 0.01
        # T is the adjacency matrix, will be NxN (with N as num-white-pixels)
        # and element ij is 1 if pix i and pix j satisfy neighbour threshold.
        T[neighbour_x_y[0][i]][neighbour_x_y[1][i]] = 1
    # return both T (the adjacency matrix) and W (similar but with some
    # elements set to 0.01)
    return T, W


def calc_and_sort_SC_indices(adjacency_matrix):
    """
    Given an input adjacency matrix, calculate eigenvalues and eigenvectors,
    calculate the subgraph centrality (ref: <== ADD REF), then sort.
    """

    # get eigenvalues
    am_lambda, am_phi = np.linalg.eigh(adjacency_matrix)

    # calculate the subgraph centrality (SC)
    phi2_explamba = np.dot(am_phi * am_phi, np.exp(am_lambda))

    # order the pixels by subgraph centrality, then find their indices
    # (corresponding to their position in the 1D list of white pixels)
    indices= np.argsort(phi2_explamba)[::-1]
    return indices


def find_sc_quantiles(pix_indices, adj_matrix, num_quantiles):
    """
    Given indices of white pixels ordered by SC value,
    do ...
    """

    # adj_matrix will be square - take the length of a side
    n = max(adj_matrix.shape)
    # set the diagonals of the adjacency matrix to 1 (previously
    # zero by definition because a pixel can't be adjacent to itself)
    for j in range(n):
        adj_matrix[j][j] = 1


    # find the different quantiles
    start = 0
    end = 100
    step = (end-start)/num_quantiles
    x = [i for i in range(start,end+1,step)]
    # create feature vector of size num_quantiles
    feature_vector = np.zeros(num_quantiles);

    # Loop through the quantiles to fill the feature vector
    for i in range(1,len(feature_vector)):
        print("calculating subregion {} of {}".format(i, num_quantiles))
        # how many pixels in this sub-region?
        n_pix = round(x[i] * n / 100)

        sub_region = pix_indices[0:n_pix]
        print (sub_region)
        # calculate the Dulmage-Mendelsohn decomposition (ref <== REF)
        S = casadi.DM(adj_matrix[np.ix_(sub_region,sub_region)])
        # calculate the block-triangular form
        nb, rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock = S.sparsity().btf()

        p = rowperm
        q = colperm
        r = np.array(rowblock)
        s = colblock

        print (r)
        feature_vector[i] = len(r) - 1 # maybe 0 ?
        r2 = (r[0:(len(r) - 1)]) # used for kicking out too small component
        k = np.where((r[1:end]-r2) < 1); # value 1  can be changed to other values
        feature_vector[i] = feature_vector[i] - len(k)

    feature_vector = feature_vector[1:end]
    return_feature_vector


if __name__ == "__main__":

    image_matrix = read_file("binary_image.txt")

    mao_pollen(image_matrix)
