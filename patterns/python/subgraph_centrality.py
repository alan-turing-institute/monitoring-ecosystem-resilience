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


def crop_image(input_image, x_range, y_range):
    """
    return a new image from specified pixel range of input image
    """
    return input_image[x_range[0]:x_range[1], y_range[0]:y_range[1]]


def get_signal_pixels(input_array, threshold=255, lower_threshold=True):
    """
    Find coordinates of all pixels within the image that are > or <
    the threshold ( require < threshold if lower_threshold==True)
    NOTE - we make the second coordinate negative, for reasons.
    """
    # find all the "white" pixels
    signal_x_y = np.where(input_array>=threshold)

    # convert ([x1, x2, ...], [y1,y2,...]) to ([x1,y1],...)
    signal_coords = [(signal_x_y[0][i], -1*signal_x_y[1][i]) \
                    for i in range(len(signal_x_y[0]))]
    return signal_coords



def calc_distance_matrix(signal_coords):
    """
    calculate the distances between all signal pixels in the original image
    """
    # find all the distances between pairs of white pixels
    # (if there are N white pixels, this will be a
    # vector of length N(N-1)/2 )
    distances = spatial.distance.pdist(signal_coords)
    # convert this into an NxN distance matrix, with each element ij
    # as the distance between pixel i and pixel j
    dist_square = spatial.distance.squareform(distances)
    return distances, dist_square


def get_neighbour_elements(distance_vector, include_diagonal_neighbours=False):
    """
    Given a vector of distances, return element indices for where
    the distance is 1 (if include_diagonal_neighbours is false)
    or 0 < distance < 2 (if include_diagonal_neighbours is true)
    """
    if not include_diagonal_neighbours:
        return  np.where(distance_vector==1)
    else:
        return  np.where((distance_vector>0) & (distance_vector < 2))


def calc_adjacency_matrix(distance_matrix,
                          input_image,
                          include_diagonal_neighbours=False):
    """
    Return a symmetric matrix of
    (n-pixels-over-threshold)x(n-pixel-over-threshold)
    where each element ij is 0 or 1 depending on whether the distance between
    pixel i and pixel j is < or > neighbour_threshold
    """

    # prepare empty arrays
    W = np.zeros(distance_matrix.shape)
    T = np.zeros(distance_matrix.shape)
    # get 2xM array where M is the number of elements of
    # the distance matrix that satisfy the "neighbour threshold" criterion.
    # i.e. the x,y coordinates of all "neighbours" in the distance matrix.
    if not include_diagonal_neighbours:
        neighbour_x_y = np.where(distance_matrix==1)
    else:
        neighbour_x_y = np.where((distance_matrix>0) & (distance_matrix < 2))

    # get the flattened input image for the next step
    input_flat = input_image.flatten()

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


def calc_and_sort_sc_indices(adjacency_matrix):
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


def find_ec_quantiles(coords, pix_indices, num_quantiles=20):
    # find the different quantiles
    start = 0
    end = 100
    step = (end-start)/num_quantiles
    x = [i for i in range(start,end+1,int(step))]
    # how many signal pixels?
    n = len(coords)
    # create feature vector of size num_quantiles
    feature_vector = np.zeros(num_quantiles+1)
    selected_pixels = {}
    # Loop through the quantiles to fill the feature vector
    for i in range(1,len(feature_vector)):
        print("calculating subregion {} of {}".format(i, num_quantiles))
        # how many pixels in this sub-region?
        n_pix = round(x[i] * n / 100)
        sub_region = pix_indices[0:n_pix]
        sel_pix = [coords[j] for j in sub_region]
        try:
            sub_dist, sub_dist_matrix = calc_distance_matrix(sel_pix)
            sub_neighbours = get_neighbour_elements(sub_dist)[0]
            sub_coords = get_neighbour_elements(sub_dist_matrix)
            # ==> NOTE we don't understand the next steps!
            nb = np.where((pix_indices[sub_coords[1]] \
                           - pix_indices[sub_coords[0]]) == 1)[0]
            f = 0
            for j in range(len(nb)):
                tmp1 = [ sel_pix[k] for k in sub_coords[0][nb] ]
                tmp2 = np.add(np.array([ sel_pix[k] \
                                         for k in sub_coords[0][nb]][j]),
                              np.array([1,0]))
                tmp3 = np.tile(tmp2, (len(nb),1))
                tmp = tmp1 - tmp3
                ff1 = np.where((tmp[:,0]==0) & (tmp[:,1]==0))
                f += len(ff1[0])
            feature_vector[i] = n_pix - len(sub_neighbours) + f
        except:

            feature_vector[i] = 0

        selected_pixels[x[i]] = sel_pix

    return feature_vector, selected_pixels


def do_everything(input_filename):
    # flatten the input image for later use
    input_flat = input_array.flatten()

    # get the coordinates of all the signal pixels
    signal_coords = get_signal_pixels(input_array, threshold, lower_threshold)



    pass


if __name__ == "__main__":

    image_matrix = read_file("binary_image.txt")

    do_everything(input_image)



    mao_pollen(image_matrix)
