"""
Python version of mao_pollen.m matlab code to look at connectedness of
pixels on a binary image, using "Subgraph Centrality" as described in:

Mander et.al. "A morphometric analysis of vegetation patterns in dryland ecosystems",
R. Soc. open sci. (2017)
https://royalsocietypublishing.org/doi/10.1098/rsos.160443

Mander et.al. "Classification of grass pollen through the quantitative
analysis of surface ornamentation and texture", Proc R Soc B 280: 20131905.
https://royalsocietypublishing.org/doi/pdf/10.1098/rspb.2013.1905

Estrada et.al. "Subgraph Centrality in Complex Networks"
https://arxiv.org/pdf/cond-mat/0504730.pdf
"""

import numpy as np
from PIL import Image
from scipy import spatial
import argparse
import igraph

from .image_utils import image_from_array, image_file_to_array


def make_graph(adj_matrix):
    """
    Use igraph to create a graph from our adjacency matrix
    """
    graph = igraph.Graph.Adjacency((adj_matrix > 0).tolist())
    return graph


def calc_euler_characteristic(pix_indices, graph):
    """
    Find the edges where both ends are within the pix_indices list
    """
    V = len(pix_indices)
    edges = []
    for edge in graph.get_edgelist():
        if edge[0] in pix_indices and edge[1] in pix_indices:
            edges.append(edge)
    E = len(edges) / 2
    return V - E


def write_csv(feature_vec, output_filename):
    """
    Write the feature vector to a 1-line csv
    """
    output_string = ""
    for feat in feature_vec:
        output_string += str(feat) + ","
    output_string = output_string[:-1] + "\n"
    with open(output_filename, "w") as outfile:
        outfile.write(output_string)
    return True


def write_dict_to_csv(metrics_dict, output_filename):
    with open(output_filename, "w") as f:
        for key in metrics_dict.keys():
            f.write("%s,%s\n" % (key, metrics_dict[key]))
    return True


def text_file_to_array(input_filename):
    """
    Read a csv-like representation of an image, where each row (representing a row
    of pixels in the image) is a comma-separated list of pixel values 0 (for black)
    or 255 (for white).
    """
    input_image = np.loadtxt(input_filename, dtype="i", delimiter=",")
    return input_image


def crop_image_array(input_image, x_range, y_range):
    """
    return a new image from specified pixel range of input image
    """
    return input_image[x_range[0] : x_range[1], y_range[0] : y_range[1]]


def fill_sc_pixels(sel_pixels, orig_image, val=200):
    """
    Given an original 2D array where all the elements are 0 (background)
    or 255 (signal), fill in a selected subset of signal pixels as 123 (grey).
    """
    new_image = np.copy(orig_image)
    for pix in sel_pixels:
        new_image[pix] = 200
    return new_image


def generate_sc_images(sel_pixels, orig_image, val=200):
    """
    Return a dict of images with the selected subsets of signal
    pixels filled in in cyan.
    """
    image_dict = {}
    for k, v in sel_pixels.items():
        new_image_array = fill_sc_pixels(v, orig_image, val)
        new_image = image_from_array(new_image_array, 400)
        image_dict[k] = new_image
    return image_dict


def save_sc_images(image_dict, file_prefix):
    """
    Saves images from dictionary.
    """
    for key in image_dict:
        im = image_dict[key]
        im.save(file_prefix + "_" + str(key), "PNG")


def get_signal_pixels(input_array, threshold=255, lower_threshold=True, invert_y=False):
    """
    Find coordinates of all pixels within the image that are > or <
    the threshold ( require < threshold if lower_threshold==True)
    NOTE - if invert_y is set, we make the second coordinate negative, for reasons.
    """
    y_sign = -1 if invert_y else 1
    # find all the "white" pixels
    signal_x_y = np.where(input_array >= threshold)

    # convert ([x1, x2, ...], [y1,y2,...]) to ([x1,y1],...)
    signal_coords = [
        (signal_x_y[0][i], y_sign * signal_x_y[1][i]) for i in range(len(signal_x_y[0]))
    ]
    return signal_coords


def invert_y_coord(coord_list):
    """
    Convert [(x1,y1),(x2,y2),...] to [(x1,-y1),(x2,-y2),...]
    """
    return [(x, -1 * y) for (x, y) in coord_list]


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


def feature_vector_metrics(feature_vector, output_csv=None):
    """
    Calculate different metrics for the feature vector
    """
    feature_vec_metrics = {}

    if len(feature_vector) == 0:
        raise RuntimeError("Empty feature vector")

    # slope of the vector
    feature_vec_metrics["slope"] = (feature_vector[-1] - feature_vector[0]) / len(
        feature_vector
    )

    # difference between last and first indexes
    feature_vec_metrics["offset"] = feature_vector[-1] - feature_vector[0]

    # difference between last and middle indexes
    feature_vec_metrics["offset50"] = (
        feature_vector[-1] - feature_vector[len(feature_vector) // 2]
    )

    # mean value on the feature_vector
    feature_vec_metrics["mean"] = np.mean(feature_vector)

    # std value on the feature_vector
    feature_vec_metrics["std"] = np.std(feature_vector)

    if output_csv:
        output_csv = output_csv[:-4] + "_metrics.csv"
        # write the feature vector to a csv file
        write_dict_to_csv(feature_vec_metrics, output_csv)

    return feature_vec_metrics


def calc_adjacency_matrix(distance_matrix, include_diagonal_neighbours=False):
    """
    Return a symmetric matrix of
    (n-pixels-over-threshold)x(n-pixel-over-threshold)
    where each element ij is 0 or 1 depending on whether the distance between
    pixel i and pixel j is < or > neighbour_threshold.
    """

    # prepare empty array
    adj_matrix = np.zeros(distance_matrix.shape)
    # get 2xM array where M is the number of elements of
    # the distance matrix that satisfy the "neighbour threshold" criterion.
    # i.e. the x,y coordinates of all "neighbours" in the distance matrix.
    if not include_diagonal_neighbours:
        neighbour_x_y = np.where(distance_matrix == 1)
    else:
        neighbour_x_y = np.where((distance_matrix > 0) & (distance_matrix < 2))

    # loop over coordinates of neighbours in distance matrix
    for i in range(len(neighbour_x_y[0])):
        adj_matrix[neighbour_x_y[0][i]][neighbour_x_y[1][i]] = 1

    return adj_matrix


def calc_and_sort_sc_indices(adjacency_matrix):
    """
    Given an input adjacency matrix, calculate eigenvalues and eigenvectors,
    calculate the subgraph centrality (ref: <== ADD REF), then sort.
    """

    # get eigenvalues
    am_lambda, am_phi = np.linalg.eigh(adjacency_matrix)

    # calculate the subgraph centrality (SC)
    phi2_explambda = np.dot(am_phi * am_phi, np.exp(am_lambda))

    # order the pixels by subgraph centrality, then find their indices
    # (corresponding to their position in the 1D list of white pixels)
    indices = np.argsort(phi2_explambda)[::-1]
    return indices


def fill_feature_vector(pix_indices, coords, adj_matrix, num_quantiles=20):
    """
    Given indices and coordinates of signal pixels ordered by SC value, put them into
    quantiles and calculate an element of a feature vector for each quantile.
    by using the Euler Characteristic.

    Will return:
        selected_pixels, feature_vector
    where selected_pixels is a vector of the pixel coordinates in each quantile,
    and a feature_vector is either num-connected-components or Euler characteristic,
    for each quantile.
    """
    # adj_matrix will be square - take the length of a side
    n = max(adj_matrix.shape)

    graph = make_graph(adj_matrix)
    # set the diagonals of the adjacency matrix to 1 (previously
    # zero by definition because a pixel can't be adjacent to itself)
    for j in range(n):
        adj_matrix[j][j] = 1

    # find the different quantiles
    start = 0
    end = 100
    step = (end - start) / num_quantiles
    x = [i for i in range(start, end + 1, int(step))]
    # create feature vector of size num_quantiles
    feature_vector = np.zeros(num_quantiles)
    # create a dictionary of selected pixels for each quantile.
    selected_pixels = {}
    # Loop through the quantiles to fill the feature vector
    for i in range(1, len(feature_vector)):
        # print("calculating subregion {} of {}".format(i, num_quantiles))
        # how many pixels in this sub-region?
        n_pix = round(x[i] * n / 100)
        sub_region = pix_indices[0:n_pix]
        sel_pix = [coords[j] for j in sub_region]
        selected_pixels[x[i]] = sel_pix
        # now calculate the feature vector element using the selected method
        feature_vector[i] = calc_euler_characteristic(sub_region, graph)

    # fill in the last quantile (100%) of selected pixels
    selected_pixels[100] = coords

    return feature_vector, selected_pixels


def subgraph_centrality(
    image,
    use_diagonal_neighbours=False,
    num_quantiles=20,
    threshold=255,  # what counts as a signal pixel?
    lower_threshold=True,
    output_csv=None,
):
    """
    Go through the whole calculation, from input image to output vector of
    pixels in each SC quantile, and feature vector (either connected-components
    or Euler characteristic).
    """

    feature_vec = [
        0 for i in range(0, num_quantiles)
    ]  # need the "+1" to include 100% quantile
    sel_pixels = {}

    # making sure that there image is not empty (all black)
    if np.array(np.where(image != 0)).size != 0:

        # get the coordinates of all the signal pixels
        signal_coords = get_signal_pixels(image, threshold, lower_threshold)
        # get the distance matrix
        dist_vec, dist_matrix = calc_distance_matrix(signal_coords)

        # will use to fill our feature vector
        adj_matrix = calc_adjacency_matrix(dist_matrix, use_diagonal_neighbours)
        # calculate the subgraph centrality and order our signal pixels accordingly
        sorted_pix_indices = calc_and_sort_sc_indices(adj_matrix)
        # calculate the feature vector and get the subsets of pixels in each quantile
        feature_vec, sel_pixels = fill_feature_vector(
            sorted_pix_indices, signal_coords, adj_matrix, num_quantiles
        )
    # write the feature vector to a csv file
    if output_csv:
        write_csv(feature_vec, output_csv)

    return feature_vec, sel_pixels
