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
import matplotlib.pyplot as plt
import casadi
import argparse


def image_from_array(input_array, output_size=None):
    """
    convert a 2D numpy array of values into
    an image where each pixel has r,g,b set to
    the corresponding value in the array.
    If an output size is specified, rescale to this size.
    """
    size_x, size_y = input_array.shape
    new_img = Image.new("RGB", (size_x,size_y))
    for ix in range(size_x):
        for iy in range(size_y):
            val = input_array[ix,iy]
            if val == input_array.max():
                new_img.putpixel((ix,iy),(val,val,val))
            else:
                new_img.putpixel((ix,iy),(0,val,val))
    if output_size:
        new_img = new_img.resize((output_size, output_size), Image.ANTIALIAS)
    return new_img



def read_image_file(input_filename):
    """
    Read an image and convert to a 2D numpy array, with values
    0 for background pixels and 255 for signal.
    Assume that the input image has only two colours, and take
    the one with higher sum(r,g,b) to be "signal".
    """
    im = Image.open(input_filename)
    x_size, y_size = im.size
    pix = im.load()
    sig_col = None
    bkg_col = None
    max_sum_rgb = 0
    # loop through all the pixels and find the colour with the highest sum(r,g,b)
    for ix in range(x_size):
        for iy in range(y_size):
            col = pix[ix,iy]
            if sum(col) > max_sum_rgb:
                max_rgb = sum(col)
                if sig_col:
                    bkg_col = sig_col
                sig_col = col
    # ok, we now know what sig_col is - loop through pixels again and set any that
    # match this colour to 255.
    rows = []
    for iy in range(y_size):
        row = np.zeros(x_size)
        for ix in range(x_size):
            if pix[ix,iy] == sig_col:
                row[ix] = 255
        rows.append(row)
    return rows


def read_text_file(input_filename):
    """
    Read a csv-like representation of an image, where each row (representing a row
    of pixels in the image) is a comma-separated list of pixel values 0 (for black)
    or 255 (for white).
    """
    input_image = np.loadtxt(input_filename, dtype='i', delimiter=',')
    return input_image


def crop_image(input_image, x_range, y_range):
    """
    return a new image from specified pixel range of input image
    """
    return input_image[x_range[0]:x_range[1], y_range[0]:y_range[1]]


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
    for k,v in sel_pixels.items():
        new_image_array = fill_sc_pixels(v, orig_image, val)
        new_image = image_from_array(new_image_array,400)
        image_dict[k] = new_image
    return image_dict


def get_signal_pixels(input_array, threshold=255, lower_threshold=True, invert_y=False):
    """
    Find coordinates of all pixels within the image that are > or <
    the threshold ( require < threshold if lower_threshold==True)
    NOTE - if invert_y is set, we make the second coordinate negative, for reasons.
    """
    y_sign = -1 if invert_y else 1
    # find all the "white" pixels
    signal_x_y = np.where(input_array>=threshold)

    # convert ([x1, x2, ...], [y1,y2,...]) to ([x1,y1],...)
    signal_coords = [(signal_x_y[0][i], y_sign*signal_x_y[1][i]) \
                    for i in range(len(signal_x_y[0]))]
    return signal_coords


def invert_y_coord(coord_list):
    """
    Convert [(x1,y1),(x2,y2),...] to [(x1,-y1),(x2,-y2),...]
    """
    return [(x,-1*y) for (x,y) in coord_list]


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


def calc_ec(sel_pix, pix_indices):
    """
    calculate the Euler characteristic for a selected subset of pixels.
    Takes arguments:
    sel_pix = coordinates of signal pixels in the selected sub-region.
    pix_indices = indices of all signal pixels in the original image.
    """
    try:
        sel_pix = invert_y_coord(sel_pix)
        n_pix = len(pix_indices)
        sub_dist, sub_dist_matrix = calc_distance_matrix(sel_pix)
        sub_neighbours = get_neighbour_elements(sub_dist)[0]
        # get x,y coordinates of neighbouring pixels within this subregion
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
        val = n_pix - len(sub_neighbours) + f
        print("Euler characteristic is {}".format(val))
        return val
    except Exception as e:
        print("{}".format(e))
        return 0


def calc_connected_components(sub_region, adj_matrix):
    """
    Calculate something (V-E ?) for a selected subregion.
    Takes as arguments:
    sub_region = list of signal pixels in this subregion (ordered by SC)
    adj_matrix = adjacency matrix from the whole image (we use the "W" weighted one here).
    """
    # calculate the Dulmage-Mendelsohn decomposition (ref <== REF)
    S = casadi.DM(adj_matrix[np.ix_(sub_region,sub_region)])
    # calculate the block-triangular form
    nb, rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock = S.sparsity().btf()

    p = rowperm
    q = colperm
    r = np.array(rowblock)
    s = colblock

    connected_components = len(r) - 1 # maybe 0 ?
    r2 = (r[0:(len(r) - 1)]) # used for kicking out too small component

    k = np.where((r[1:]-r2) < 1); # value 0  can be changed to other values
    connected_components = connected_components - len(k)
    return connected_components


def fill_feature_vector(pix_indices, coords, adj_matrix, do_EC=False, num_quantiles=20):
    """
    Given indices and coordinates of signal pixels ordered by SC value, put them into
    quantiles and calculate an element of a feature vector for each quantile.
    These feature vector values can be either:
    Number-of-connected-components (default), if do_EC is set to False, or:
    Euler Characteristic, if do_EC is set to True.

    Will return:
        selected_pixels, feature_vector
    where selected_pixels is a vector of the pixel coordinates in each quantile,
    and a feature_vector is either num-connected-components or Euler characteristic,
    for each quantile.
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
    x = [i for i in range(start,end+1,int(step))]
    # create feature vector of size num_quantiles
    feature_vector = np.zeros(num_quantiles);
    # create a dictionary of selected pixels for each quantile.
    selected_pixels = {}
    # Loop through the quantiles to fill the feature vector
    for i in range(1,len(feature_vector)):
        print("calculating subregion {} of {}".format(i, num_quantiles))
        # how many pixels in this sub-region?
        n_pix = round(x[i] * n / 100)
        sub_region = pix_indices[0:n_pix]
        sel_pix = [coords[j] for j in sub_region]
        selected_pixels[x[i]] = sel_pix
        # now calculate the feature vector element using the selected method
        if do_EC:
            feature_vector[i] = calc_ec(sel_pix, pix_indices)
        else:
            feature_vector[i] = calc_connected_components(sub_region, adj_matrix)
    # fill in the last quantile (100%) of selected pixels
    selected_pixels[100] = coords

    return feature_vector, selected_pixels


def find_cc_quantiles(pix_indices, adj_matrix, num_quantiles):
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

#
#def find_ec_quantiles(coords, pix_indices, num_quantiles=20):
#    # find the different quantiles
#    start = 0
#    end = 100
#    step = (end-start)/num_quantiles
#    x = [i for i in range(start,end+1,int(step))]
#    # how many signal pixels?
#    n = len(coords)
#    # create feature vector of size num_quantiles
#    feature_vector = np.zeros(num_quantiles+1)
#    selected_pixels = {}
#    # Loop through the quantiles to fill the feature vector
#    for i in range(1,len(feature_vector)):
#        print("calculating subregion {} of {}".format(i, num_quantiles))
#        # how many pixels in this sub-region?
#        n_pix = round(x[i] * n / 100)
#        sub_region = pix_indices[0:n_pix]
#        sel_pix = [coords[j] for j in sub_region]
#        try:
#            sub_dist, sub_dist_matrix = calc_distance_matrix(sel_pix)
#            sub_neighbours = get_neighbour_elements(sub_dist)[0]
#            sub_coords = get_neighbour_elements(sub_dist_matrix)
#            # ==> NOTE we don't understand the next steps!
#            nb = np.where((pix_indices[sub_coords[1]] \
#                           - pix_indices[sub_coords[0]]) == 1)[0]
#            f = 0
#            for j in range(len(nb)):
#                tmp1 = [ sel_pix[k] for k in sub_coords[0][nb] ]
#                tmp2 = np.add(np.array([ sel_pix[k] \
#                                         for k in sub_coords[0][nb]][j]),
#                              np.array([1,0]))
#                tmp3 = np.tile(tmp2, (len(nb),1))
#                tmp = tmp1 - tmp3
#                ff1 = np.where((tmp[:,0]==0) & (tmp[:,1]==0))
#                f += len(ff1[0])
#            feature_vector[i] = n_pix - len(sub_neighbours) + f
#        except:
#
#            feature_vector[i] = 0
#
#        selected_pixels[x[i]] = sel_pix
#
#    return feature_vector, selected_pixels





def subgraph_centrality(image, do_EC=False,
                        use_diagonal_neighbours=False,
                        num_quantiles=20,
                        threshold=255, # what counts as a signal pixel?
                        lower_threshold=True):
    """
    Go through the whole calculation, from input image to output vector of
    pixels in each SC quantile, and feature vector (either connected-components
    or Euler characteristic).
    """
    # flatten the input image for later use
    image_flat = image.flatten()
    # get the coordinates of all the signal pixels
    signal_coords = get_signal_pixels(image, threshold, lower_threshold)
    # get the distance matrix
    dist_vec, dist_matrix = calc_distance_matrix(signal_coords)
    T, W = calc_adjacency_matrix(dist_matrix, image, use_diagonal_neighbours)
    # Use weighted or unweighted adjacency matrix depending on which algorithm we
    # will use to fill our feature vector
    adj_matrix = T if do_EC else W
    # calculate the subgraph centrality and order our signal pixels accordingly
    sorted_pix_indices = calc_and_sort_sc_indices(adj_matrix)
    # calculate the feature vector and get the subsets of pixels in each quantile
    feature_vec, sel_pixels = fill_feature_vector(sorted_pix_indices,
                                                  signal_coords,
                                                  adj_matrix,
                                                  do_EC,
                                                  num_quantiles)
    return feature_vec, sel_pixels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Look at subgraph centrality of signal pixels in an image")
    parser.add_argument("--input_txt",help="input image as a csv file")
    parser.add_argument("--input_img",help="input image as an image file")
    parser.add_argument("--do_EC",help="calculate Euler characteristic for feature vector",
                        action='store_true')
    parser.add_argument("--use_diagonal_neighbours", help="use 8-neighbours rather than 4-neighbours",
                        action='store_true')
    parser.add_argument("--num_quantiles", help="number of elements of feature vector",
                        type=int, default=20)
    parser.add_argument("--sig_threshold", help="threshold for signal pixel",
                        type=int, default=255)
    parser.add_argument("--upper_threshold", help="threshold for signal pixel is an upper limit",
                        action='store_true')
    args = parser.parse_args()
    image_array = None
    if args.input_txt:
        image_array = read_text_file(args.input_txt)
    elif args.input_img:
        image_array = read_image_file(args.input_img)
    else:
        raise RuntimeError("Need to specify input_txt or input_img")
    do_EC = True if args.do_EC else False
    use_diagonal_neighbours = True if args.use_diagonal_neighbours else False
    num_quantiles = args.num_quantiles
    threshold = args.sig_threshold
    is_lower_limit = True if not args.upper_threshold else False
    feature_vec, sel_pixels = subgraph_centrality(image_array,
                                                  do_EC,
                                                  use_diagonal_neighbours,
                                                  num_quantiles,
                                                  threshold,
                                                  is_lower_limit)
    sc_images = generate_sc_images(sel_pixels, image_array)
