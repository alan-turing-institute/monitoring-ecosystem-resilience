from subgraph_centrality import *
IMG_FILE = "../binary_image.txt"
FULL_IMG = read_text_file(IMG_FILE)
IMG = crop_image(FULL_IMG,(0,5),(0,5))
sig_pix = get_signal_pixels(IMG)
d,dsq = calc_distance_matrix(sig_pix)
T, W = calc_adjacency_matrix(dsq, IMG, False)
# use the T matrix
indices = calc_and_sort_sc_indices(T)
selected_pixels = find_ec_quantiles(sig_pix, indices, 20)
sel_pix = selected_pixels[75]



sub_dist, sub_dist_matrix = calc_distance_matrix(sel_pix)
sub_neighbours = get_neighbour_elements(sub_dist)[0]
sub_coords = get_neighbour_elements(sub_dist_matrix)
nb = np.where((indices[sub_coords[1]] \
               - indices[sub_coords[0]]) == 1)[0]
