"""
Test the functions in subgraph_centrality.py
"""

import os
import numpy as np
from subgraph_centrality import *

IMG_FILE = "../binary_image.txt"
FULL_IMG = read_text_file(IMG_FILE)
IMG = crop_image(FULL_IMG,(0,5),(0,5))


def test_load_image():
    img = read_text_file(IMG_FILE)
    assert(isinstance(img, np.ndarray))
    assert(img.shape == (100,100))


def test_crop_image():
    img = read_text_file(IMG_FILE)
    new_img = crop_image(img,(0,5),(0,5))
    assert(isinstance(new_img, np.ndarray))
    assert(new_img.shape ==(5,5))


def test_get_signal_pixels():
    sig_pix = get_signal_pixels(IMG)
    assert(len(sig_pix)==9)
    for i in sig_pix:
        assert(len(i)==2)


def test_calc_distance_matrix():
    sig_pix = get_signal_pixels(IMG)
    d,dsq = calc_distance_matrix(sig_pix)
    expected_len = (len(sig_pix)*(len(sig_pix)-1))/2
    assert(len(d)== expected_len)
    assert(isinstance(dsq, np.ndarray))
    assert(dsq.shape==(len(sig_pix), len(sig_pix)))


def test_calc_adjacency_matrix_with_diagonals():
    sig_pix = get_signal_pixels(IMG)
    d,dsq = calc_distance_matrix(sig_pix)
    adj_matrix = calc_adjacency_matrix(dsq, weighted=False,
                                       include_diagonal_neighbours=True)
    assert(isinstance(adj_matrix, np.ndarray))
    assert(adj_matrix.shape==dsq.shape)
    # check that all diagonals are zero - pixels
    # can't be their own neighbours
    for i in range(adj_matrix.shape[0]):
        assert(adj_matrix[i,i] == 0)
    # check that we have some non-zero elements
    assert(adj_matrix.sum() == 24)


def test_calc_adjacency_matrix_no_diagonals():
    sig_pix = get_signal_pixels(IMG)
    d,dsq = calc_distance_matrix(sig_pix)
    adj_matrix = calc_adjacency_matrix(dsq, weighted=False)
    assert(isinstance(adj_matrix, np.ndarray))
    assert(adj_matrix.shape==dsq.shape)
    # check that all diagonals are zero - pixels
    # can't be their own neighbours
    for i in range(adj_matrix.shape[0]):
        assert(adj_matrix[i,i] == 0)
    # check that we have some non-zero elements
    assert(adj_matrix.sum() == 16)


def test_calc_and_sort_indices():
    sig_pix = get_signal_pixels(IMG)
    d,dsq = calc_distance_matrix(sig_pix)
    adj_matrix = calc_adjacency_matrix(dsq, False)
    indices = calc_and_sort_sc_indices(adj_matrix)
    assert(len(indices)==9)
    assert(indices[0]==1)


def test_calc_connected_components():
    sig_pix = get_signal_pixels(IMG)
    d,dsq = calc_distance_matrix(sig_pix)
    adj_matrix = calc_adjacency_matrix(dsq, weighted=False)
    indices = calc_and_sort_sc_indices(adj_matrix)
    cc = calc_connected_components(indices, adj_matrix)
    assert(cc == 0)


def test_get_neighbour_elements():
    test_vec = np.array([0.,1.,2.,3.,np.sqrt(2.)])
    av_with_diagonals = get_neighbour_elements(test_vec, True)[0]
    assert(len(av_with_diagonals)==2)
    av_no_diagonals = get_neighbour_elements(test_vec, False)[0]
    assert(len(av_no_diagonals)==1)


def test_calc_ec():
    sig_pix = get_signal_pixels(IMG)
    d,dsq = calc_distance_matrix(sig_pix)
    adj_matrix = calc_adjacency_matrix(dsq, False)
    # use the T matrix
    indices = calc_and_sort_sc_indices(adj_matrix)
    # look at the top half of ordered list
    sub_region = indices[0: len(indices)//2]
    sel_pix = [sig_pix[j] for j in sub_region]
    ec = calc_ec(sel_pix, sub_region)
    assert(ec==1)


def test_fill_feature_vector_connected_components():
    sig_pix = get_signal_pixels(IMG)
    d,dsq = calc_distance_matrix(sig_pix)
    adj_matrix = calc_adjacency_matrix(dsq, True, IMG, False)
    indices = calc_and_sort_sc_indices(adj_matrix)
    feature_vec, sel_pix = fill_feature_vector(indices,
                                               sig_pix,
                                               adj_matrix)
    assert(len(feature_vec)==20)
    assert(len(sel_pix)==20)
    pass


def test_fill_feature_vector_EC():
    sig_pix = get_signal_pixels(IMG)
    d,dsq = calc_distance_matrix(sig_pix)
    adj_matrix = calc_adjacency_matrix(dsq)
    indices = calc_and_sort_sc_indices(adj_matrix)
    feature_vec, sel_pix = fill_feature_vector(indices,
                                               sig_pix,
                                               adj_matrix,
                                               True
    )
    assert(len(feature_vec)==20)
    assert(len(sel_pix)==20)
    pass


def test_full_calculation_connected_components():
    feat_vec, sel_pix = subgraph_centrality(IMG)
    assert(len(feat_vec)==20)
    assert(len(sel_pix)==20)


def test_full_calculation_EC():
    feat_vec, sel_pix = subgraph_centrality(IMG, True)
    assert(len(feat_vec)==20)
    assert(len(sel_pix)==20)


def test_fill_sc_pixels():
    sig_pix = get_signal_pixels(IMG)
    new_img = fill_sc_pixels(sig_pix, IMG)
    assert((IMG==255).sum() == (new_img==200).sum())
