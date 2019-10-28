"""
Test the functions in subgraph_centrality.py
"""

import os
import numpy as np
from subgraph_centrality import *

IMG_FILE = "../binary_image.txt"
FULL_IMG = read_file(IMG_FILE)
IMG = crop_image(FULL_IMG,(0,5),(0,5))


def test_load_image():
    img = read_file(IMG_FILE)
    assert(isinstance(img, np.ndarray))
    assert(img.shape == (100,100))


def test_crop_image():
    img = read_file(IMG_FILE)
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


        #
def test_calc_adjacency_matrix_with_diagonals():
    sig_pix = get_signal_pixels(IMG)
    d,dsq = calc_distance_matrix(sig_pix)
    T, W = calc_adjacency_matrix(dsq, IMG,True)
    assert(isinstance(T, np.ndarray))
    assert(isinstance(W, np.ndarray))
    assert(T.shape==dsq.shape)
    assert(W.shape==dsq.shape)
    # check that all diagonals are zero - pixels
    # can't be their own neighbours
    for i in range(T.shape[0]):
        assert(T[i,i] == 0)
        assert(W[i,i] == 0)
    # check that we have some non-zero elements
    assert(T.sum() == 24)
    assert(W.sum() == 20.04)


def test_calc_adjacency_matrix_no_diagonals():
    sig_pix = get_signal_pixels(IMG)
    d,dsq = calc_distance_matrix(sig_pix)
    T, W = calc_adjacency_matrix(dsq, IMG,False)
    assert(isinstance(T, np.ndarray))
    assert(isinstance(W, np.ndarray))
    assert(T.shape==dsq.shape)
    assert(W.shape==dsq.shape)
    # check that all diagonals are zero - pixels
    # can't be their own neighbours
    for i in range(T.shape[0]):
        assert(T[i,i] == 0)
        assert(W[i,i] == 0)
    # check that we have some non-zero elements
    assert(T.sum() == 16)
    assert(W.sum() == 14.02)


def test_calc_and_sort_indices():
    sig_pix = get_signal_pixels(IMG)
    d,dsq = calc_distance_matrix(sig_pix)
    T, W = calc_adjacency_matrix(dsq, IMG, False)
    indices = calc_and_sort_sc_indices(W)
    assert(len(indices)==9)
    assert(indices[0]==1)


def test_find_ec_quantiles():
    sig_pix = get_signal_pixels(IMG)
    d,dsq = calc_distance_matrix(sig_pix)
    T, W = calc_adjacency_matrix(dsq, IMG, False)
    # use the T matrix
    indices = calc_and_sort_sc_indices(T)
    selected_pixels = find_ec_quantiles(sig_pix, indices, 20)
    assert(len(selected_pixels)==20)
    assert(len(selected_pixels[100]) == len(indices))


def test_get_neighbour_elements():
    test_vec = np.array([0.,1.,2.,3.,np.sqrt(2.)])
    av_with_diagonals = get_neighbour_elements(test_vec, True)[0]
    assert(len(av_with_diagonals)==2)
    av_no_diagonals = get_neighbour_elements(test_vec, False)[0]
    assert(len(av_no_diagonals)==1)
