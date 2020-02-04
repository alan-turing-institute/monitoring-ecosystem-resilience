"""
Test the functions in subgraph_centrality.py
"""

import os
import numpy as np
from pyveg.src.image_utils import *
import igraph


def test_image_all_white():
    img = Image.open(os.path.join(os.path.dirname(__file__),"..","..","testdata","white.png"))
    assert(image_all_same_colour(img,(255,255,255)))
    assert(not image_all_same_colour(img,(0,0,0)))


def test_image_all_black():
    img = Image.open(os.path.join(os.path.dirname(__file__),"..","..","testdata","black.png"))
    assert(image_all_same_colour(img,(0,0,0)))
    assert(not image_all_same_colour(img,(255,255,255)))


def test_image_not_all_same():

    img = Image.open(os.path.join(os.path.dirname(__file__),"..","..","testdata","black_and_white_diagonal.png"))
    assert(not image_all_same_colour(img,(0,0,0)))
    assert(not image_all_same_colour(img,(255,255,255)))


def test_compare_same_image():

    img1 = Image.open(os.path.join(os.path.dirname(__file__),"..","..","testdata","black_and_white_diagonal.png"))
    img2 = Image.open(os.path.join(os.path.dirname(__file__),"..","..","testdata","black_and_white_diagonal.png"))
    assert(compare_binary_images(img1,img2) == 1.)


def test_compare_opposite_images():
    img1 = Image.open(os.path.join(os.path.dirname(__file__),"..","..","testdata","black_and_white_diagonal.png"))
    img2 = Image.open(os.path.join(os.path.dirname(__file__),"..","..","testdata","black_and_white_diagonal_2.png"))
    assert(compare_binary_images(img1,img2) < 0.1)

def test_create_gif_from_images():
    path_dir = os.path.join(os.path.dirname(__file__), "..","..", "testdata/")
    create_gif_from_images(path_dir,'test','black_and_white')
    list_png_files = [f for f in os.listdir(path_dir) if (os.path.isfile(os.path.join(path_dir, f)) and f=="test.gif")]
    assert(len(list_png_files) == 1)
