"""
Collection of utility functions for manipulating coordinates
and their string representations.,
"""

import re


def find_coords_string(file_path):
    """
    Parse a file path using a regular expresion to find a substring
    that looks like a set of coordinates,
    and return that.
    """

    match = re.search(
        "([-]?[\d]{1,3}\.[\d]{1,3}[_,][-]?[\d]{1,3}\.[\d]{1,3})", file_path
    )
    if not match:
        return None
    coords_string = match.groups()[0]
    return coords_string


def get_sub_image_coords(bounds, x_parts, y_parts):
    """
    If an image is divided into sub_images, return a list of coordinates
    for all the sub-images. Coordinates will be defined as the bottom left corner of each image.

    Parameters
    ==========
    bounds: list with coordinates, e.g.  [left, bottom, right, top]
    x_parts: int, number of sub-images in x-direction
    y_parts: int, number of sub-images in y-direction

    Returns
    =======
    sub_image_coords: list, of lists of floats [[long,lat],...]
    """
    sub_image_coords = []
    if bounds:
        left_start = bounds[0]
        top_start = bounds[3]
        sub_image_size_x = (bounds[2] - bounds[0]) / x_parts
        sub_image_size_y = (bounds[3] - bounds[1]) / y_parts
        for ix in range(x_parts):
            for iy in range(y_parts):
                sub_image_coords.append(
                    (
                        left_start + (ix * sub_image_size_x),
                        top_start - sub_image_size_y - (iy * sub_image_size_y),
                    )
                )
    return sub_image_coords
