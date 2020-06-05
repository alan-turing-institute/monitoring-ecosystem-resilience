"""
Collection of utility functions for manipulating coordinates
and their string representations.,
"""

import re

def get_region_string(coords, region_size):
    """
    Given a set of (long,lat) coordinates, and the size
    of a square region in long,lat space, return
    a string in the format expected by GEE.

    Parameters
    ==========
    coords: list of floats, [longitude,latitude]
    region_size: float,  size of each side of the region, in degrees

    Returns
    =======
    region_string: str, string representation of list of four coordinates,
                   representing four corners of the region.
    """
    left = coords[0] - region_size/2
    right = coords[0] + region_size/2
    top = coords[1] + region_size/2
    bottom = coords[1] - region_size/2
    region_string =  str([[left,top],[right,top],[right,bottom],[left,bottom]])
    return region_string


def find_coords_string(file_path):
    """
    Parse a file path using a regular expresion to find a substring
    that looks like a set of coordinates,
    and return that.
    """

    match = re.search("([-]?[\d]{1,3}\.[\d]{1,3}[_,][-]?[\d]{1,3}\.[\d]{1,3})",file_path)
    if not match:
        return None
    coords_string = match.groups()[0]
    return coords_string


def get_sub_image_coords(coords, region_size, x_parts, y_parts):
    """
    If an image is divided into sub_images, return a list of coordinates
    for all the sub-images.

    Parameters
    ==========
    coords: list of floats, [long,lat]
    region_size: float, size of square image in degrees long,loat
    x_parts: int, number of sub-images in x-direction
    y_parts: int, number of sub-images in y-direction

    Returns
    =======
    sub_image_coords: list, of lists of floats [[long,lat],...]
    """
    sub_image_coords = []
    if coords and region_size:
        left_start = coords[0] - region_size/2
        top_start = coords[1] + region_size/2
        sub_image_size_x = region_size / x_parts
        sub_image_size_y = region_size / y_parts
        for ix in range(x_parts):
            for iy in range(y_parts):
                sub_image_coords.append(
                    (left_start + sub_image_size_x/2 + (ix*sub_image_size_x),
                     top_start - sub_image_size_y/2 - (iy*sub_image_size_y))
                )
    return sub_image_coords
