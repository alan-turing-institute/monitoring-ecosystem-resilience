import os
import datetime
import dateparser
import json
import re


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



def save_json(out_dict, output_dir, output_filename):
    """
    Given a dictionary, save
    to requested filename -
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as fp:
        json.dump(out_dict, fp, indent=2)

    print("Saved json file '{}'".format(output_path))


def construct_image_savepath(output_dir, collection_name, coords, date_range, image_type):
    """
    Function to abstract output image filename construction. Current approach is to create
    a new dir inside `output_dir` for the satellite, and then save date and coordinate
    stamped images in this dir.
    """

    # get the mid point of the date range
    mid_period_string = find_mid_period(date_range[0], date_range[1])

    # filename is the date, coordinates, and image type
    filename = f'{mid_period_string}_{coords[0]}-{coords[1]}_{image_type}.png'

    # full path is dir + filename
    full_path = os.path.join(output_dir, filename)

    return full_path


def consolidate_json_to_list(json_dir, output_dir, output_filename="network_centralities.json"):
    """
    Load all the json files (e.g. from individual sub-images), and return
    a list of dictionaries, to be written out into one json file.

    Parameters
    ==========
    json_dir: str, full path to directory containing temporary json files
    output_dir: str, full path to desired output directory
    output_filename: str, name of the output json file.
    """
    results = []

    for filename in os.listdir(json_dir):
        results.append(json.load(open(os.path.join(json_dir,filename))))
    save_json(results, output_dir, output_filename)
    return results
