import datetime
import json
import os
import re
import subprocess
from zipfile import BadZipFile, ZipFile

import dateparser
import requests

from .date_utils import find_mid_period

if os.name == "posix":
    TMPDIR = "/tmp/"
else:
    TMPDIR = "%TMP%"

LOGFILE = os.path.join(TMPDIR, "failed_downloads.log")


def split_filepath(path):
    allparts = []
    if path.endswith("/") or path.endswith("\\"):
        path = path[:-1]
    while True:
        parts = os.path.split(path)
        if parts[0] == path:  # for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def get_filepath_after_directory(path, dirname, include_dirname=False):
    """
    Return part of a filepath from a certain point onwards.
    e.g. if we have path /a/b/c/d/e/f  and we say dirname=c,
    then this will return d/e/f if include_dirname==False,
    or c/d/e/f if it is True.

    Parameters
    ==========
    path: str, full filepath
    dirname: str, delimeter, from where we will take the remaining filepath
    include_dirname: bool, if True, the returned path will have dirname as its root.
    """
    path_parts = split_filepath(path)
    output_parts = []
    dirname_found = False
    for part in path_parts:
        if part == dirname:
            dirname_found=True
            if include_dirname:
                output_parts.append(part)
        else:
            if dirname_found:
                output_parts.append(part)
    return os.path.join(*output_parts)


def download_and_unzip(url, output_tmpdir):
    """
    Given a URL from GEE, download it (will be a zipfile) to
    a temporary directory, then extract archive to that same dir.
    Then find the base filename of the resulting .tif files (there
    should be one-file-per-band) and return that.

    Parameters
    ==========
    url: str, URL of zipfile on GEE server.
    output_tmpdir: str, full path of directory into which to unpack zipfile.

    Returns
    =======
    tif_filenames: list of strings, the full paths to unpacked tif files.
    """

    # GET the URL
    r = requests.get(url)
    if not r.status_code == 200:
        raise RuntimeError(" HTTP Error {} getting download link {}".format(r.status_code,
                                                                            url))
    os.makedirs(output_tmpdir, exist_ok=True)
    output_zipfile = os.path.join(output_tmpdir, "gee.zip")
    with open(output_zipfile, "wb") as outfile:
        outfile.write(r.content)
    ## catch zipfile-related exceptions here, and if they arise,
    ## write the name of the zipfile and the url to a logfile
    try:
        with ZipFile(output_zipfile, "r") as zip_obj:
            zip_obj.extractall(path=output_tmpdir)
    except (BadZipFile):
        with open(LOGFILE, "a") as logfile:
            logfile.write(
                "{}: {} {}\n".format(str(datetime.now()), output_zipfile, url)
            )
        return None
    tif_files = [
        filename for filename in os.listdir(output_tmpdir) if filename.endswith(".tif")
    ]
    if len(tif_files) == 0:
        raise RuntimeError("No files extracted")

    # get the filename before the "Bx" band identifier
    tif_filebases = [tif_file.split(".")[0] for tif_file in tif_files]

    # get the unique list
    tif_filebases = set(tif_filebases)

    # prepend the directory name to each of the filebases
    tif_filenames = [
        os.path.join(output_tmpdir, tif_filebase) for tif_filebase in tif_filebases
    ]
    return tif_filenames


def save_json(out_dict, output_dir, output_filename, verbose=False):
    """
    Given a dictionary, save
    to requested filename -
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w") as fp:
        json.dump(out_dict, fp, indent=2)
    if verbose:
        print("Saved json file '{}'".format(output_path))


def save_image(image, output_dir, output_filename, verbose=False):
    """
    Given a PIL.Image (list of pixel values), save
    to requested filename - note that the file extension
    will determine the output file type, can be .png, .tif,
    probably others...
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    image.save(output_path)
    if verbose:
        print("Saved image '{}'".format(output_path))


def construct_image_savepath(
    output_dir, collection_name, coords, date_range, image_type
):
    """
    Function to abstract output image filename construction. Current approach is to create
    a new dir inside `output_dir` for the satellite, and then save date and coordinate
    stamped images in this dir.
    """

    # get the mid point of the date range
    mid_period_string = find_mid_period(date_range[0], date_range[1])

    # filename is the date, coordinates, and image type
    filename = f"{mid_period_string}_{coords[0]}-{coords[1]}_{image_type}.png"

    # full path is dir + filename
    full_path = os.path.join(output_dir, filename)

    return full_path


def consolidate_json_to_list(json_dir, output_dir=None, output_filename=None):
    """
    Load all the json files (e.g. from individual sub-images), and return
    a list of dictionaries, to be written out into one json file.

    Parameters
    ==========
    json_dir: str, full path to directory containing temporary json files
    output_dir: str, full path to desired output directory.
                     Can be None, in which case no output written to disk.
    output_filename: str, name of the output json file.
                     Can be None, in which case no output written to disk.

    Returns
    =======
    results: list of dicts.
    """
    results = []

    # if input dir doesn't exist, return
    if not os.path.exists(json_dir):
        print("No sub-images processed!")
        return results

    for filename in os.listdir(json_dir):
        results.append(json.load(open(os.path.join(json_dir, filename))))
    if output_dir and output_filename:
        save_json(results, output_dir, output_filename)
    return results


def get_tag():
    """
    Get the git tag currently checked out.
    """
    p=subprocess.Popen(["git","describe","--tags"],stdout=subprocess.PIPE)
    tag = p.communicate()[0].decode("utf-8").strip()
    return tag


def construct_filename_from_metadata(metadata, suffix):
    """
    Given a dictionary of metadata, construct a filename.
    Will be used for the results summary json, and the summary stats csv
    as they are uploaded to Zenodo.
    """
    if "coords_id" in metadata.keys():
        filename = metadata["coords_id"]
    else:
        filename = "coords"
    filename += "_{}N_{}E_{}_freq-{}".format(metadata["latitude"], metadata["longitude"],
                                        metadata["collection"], metadata["time_per_point"])
    if "region_size" in metadata.keys():
        filename += "region-{}".format(region_size)
    if "tag" in metadata.keys():
        filename += "_{}".format(metadata["tag"])
    filename += "_{}".format(suffix)
    filename = filename.replace("/","")
    return filename
