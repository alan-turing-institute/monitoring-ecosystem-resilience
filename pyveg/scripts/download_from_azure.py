#!/usr/bin/env python

"""
Download files from Azure blob storage that we will want for further analysis, or
for putting into presentations.
i.e.
* the 'results_summary.json'
* Full-sized RGB images
* (optionally) the RGB sub-images.
* (optionally) NDVI and/or BWNDVI full-sized or sub-images.

All go into a zip file.

This script relies on the blob structure being like the one produced from
running pyveg_run_pipeline.

"""

import os
import tempfile
import argparse
import subprocess

from azure.storage.blob import BlockBlobService

from pyveg.src.azure_utils import list_directory, retrieve_blob
from pyveg.azure_config import config


def get_summary_json(container, json_dir):
    """
    Parameters
    ==========
    container: str, the container name
    json_dir: str, temporary directory into which to put json file.
    """

    print("Getting summary JSON file  to {}".format(json_dir))
    blob_dirs = list_directory(container, container)
    json_blob_dir = None
    for b in blob_dirs:
        if b.endswith("combine"):
            json_blob_dir = b
    json_blob_file = list_directory(json_blob_dir, container)[0]
    retrieve_blob(os.path.join(json_blob_dir, json_blob_file), container, json_dir)


def get_rgb(container, rgb_dir):
    """
    Parameters
    ==========
    container: str, the container name
    rgb_dir: str, temporary directory into which to put image files.
    """
    print("Getting RGB images to {}".format(rgb_dir))
    bbs = BlockBlobService(
        account_name=config["storage_account_name"],
        account_key=config["storage_account_key"],
    )
    blob_names = bbs.list_blob_names(container)
    rgb_names = [b for b in blob_names if "PROCESSED" in b and b.endswith("RGB.png")]
    print("Found {} images".format(len(rgb_names)))
    for blob in rgb_names:
        retrieve_blob(blob, container, rgb_dir)


def create_zip_archive(temp_dir, output_zipname):
    """
    Parameters
    ==========
    temp_dir: str, tempdir that has JSON and RGB subdirs
    output_zip: str, path to output zipfile.

    """
    subprocess.run(["zip", "-r", os.path.basename(output_zipname), "."], cwd=temp_dir)
    subprocess.run(
        ["cp", os.path.join(temp_dir, os.path.basename(output_zipname)), output_zipname]
    )


def main():
    parser = argparse.ArgumentParser(
        description="download from blob storage to zipfile"
    )
    parser.add_argument(
        "--container", help="name of blob storage container", required=True
    )
    parser.add_argument(
        "--output_zipfile", help="name of zipfile to write to", default="./results.zip"
    )
    args = parser.parse_args()

    td = tempfile.mkdtemp()
    print("Created temp dir {}", format(td))

    json_dir = os.path.join(td, "JSON")
    os.makedirs(json_dir)
    rgb_dir = os.path.join(td, "RGB")
    os.makedirs(rgb_dir)

    get_summary_json(args.container, json_dir)

    get_rgb(args.container, rgb_dir)
    current_dir = os.getcwd()
    create_zip_archive(td, args.output_zipfile)


if __name__ == "__main__":
    main()
