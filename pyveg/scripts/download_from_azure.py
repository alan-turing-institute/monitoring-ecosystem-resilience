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

import argparse
import os
import subprocess
import tempfile

from azure.storage.blob import BlockBlobService

from pyveg.azure_config import config
from pyveg.src.azure_utils import download_rgb, download_summary_json


def create_zip_archive(temp_dir, output_zipname, json_dir=None, rgb_dir=None):
    """
    Parameters
    ==========
    temp_dir: str, tempdir that has JSON and RGB subdirs
    output_zip: str, path to output zipfile.

    """
    if json_dir:
        subprocess.run(["cp", "-r", json_dir, temp_dir])
    if rgb_dir:
        subprocess.run(["cp", "-r", rgb_dir, temp_dir])
    subprocess.run(
        ["zip", "-r", os.path.basename(output_zipname), "."], cwd=temp_dir)
    subprocess.run(
        ["cp", os.path.join(temp_dir, os.path.basename(
            output_zipname)), output_zipname]
    )


def main():
    parser = argparse.ArgumentParser(
        description="download from blob storage to zipfile"
    )
    parser.add_argument(
        "--container", help="name of blob storage container", required=True
    )
    parser.add_argument(
        "--output_dir", help="name of output_directory", required=True
    )
    parser.add_argument(
        "--summary_json", help="download results_summary.json?", action='store_true'
    )
    parser.add_argument(
        "--rgb", help="download RGB images?", action='store_true'
    )
    parser.add_argument(
        "--output_zipfile", help="name of zipfile to write to"
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = tempfile.mkdtemp()
        print("Created temp dir {}", format(output_dir))

    json_dir = os.path.join(output_dir, "JSON")
    os.makedirs(json_dir)
    rgb_dir = os.path.join(output_dir, "RGB")
    os.makedirs(rgb_dir)

    if args.summary_json:
        download_summary_json(args.container, json_dir)
    if args.rgb:
        download_rgb(args.container, rgb_dir)
    if args.output_zipfile:
        create_zip_archive(output_dir, args.output_zipfile, json_dir, rgb_dir)


if __name__ == "__main__":
    main()
