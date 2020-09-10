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

from pyveg.src.azure_utils import download_summary_json, download_rgb
from pyveg.azure_config import config


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

    download_summary_json(args.container, json_dir)

    download_rgb(args.container, rgb_dir)
    current_dir = os.getcwd()
    create_zip_archive(td, args.output_zipfile)


if __name__ == "__main__":
    main()
