import os
import argparse

from pyveg.src.azure_utils import download_images_from_container
from pyveg.src.zenodo_utils import upload_file, config


def process_container(container_name,
                      deposition_id,
                     types=["PROCESSED/RGB",
                            "PROCESSED/NDVI",
                            "PROCESSED/BWNDVI"
                     ],
                     test=False):
    tarfiles = download_images_from_container(container_name,
                                              types)
    for tf in tarfiles:
        upload_file(tf, deposition_id, test)


def main():
    parser = argparse.ArgumentParser("upload image files to Zenodo")
    parser.add_argument("--inputfile", type=str, help="text file containing list of containers")
    parser.add_argument("--test", help="Use the sandbox repository", action="store_true")
    parser.add_argument("--types", help="What types of file? comma-separated list", default="PROCESSED/RGB,PROCESSED/NDVI,SPLIT/BWNDVI")
    parser.add_argument("--start_index", help="Where to start in the list", type=int, default=0)
    parser.add_argument("--end_index", help="Where to end in the list", type=int, default=None)
    args = parser.parse_args()
    test = args.test if args.test else False
    if test:
        dep_id = config.test_api_credentials["deposition_id_images"]
    else:
        dep_id = config.prod_api_credentials["deposition_id_images"]
    types = args.types.split(",")
    containers = open(args.inputfile).readlines()
    containers = [c.strip() for c in containers]
    containers = containers[args.start_index:args.end_index]
    for container in containers:
        print("Processing container {}".format(container))
        process_container(container, dep_id, types, test)


if __name__ == "__main__":
    main()
