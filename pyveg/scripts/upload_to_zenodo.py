"""
Upload the results_summary.json, and the outputs of the time-series analysis
to the Zenodo open source platform for data www.zenodo.org.

Will create a zipfile, with a name based upon the coordinates and
satellite collection, and upload it to a "deposition" via the zenodo API.

In addition to the main production zenodo API, there is also a "sandbox"
for testing.  Use the '--test_api' argument to use this.
"""

import os
import argparse

from pyveg.src.zenodo_utils import (
    get_deposition_id,
    prepare_results_zipfile,
    upload_file
)


def upload_results(collection,
                   png_location,
                   png_location_type,
                   json_location,
                   json_location_type,
                   use_test_api):
    deposition_id = get_deposition_id(use_test_api)
    zipfile = prepare_results_zipfile(collection,
                                      png_location,
                                      png_location_type,
                                      json_location,
                                      json_location_type )
    uploaded_ok = upload_file(zipfile, deposition_id, use_test_api)
    return uploaded_ok


def main():
    parser = argparse.ArgumentParser(description="upload to zenodo")
    parser.add_argument("--input_png_loc",help="path to analysis/ subdirectory", required=True)
    parser.add_argument("--png_loc_type",help="'local' or 'azure'", default="local")
    parser.add_argument("--input_json_loc",help="path to dir containing results_summary.json")
    parser.add_argument("--json_loc_type",help="'local' or 'azure'", default="local")
    parser.add_argument("--collection", help="Collection name, e.g. 'Sentinel2'",required=True)
    parser.add_argument("--test_api", help="use the test API", action="store_true")
    args = parser.parse_args()
    png_location = args.input_png_loc
    png_location_type = args.png_loc_type
    if args.input_json_loc:
        json_location = args.input_json_loc
    else:
        json_location = png_location
    json_loc_type = args.json_loc_type
    collection = args.collection,
    test_api = True if args.test_api else False
    result = upload_results(collection,
                            png_location,
                            png_location_type,
                            json_location,
                            json_location_type,
                            test_api)
    if result:
        print("Uploaded OK")
    else:
        print("Problem uploading")

if __name__ == "__main__":
    main()
