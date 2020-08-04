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
    upload_file,
    create_deposition,
    upload_standard_metadata
)


def create_new_deposition(use_sandbox=False):
    """
    Create a new deposition, and populate it with the metadata from zenodo_config.py.

    Parameters
    ==========
    use_sandbox: bool, if True use the "sandbox" API rather than the production one

    Returns
    =======
    deposition_id: int, the ID of the newly created deposition.  Recommended that this
               is then copied into zenodo_config.py
    """
    section_str = "test_api_credentials" if use_sandbox else "prod_api_credentials"
    deposition_json = create_deposition(use_sandbox)
    deposition_id = deposition_json["id"]
    metadata_response = upload_standard_metadata(deposition_id, use_sandbox)
    print("Created new deposition with deposition_id={}.  We recommend that you now copy this deposition_id into the {} section of pyveg/zenodo_config.py".format(deposition_id, section_str))
    return deposition_id


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
    parser.add_argument("--create_deposition", help="create a new deposition", action="store_true")
    parser.add_argument("--input_png_loc",help="path to analysis/ subdirectory")
    parser.add_argument("--png_loc_type",help="'local' or 'azure'", default="local")
    parser.add_argument("--input_json_loc",help="path to dir containing results_summary.json")
    parser.add_argument("--json_loc_type",help="'local' or 'azure'", default="local")
    parser.add_argument("--collection", help="Collection name, e.g. 'Sentinel2'")
    parser.add_argument("--test_api", help="use the test API", action="store_true")
    args = parser.parse_args()
    if args.create_deposition:
        repo_string = "sandbox" if args.test_api else "production"
        confirm = input("""
Are you sure you want to create a new deposition on the {} repository?
We normally do this just once per project.  Please confirm (y/n)
""".format(repo_string))
        if confirm in ["y", "Y", "yes", "Yes"]:
            deposition_id = create_new_deposition(args.test_api)
            print("Created new deposition in {} repository with deposition_id={}."\
                  .format(repo_string, deposition_id))
            return
        else:
            print("Confirmation not received - exiting with no action")
            return
    if not args.input_png_loc and args.collection:
        raise RuntimeError("--input_png_loc and --collection are required arguments for uploading data")
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
