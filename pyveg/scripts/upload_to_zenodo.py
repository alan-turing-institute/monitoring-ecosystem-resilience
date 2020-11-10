"""
Upload the results_summary.json or the outputs of the time-series analysis
to the Zenodo open source platform for data www.zenodo.org.

Will create a zipfile, with a name based upon the coordinates and
satellite collection, and upload it to a "deposition" via the zenodo API.

In addition to the main production zenodo API, there is also a "sandbox"
for testing.  Use the '--test_api' argument to use this.
"""

import os
import argparse
import tempfile
import json

from pyveg.src.file_utils import get_tag, construct_filename_from_metadata
from pyveg.src.analysis_preprocessing import read_results_summary
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


def upload_results_summary(json_location,
                           json_location_type,
                           use_test_api):
    """
    Upload the results summary json from running pyveg pipeline to download and process data from GEE.
    """
    deposition_id = get_deposition_id("json", test=use_test_api)

    # read in the json
    results_summary = read_results_summary(json_location, input_location_type=json_location_type)
    if (not results_summary) or (not "metadata" in results_summary.keys()):
        print("Unable to find metadata in json file in {}".format(json_location))
        return False
    filename = construct_filename_from_metadata(results_summary["metadata"],"results_summary.json")
    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, filename)
    with open(filepath, "w") as outfile:
        json.dump(results_summary, outfile)
    print("Uploading {} to Zenodo".format(filename))
    uploaded_ok = upload_file(filepath, deposition_id, use_test_api)
    return uploaded_ok


def upload_summary_stats(csv_filepath, use_test_api):
    """
    Typically called by the analyse_gee_data script, upload the
    results summary csv file.
    """
    deposition_id = get_deposition_id("csv", use_test_api)
    uploaded_ok = upload_file(csv_filepath, deposition_id, use_test_api)
    return uploaded_ok


def main():
    parser = argparse.ArgumentParser(description="upload to zenodo")
    parser.add_argument("--create_deposition", help="create a new deposition", action="store_true")
    parser.add_argument("--input_location",help="directory or container with file of interest",required=True)
    parser.add_argument("--input_location_type",help="'local' or 'azure'", default="local")
    parser.add_argument("--test_api", help="use the test API", action="store_true")
    parser.add_argument("--summary_csv", help="upload the summary stats csv rather than the results_summary.json", action="store_true")
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

    test_api = True if args.test_api else False
    if args.summary_csv:
        result = upload_summary_stats(args.input_location, test_api)
    else:
        result = upload_results_summary(args.input_location,
                                        args.input_location_type,
                                        test_api)
    if result:
        print("Uploaded OK")
    else:
        print("Problem uploading")

if __name__ == "__main__":
    main()
