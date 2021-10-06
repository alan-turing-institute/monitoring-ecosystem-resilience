#!/usr/bin/env python

"""
Use the Zenodo API to deposit or retrieve data.

Needs an API token - to create one:
Sign-in or create an account at https://zenodo.org
Create an API token by going to this page:
https://zenodo.org/account/settings/applications/tokens/new/
 tick "deposit:actions" and "deposit:write" in the "Scopes" section
and click Create.
Then copy the created token into a file called "zenodo_api_token" in the pyveg/configs/ directory.

OR, to use the "Sandbox" API for testing, follow the same steps but replacing "zenodo.org"
with "sandbox.zenodo.org" in the URLs, and put the token into a file named "zenodo_test_api_token"
then call the functions in this module with the "test" argument set to True.
"""
import os
import shutil
import json
import re
import requests
import tempfile
from zipfile import ZipFile, BadZipFile

from pyveg.src.file_utils import get_filepath_after_directory
from pyveg.src.coordinate_utils import find_coords_string
ZENODO_CONFIG_FOUND=False
try:
    import pyveg.zenodo_config as config
    ZENODO_CONFIG_FOUND=True
except:
    pass


def get_base_url_and_token(test=False):
    """
    Get the base URL for the API, and the API token, for use in requests.

    Parameters
    ==========
    test: bool, True if we will use the sandbox API, False otherwise

    Returns
    =======
    base_url: str, the first part of the URL for the API
    api_token: str, the personal access token, read from a file.
    """
    if not ZENODO_CONFIG_FOUND:
        raise RuntimeError("File zenodo_config.py not found - will not be able to access the Zenodo API")
    if test:
        base_url = config.test_api_credentials["base_url"]
        token = config.test_api_credentials["api_token"]
    else:
        base_url = config.prod_api_credentials["base_url"]
        token = config.prod_api_credentials["api_token"]
    return base_url, token


def get_deposition_id(file_type="json", test=False):
    """
    If we have previously created a deposition, we hopefully stored its ID in
    the zenodo_config.py file.
    """
    if test:
        credentials = config.test_api_credentials

    else:
        credentials =  config.prod_api_credentials
    if file_type == "json":
        return credentials["deposition_id_summary_json"]
    elif file_type == "csv":
        return credentials["deposition_id_ts_csv"]
    elif file_type == "images":
        return credentials["deposition_id_images"]


def list_depositions(test=False):
    """
    List all the depositions created by this account.

    Parameters
    ==========
    test: bool, True if we will use the sandbox API, False otherwise

    Returns
    =======
    r: list of dicts, response from the API with info about the depositions
    """
    base_url, api_token = get_base_url_and_token(test)
    r = requests.get('{}/deposit/depositions'.format(base_url),
                     params={'access_token': api_token})
    if r.status_code != 200:
        print("Error retrieving depositions", r.content)
        return False
    return r.json()


def create_deposition(test=False):
    """
    Create a new, empty deposition.

    Parameters
    ==========
    test: bool, True if we will use the sandbox API, False otherwise

    Returns
    =======
    r: dict, response from the API with info about the newly created deposition
    """
    headers = {"Content-Type": "application/json"}
    base_url, api_token = get_base_url_and_token(test)
    params = {'access_token': api_token}
    r = requests.post('{}/deposit/depositions'.format(base_url),
                   params=params,
                   json={},
                   headers=headers)
    if r.status_code != 201:
        print("Error creating deposition", r.content)
        return None
    return r.json()


def get_deposition_info(deposition_id, test=False):
    """
    Get the JSON object containing details of a deposition.

    Parameters
    ==========
    deposition_id: int, ID of the deposition.
    test: bool, if True use the sandbox API, if False will use the real one.

    Returns
    =======
    dep_info: dict, information about the deposition
    """
    headers = {"Content-Type": "application/json"}
    base_url, api_token = get_base_url_and_token(test)
    params = {'access_token': api_token}
    r = requests.get('{}/deposit/depositions/{}'.format(base_url, deposition_id),
                   params=params,
                   json={},
                   headers=headers)
    if r.status_code != 200:
        print("Error getting deposition", r.content)
        return {}
    return r.json()


def get_bucket_url(deposition_id, test=False):
    """
    For a given deposition_id, find the URL needed to upload a file.

    Parameters
    ==========
    deposition_id: int, ID of the deposition.
    test: bool, if True use the sandbox API, if False will use the real one.

    Returns
    =======
    bucket_url: str, the URL of the bucket for this deposition, or empty string if id not found
    """
    dep_info = get_deposition_info(deposition_id, test)
    if not dep_info:
        print("Deposition {} not found".format(deposition_id))
        return ""
    return dep_info["links"]["bucket"]


def upload_file(filename, deposition_id, test=False):
    """
    Upload a file to a deposition.

    Parameters
    ==========
    filename: str, full path to the file to be uploaded
    deposition_id: int, ID of the deposition to which we want to upload.
    test: bool, True if we will use the sandbox API, False otherwise

    Returns
    =======
    uploaded_ok: bool, True if we get status code 200 from the API
    """
    base_url, api_token = get_base_url_and_token(test)
    bucket_url = get_bucket_url(deposition_id, test)

    with open(filename, "rb") as f:
        r = requests.put("{}/{}".format(bucket_url, os.path.basename(filename)),
                         data=f,
                         params = {'access_token': api_token})
        if r.status_code != 200:
            print("Error uploading file", r.content)
            return False
        return True


def list_files(deposition_id, file_type="json", test=False):
    """
    List all the files in a deposition.

    Parameters
    ==========
    deposition_id: int, ID of the deposition on which to list files
    file_type: str, 'json', 'csv', or 'images'.
                 if 'json', list the deposition containing the results_summary.json
                 if 'csv' list the one containing ts_summary_stats.csv
                 if 'images' list the tarfiles of images.
    test: bool, True if using the sandbox API, False otherwise

    Returns
    =======
    files: list[str], list of all filenames in the deposition.
    """
    if file_type not in ['csv', 'json', 'images']:
        raise RuntimeError("File type must be 'csv', 'json', or 'images'")
    base_url, api_token = get_base_url_and_token(test)
    deposition_id = get_deposition_id(file_type, test=test)
    r = requests.get("{}/deposit/depositions/{}/files".format(base_url, deposition_id),
                     params={"access_token": api_token})
    if r.status_code != 200:
        print("Error getting file list for deposition {}".format(deposition_id))
    return [f["filename"] for f in r.json()]


def download_file(filename, deposition_id, destination_path=".", test=False):
    """
    Upload a file to a deposition.

    Parameters
    ==========
    filename: str, full path to the file to be uploaded
    deposition_id: int, ID of the deposition containing this file
    destination_path: str, where to put the downloaded file
    test: bool, True if we will use the sandbox API, False otherwise

    Returns
    =======
    filepath: str, location of downloaded file.
    """
    base_url, api_token = get_base_url_and_token(test)
    bucket_url = get_bucket_url(deposition_id, test)
    r = requests.get("{}/{}".format(bucket_url, os.path.basename(filename)),
                 params = {'access_token': api_token})
    if r.status_code != 200:
        print("Error downloading file", r.content)
        return {}
    os.makedirs(destination_path, exist_ok=True)
    destination = os.path.join(destination_path, filename)
    with open(destination, "wb") as output:
        output.write(r.content)
    return destination


def delete_file(filename, deposition_id, test=False):
    """
    Delete a file from a deposition.

    Parameters
    ==========
    filename: str, full path to the file to be deleted
    deposition_id: int, ID of the deposition containing this file
    test: bool, True if we will use the sandbox API, False otherwise

    Returns
    =======
    True if file was deleted OK, False otherwise.
    """
    base_url, api_token = get_base_url_and_token(test)
    r = requests.delete("{}/deposit/depositions/{}/files/{}".format(base_url, deposition_id, filename),
                        params = {'access_token': api_token})

    if r.status_code != 204:
        print("Error deleting file", r.content)
        return False
    return True


def upload_standard_metadata(deposition_id, json_or_csv="json", test=False):
    """
    Upload the metadata dict defined in zenodo_config.py to the
    specified deposition ID.Kcontaining metadata with the format:

    Parameters:
    ==========
    deposition_id: int, ID of the deposition to which to upload
    json_or_csv: str, can be either 'json' to upload the metadata for `results_summary.json`
                   or `csv` to upload the metadata for `ts_summary_stats.csv`
    test: if True, use the sandbox API, if False use the production one.

    Returns
    =======
    r: dict, JSON response from the API.
    """
    if json_or_csv == "json":
        metadata_dict = config.metadata_dict_summary_json
    else:
        metadata_dict = config.metadata_dict_ts_csv
    base_url, api_token = get_base_url_and_token(test)
    r = requests.put("{}/deposit/depositions/{}".format(base_url, deposition_id),
                     params={"access_token": api_token},
                     json=metadata_dict)
    if r.status_code != 200:
        print("Error uploading metadata", r.content)
        return False
    return r.json()


def upload_custom_metadata(title, upload_type, description, creators, deposition_id, test=False):
    """
    Upload a dict to the deposition containing metadata with the format:


    {
       'metadata': {
         'title': 'My first upload',
         'upload_type': 'poster',
         'description': 'This is my first upload',
         'creators': [{'name': 'Doe, John',
                       'affiliation': 'Zenodo'}]
       }
    }

    Parameters:
    ==========
    title: str, title of the deposition
    upload_type: str, type of upload, typically "dataset"
    description: str, description of the deposition
    creators: dict, format {"name": <str:name>, "affiliation": <str:affiliation>}

    Returns
    =======
    r: dict, JSON response from the API.
    """
    metadata_dict = {
        "metadata": {
            "title": title,
            "upload_type": upload_type,
            "description": description,
            "creators": creators
            }
    }
    base_url, api_token = get_base_url_and_token(test)
    r = requests.put("{}/deposit/depositions/{}".format(base_url, deposition_id),
                     params={"access_token": api_token},
                     json=metadata_dict)
    if r.status_code != 200:
        print("Error uploading metadata", r.content)
        return False
    return r.json()


def publish_deposition(deposition_id, test=False):
    """
    Submit the deposition, so it will be findable on Zenodo and have a DOI.
    """
    base_url, api_token = get_base_url_and_token(test)
    r = requests.post("{}/deposit/depositions/{}/actions/publish".format(base_url, deposition_id),
                      params={"access_token": api_token})
    if r.status_code != 202:
        print("Error publishing", r.content)
        return False
    return r.json()


def unlock_deposition(deposition_id, test=False):
    """
    Unlock a previously submitted deposition, so we can add to it.
    """
    base_url, api_token = get_base_url_and_token(test)
    r = requests.post("{}/deposit/depositions/{}/actions/edit".format(base_url, deposition_id),
                      params={"access_token": api_token})
    if r.status_code != 201:
        print("Error unlocking", r.content)
        return False
    return r.json()


def prepare_results_zipfile(collection_name,
                            png_location,
                            png_location_type="local",
                            json_location=None,
                            json_location_type="local"):
    """
    Create a zipfile called <results_long_lat_collection> containing the 'results_summary.json',
    and the outputs of the analysis.

    Parameters
    ==========
    collection_name: str, typically "Sentinel2" or "Landsat8" or similar
    base_png_location: str, directory containing analysis/ subdirectory
    png_location_type: str, either "local" or "azure"
    base_json_location: str, directory containing "results_summary.json.
                        If not specified, assume same as base_png_location
    json_location_type: str, either "local" or "azure"

    Returns
    =======
    zip_filename: str, location of the produced zipfile
    """
    tmpdir = tempfile.mkdtemp()
    zip_filename = os.path.join(tmpdir,"results_")
    if find_coords_string(png_location):
        zip_filename += find_coords_string(png_location) + "_"
    zip_filename += collection_name + ".zip"
    zf = ZipFile(zip_filename, mode="w")
    if not json_location:
        # assume json and png are in the same directory
        json_location = png_location
    if json_location_type == "local":
        if not os.path.isdir(json_location):
            raise RuntimeError("{} is not a directory".format(json_location))
        dir_contents = os.listdir(json_location)
        if not os.path.exists(os.path.join(json_location, "results_summary.json")):
            raise RuntimeError("Could not find results_summary.json in {}".format(json_location))
        zf.write(os.path.join(json_location, "results_summary.json"),
                 arcname="results_summary.json")
    if png_location_type == "local":
        if not os.path.exists(os.path.join(png_location, "analysis")):
            raise RuntimeError("Could not find analysis dir in {}".format(png_location))
        for root, dirnames, filenames in os.walk(os.path.join(png_location, "analysis")):
            for filename in filenames:
                full_filepath = os.path.join(root, filename)
                short_filepath = get_filepath_after_directory(full_filepath, "analysis")
                zf.write(full_filepath, arcname=short_filepath)
        zf.close()
    return zip_filename

def get_results_summary_json(coords_string, collection, deposition_id, test=False):
    """
    Assuming the zipfile is named following the convention
    results_<long>_<lat>_<collection>.zip
    download this from the deposition, and extract the
    results_summary.json.
    """
    zip_filename = "results_{}_{}.zip".format(coords_string, collection)
    if not zip_filename in list_files(deposition_id, test):
        print("Unable to find file {} in deposition {}".format(zip_filename, deposition_id))
        return None

    data = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_zipfile = download_file(zip_filename, deposition_id, tmpdir, test)
        zf = ZipFile(tmp_zipfile)
        try:
            data = zf.read("results_summary.json")
        except KeyError:
            print("results_summary.json not found in {}".format(zip_filename))
            return {}
    return json.loads(data)



def download_results_by_coord_id(coords_id, json_or_csv="json", destination_path=None, deposition_id=None, test=False):
    """
    Search the deposition (defined by the deposition_id in zenodo_config.py)
    for results_summary json or summary_stats csv files beginning with 'coord_id'
    and download the most recent one.

    Parameters
    ==========
    coords_id: str, two-digit string identifiying the row of the location in coordinates.py
    json_or_csv: str, if "json", download 'results_summary.json', otherwise download 'ts_summary_stats.csv'.
    destination_path: str, directory to download to.  If not given, put in temporary dir
    deposition_id: str, deposition ID in Zenodo.  If not given, use the one from zenodo_config.py
    test: bool, if True, use the sandbox Zenodo repository
    """
    # coords_id should be two digits, e.g. '00'
    if not re.search('[\d]{2}', coords_id):
        raise RuntimeError("coords_id should be a 2-digit string")
    if not deposition_id:
        deposition_id = get_deposition_id(json_or_csv, test=test)
    if not destination_path:
        destination_path = tempfile.TemporaryDirectory().name
    elif not os.path.exists(destination_path):
        os.makedirs(destination_path)
    # list the files in the deposition
    file_list = [f for f in list_files(deposition_id, test=test) \
                 if f.startswith(coords_id) and "results_summary" in f]
    if len(file_list)==0:
        print("No files for coords_id {} found.".format(coords_id))
        return ""
    # files should follow the same naming convention, and have the date at the end.
    # this means they should be sort-able.  Find the most recent:
    file_list.sort()
    latest_file = file_list[-1]

    # download this
    destination = download_file(latest_file, deposition_id, destination_path, test)
    return destination
