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
import requests

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
    token_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    token_filename = os.path.join(token_dir, "zenodo_test_api_token") if test \
        else os.path.join(token_dir, "zenodo_api_token")
    if not os.path.exists(token_filename):
        raise FileNotFoundError("Unable to find file containing Zenodo API token: {}".format(token_filename))
    token = open(token_filename).read().strip()
    base_url = 'https://sandbox.zenodo.org/api/' if test else 'https://zenodo.org/api/'
    return base_url, token


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



def upload_file(filename, bucket_url, test=False):
    """
    Upload a file to a deposition.

    Parameters
    ==========
    filename: str, full path to the file to be uploaded
    bucket_url: str, obtained from r["links"]["bucket"] where r is the return val from create_deposition
    test: bool, True if we will use the sandbox API, False otherwise

    Returns
    =======
    r: dict, response from the API with details about the newly uploaded file
    """
    base_url, api_token = get_base_url_and_token(test)
    with open(filename, "rb") as f:
        r = requests.put("{}/{}".format(bucket_url, os.path.basename(filename)),
                         data=f,
                         params = {'access_token': api_token})
        if r.status_code != 200:
            print("Error uploading file", r.content)
            return {}
        return r.json()


def upload_metadata(metadata_dict, deposition_id, test=False):
    base_url, api_token = get_base_url_and_token(test)
    r = requests.put("{}/deposit/{}".format(base_url, deposition_id),
                     params={"access_token": api_token},
                     json=metadata_dict)
    if r.status_code != 200:
        print("Error uploading metadata", r.content)
        return False
    return r.json()


def publish(deposition_id, test=False):
    base_url, api_token = get_base_url_and_token(test)
    r = requests.post("{}/deposit/depositions/{}/actions/publish".format(base_url, deposition_id),
                      params={"access_token": api_token})
    if r.status_code != 202:
        print("Error publishing", r.content)
        return False
    return r.json()
