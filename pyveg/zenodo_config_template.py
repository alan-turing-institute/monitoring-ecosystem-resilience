"""
Configuration for uploading and downloading results to/from the Zenodo API.

The data for each paper will be stored in a single "deposition", to be created once,
and the deposition ID will be one of the values below.
This deposition will have some metadata, also specified below.

The deposition will eventually contain multiple zip files, each of which contains the file
"results_summary.json", which is the input to the time-series analysis, and also a directory
structure containing the png files that are the output of the time series analysis.

"""


metadata_dict = {
    "metadata": {
        "title": < title >
        "upload_type": "dataset",
        "description": < description > ,
        "creators": [
                {"name": < name > , "affiliation": < affiliation > },
        ]
    }
}

test_api_credentials = {
    "base_url": "https://sandbox.zenodo.org/api/",
    "api_token": < api_token >,
    "deposition_id": None
}

prod_api_credentials = {
    "base_url": "https://zenodo.org/api/",
    "api_token": < api_token >
    "deposition_id": None
}
