# Uploading results to the Zenodo open source data repository

Zenodo ([https://zenodo.org]) is a free and open source repository for research data, hosted by CERN.

Data is organized into `depositions`, each of which has a Digital Object Identifier (DOI) which can then be cited.

For the purposes of this package, we make the assumption that we will keep all the data for a single paper in one deposition.

## Prerequisites

In order to use the functions in this package to automatically upload data to Zenodo, and to download specific files to rerun analysis, you will need:
* Sign up for a Zenodo account by clicking the "Sign up" button on the top right of [https://zenodo.org/]
* If you want to use the "sandbox" repository for testing functionality (recommended!) you'll also need to sign up separately here: [https://sandbox.zenodo.org/]
* For both the production and sandbox versions, once you are signed in, go to [https://zenodo.org/account/settings/applications/tokens/new/] to create an API token.  Write any name for your token in the box, and tick the boxes for "deposit:actions" and "deposit:write" before clicking the "Create" button.  Keep this tab open until you have copied the token into `zenodo_config.py` (see below).

## How to fill `zenodo_config.py`

In the `pyveg/` directory there is a file `zenodo_config_template.py`.  Copy this to `zenodo_config.py` and fill in the various fields:
* The `metadata_dict` is the metadata that will be stored with your deposition.  Put the title and description of your study here, and "upload_type" as "dataset".  List the authors, giving names and affiliations as you would like them to appear on Zenodo.
* For the "test_api_credentials", if you plan to use the "sandbox" repository for testing, and if you have signed up for this and created an API token as described in the section above, copy/paste the personal access token to the "api_token" field here.
* Similarly for the "prod_api_credentials" do the same, but with the main Zenodo site.
* Leave the "deposition_id" as None for now - we will create a deposition in the next step.

## Create a deposition to hold our data

Once we have filled in the "api_token" in `zenodo_config.py` we can do:
```
pip install .
```
then you can run the following command to create a new deposition:
```
pyveg_zenodo_upload --create_deposition [--test_api]
```
where the final `--test_api` argument should be included if you want to use the sandbox repository, or omitted to use the production one.
The output from this command should give you the deposition_id that you can then paste into the appropriate section of `zenodo_config.py`, and then do
```
pip install .
```
once more.

## Uploading analysis results to Zenodo

Once all the necessary fields (the "api_token" and "deposition_id") are present in `zenodo_config.py`, then uploading analysis results, plus the "results_summary.json" file (the output of the image downloading and processing that is the input to the analysis) should be straightforward:
* When running ```pyveg_gee_analysis``` you can add the argument ```--upload_to_zenodo``` to upload the results to the deposition on the production Zenodo repository, or ```--upload_to_zenodo_sandbox``` to use the sandbox repository instead.
* Alternatively, if you have previously run ```pyveg_gee_analysis``` you can run the command:
```
pyveg_zenodo_upload --input_png_loc <path-to-analysis-subdir> --input_json_loc <path-to-dir-containing-results_summary.json> --json_loc_type <'local' or 'azure'> --collection <collection-name>
```
Here, we assume that the png files from running the analysis are in a local directory, while the "results_summary.json" can be either in a local directory (in which case specify this directory as the ```--input_json_loc``` argument and specify ```--json_loc_type local```), or on Azure (in which case use the blob storage container as the ```--input_json_loc``` argument and specify ```--json_loc_type azure```)
