"""
Test the functions in data_analysis_utils.py
"""

import os
import shutil
import json

from pyveg.src.data_analysis_utils import *
from pyveg.src.analysis_preprocessing import *


def test_coarse_dataframe():

    json_filename = os.path.join(
        os.path.dirname(__file__),
        "..",
        "testdata",
        "network_json_data/test-results-summary.json",
    )
    results_dict = json.load(open(json_filename))
    test_df = read_json_to_dataframes(results_dict)

    data_df = convert_to_geopandas(test_df["COPERNICUS/S2"])

    data_df = coarse_dataframe(data_df, 11)

    n_blocks = len(np.unique([i for i in data_df["category"]])) / len(
        np.unique(data_df["date"])
    )

    assert n_blocks == 2.0


def test_create_lat_long_metric_figures():

    dir_path = os.path.join(
        os.path.dirname(__file__), "..", "testdata", "network_json_data/"
    )

    json_filename = os.path.join(
        os.path.dirname(__file__),
        "..",
        "testdata",
        "network_json_data/test-results-summary.json",
    )
    summary_json = json.load(open(json_filename))
    test_df = read_json_to_dataframes(summary_json)

    data_df = convert_to_geopandas(test_df["COPERNICUS/S2"])

    for filename in os.listdir(dir_path):
        if filename.endswith(".png"):
            os.unlink(os.path.join(dir_path, filename))
    tmp_png_path = os.path.join(dir_path, "tmp_png")
    create_lat_long_metric_figures(data_df, "offset50", tmp_png_path)

    list_png_files = [
        f
        for f in os.listdir(tmp_png_path)
        if (os.path.isfile(os.path.join(tmp_png_path, f)) and f.endswith(".png"))
    ]
    len_dates = len(np.unique(data_df["date"]))

    assert len(list_png_files) == len_dates
    # now delete the test png files
    shutil.rmtree(tmp_png_path, ignore_errors=True)


def test_moving_window_analysis():

    path_to_dict = os.path.join(
        os.path.dirname(__file__),
        "..",
        "testdata",
        "network_json_data/results_summary.json",
    )
    results_json = json.load(open(path_to_dict))
    dfs = read_json_to_dataframes(results_json)
    time_series_dfs = make_time_series(dfs.copy())

    ar1_var_df = moving_window_analysis(
        time_series_dfs, os.path.dirname(path_to_dict), 0.5
    )

    keys_ar1 = list(ar1_var_df.keys())
    assert ar1_var_df.shape == (38, 9)
