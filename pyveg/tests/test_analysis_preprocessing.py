"""
Test the functions in analysis_preprocessing.py
"""
import json
import os

from pyveg.src.analysis_preprocessing import (read_json_to_dataframes,
                                              read_results_summary)


def test_read_json_to_dataframes():

    json_filename = os.path.join(
        os.path.dirname(__file__),
        "..",
        "testdata",
        "network_json_data",
        "test-results-summary.json",
    )
    summary_json = json.load(open(json_filename))
    test_df_dict = read_json_to_dataframes(summary_json)

    dict_len = len(test_df_dict.keys())
    test_df = test_df_dict["COPERNICUS/S2"]
    assert test_df.shape == (120, 5)
    assert dict_len == 2
