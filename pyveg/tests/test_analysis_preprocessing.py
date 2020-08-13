"""
Test the functions in analysis_preprocessing.py
"""
import os
import json

from pyveg.src.analysis_preprocessing import (
    read_results_summary,
    read_json_to_dataframes,
    read_label_mask
)


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


def test_read_label_mask():

    json_filename = os.path.join(
        os.path.dirname(__file__),
        "..",
        "testdata",
        "network_json_data",
        "image-labeller-output.json"
    )
    mask = read_label_json(json_filename)
    assert isinstance(mask, list)
    assert len(mask) > 0
    for m in mask:
        assert isinstance(m, tuple)
        assert len(m) == 2
        assert isinstance(m[0], float)
        assert isinstance(m[1], float)
