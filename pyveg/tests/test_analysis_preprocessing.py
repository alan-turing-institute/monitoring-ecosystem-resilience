"""
Test the functions in analysis_preprocessing.py
"""

from pyveg.src.analysis_preprocessing import *


def test_read_json_to_dataframes():
    test_df_dict = read_json_to_dataframes(
        os.path.join(os.path.dirname(__file__), "..", "testdata", "network_json_data/test-results-summary.json"))

    dict_len = len(test_df_dict.keys())
    test_df = test_df_dict['COPERNICUS/S2']
    assert (test_df.shape == (120, 5))
    assert (dict_len == 2)