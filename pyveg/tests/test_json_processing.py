"""
Test the functions in process_network_metrics.py
"""

from pyveg.src.process_network_metrics import *

def test_process_json_metrics_to_dataframe():

    dir_path = os.path.join(os.path.dirname(__file__),"..","testdata","network_json_data")

    test_df = process_json_metrics_to_dataframe(dir_path)

    print (test_df.shape)
    assert (test_df.shape[0]==8)