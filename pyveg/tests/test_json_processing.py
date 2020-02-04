"""
Test the functions in process_network_metrics.py
"""

from pyveg.src.process_network_metrics import *



def test_process_json_metrics_to_dataframe():
    test_df = process_json_metrics_to_dataframe(os.path.join(os.path.dirname(__file__),"..","..","testdata","network_json_data/"))
    assert (test_df.shape[0]==5)
    assert (test_df.shape[1]>=8)


def test_create_network_figures():
    dir_path = os.path.join(os.path.dirname(__file__),"..","..","testdata","network_json_data/")
    test_df = process_json_metrics_to_dataframe(dir_path)
    create_network_figures(test_df, 'offset50', dir_path, 'test')

    list_png_files = [f for f in os.listdir(dir_path) if (isfile(join(dir_path, f)) and f.endswith(".png"))]
    list_json_files = [f for f in os.listdir(dir_path) if (isfile(join(dir_path, f)) and f.endswith(".json"))]
    assert (len(list_png_files)==len(list_json_files))

