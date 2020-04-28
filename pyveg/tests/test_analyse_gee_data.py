import shutil
from pyveg.scripts.analyse_gee_data import *

def test_analyse_gee_data():

    input_dir = os.path.join(os.path.dirname(__file__), "..", "testdata", "network_json_data")
    do_spatial_plot = True
    do_time_series_plot = True

    analysis_path = os.path.join(input_dir, 'analysis')

    if os.path.exists(analysis_path):
        shutil.rmtree(analysis_path)


    analyse_gee_data(input_dir, do_spatial_plot, do_time_series_plot)


    assert (os.path.exists(analysis_path) == True )
    assert (os.path.exists(os.path.join(analysis_path, 'spatial')) == True )

    list_png_files = [f for f in os.listdir(os.path.join(input_dir, 'analysis')) if
                      (os.path.isfile(os.path.join(os.path.join(input_dir, 'analysis'), f)) and f.endswith(".png"))]

    assert (len(list_png_files) > 0 )

