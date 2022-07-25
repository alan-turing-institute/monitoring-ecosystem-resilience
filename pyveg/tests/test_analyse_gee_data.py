"""
Test the functions in analyse_gee_data.py
"""
import json
import os
import shutil

from pyveg.scripts.analyse_gee_data import analyse_gee_data


def test_analyse_gee_data():

    input_dir = os.path.join(
        os.path.dirname(__file__), "..", "testdata", "network_json_data"
    )
    analysis_path = os.path.join(input_dir, "analysis")

    # remove old tests
    if os.path.exists(analysis_path):
        shutil.rmtree(analysis_path)

    # run script
    analyse_gee_data(input_dir, do_spatial=True)

    # assert script produced output
    assert os.path.exists(analysis_path) == True
    assert os.path.exists(os.path.join(analysis_path, "spatial")) == True

    list_png_files = [
        f
        for f in os.listdir(os.path.join(input_dir, "analysis", "time-series"))
        if (
            os.path.isfile(
                os.path.join(os.path.join(
                    input_dir, "analysis", "time-series"), f)
            )
            and f.endswith(".png")
        )
    ]

    assert len(list_png_files) > 0
