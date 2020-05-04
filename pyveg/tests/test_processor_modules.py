"""
Tests for the modules that process the images downloaded from GEE
"""

import os
import pytest

import shutil

from pyveg.src.processor_modules import (
    VegetationImageProcessor,
    NetworkCentralityCalculator,
    WeatherImageToJSON
)


def test_Sentinel2_image_processor():
    """
    Should combine tif files into RGB, NDVI, and BWNDVI
    big images, and split RGB and BWNVI into sub-images.
    """
    dir_path = os.path.join(os.path.dirname(__file__), "..", "testdata", "test_tif")
    tmp_png_path = os.path.join(os.path.dirname(__file__), "..", "testdata", "tmp_png")

    vip = VegetationImageProcessor()
    vip.input_dir = dir_path
    vip.output_dir = tmp_png_path
    vip.coords = [11.58,27.95]
    vip.configure()
    vip.run()
    assert os.path.exists(os.path.join(tmp_png_path, "2018-03-01", "PROCESSED"))
