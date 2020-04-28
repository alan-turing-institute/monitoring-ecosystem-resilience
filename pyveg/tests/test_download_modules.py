import pytest
from pyveg.src.download_modules import VegetationDownloader, WeatherDownloader


def test_instantiate_vegetation_downloader():

    veg_downloader = VegetationDownloader("Sentinel2")
    assert isinstance(veg_downloader, VegetationDownloader)
