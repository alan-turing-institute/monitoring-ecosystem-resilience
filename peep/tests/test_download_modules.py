import os
import shutil
import unittest

import pytest

if os.name == "posix":
    TMPDIR = "/tmp/"
else:
    TMPDIR = "%TMP%"

if not os.environ.get("CI") == "true":
    from peep.src.download_modules import ImageDownloader, WeatherDownloader


@unittest.skipIf(
    os.environ.get("CI") == "true",
    "Skipping this test in a Continuous Integration environment.",
)
def test_instantiate_vegetation_downloader():
    veg_downloader = ImageDownloader("Sentinel2")
    assert isinstance(veg_downloader, ImageDownloader)


@unittest.skipIf(
    os.environ.get("CI") == "true",
    "Skipping this test in a Continuous Integration environment.",
)
def test_veg_downloader_unconfigured_exception():
    veg_downloader = ImageDownloader("Sentinel2")
    with pytest.raises(RuntimeError):
        veg_downloader.run()


@unittest.skipIf(
    os.environ.get("CI") == "true",
    "Skipping this test in a Continuous Integration environment.",
)
def test_veg_downloader_variables_not_set_exception():
    veg_downloader = ImageDownloader("Sentinel2")
    with pytest.raises(RuntimeError):
        veg_downloader.configure()


@unittest.skipIf(
    os.environ.get("CI") == "true",
    "Skipping this test in a Continuous Integration environment.",
)
def test_veg_downloader_configure_sentinel2():
    veg_downloader = ImageDownloader("Sentinel2")
    veg_downloader.data_type = "vegetation"
    veg_downloader.collection_name = "COPERNICUS/S2"
    veg_downloader.RGB_bands = ["B4", "B3", "B2"]
    veg_downloader.NIR_band = "B8"
    veg_downloader.bounds = [532480.0, 174080.0, 542720.0, 184320.0]
    veg_downloader.date_range = ["2017-03-03", "2018-12-01"]
    veg_downloader.time_per_point = "1m"
    veg_downloader.do_network_centrality = True
    veg_downloader.cloudy_pix_frac = 50
    veg_downloader.cloudy_pix_flag = "CLOUDY_PIXEL_PERCENTAGE"
    veg_downloader.mask_cloud = True
    veg_downloader.configure()
    assert veg_downloader.is_configured == True


@unittest.skipIf(
    os.environ.get("CI") == "true",
    "Skipping this test in a Continuous Integration environment.",
)
def test_veg_downloader_run_sentinel2():
    veg_downloader = ImageDownloader("Sentinel2")
    veg_downloader.data_type = "vegetation"
    veg_downloader.collection_name = "COPERNICUS/S2"
    veg_downloader.RGB_bands = ["B4", "B3", "B2"]
    veg_downloader.NIR_band = "B8"
    veg_downloader.bounds = [532480.0, 174080.0, 542720.0, 184320.0]
    veg_downloader.date_range = ["2017-01-01", "2017-02-01"]
    veg_downloader.time_per_point = "1m"
    veg_downloader.do_network_centrality = True
    veg_downloader.cloudy_pix_frac = 50
    veg_downloader.cloudy_pix_flag = "CLOUDY_PIXEL_PERCENTAGE"
    veg_downloader.mask_cloud = True
    veg_downloader.ndvi = True
    veg_downloader.count = False
    veg_downloader.output_location = os.path.join(TMPDIR, "testveg")
    veg_downloader.configure()
    veg_downloader.run()
    tif_dir = os.path.join(TMPDIR, "testveg", "2017-01-01_2017-02-01", "RAW")
    assert os.path.exists(tif_dir)
    tif_files = [
        filename for filename in os.listdir(tif_dir) if filename.endswith(".tif")
    ]
    assert len(tif_files) == 4  # B2, B3, B4, NDVI
    shutil.rmtree(tif_dir, ignore_errors=True)


@unittest.skipIf(
    os.environ.get("CI") == "true",
    "Skipping this test in a Continuous Integration environment.",
)
def test_instantiate_weather_downloader():
    weather_downloader = WeatherDownloader("ERA5")
    assert isinstance(weather_downloader, WeatherDownloader)


@unittest.skipIf(
    os.environ.get("CI") == "true",
    "Skipping this test in a Continuous Integration environment.",
)
def test_weather_downloader_unconfigured_exception():
    weather_downloader = WeatherDownloader("ERA5")
    with pytest.raises(RuntimeError):
        weather_downloader.run()


@unittest.skipIf(
    os.environ.get("CI") == "true",
    "Skipping this test in a Continuous Integration environment.",
)
def test_weather_downloader_variables_not_set_exception():
    weather_downloader = WeatherDownloader("ERA5")
    with pytest.raises(RuntimeError):
        weather_downloader.configure()


@unittest.skipIf(
    os.environ.get("CI") == "true",
    "Skipping this test in a Continuous Integration environment.",
)
def test_weather_downloader_configure_era5():
    weather_downloader = WeatherDownloader("ERA5")
    weather_downloader.data_type = "weather"
    weather_downloader.collection_name = "ECMWF/ERA5/MONTHLY"
    weather_downloader.precipitation_band = ["total_precipitation"]
    weather_downloader.temperature_band = ["mean_2m_air_temperature"]
    weather_downloader.bounds = [532480.0, 174080.0, 542720.0, 184320.0]
    weather_downloader.date_range = ["2017-03-03", "2018-12-01"]
    weather_downloader.time_per_point = "1m"
    weather_downloader.configure()
    assert weather_downloader.is_configured == True


@unittest.skipIf(
    os.environ.get("CI") == "true",
    "Skipping this test in a Continuous Integration environment.",
)
def test_weather_downloader_run_era5():
    weather_downloader = WeatherDownloader("ERA5")
    weather_downloader.data_type = "weather"
    weather_downloader.collection_name = "ECMWF/ERA5/MONTHLY"
    weather_downloader.precipitation_band = ["total_precipitation"]
    weather_downloader.temperature_band = ["mean_2m_air_temperature"]
    weather_downloader.bounds = [532480.0, 174080.0, 542720.0, 184320.0]
    weather_downloader.date_range = ["2017-01-01", "2017-02-01"]
    weather_downloader.time_per_point = "1m"
    weather_downloader.output_location = os.path.join(TMPDIR, "testweather")
    weather_downloader.configure()
    weather_downloader.run()
    tif_dir = os.path.join(TMPDIR, "testweather", "2017-01-01_2017-02-01", "RAW")
    assert os.path.exists(tif_dir)
    tif_files = [
        filename for filename in os.listdir(tif_dir) if filename.endswith(".tif")
    ]
    assert len(tif_files) == 2  # temp, precipitation
    shutil.rmtree(tif_dir, ignore_errors=True)
