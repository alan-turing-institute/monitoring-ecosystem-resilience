import pytest
from pyveg.src.download_modules import VegetationDownloader, WeatherDownloader


def test_instantiate_vegetation_downloader():
    veg_downloader = VegetationDownloader("Sentinel2")
    assert isinstance(veg_downloader, VegetationDownloader)


def test_veg_downloader_unconfigured_exception():
    veg_downloader = VegetationDownloader("Sentinel2")
    with pytest.raises(RuntimeError):
        veg_downloader.run()

def test_veg_downloader_variables_not_set_exception():
    veg_downloader = VegetationDownloader("Sentinel2")
    with pytest.raises(RuntimeError):
        veg_downloader.configure()


def test_veg_downloader_configure():
    veg_downloader = VegetationDownloader("Sentinel2")
    veg_downloader.data_type="vegetation"
    veg_downloader.collection_name = "COPERNICUS/S2"
    veg_downloader.RGB_bands = ["B4","B3","B2"]
    veg_downloader.NIR_band = "B8"
    veg_downloader.coords = (11.45,27.5)
    veg_downloader.date_range = ["2017-03-03","2018-12-01"]
    veg_downloader.time_per_point = "1m"
    veg_downloader.do_network_centrality = True
    veg_downloader.cloudy_pix_frac = 50
    veg_downloader.cloudy_pix_flag = "CLOUDY_PIXEL_PERCENTAGE"
    veg_downloader.mask_cloud = True
    veg_downloader.configure()
    assert veg_downloader.is_configured == True


def test_instantiate_weather_downloader():
    weather_downloader = WeatherDownloader("ERA5")
    assert isinstance(weather_downloader, WeatherDownloader)
