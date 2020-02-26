from pyveg.src.satellite_data_analysis import get_vegetation, get_weather

coordinates = (27.99,11.29)
date_range = ('2016-01-01', '2017-01-01')
num_days_per_point = 30

data_collections = {
    'Copernicus' : {
        'collection_name': 'COPERNICUS/S2',
        'type': 'vegetation',
        'RGB_bands': ('B4','B3','B2'),
        'NIR_band': 'B8',
        'cloudy_pix_flag': 'CLOUDY_PIXEL_PERCENTAGE'
    },
    'Landsat' : {
        'collection_name': 'LANDSAT/LC08/C01/T1_SR',
        'type': 'vegetation',
        'RGB_bands': ('B4','B3','B2'),
        'NIR_band': 'B5',
        'cloudy_pix_flag': 'CLOUD_COVER'
    },
    'NOAA' : {
        'collection_name': 'NOAA/PERSIANN-CDR',
        'type': 'weather',
        'precipitation_band': ['precipitation']
    },
    'NASA' : {
        'collection_name': 'NASA/GPM_L3/IMERG_V06',
        'type': 'weather',
        'precipitation_band': ['precipitationCal'],
        'temperature_band': ['probabilityLiquidPrecipitation']

    },
    'unsupported' : {
        'collection_name': "ECMWF/ERA5/DAILY",
        'type': 'weather',
        'precipitation_band': ['total_precipitation'],
        'temperature_band': ['mean_2m_air_temperature']
    }
}


def test_get_vegetation():
    result = get_vegetation(data_collections['Copernicus'], coordinates, date_range)
    assert(isinstance(result, float))


def test_get_rainfall():

    result = get_weather('test',data_collections['NOAA'], coordinates, date_range)

    assert (len(result.items())!=0)

