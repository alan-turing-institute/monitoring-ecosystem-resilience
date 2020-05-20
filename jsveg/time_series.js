// GEE script to get time series for precipitation and NDVI at a single point.
// Usage: copy and paste the script into the Google Earth Engine Editor:
// https://code.earthengine.google.com/

// specify dates
var date_1 = '2000-01-01';
var date_2 = '2020-01-01';

// specify precipitation data source
var precip_collection = "ECMWF/ERA5/MONTHLY";
var precip_bandname = "total_precipitation";

// specify NDVI data source
var vi_collection = "LANDSAT/LE07/C01/T1_8DAY_EVI"; // or "LANDSAT/LE07/C01/T1_8DAY_NDVI"
var vi_bandname = "EVI"; // or "NDVI"


// fetch prcipitation time series
var precipitation = ee.ImageCollection(precip_collection)
                        .select(precip_bandname)
                        .filter(ee.Filter.date(date_1, date_2));
                        
// vi time series
var vi = ee.ImageCollection(vi_collection)
                        .select(vi_bandname)
                        .filter(ee.Filter.date(date_1, date_2));

// Define a region of interest as a buffer around a point.
var geom = geometry.buffer(1e4); // default unit of the buffer is meters

// create and print charts
print(ui.Chart.image.series(precipitation, geom, ee.Reducer.mean(), 1e4));
print(ui.Chart.image.series(vi, geom, ee.Reducer.mean(), 1e4));
    