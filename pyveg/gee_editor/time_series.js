// GEE script to get time series for precipitation and NDVI at a single point.
// Usage: copy and paste the script into the Google Earth Engine Editor:
// https://code.earthengine.google.com/

// overwrite selection from map if you already have a point in mind
//var long = 12.76;
//var lat = -18.05;
//geometry = ee.Geometry.Point([long, lat]);
//Map.setCenter(long, lat, 17);

// specify dates
var date_1 = '2000-01-01';
var date_2 = '2020-01-01';

// specify precipitation data source
var precip_collection = "ECMWF/ERA5/MONTHLY";
var precip_bandname = "total_precipitation";

// specify NDVI data source
//var veg_collection = "LANDSAT/LE07/C01/T1_8DAY_EVI"; 
//var veg_collection = "LANDSAT/LE07/C01/T1_8DAY_NDVI"
//var veg_collection = "LANDSAT/LT05/C01/T1_32DAY_NDVI"
var veg_collection = "MODIS/006/MOD13Q1"; // MODIS seems to be cleaner
var veg_bandname = "NDVI"; // or "EVI"

// fetch prcipitation time series
var precipitation = ee.ImageCollection(precip_collection)
                        .select(precip_bandname)
                        .filter(ee.Filter.date(date_1, date_2));
                        
// veg time series
var veg = ee.ImageCollection(veg_collection)
                        .select(veg_bandname)
                        .filter(ee.Filter.date(date_1, date_2));
                        
/*
// L7 scan line masking as per https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6544617/
var veg = veg.map(function(img) {
                   var mask = img.select(veg_bandname).lt(0.9)
                                 .and(img.select(veg_bandname).gt(-0.9))
                   return img.updateMask(mask)
                   })
*/

// Define a region of interest as a buffer around a point.
var region_size = 1e3; // units are meters
var geom = geometry.buffer(region_size);

// create and print charts
print('Precipitation')
print(ui.Chart.image.series(precipitation, geom, ee.Reducer.mean(), region_size));
print('Vegetation')
print(ui.Chart.image.series(veg, geom, ee.Reducer.median(), region_size));
    