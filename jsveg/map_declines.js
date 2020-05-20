// GEE script to map locations which have seen a decrase in precipitation and NDVI.
// Usage: copy and paste the script into the Google Earth Engine Editor:
// https://code.earthengine.google.com/

// set dates
var date_1 = '2000-01-01';
var date_2 = '2010-01-01';
var date_3 = '2020-01-01';

// specify precipitation data source
var precip_collection = "ECMWF/ERA5/MONTHLY";
var precip_bandname = "total_precipitation";

// calculate mean total precipitation
var precip_now = ee.ImageCollection(precip_collection)
                  .select(precip_bandname)
                  .filter(ee.Filter.date(date_1, date_2)).mean();

var precip_before = ee.ImageCollection(precip_collection)
                  .select(precip_bandname)
                  .filter(ee.Filter.date(date_2, date_3)).mean();  

// calculate precipitation percentage change
var precipitation_change = precip_before
                            .subtract(precip_now)
                            .divide(precip_before);
                


// specify NDVI data source
var vi_collection = "LANDSAT/LE07/C01/T1_8DAY_EVI"; // or "LANDSAT/LE07/C01/T1_8DAY_NDVI"
var vi_bandname = "EVI"; // or "NDVI"


// calculate mean vegetation index
var vi_now = ee.ImageCollection(vi_collection)
                .select(vi_bandname)
                .filter(ee.Filter.date(date_1, date_2)).mean();

var vi_before = ee.ImageCollection(vi_collection)
                  .select(vi_bandname)
                  .filter(ee.Filter.date(date_2, date_3)).mean();  

// calculate vegetation index percentage change
var vi_change = vi_before
                .subtract(vi_now)
                .divide(vi_before);


// sum and mulitply the two percentage changes
var sum_changes = precipitation_change.add(vi_change)
var prod_changes = precipitation_change.multiply(vi_change)

// Visualization palette for total precipitation
var vis_eco = {
  min: -0.2,
  max: 0.2,
  palette: [
    '#4000FF', '#8000FF', '#0080FF', '#00FFFF', '#00FF80',
    '#80FF00', '#FFF500', '#FFDA00', '#FFA400',
    '#FF4F00', '#FF2500'
  ].reverse()
};

var vis_grayscale = {
  min: -0.5,
  max: 0.5,
  palette: [
    '#000000', '#333333', '#666666',
    '#999999', '#CCCCCC', '#EEEEEE'
  ].reverse()
};


// Add layer to map
Map.addLayer(
    precipitation_change, vis_grayscale,
    'Precip Change');

Map.addLayer(
    vi_change, vis_grayscale,
    'VI Change');

Map.addLayer(
    sum_changes, vis_grayscale,
    'Sum Change');
    
Map.addLayer(
    prod_changes, vis_grayscale,
    'Prod Change');

Map.setCenter(21.2, 22.2, 2);
