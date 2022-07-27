// GEE script to map locations which have seen a decrase in precipitation and NDVI.
// Usage: copy and paste the script into the Google Earth Engine Editor:
// https://code.earthengine.google.com/

// GEE script to map locations which have seen a decrase in precipitation and NDVI

// set dates
var date_1 = '2000-01-01';
var date_2 = '2005-01-01';
var date_3 = '2015-01-01';
var date_4 = '2020-01-01';

// specify precipitation data source
var precip_collection = "ECMWF/ERA5/MONTHLY";
var precip_bandname = "total_precipitation";

// calculate mean total precipitation
var precip_before = ee.ImageCollection(precip_collection)
                  .select(precip_bandname)
                  .filter(ee.Filter.date(date_1, date_2)).mean();

var precip_now = ee.ImageCollection(precip_collection)
                  .select(precip_bandname)
                  .filter(ee.Filter.date(date_3, date_4)).mean();

// calculate precipitation percentage change
var precipitation_change = precip_now
                            .subtract(precip_before)
                            .divide(precip_before);

// mask regions that are initially too dry or too wet
var precip_mask = precip_before.gt(0.0001).and(precip_before.lt(0.05));
precipitation_change = precipitation_change.multiply(precip_mask);

// mask small absolute changes
//var precip_change_abs = precip_now.subtract(precip_before);
//precipitation_change = precipitation_change.multiply(precip_change_abs.lt(-0.001));



// specify NDVI data source
var veg_collection = "LANDSAT/LE07/C01/T1_8DAY_EVI"; // or "LANDSAT/LE07/C01/T1_8DAY_NDVI"
var veg_bandname = "EVI"; // or "NDVI"

// calculate mean vegetation index
var veg_before = ee.ImageCollection(veg_collection)
                  .select(veg_bandname)
                  .filter(ee.Filter.date(date_1, date_2));

var veg_now = ee.ImageCollection(veg_collection)
                .select(veg_bandname)
                .filter(ee.Filter.date(date_3, date_4));

// scan line masking as per https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6544617/
var veg_now = veg_now.map(function(img) {
                   var mask = img.select(veg_bandname).lt(0.9)
                                 .and(img.select(veg_bandname).gt(-0.9))
                   return img.updateMask(mask)
                   }).mean()

var veg_before = veg_before.map(function(img) {
                   var mask = img.select(veg_bandname).lt(0.9)
                                 .and(img.select(veg_bandname).gt(-0.9))
                   return img.updateMask(mask)
                   }).mean()

// calculate vegetation index percentage change
var veg_change = veg_now
                .subtract(veg_before)
                .divide(veg_before);

// mask areas with no vegetation
var veg_mask = veg_before.gt(0.05);
veg_change = veg_change.multiply(veg_mask);

// sum and mulitply the two percentage changes
var sum_changes = precipitation_change.add(veg_change).multiply(veg_change.abs()).multiply(precipitation_change.lt(0.0).and(veg_change.lt(0.0)))
var prod_changes = precipitation_change.abs().multiply(veg_change.abs()).multiply(precipitation_change.lt(0.0).and(veg_change.lt(0.0)))

// Visualization palettes
var vis_eco = { //unused colour palette
  min: -0.8,
  max: 0.8,
  palette: [
    '#4000FF', '#8000FF', '#0080FF', '#00FFFF', '#00FF80',
    '#80FF00', '#FFF500', '#FFDA00', '#FFA400',
    '#FF4F00', '#FF2500'
  ].reverse()
};

var vis_grayscale = {
  min: -1.0,
  max: 1.0,
  palette: [
    '#000000', '#111111', '#222222', '#333333',
    '#444444', '#555555', '#666666', '#777777',
    '#888888', '#999999', '#AAAAAA', '#BBBBBB',
    '#CCCCCC', '#DDDDDD', '#EEEEEE', '#FFFFFF'
  ].reverse()
};

var vis_grayscale_sum = {
  min: -1.0,
  max: 1.0,
  palette: [
    '#000000', '#333333', '#666666',
    '#999999', '#CCCCCC', '#EEEEEE'
  ].reverse()
};

var vis_grayscale_prod = {
  min: -0.35,
  max: 0.35,
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
    veg_change, vis_grayscale,
    'Veg Change');

Map.addLayer(
    sum_changes, vis_grayscale_sum,
    'Sum Change');

Map.addLayer(
    prod_changes, vis_grayscale_prod,
    'Prod Change');

Map.setCenter(0, 0, 3);
