"""
Classes for modules that download from GEE
"""

import os
import requests
from datetime import datetime, timedelta
import dateparser
from zipfile import ZipFile, BadZipFile


from geetools import cloud_mask
import cv2 as cv

import ee
ee.Initialize()

if os.name == "posix":
    TMPDIR = "/tmp/"
else:
    TMPDIR = "%TMP%"

LOGFILE = os.path.join(TMPDIR, "failed_downloads.log")

# Hack to silence google API WARNING
import logging
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

from pyveg_sequence import BaseModule



class BaseDownloader(BaseModule):

    def __init__(self, name):
        super().__init__(name)
        # some parameters that are common to all Downloaders
        self.params +=  [("collection_name",str),
                         ("coords", list),
                         ("date_range", list),
                         ("type", str),
                         ("region_size", float),
                         ("scale", int),
                         ("output_basedir", str)]
        return

    def set_default_parameters(self):
        """
        Set some basic defaults that should be common to all downloaders
        """
        if not "region_size" in vars(self):
            self.region_size = 0.1
        if not "scale" in vars(self):
            self.scale = 10
        if not "output_basedir" in vars(self):
            self.output_basedir = "."
        return

    def configure(self, config_dict=None):
        super().configure(config_dict)
        # construct name of output directory from coords if not set
        if not "output_dir" in vars(self):
            self.set_output_dir()

    def get_region_string(self):
        """
        Construct a string of coordinates that create a box
        around the specified point.

        Returns
        ----------
        str
        A string with coordinates for each corner of the box.
        Can be passed to Earth
        Engine.
        """
        left = self.coords[0] - self.region_size/2
        right = self.coords[0] + self.region_size/2
        top = self.coords[1] + self.region_size/2
        bottom = self.coords[1] - self.region_size/2
        coords =  str([[left,top],[right,top],[right,bottom],[left,bottom]])
        return coords

    def find_mid_period(self, start_time, end_time):
        """
        Given two strings in the format YYYY-MM-DD return a
        string in the same format representing the middle (to
        the nearest day)
        """
        t0 = dateparser.parse(start_time)
        t1 = dateparser.parse(end_time)
        td = (t1 - t0).days
        mid = (t0 + timedelta(days=(td//2))).isoformat()
        return mid.split("T")[0]


    def ee_prep_data(self, date_range):
        """
        Interact with the Google Earth Engine API to get in ImageCollection,
        filter it, and convert (e.g. via median or sum) into a list of Images,
        then get the download URLs for those.

        Parameters
        ----------
        date_range: list of strings 'YYYY-MM-DD'.  Note that this will generally
                    be a sub-range of the overall date-range, as this function
                    is called in the loop over time slices.

        Returns
        -------
        url_list:  a list of URLs from which zipfiles can be downloaded from GEE.
        """
        region  = self.get_region_string()
        start_date, end_date = date_range

        image_coll = ee.ImageCollection(self.collection_name)
        geom = ee.Geometry.Point(self.coords)

        dataset = image_coll.filterBounds(geom).filterDate(start_date,end_date)
        dataset_size = dataset.size().getInfo()

        if dataset_size == 0:
            print('No images found in this date rage, skipping.')
            log_msg = 'WARN >>> No data found.'
            return []
        # concrete class will do more filtering, and prepare Images for download
        image_list = self.ee_prep_images(dataset)
        url_list =[]
        for image in image_list:
            # get a URL from which we can download the resulting data
            try:
                url = image.getDownloadURL(
                    {'region': region,
                     'scale': self.scale}
                )
                url_list.append(url)
            except Exception as e:
                print("Unable to get URL: {}".format(e))

            logging.info(f'OK   >>> Found {dataset.size().getInfo()}/{dataset_size} valid images after cloud filtering.')
        return url_list


    def ee_download(self, download_urls, date_range):
        """
        Download zip file(s) from GEE to configured output location.

        Parameters
        ---------
        download_urls: list of strings (URLs) from gee_prep_data
        date_range: list of strings 'YYYY-MM-DD' - note that this will
                    generally be a sub range of the object's overall
                    date_range.
        """
        if len(download_urls) == 0:
            return None, "{}: No URLs found for {} {}".format(self.name,
                                                              self.coords,
                                                              date_range)

        mid_date = self.find_mid_period(date_range[0], date_range[1])
        download_dir = os.path.join(self.output_dir, mid_date, "RAW")

        # download files and unzip to temporary directory
        for download_url in download_urls:
            self.download_and_unzip(download_url, download_dir)

        # return the path so downloaded files can be handled by caller
        return download_dir


    def download_and_unzip(self, url, output_tmpdir):
        """
        Given a URL from GEE, download it (will be a zipfile) to
        a temporary directory, then extract archive to that same dir.
        Then find the base filename of the resulting .tif files (there
        should be one-file-per-band) and return that.
        """
        print("Will download {} to {}".format(url, output_tmpdir))
        # GET the URL
        r = requests.get(url)
        if not r.status_code == 200:
            raise RuntimeError(" HTTP Error getting download link {}".format(url))
        os.makedirs(output_tmpdir, exist_ok=True)
        output_zipfile = os.path.join(output_tmpdir,"gee.zip")
        with open(output_zipfile, "wb") as outfile:
            outfile.write(r.content)
        ## catch zipfile-related exceptions here, and if they arise,
        ## write the name of the zipfile and the url to a logfile
        try:
            with ZipFile(output_zipfile, 'r') as zip_obj:
                zip_obj.extractall(path=output_tmpdir)
        except(BadZipFile):
            with open(LOGFILE, "a") as logfile:
                logfile.write("{}: {} {}\n".format(str(datetime.now()),
                                               output_zipfile,
                                               url))
            return None
        tif_files = [filename for filename in os.listdir(output_tmpdir) \
                     if filename.endswith(".tif")]
        if len(tif_files) == 0:
            raise RuntimeError("No files extracted")

        # get the filename before the "Bx" band identifier
        tif_filebases = [tif_file.split(".")[0] for tif_file in tif_files]

        # get the unique list
        tif_filebases = set(tif_filebases)

        # prepend the directory name to each of the filebases
        return [os.path.join(output_tmpdir, tif_filebase) \
                for tif_filebase in tif_filebases]


    def get_num_n_day_slices(self):
        """
        Divide the full period between the start_date and end_date into n equal-length
        (to the nearest day) chunks. The size of the chunk is defined by days_per_point.
        Takes start_date and end_date as strings 'YYYY-MM-DD'.
        Returns an integer with the number of possible points avalaible in that time period]
        """
        start = dateparser.parse(self.date_range[0])
        end = dateparser.parse(self.date_range[1])
        if (not isinstance(start, datetime)) or (not isinstance(end, datetime)):
            raise RuntimeError("invalid time strings")
        td = end - start
        if td.days <= 0:
            raise RuntimeError("end_date must be after start_date")
        n = td.days//self.num_days_per_point
        return  n


    def slice_time_period(self,n):
        """
        Divide the full period between the start_date and end_date into n equal-length
        (to the nearest day) chunks.
        Takes start_date and end_date as strings 'YYYY-MM-DD'.
        Returns a list of tuples
        [ (chunk0_start,chunk0_end),...]
        """
        start = dateparser.parse(self.date_range[0])
        end = dateparser.parse(self.date_range[1])
        if (not isinstance(start, datetime)) or (not isinstance(end, datetime)):
            raise RuntimeError("invalid time strings")
        td = end - start
        if td.days <= 0:
            raise RuntimeError("end_date must be after start_date")
        days_per_chunk = td.days // n
        output_list = []
        for i in range(n):
            chunk_start = start + timedelta(days=(i*days_per_chunk))
            chunk_end = start + timedelta(days=((i+1)*days_per_chunk))
            ## unless we are in the last chunk, which should finish at end_date
            if i == n-1:
                chunk_end = end
            output_list.append((chunk_start.isoformat().split("T")[0],
                                chunk_end.isoformat().split("T")[0]))
        return output_list


    def set_output_dir(self, output_dir=None):
        """
        If provided an output directory name, set it here,
        otherwise, construct one from coords and collection name.
        """
        if output_dir:
            self.output_dir = output_dir
        else:
            sub_dir = f'gee_{self.coords[0]}_{self.coords[1]}'\
                +"_"+self.collection_name.replace('/', '-')
            self.output_dir = os.path.join(self.output_basedir, sub_dir)


    def run(self):

        num_slices = self.get_num_n_day_slices()
        date_ranges = self.slice_time_period(num_slices)
        download_dirs = []
        for date_range in date_ranges:
            urls = self.ee_prep_data(date_range)
            print("{}: got URL {} for date range {}".format(self.name,
                                                            urls,
                                                            date_range))
            download_dir = self.ee_download(urls, date_range)
            download_dirs.append(download_dir)
        return download_dirs


##############################################################################
# Below here are specializations of the BaseDownloader class.
# e.g. for downloading vegetation imagery, or weather data.
##############################################################################


class VegetationDownloader(BaseDownloader):
    """
    Specialization of the BaseDownloader class, to deal with
    imagery from Sentinel 2 or Landsat 5-8 satellites, and
    get NDVI band from combining red and near-infra-red.
    """

    def __init__(self, name):
        super().__init__(name)
        self.params += [("mask_cloud", bool),
                        ("cloudy_pix_flag", str),
                        ("cloudy_pix_frac", int),
                        ("RGB_bands", list),
                        ("NIR_band", str),
                        ("num_days_per_point", int)
        ]


    def set_default_parameters(self):
        """
        Set some defaults for the chosen satellite
        """
        # set basic things like region_size and scale in the base class
        super().set_default_parameters()
        if not "data_type" in vars(self):
            self.type = "vegetation"
        if "Sentinel2" in self.name:
            self.collection_name = "COPERNICUS/S2"
            self.RGB_bands = ["B4","B3","B2"]
            self.NIR_band = "B8"
            self.mask_cloud = True
            self.cloudy_pix_flag = "CLOUDY_PIXEL_PERCENTAGE"
            self.cloudy_pix_frac = 50
            self.num_days_per_point = 30


    def apply_mask_cloud(self, image_coll):
        """
        Different input_collections need different steps to be taken to
        handle cloudy images.
        The first step is to reject images that more than X% cloudy pixels.
        The next step is to mask cloudy pixels.
        This will hopefully mean that when we take the median of the
        ImageCollection, we ignore cloudy pixels.

        Parameters
        ----------
        image_coll : ee.ImageCollection
            The ImageCollection of images from which we want to remove cloud.

        Returns
        ----------
        image_coll
            Image collection with very cloudy images removed, and masked images
            containing a tolerable amount of cloud.
        """
        # construct cloud mask if availible
        if self.collection_name == 'COPERNICUS/S2':
            mask_func = cloud_mask.sentinel2()
        elif self.collection_name == 'LANDSAT/LC08/C01/T1_SR':
            mask_func = cloud_mask.landsat8SRPixelQA()
        elif ( self.collection_name == 'LANDSAT/LE07/C01/T1_SR' or
               self.collection_name == 'LANDSAT/LT05/C01/T1_SR' or
               self.collection_name == 'LANDSAT/LT04/C01/T1_SR' ):
            mask_func = cloud_mask.landsat457SRPixelQA()
        else:
            print("No cloud mask logic defined for input collection {}"\
                  .format(collection_name))
            return image_coll

        # remove images that have more than `cloud_pix_frac`% cloudy pixels
        if self.cloudy_pix_flag != 'None':
            image_coll = image_coll.filter(ee.Filter.lt(self.cloudy_pix_flag,
                                                        self.cloudy_pix_frac))

        # apply per pixel cloud mask
        image_coll = image_coll.map(mask_func)

        return image_coll


    def add_NDVI(self, image, red_band, near_infrared_band):
        try:
            image_ndvi = image.normalizedDifference([near_infrared_band,
                                                     red_band])\
                              .rename('NDVI')
            return ee.Image(image).addBands(image_ndvi)
        except:
            print ("Something went wrong in the NDVI variable construction")
            return image


    def ee_prep_images(self, dataset):
        """
        Take a dataset that has already been filtered by date and location.
        Then apply specific filters, take the median, and calculate NDVI.

        Parameters
        ----------
        dataset : ee.ImageCollection
            The ImageCollection of images filtered by location and date.

        Returns
        ----------
        image_list : list(ee.Image)
            List of Images to be downloaded
        """
        # Apply cloud mask
        dataset = self.apply_mask_cloud(dataset)
        # Take median
        image = dataset.median()
        # Calculate NDVI
        image = self.add_NDVI(image, self.RGB_bands[0], self.NIR_band)
        # select only RGB + NDVI bands to download
        bands_to_select = self.RGB_bands + ['NDVI']
        image = image.select(bands_to_select)
        return [image]



class WeatherDownloader(BaseDownloader):
    """
    Download precipitation and temperature data.
    """

    def __init__(self,name):
        super().__init__(name)
        self.params += [("temperature_band", list),
                        ("precipitation_band", list),
                        ("num_days_per_point", int)
        ]


    def set_default_parameters(self):
        """
        Set some defaults for the chosen satellite
        """
        # set basic things like region_size and scale in the base class
        super().set_default_parameters()
        if not "data_type" in vars(self):
            self.type = "weather"
        if "ERA5" in self.name:
            self.collection_name = "ECMWF/ERA5/MONTHLY"
            self.temperature_band = ['mean_2m_air_temperature']
            self.precipitation_band = ['total_precipitation']
            self.num_days_per_point = 30


    def ee_prep_images(self, dataset):
        """
        Take a dataset that has already been filtered by date and location,
        and combine into Images by summing (for precipitation) and taking
        average (for temperature).

        Parameters
        ----------
        dataset : ee.ImageCollection
            The ImageCollection of images filtered by location and date.

        Returns
        ----------
        image_list : list(ee.Image)
            List of Images to be downloaded
        """
        image_list = []
        image_weather = dataset.select(self.precipitation_band).sum()
        image_list.append(image_weather)
        image_temp = dataset.select(self.temperature_band).mean()
        image_list.append(image_temp)

        return image_list
