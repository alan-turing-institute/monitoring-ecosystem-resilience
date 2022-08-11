"""
Classes for modules that download from GEE
"""

import imp
import logging
import tempfile

import ee

from pyveg.src.date_utils import slice_time_period
from pyveg.src.file_utils import download_and_unzip
from pyveg.src.gee_interface import add_NDVI, apply_mask_cloud
from pyveg.src.pyveg_pipeline import BaseModule, logger

ee.Initialize()

# silence google API WARNING


logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)


class DownloaderModule(BaseModule):
    """
    Most of the code needed to download images from GEE is common to all
    types of data, so we put it in this base class, and have data-type-specific
    code in subclasses.
    """

    def __init__(self, name):
        super().__init__(name)
        # some parameters that are common to all Downloaders
        self.params += [
            ("collection_name", [str]),
            ("date_range", [list, tuple]),
            ("scale", [int]),
            ("output_location", [str]),
            ("output_location_type", [str]),
            ("replace_existing_files", [bool]),
            ("ndvi", [bool]),
            ("count", [bool]),
            ("bounds", [list]),
            ("projection", [str]),
        ]
        return

    def set_default_parameters(self):
        """
        Set some basic defaults that should be common to all downloaders.
        Note that these can be overriden by either values held by a parent Sequence
        or by calling configure() with a configuration dictionary.
        """
        super().set_default_parameters()
        if not "scale" in vars(self):
            self.scale = 10
        if not "output_location" in vars(self):
            self.set_output_location()
        if not "output_location_type" in vars(self) or not self.output_location_type:
            self.output_location_type = "local"
        if not "replace_existing_files" in vars(self):
            self.replace_existing_files = False
        if not "ndvi" in vars(self):
            self.ndvi = False
        if not "count" in vars(self):
            self.count = True
        if not "bounds" in vars(self):
            self.bounds = []
        if not "projection" in vars(self):
            self.projection = "EPSG:27700"

        return

    def set_output_location(self, output_location=None):
        """
        If provided an output directory name, set it here,
        otherwise, construct one from bounds and collection name.

        Parameters
        ==========
        output_location: tuple of strings (location, location_type)

        """
        if output_location:
            self.output_location = output_location[0]

        elif ("bounds" in vars(self)) and ("collection_name" in vars(self)):

            self.output_location = (
                "gee_{:0>6}_{:0>7}_{:0>6}_{:0>7}".format(
                    round(self.bounds[0]),
                    round(self.bounds[1]),
                    round(self.bounds[2]),
                    round(self.bounds[3]),
                )
                + "_"
                + self.collection_name.replace("/", "-")
            )

        else:
            raise RuntimeError(
                """
            {}: need to set collection_name and bounds before calling set_output_location()
            """.format(
                    self.name
                )
            )

    def prep_data(self, date_range):
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
        start_date, end_date = date_range

        image_coll = ee.ImageCollection(self.collection_name)
        ll_point = ee.Geometry.Point(
            (self.bounds[0], self.bounds[1]), proj=self.projection
        )
        tr_point = ee.Geometry.Point(
            (self.bounds[2], self.bounds[3]), proj=self.projection
        )
        geom = ee.Geometry.Rectangle(
            coords=(ll_point, tr_point), proj=self.projection, evenOdd=False
        )

        dataset = image_coll.filterBounds(geom).filterDate(start_date, end_date)
        dataset_size = dataset.size().getInfo()

        if dataset_size == 0:
            logger.info("No images found in this date rage, skipping.")
            log_msg = "WARN >>> No data found."
            return []
        # concrete class will do more filtering, and prepare Images for download
        image_list = self.prep_images(dataset)
        url_list = []
        for image in image_list:
            # get a URL from which we can download the resulting data
            try:
                url = image.getDownloadURL(
                    {"region": geom, "scale": self.scale, "crs": self.projection}
                )
                url_list.append(url)
            except Exception as e:
                logger.info("Unable to get URL: {}".format(e))

            logging.info(
                f"OK   >>> Found {dataset.size().getInfo()}/{dataset_size} valid images after cloud filtering."
            )
        return url_list

    def download_data(self, download_urls, download_location):
        """
        Download zip file(s) from GEE to configured output location.

        Parameters
        ---------
        download_urls: list of strings (URLs) from gee_prep_data
        download_location: str, this will generally be <base_dir>/<date>/RAW

        Returns:
        --------
        bool, True if downloaded something, False otherwise
        """
        if len(download_urls) == 0:
            logger.info("{}: No URLs found for {}".format(self.name, self.bounds))
            return False

        # download files and unzip to temporary directory
        tempdir = tempfile.TemporaryDirectory()
        for download_url in download_urls:
            try:
                download_and_unzip(download_url, tempdir.name)
            except RuntimeError as e:
                return False
        logger.debug("{}: Wrote zipfiles to {}".format(self.name, tempdir.name))
        logger.info("{}: Will download to {}".format(self.name, download_location))
        self.copy_to_output_location(tempdir.name, download_location, [".tif"])
        return True

    def run(self):
        self.prepare_for_run()

        start_date, end_date = self.date_range
        date_ranges = slice_time_period(start_date, end_date, self.time_per_point)
        download_locations = []
        for date_range in date_ranges:
            mid_date = "{}_{}".format(
                date_range[0], date_range[1]
            )  # find_mid_period(date_range[0], date_range[1])
            location = self.join_path(self.output_location, mid_date, "RAW")
            logger.debug(
                "{} Will check for existing files in {}".format(self.name, location)
            )
            if not self.replace_existing_files and self.check_for_existing_files(
                location, self.num_files_per_point
            ):
                continue
            urls = self.prep_data(date_range)
            logger.debug(
                "{}: got URL {} for date range {}".format(self.name, urls, date_range)
            )
            downloaded_ok = self.download_data(urls, location)
            if downloaded_ok:
                self.run_status["succeeded"] += 1
                logger.info(
                    "{}: download succeeded for date range {}".format(
                        self.name, date_range
                    )
                )
                download_locations.append(location)
            else:
                self.run_status["failed"] += 1
                logger.error(
                    "{}: download did not succeed for date range {}".format(
                        self.name, date_range
                    )
                )
        self.is_finished = True
        return self.run_status


##############################################################################
# Below here are specializations of the BaseDownloader class.
# e.g. for downloading vegetation imagery, or weather data.
##############################################################################


class VegetationDownloader(DownloaderModule):
    """
    Specialization of the DownloaderModule class, to deal with
    imagery from Sentinel 2 or Landsat 5-8 satellites,
    get NDVI band from combining red and near-infra-red and create a
    COUNT band with number of images per pixel in the composite.
    """

    def __init__(self, name=None):
        super().__init__(name)
        self.params += [
            ("mask_cloud", [bool]),
            ("cloudy_pix_flag", [str]),
            ("cloudy_pix_frac", [int]),
            ("RGB_bands", [list]),
            ("NIR_band", [str]),
            ("time_per_point", [str]),
            ("num_files_per_point", [int]),
        ]

    def set_default_parameters(self):
        """
        Set some defaults.  Note that these can be overriden, either
        by parent Sequence, or by calling configure() with a dict.
        """
        super().set_default_parameters()
        self.mask_cloud = True
        self.cloudy_pix_frac = 50
        self.num_files_per_point = 4

    def prep_images(self, dataset):
        """
        Take a dataset that has already been filtered by date and location.
        Then apply specific filters, take the median, and calculate NDVI and create
        the COUNT band.

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
        dataset = apply_mask_cloud(dataset, self.collection_name, self.cloudy_pix_flag)
        # Take median
        image = dataset.median()
        bands_to_select = self.RGB_bands

        if self.ndvi:
            # Calculate NDVI
            image = add_NDVI(image, self.RGB_bands[0], self.NIR_band)
            # select only RGB + NDVI bands to download
            bands_to_select = bands_to_select + ["NDVI"]

        if self.count:
            # create pixel level count image
            count = dataset.count()
            # sum counts on each band into one image
            image_count = count.select(self.RGB_bands[0]).rename("COUNT")
            # add count image as a band
            image = image.addBands(image_count)

            bands_to_select = bands_to_select + ["COUNT"]

        # select relevant bands
        image = image.select(bands_to_select)
        return [image]


class WeatherDownloader(DownloaderModule):
    """
    Download precipitation and temperature data.
    """

    def __init__(self, name=None):
        super().__init__(name)
        self.params += [
            ("temperature_band", [list]),
            ("precipitation_band", [list]),
            ("time_per_point", [str]),
            ("num_files_per_point", [int]),
        ]

    def set_default_parameters(self):
        """
        Set some defaults.  Note that these can be overriden, either
        by parent Sequence, or by calling configure() with a dict.
        """
        super().set_default_parameters()
        self.num_files_per_point = 2

    def prep_images(self, dataset):
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
