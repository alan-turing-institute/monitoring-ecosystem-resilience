"""
Class for holding analysis modules
that can be chained together to build a sequence.
"""

import os
import re
import tempfile

import cv2 as cv

from multiprocessing import Pool

from pyveg.src.image_utils import *
from pyveg.src.file_utils import *
from pyveg.src.coordinate_utils import *
from pyveg.src.subgraph_centrality import (
    subgraph_centrality,
    feature_vector_metrics,
)
from pyveg.src import azure_utils

from pyveg.src.pyveg_pipeline import BaseModule


class ProcessorModule(BaseModule):

    def __init__(self, name):
        super().__init__(name)

    def save_image(self, image, output_location, output_filename, verbose=True):
        if self.output_location_type == "local":
            # use the file_utils function
            save_image(image, output_location, output_filename, verbose)
        elif self.output_location_type == "azure":
            # container name will be the first bit of self.output_location.
            container_name = self.output_location.split("/")[0]
            azure_utils.save_image(image, output_location, output_filename,
                                   container_name)
        else:
            raise RuntimeError("Unknown output location type {}"\
                               .format(self.output_location_type))



class VegetationImageProcessor(ProcessorModule):
    """
    Class to convert tif files downloaded from GEE into png files
    that can be looked at or used as input to further analysis.

    Current default is to output:
    1) Full-size RGB image
    2) Full-size NDVI image (greyscale)
    3) Full-size black+white NDVI image (after processing, thresholding, ...)
    4) Many 50x50 pixel sub-images of RGB image
    5) Many 50x50 pixel sub-images of black+white NDVI image.

    """
    def __init__(self, name=None):
        super().__init__(name)
        self.params += [("input_location", [str]),
                        ("input_location_type", [str]),
                        ("output_location", [str]),
                        ("output_location_type", [str]),
                        ("region_size", [float]),
                        ("RGB_bands", [list]),
                        ("split_RGB_images", [bool])
        ]


    def set_default_parameters(self):
        """
        Set some basic defaults.  Note that these might get overriden
        by a parent Sequence, or by calling configure() with a dict of values
        """
        super().set_default_parameters()
        if not "region_size" in vars(self):
            self.region_size = 0.1
        if not "RGB_bands" in vars(self):
            self.RGB_bands = ["B4","B3","B2"]
        if not "split_RGB_images" in vars(self):
            self.split_RGB_images = True

    def construct_image_savepath(self, date_string, coords_string, image_type):
        """
        Function to abstract output image filename construction.
        Current approach is to create a 'PROCESSED' subdir inside the
        sub-directory corresponding to the mid-period of the date range
        for the full-size images and a 'SPLIT' subdirectory for the
        sub-images.
        """

        if "SUB" in image_type:
            output_location = os.path.join(self.output_location, date_string, "SPLIT")
        else:
            output_location = os.path.join(self.output_location, date_string, "PROCESSED")
        # filename is the date, coordinates, and image type
        filename = f'{date_string}_{coords_string}_{image_type}.png'

        # full path is dir + filename
        full_path = os.path.join(output_location, filename)

        return full_path


    def save_rgb_image(self, band_dict,
                       date_string, coords_string):
        """
        Merge the seperate tif files for the R,G,B bands into
        one image, and save it.
        """
        print("{}: Saving RGB image for {} {}".format(self.name, date_string, coords_string))
        rgb_image = convert_to_rgb(band_dict)

        # check image quality on the colour image
        if not check_image_ok(rgb_image, 1.0):
            print('Detected a low quality image, skipping to next date.')
            return False
        rgb_filepath = self.construct_image_savepath(date_string,
                                                     coords_string,
                                                     'RGB')
        print("Will save image to {} / {}".format(os.path.dirname(rgb_filepath),
                                                  os.path.basename(rgb_filepath)))
        self.save_image(rgb_image, os.path.dirname(rgb_filepath),
                   os.path.basename(rgb_filepath))
        if self.split_RGB_images:
            self.split_and_save_sub_images(rgb_image,
                                           date_string,
                                           coords_string,
                                           "RGB")
        return True


    def split_and_save_sub_images(self, image,
                                  date_string,
                                  coords_string,
                                  image_type,
                                  npix=50):
        """
        Split the full-size image into lots of small sub-images

        Parameters:
        ===========
        image: pillow Image
        date_string: str, format YYYY-MM-DD
        coords_string: str, format long_lat
        image_type: str, typically 'RGB' or 'BWNDVI'
        npix: dimension in pixels of side of sub-image.  Default is 50x50

        Returns:
        ========
        True if all sub-images saved correctly.
        """

        coords = [float(coord) for coord in coords_string.split("_")]
        sub_images = crop_image_npix(image, npix,
                                     region_size=self.region_size,
                                     coords = coords)

        output_location = os.path.dirname(self.construct_image_savepath(date_string,
                                                                   coords_string,
                                                                   'SUB_'+image_type))
        for i, sub in enumerate(sub_images):
            # sub will be a tuple (image, coords) - unpack it here
            sub_image, sub_coords = sub
            output_filename = f'sub{i}_'
            output_filename += "{0:.3f}_{1:.3f}".format(sub_coords[0],
                                                           sub_coords[1])
            output_filename += "_{}".format(image_type)
            output_filename += '.png'
            self.save_image(sub_image, output_location, output_filename, verbose=False)
        return True


    def process_single_date(self, input_filepath):
        """
        For a single set of .tif files corresponding to a date range
        (normally a sub-range of the full date range for the pipeline),
        construct RGB, and NDVI greyscale images.
        Then do processing and thresholding to make black+white NDVI images.
        Split the RGB and black+white NDVI ones into small (50x50pix)
        sub-images.

        Parameters
        ==========
        input_filepath: str, full path to directory containing tif files
                       downloaded from GEE.  This will normally be
                       self.output_location/<mid-point-date>/RAW

        Returns
        =======
        True if everything was processed and saved OK, False otherwise.
        """
        filenames = [filename for filename in self.list_directory(input_filepath,
                                                                  self.input_location_type) \
                     if filename.endswith(".tif")]

        # extract this to feed into `convert_to_rgb()`
        band_dict = {}
        for icol, col in enumerate('rgb'):
            band = self.RGB_bands[icol]
            filename = self.get_file(os.path.join(input_filepath,
                                                  "download.{}.tif".format(band)),
                                     self.input_location_type)
            band_dict[col] = {"band": band,
                              "filename": filename
                              }

        tif_filebase = os.path.join(input_filepath,
                                    filenames[0].split('.')[0])
        # normally the coordinates will be part of the file path
        coords_string = find_coords_string(input_filepath)
        # if not though, we might have coords set explicitly
        if (not coords_string) and "coords" in vars(self):
            coords_string = "{}_{}".format(self.coords[0],self.coords[1])
        date_string = input_filepath.split("/")[-2]
        if not re.search("[\d]{4}-[\d]{2}-[\d]{2}", date_string):
            if date_range in vars(self):
                date_string = fid_mid_period(self.date_range[0], self.date_range[1])
            else:
                date_String = None
        if not coords_string and date_string:
            raise RuntimeError("{}: coords and date need to be defined, through file path or explicitly set")
        # save the rgb image
        rgb_ok = self.save_rgb_image(band_dict,
                                     date_string,
                                     coords_string)
        if not rgb_ok:
            return False

        # save the NDVI image
        ndvi_tif = self.get_file(os.path.join(input_filepath,
                                              "download.NDVI.tif"),
                                 self.input_location_type)
        ndvi_image = scale_tif(ndvi_tif)
        ndvi_filepath = self.construct_image_savepath(date_string,
                                                      coords_string,
                                                      'NDVI')
        self.save_image(ndvi_image,
                        os.path.dirname(ndvi_filepath),
                        os.path.basename(ndvi_filepath))

        # preprocess and threshold the NDVI image
        processed_ndvi = process_and_threshold(ndvi_image)
        ndvi_bw_filepath = self.construct_image_savepath(date_string,
                                                         coords_string,
                                                         'BWNDVI')
        self.save_image(processed_ndvi,
                        os.path.dirname(ndvi_bw_filepath),
                        os.path.basename(ndvi_bw_filepath))

        # split and save sub-images
        self.split_and_save_sub_images(ndvi_image,
                                       date_string,
                                       coords_string,
                                       "NDVI")

        self.split_and_save_sub_images(processed_ndvi,
                                       date_string,
                                       coords_string,
                                       "BWNDVI")

        return True


    def run(self):
        """"
        Function to run the module.  Loop over all date-sub-ranges and
        call process_single_date() on each of them.
        """
        super().run()
        date_subdirs = sorted(self.list_directory(self.input_location,
                                                  self.input_location_type))
        for date_subdir in date_subdirs:
            if not re.search("^([\d]{4}-[\d]{2}-[\d]{2})", date_subdir):
                print("{}: Directory name {} not in YYYY-MM-DD format"\
                      .format(self.name, date_subdir))
                continue
            date_path = os.path.join(self.input_location, date_subdir, "RAW")
            processed_ok = self.process_single_date(date_path)
            if not processed_ok:
                continue


class WeatherImageToJSON(ProcessorModule):
    """
    Read the weather-related tif files downloaded from GEE, and
    write the temp and precipitation values out as a JSON file.
    """

    def __init__(self, name=None):
        super().__init__(name)
        self.params += [("input_location", [str]),
                        ("output_location", [str]),
                        ("input_location_type", [str]),
                        ("output_location_type", [str])
        ]


    def set_default_parameters(self):
        """
        Set some basic defaults.  Note that these might get overriden
        by a parent Sequence, or by calling configure() with a dict of values
        """
        super().set_default_parameters()


    def process_one_date(self, date_string):
        """
        Read the tif files downloaded from GEE and extract the values
        (should be the same for all pixels in the image, so just take mean())

        Parameters
        ----------
        date_string: str, format "YYYY-MM-DD"

        Returns:
        --------
        metrics_dict: dict, typically 2 keys, for precipitation and temp,
                           and values as floats.
        """
        metrics_dict = {}
        print("Processing date {}".format(date_string))
        input_location = os.path.join(self.input_location, date_string, "RAW")
        for filename in self.list_directory(input_location, self.input_location_type):
            if filename.endswith(".tif"):
                name_variable = (filename.split('.'))[1]
                variable_array = cv.imread(self.get_file(os.path.join(input_location,
                                                                      filename),
                                                         self.input_location_type),
                                           cv.IMREAD_ANYDEPTH)

                metrics_dict[name_variable] = variable_array.mean()\
                                                            .astype(np.float64)
        return metrics_dict


    def run(self):
        super().run()
        # sub-directories of our input directory should be dates.
        time_series_data = {}
        date_strings = self.list_directory(self.input_location, self.input_location_type)
        date_strings.sort()
        for date_string in date_strings:
            if date_string == "RESULTS":
                continue
            time_series_data[date_string] = self.process_one_date(date_string)

        self.save_json(time_series_data,
                       "weather_data.json",
                       os.path.join(self.output_location, "RESULTS"),
                       self.output_location_type
                       )



#######################################################################

def process_sub_image(i, input_filename, input_location, output_location):
    """
    Read file and run network centrality
    """
    date_string = input_location.split("/")[-2]

    # open BWNDVI image
    sub_image = Image.open(os.path.join(input_location, input_filename))

    # open NDVI image
    ndvi_sub_image = Image.open(os.path.join(input_location, input_filename.replace('BWNDVI', 'NDVI')))

    # get average NDVI across the whole image (in case there is no patterned veg)
    ndvi_mean = round(pillow_to_numpy(ndvi_sub_image).mean(), 4)
    #ndvi_std = round(pillow_to_numpy(ndvi_sub_image).std(), 4)

    # use the BWDVI to mask the NDVI and calculate the average
    # pixel value of veg pixels
    veg_mask = (pillow_to_numpy(sub_image) == 0)
    ndvi_veg_mean = round(pillow_to_numpy(ndvi_sub_image)[veg_mask].mean(), 4)
    #veg_ndvi_std = round(pillow_to_numpy(ndvi_sub_image)[veg_mask].std(), 4)

    image_array = pillow_to_numpy(sub_image)
    feature_vec, _ = subgraph_centrality(image_array)
    # coords should be part of the filepath
    coords_string = find_coords_string(input_filename)
    if not coords_string:
        raise RuntimeError("Unable to find coordinates in {}"\
                           .format(input_filename))
    coords = [float(c) for c in coords_string.split("_")]

    nc_result = feature_vector_metrics(feature_vec)
    nc_result['feature_vec'] = list(feature_vec)
    nc_result['date'] = date_string
    nc_result['latitude'] = coords[1]
    nc_result['longitude'] = coords[0]
    nc_result['ndvi'] = ndvi_mean
    #nc_result['ndvi_std'] = ndvi_std
    nc_result['ndvi_veg'] = ndvi_veg_mean
    #nc_result['veg_ndvi_std'] = veg_ndvi_std

    # save individual result for sub-image to tmp json, will combine later.
    save_json(nc_result, output_location,
              f"network_centrality_sub{i}.json", verbose=False)
    # count and print how many sub-images we have done.
    n_processed = len(list_directory(output_location))
    print(f'Processed {n_processed} sub-images...', end='\r')
    return True


class NetworkCentralityCalculator(ProcessorModule):
    """
    Class to run network centrality calculation on small black+white
    images, and return the results as json.
    Note that the input directory is expected to be the level above
    the subdirectories for the date sub-ranges.
    """

    def __init__(self, name=None):
        super().__init__(name)
        self.params += [
            ("input_location", [str]),
            ("output_location", [str]),
            ("input_location_type", [str]),
            ("output_location_type", [str]),
            ("n_threads", [int]),
            ("n_sub_images", [int])
        ]

    def set_default_parameters(self):
        """
        Default values. Note that these can be overridden by parent Sequence
        or by calling configure().
        """
        super().set_default_parameters()
        self.n_threads = 4
        if not "n_sub_images" in vars(self):
            self.n_sub_images = -1 # do all-sub-images
        if not "input_location_type" in vars(self):
            self.input_location_type = "local"
        if not "output_location_type" in vars(self):
            self.output_location_type = "local"


    def check_sub_image(self, ndvi_filename, input_path):
        """
        Check the RGB sub-image corresponding to this NDVI image
        looks OK.
        """
        rgb_filename = re.sub("BWNDVI","RGB",ndvi_filename)
        rgb_img = Image.open(os.path.join(input_path, rgb_filename))
        img_ok = check_image_ok(rgb_img, 0.05)
        return img_ok


    def process_single_date(self, date_string):
        """
        Each date will have a subdirectory called 'SPLIT' with ~500
        sub-images.
        """

        input_path = os.path.join(self.input_location, date_string, "SPLIT")
        if not os.path.exists(input_path):
            print("{}: No sub-images for date {}".format(self.name,
                                                         date_string))
            return
        # list all the "BWNDVI" sub-images where RGB image passes quality check
        input_files = [filename for filename in self.list_directory(input_path) \
                       if "BWNDVI" in filename and \
                       self.check_sub_image(filename,input_path)]
        tmp_json_dir = os.path.join(self.input_location, date_string,"tmp_json")

        # if we only want a subset of sub-images, truncate the list here
        if self.n_sub_images > 0:
            input_files = input_files[:self.n_sub_images]

        # create a multiprocessing pool to handle each sub-image in parallel
        with Pool(processes=self.n_threads) as pool:
            # prepare the arguments for the process_sub_image function
            arguments=[(i, filename, input_path, tmp_json_dir) \
                   for i, filename in enumerate(input_files)]
            pool.starmap(process_sub_image, arguments)
        # put all the output json files for subimages together into one for this date
        consolidate_json_to_list(os.path.join(self.input_location,
                                              date_string,
                                              "tmp_json"),
                                 os.path.join(self.output_location,
                                              date_string),
                                 "network_centralities.json")


    def run(self):
        super().run()
        if "list_of_dates" in vars(self):
            date_strings = self.list_of_dates
        else:
            date_strings = sorted(self.list_directory(self.input_location,
                                                      self.input_location_type))
        for date_string in date_strings:
            self.process_single_date(date_string)
