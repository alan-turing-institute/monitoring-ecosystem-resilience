"""
Class for holding analysis modules
that can be chained together to build a sequence.
"""

import os
import re

from image_utils import *
from file_utils import *

from .subgraph_centrality import (
    subgraph_centrality,
    feature_vector_metrics,
)

from pyveg_sequence import BaseModule


class AnalysisModule(BaseModule):

    def __init__(self, name):
        super().__init__(name)


class VegetationImageProcessor(AnalysisModule):

    def __init__(self, name):
        super().__init__(name)
        self.params += [("input_dir", str),
                        ("output_dir", str),
                        ("RGB_bands", list),
                        ("NDVI_band", str),
                        ("split_RGB_images", bool)
        ]
        self.set_default_parameters()


    def set_default_parameters(self):
        """
        Set some basic defaults
        """
        if not "region_size" in vars(self):
            self.region_size = 0.1
        if not "output_dir" in vars(self):
            self.output_dir = "."
        if not "split_RGB_images" in vars(self):
            self.split_RGB_images = True
        return


    def construct_image_savepath(self, date_string, coords_string, image_type):
        """
        Function to abstract output image filename construction.
        Current approach is to create a 'PROCESSED' subdir inside the sub-directory
        corresponding to the mid-period of the date range for the full-size images
        and a 'SPLIT' subdirectory for the sub-images.
        """

        if "SUB" in image_type:
            output_dir = os.path.join(self.output_dir, date_string, "SPLIT")
        else:
            output_dir = os.path.join(self.output_dir, date_string, "PROCESSED")
        # filename is the date, coordinates, and image type
        filename = f'{date_string}_{coords_string}_{image_type}.png'

        # full path is dir + filename
        full_path = os.path.join(output_dir, filename)

        return full_path


    def save_rgb_image(self, tif_filebase, input_filepath,
                       date_string, coords_string):
        """
        Merge the seperate tif files for the R,G,B bands into
        one image, and save it.
        """
        print("{}: Processing {}".format(self.name, tif_filebase))
        rgb_image = convert_to_rgb(tif_filebase, self.RGB_bands)

        # check image quality on the colour image
        if not check_image_ok(rgb_image):
            print('Detected a low quality image, skipping to next date.')
            return False
        rgb_filepath = self.construct_image_savepath(date_string, coords_string,'RGB')
        save_image(rgb_image, os.path.dirname(rgb_filepath),
                   os.path.basename(rgb_filepath))
        if self.split_RGB_images:
            self.split_and_save_sub_images(rgb_image, date_string, coords_string, "RGB")
        return True


    def split_and_save_sub_images(self, image,
                                  date_string,
                                  coords_string,
                                  image_type,
                                  npix=50):
        """
        Split the full-size image into lots of small sub-images
        """

        coords = [float(coord) for coord in coords_string.split("_")]
        sub_images = crop_image_npix(image, npix,
                                     region_size=self.region_size,
                                     coords = coords)

        output_dir = os.path.dirname(self.construct_image_savepath(date_string,
                                                                   coords_string,
                                                                   'SUB_'+image_type))
        for i, sub in enumerate(sub_images):
            # sub will be a tuple (image, coords) - unpack it here
            sub_image, sub_coords = sub
            output_filename = f'sub{i}_'
            output_filename += "{0:.3f}-{1:.3f}".format(sub_coords[0],
                                                           sub_coords[1])
            output_filename += "_{}".format(image_type)
            output_filename += '.png'
            save_image(sub_image, output_dir, output_filename)
        return True


    def process_single_date(self, input_filepath):
        filenames = [filename for filename in os.listdir(input_filepath) \
                     if filename.endswith(".tif")]

        # extract this to feed into `convert_to_rgb()`
        tif_filebase = os.path.join(input_filepath,
                                    filenames[0].split('.')[0])
        coords_string = find_coords_string(input_filepath)
        date_string = input_filepath.split("/")[-2]

        # save the rgb image
        rgb_ok = self.save_rgb_image(tif_filebase,
                                     input_filepath,
                                     date_string,
                                     coords_string)
        if not rgb_ok:
            return False

        # save the NDVI image
        ndvi_image = scale_tif(tif_filebase, "NDVI")
        ndvi_filepath = self.construct_image_savepath(date_string,
                                                      coords_string,
                                                      'NDVI')
        save_image(ndvi_image,
                   os.path.dirname(ndvi_filepath),
                   os.path.basename(ndvi_filepath))
        # preprocess and threshold the NDVI image
        processed_ndvi = process_and_threshold(ndvi_image)
        ndvi_bw_filepath = self.construct_image_savepath(date_string,
                                                         coords_string,
                                                         'BWNDVI')
        save_image(processed_ndvi,
                   os.path.dirname(ndvi_bw_filepath),
                   os.path.basename(ndvi_bw_filepath))
        self.split_and_save_sub_images(processed_ndvi,
                                       date_string,
                                       coords_string,
                                       "BWNDVI")
        return True


    def run(self):
        self.check_config()
        date_subdirs = os.listdir(self.input_dir)
        for date_subdir in date_subdirs:
            date_path = os.path.join(self.input_dir, date_subdir, "RAW")
            processed_ok = self.process_single_date(date_path)
            if not processed_ok:
                continue


class NetworkCentralityCalculator(AnalysisModule):

    def __init__(self, name):
        super().__init__(name)
        self.params += [
            ("input_dir", str),
            ("output_dir", str),
            ("n_threads", int)
                        ]
        self.set_default_parameters()


    def set_default_parameters(self):
        if not "n_threads" in vars(self):
            self.n_threads = 4
        pass


    def process_sub_image(self, input_filename):
        """
        Read file and run network centrality
        """
        img = Image.open(input_filename)
        image_array = pillow_to_numpy(sub_image)
        feature_vec, _ = subgraph_centrality(image_array)
        nc_result = feature_vector_metrics(feature_vec)
        return feature_vec, nc_result


    def process_single_date(self, date_string):
        input_path = os.path.join(self.input_dir, date_string, "SPLIT")
        input_files = [filename for filename in os.listdir(input_path) \
                       if "BWNDVI" in filename]


    def run(self):
        date_strings = os.listdir(self.input_path)
        for date_string in date_strings:
            self.process_single_date(date_string)
