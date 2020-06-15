"""
Class for holding analysis modules
that can be chained together to build a sequence.
"""

import os
import re
import shutil
import tempfile
import time

import cv2 as cv

from multiprocessing import Pool

from pyveg.src.image_utils import *
from pyveg.src.file_utils import *
from pyveg.src.coordinate_utils import *
from pyveg.src.date_utils import *
from pyveg.src.subgraph_centrality import (
    subgraph_centrality,
    feature_vector_metrics,
)
from pyveg.src import azure_utils
from pyveg.src import batch_utils
from pyveg.src.pyveg_pipeline import BaseModule


class ProcessorModule(BaseModule):

    def __init__(self, name):
        super().__init__(name)
        self.params += [
            ("replace_existing_files", [bool]),
            ("input_location", [str]),
            ("input_location_subdirs", [list, tuple]),
            ("output_location_subdirs", [list, tuple]),
            ("input_location_type", [str]),
            ("output_location", [str]),
            ("output_location_type", [str]),
            ("num_files_per_point", [int]),
            ("dates_to_process", [list, tuple]),
            ("run_mode", [str]), # batch or local
            ("n_batch_tasks", [int])
        ]


    def set_default_parameters(self):
        """
        Set some basic defaults.  Note that these might get overriden
        by a parent Sequence, or by calling configure() with a dict of values
        """
        super().set_default_parameters()
        if not "replace_existing_files" in vars(self):
            self.replace_existing_files = False
        if not "num_files_per_point" in vars(self):
            self.num_files_per_point = -1
        if not "input_location_type" in vars(self):
            self.input_location_type = "local"
        if not "output_location_type" in vars(self):
            self.output_location_type = "local"
        if not "dates_to_process" in vars(self):
            self.dates_to_process = []
        if not "run_mode" in vars(self):
            self.run_mode = "local"
        if not "n_batch_tasks" in vars(self):
            self.n_batch_tasks = 100
        if not "batch_task_dict" in vars(self):
            self.batch_task_dict = {}


    def check_input_data_exists(self, date_string):
        """
        Processor modules will look for inputs in
        <input_location>/<date_string>/<input_location_subdirs>
        Check that the subdirs exist and are not empty.

        Parameters
        ==========
        date_string: str, format YYYY-MM-DD

        Returns
        =======
        True if input directories exist and are not empty, False otherwise.
        """
        for i in range(len(self.input_location_subdirs)):
            if not self.input_location_subdirs[i] in self.list_directory(
                    os.path.join(self.input_location, date_string,
                                 *(self.input_location_subdirs[:i])),
                    self.input_location_type):
                return False
            if len(self.list_directory(
                    os.path.join(self.input_location, date_string,
                                 *(self.input_location_subdirs)),
                    self.input_location_type)) == 0:
                return False
        return True


    def check_output_data_exists(self, date_string):
        """
        Processor modules will write output to
        <output_location>/<date_string>/<output_location_subdirs>
        Check

        Parameters
        ==========
        date_string: str, format YYYY-MM-DD

        Returns
        =======
        True if expected number of output files are already in output location,
             AND self.replace_existing_files is set to False
        False otherwise
        """
        output_location = os.path.join(self.output_location,
                                       date_string,
                                       *(self.output_location_subdirs))
        return self.check_for_existing_files(output_location, self.num_files_per_point)


    def get_image(self, image_location):
        if self.input_location_type == "local":
            return Image.open(image_location)
        elif self.input_location_type == "azure":
            # container name will be the first bit of self.input_location.
            container_name = self.input_location.split("/")[0]
            return azure_utils.read_image(image_location, container_name)
        else:
            raise RuntimeError("Unknown output location type {}"\
                               .format(self.output_location_type))


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


    def run(self):
        super().run()
        job_status={}
        if self.run_mode == "local":
            job_status = self.run_local()
        elif self.run_mode == "batch":
            job_status = self.run_batch()
        else:
            raise RuntimeError("{}: Unknown run_mode {} - must be 'local' or 'batch'"\
                               .format(self.name, self.run_mode))
        return job_status


    def run_local(self):
        """
        loop over dates and call process_single_date on all of them.
        """
        print("{}: Running local".format(self.name))
        if "dates_to_process" in vars(self) and len(self.dates_to_process) > 0:
            date_strings = self.dates_to_process
        else:
            date_strings = sorted(self.list_directory(self.input_location,
                                                      self.input_location_type))
        num_succeeded = 0
        num_failed = 0
        for date_string in date_strings:
            print("date string {} input exists {} output exists {}"\
                  .format(date_string,
                  self.check_input_data_exists(date_string),
                  self.check_output_data_exists(date_string)))

            if self.check_input_data_exists(date_string) and \
               not self.check_output_data_exists(date_string):
                succeeded = self.process_single_date(date_string)
                if succeeded:
                    num_succeeded += 1
                else:
                    num_failed += 1
        self.is_finished = True
        return {
            "Succeeded": num_succeeded,
            "Failed": num_failed
        }


    def get_dependent_batch_tasks(self):
        """
        When running in batch, we are likely to depend on tasks submitted by
        the previous Module in the Sequence.  This Module should be in the
        "depends_on" attribute of this one.

        Task dependencies will be a dict of format
        {"task_id": <task_id>, "date_range": [<dates>]}
        """
        ready_to_submit = True
        task_dependencies = {}
        if len(self.depends_on) > 0:
            for dependency in self.depends_on:
                if not (self.parent and self.parent.get(dependency)):
                    print("{} couldn't retrieve dependency {}".format(self.name. dependency))
                    continue
                dependency_module = self.parent.get(dependency)
                if (not "run_mode" in vars(dependency_module)) or \
                   dependency_module.run_mode == "local":
                    continue
                while not dependency_module.all_tasks_submitted:
                    print("{}: waiting for {} to submit all batch tasks".format(self.name, dependency_module.name))
                    print('.', end='')
                    sys.stdout.flush()
                    time.sleep(1)
                task_dependencies.update(dependency_module.batch_task_dict)
        return task_dependencies


    def create_task_dict(self, task_id, date_list, dependencies=[]):
        config = self.get_config()
        config["dates_to_process"] = date_list
        # reset run_mode so that the batch jobs won't try to generate more batch jobs!
        config["run_mode"] = "local"
        config["input_location_type"]="azure"
        task_dict = {"depends_on": dependencies,
                     "task_id": task_id,
                     "config": config
        }
        return task_dict


    def run_batch(self):
        """"
        Write a config json file for each set of dates.
        If this module depends on another module running in batch, we first
        get the tasks on which this modules tasks will depend on.
        If not, we look at the input dates subdirectories and divide them up
        amongst the number of batch nodes.

        We want to create a list of dictionaries
        [{"task_id": <task_id>, "config": <config_dict>, "depends_on": [<task_ids>]}]
        to pass to the batch_utils.submit_tasks function.
        """

        print("{} running in batch!".format(self.name))
        self.all_tasks_submitted = False

        task_dicts = []
        task_dependencies = self.get_dependent_batch_tasks()


        if len(task_dependencies) == 0:
            # divide up the dates
            date_strings = sorted(self.list_directory(self.input_location,
                                                      self.input_location_type))
            print("number of date strings in input location {}".format(len(date_strings)))
            date_strings = [ds for ds in date_strings if self.check_input_data_exists(ds)]
            print("number of date strings with input data {}".format(len(date_strings)))
            date_strings = [ds for ds in date_strings if not self.check_output_data_exists(ds)]
            print("number of date strings without output data {}".format(len(date_strings)))
            # split these dates up over the batch tasks
            dates_per_task = assign_dates_to_tasks(date_strings, self.n_batch_tasks)
            # create a config dict for each task - will correspond to configuration for an
            # instance of this Module.

            for i in range(len(dates_per_task)):
                task_dict = self.create_task_dict("{}_{}".format(self.name, i), dates_per_task[i])
                task_dicts.append(task_dict)
        else:
            # we have a bunch of tasks from the previous Module in the Sequence
            for i, (k, v) in enumerate(task_dependencies.items()):
                # key k will be the task_id of the old task.  v will be the list of dates.
                task_dict = self.create_task_dict("{}_{}".format(self.name, i), v, [k])
                task_dicts.append(task_dict)
        # Take the job_id from the parent Sequence if there is one
        if self.parent and self.parent.batch_job_id:
            job_id = self.parent.batch_job_id
        else:
            # otherwise create a new job_id just for this module
            job_id = self.name +"_"+time.strftime("%Y-%m-%d_%H-%M-%S")
        submitted_ok = batch_utils.submit_tasks(task_dicts, job_id)
        if submitted_ok:
            # store the task dict so any dependent modules can query it
            self.task_dict = {td["task_id"]: td["config"]["dates_to_process"] for td in task_dicts}
            self.all_tasks_submitted = True
        return submitted_ok


    def check_if_finished(self):
        if self.run_mode == "local":
            return self.is_finished
        elif self.parent and self.parent.batch_job_id:
            job_id = self.parent.batch_job_id
            num_incomplete, num_success, num_failed = batch_utils.check_tasks_status(job_id)
            self.is_finished =  (num_incomplete == 0)
        return self.is_finished


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
        self.params += [
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
            self.region_size = 0.08
        if not "RGB_bands" in vars(self):
            self.RGB_bands = ["B4","B3","B2"]
        if not "split_RGB_images" in vars(self):
            self.split_RGB_images = True
        # in PROCESSED dir we expect RGB. NDVI, BWNDVI
        self.num_files_per_point = 3
        self.input_location_subdirs = ["RAW"]
        self.output_location_subdirs = ["PROCESSED"]


    def construct_image_savepath(self, date_string, coords_string, image_type="RGB"):
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


    def process_single_date(self, date_string):
        """
        For a single set of .tif files corresponding to a date range
        (normally a sub-range of the full date range for the pipeline),
        construct RGB, and NDVI greyscale images.
        Then do processing and thresholding to make black+white NDVI images.
        Split the RGB and black+white NDVI ones into small (50x50pix)
        sub-images.

        Parameters
        ==========
        date_string: str, format YYYY-MM-DD

        Returns
        =======
        True if everything was processed and saved OK, False otherwise.
        """
        # first see if there are already files in the output location
        # (in which case we can skip this date)

        # normally the coordinates will be part of the file path
        coords_string = find_coords_string(self.input_location)
        # if not though, we might have coords set explicitly
        if (not coords_string) and "coords" in vars(self):
            coords_string = "{}_{}".format(self.coords[0],self.coords[1])

        if not coords_string and date_string:
            raise RuntimeError("{}: coords and date need to be defined, through file path or explicitly set")

        output_location = os.path.dirname(self.construct_image_savepath(date_string,
                                                                        coords_string))
        if (not self.replace_existing_files) and \
           self.check_for_existing_files(output_location, self.num_files_per_point):
            return True

        # If no files already there, proceed.
        input_filepath = os.path.join(self.input_location,
                                      date_string,
                                      *(self.input_location_subdirs))
        print("{} processing files in {}".format(self.name, input_filepath))
        filenames = [filename for filename in self.list_directory(input_filepath,
                                                                  self.input_location_type) \
                     if filename.endswith(".tif")]
        if len(filenames) == 0:
            return

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

        print(filenames)
        tif_filebase = os.path.join(input_filepath, filenames[0].split('.')[0])

        # save the rgb image
        rgb_ok = self.save_rgb_image(band_dict,
                                     date_string,
                                     coords_string)
        if not rgb_ok:
            print("Problem with the rgb image?")
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



class WeatherImageToJSON(ProcessorModule):
    """
    Read the weather-related tif files downloaded from GEE, and
    write the temp and precipitation values out as a JSON file.
    """

    def __init__(self, name=None):
        super().__init__(name)


    def set_default_parameters(self):
        """
        Set some basic defaults.  Note that these might get overriden
        by a parent Sequence, or by calling configure() with a dict of values
        """
        super().set_default_parameters()
        self.input_location_subdirs = ["RAW"]
        self.output_location_subdirs = ["JSON","WEATHER"]


    def process_single_date(self, date_string):
        """
        Read the tif files downloaded from GEE and extract the values
        (should be the same for all pixels in the image, so just take mean())

        Parameters
        ----------
        date_string: str, format "YYYY-MM-DD"

        """
        metrics_dict = {}
        # if we are given a list of date strings to process, and this isn't
        # one of them, skip it.
        if self.dates_to_process and not date_string in self.dates_to_process:
            print("{} will not process date {}".format(self.name, date_string))
            return True
        print("Processing date {}".format(date_string))
        input_location = os.path.join(self.input_location, date_string,
                                      *(self.input_location_subdirs))
        for filename in self.list_directory(input_location, self.input_location_type):
            if filename.endswith(".tif"):
                name_variable = (filename.split('.'))[1]
                variable_array = cv.imread(self.get_file(os.path.join(input_location,
                                                                      filename),
                                                         self.input_location_type),
                                           cv.IMREAD_ANYDEPTH)

                metrics_dict[name_variable] = variable_array.mean()\
                                                            .astype(np.float64)
        self.save_json(metrics_dict, "weather_data.json",
                       os.path.join(self.output_location,
                                    date_string,
                                    *(self.output_location_subdirs)),
                       self.output_location_type)
        return True



#######################################################################

def process_sub_image(i, input_filepath, output_location, date_string, coords_string):
    """
    Read file and run network centrality
    """

    # open BWNDVI image
    sub_image = Image.open(input_filepath)
    image_array = pillow_to_numpy(sub_image)

    # run network centrality
    feature_vec, _ = subgraph_centrality(image_array)

    # format coords
    coords = [round(float(c), 4) for c in coords_string.split("_")]

    # store results in a dict
    nc_result = feature_vector_metrics(feature_vec)
    nc_result['feature_vec'] = list(feature_vec)
    nc_result['date'] = date_string
    nc_result['latitude'] = coords[1]
    nc_result['longitude'] = coords[0]

    # save individual result for sub-image to tmp json, will combine later.
    save_json(nc_result, output_location,
              f"network_centrality_sub{i}.json", verbose=False)

    # count and print how many sub-images we have done.
    n_processed = len(os.listdir(output_location))
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
            ("n_threads", [int]),
            ("n_sub_images", [int]),
        ]


    def set_default_parameters(self):
        """
        Default values. Note that these can be overridden by parent Sequence
        or by calling configure().
        """
        super().set_default_parameters()
        if not "n_threads" in vars(self):
            self.n_threads = 4
        if not "n_sub_images" in vars(self):
            self.n_sub_images = -1 # do all-sub-images
        self.num_files_per_point = 1
        self.input_location_subdirs = ["SPLIT"]
        self.output_location_subdirs = ["JSON","NC"]


    def check_sub_image(self, ndvi_filename, input_path):
        """
        Check the RGB sub-image corresponding to this NDVI image
        looks OK.
        """
        rgb_filename = re.sub("BWNDVI","RGB",ndvi_filename)
        rgb_img = Image.open(self.get_file(os.path.join(input_path, rgb_filename),
                                           self.input_location_type))
        img_ok = check_image_ok(rgb_img, 0.05)
        return img_ok


    def process_single_date(self, date_string):
        """
        Each date will have a subdirectory called 'SPLIT' with ~400 BWNDVI
        sub-images.
        """
        print("{}: processing {}".format(self.name, date_string))
        # if we are given a list of date strings to process, and this isn't
        # one of them, skip it.
        if self.dates_to_process and not date_string in self.dates_to_process:
            print("{} will not process date {}".format(self.name, date_string))
            return True
        # see if there is already a network_centralities.json file in
        # the output location - if so, skip
        output_location = os.path.join(self.output_location, date_string,
                                       *(self.output_location_subdirs))
        if (not self.replace_existing_files) and \
           self.check_for_existing_files(output_location, self.num_files_per_point):
            return True

        input_path = os.path.join(self.input_location,
                                  date_string,
                                  *(self.input_location_subdirs))
        all_input_files = self.list_directory(input_path, self.input_location_type)

        # list all the "BWNDVI" sub-images where RGB image passes quality check
        input_files = [filename for filename in all_input_files \
                       if "BWNDVI" in filename and \
                       self.check_sub_image(filename, input_path)]
        if len(input_files) == 0:
            print("{}: No sub-images for date {}".format(self.name,
                                                         date_string))
            return
        else:
            print("{} found {} sub-images".format(self.name, len(input_files)))
        tmp_json_dir = tempfile.mkdtemp()

        # if we only want a subset of sub-images, truncate the list here
        if self.n_sub_images > 0:
            input_files = input_files[:self.n_sub_images]

        # create a multiprocessing pool to handle each sub-image in parallel
        with Pool(processes=self.n_threads) as pool:
            # prepare the arguments for the process_sub_image function
            arguments=[(i,
                        self.get_file(os.path.join(input_path,filename),
                                      self.input_location_type),
                        tmp_json_dir,
                        date_string,
                        find_coords_string(filename)) \
                   for i, filename in enumerate(input_files)]
            pool.starmap(process_sub_image, arguments)
        # put all the output json files for subimages together into one for this date
        print("\n Consolidating json from all subimages")
        all_subimages = consolidate_json_to_list(tmp_json_dir)
        self.save_json(all_subimages, "network_centralities.json",
                       output_location,
                       self.output_location_type)
        shutil.rmtree(tmp_json_dir)
        return True


class NDVICalculator(ProcessorModule):
    """
    Class to look at NDVI on sub-images
    images, and return the results as json.
    Note that the input directory is expected to be the level above
    the subdirectories for the date sub-ranges.
    """

    def __init__(self, name=None):
        super().__init__(name)
        self.params += [
            ("n_sub_images", [int])
        ]


    def set_default_parameters(self):
        """
        Default values. Note that these can be overridden by parent Sequence
        or by calling configure().
        """
        super().set_default_parameters()
        if not "n_sub_images" in vars(self):
            self.n_sub_images = -1 # do all-sub-images
        self.num_files_per_point = 1
        self.input_location_subdirs = ["SPLIT"]
        self.output_location_subdirs = ["JSON","NDVI"]


    def check_sub_image(self, ndvi_filename, input_path):
        """
        Check the RGB sub-image corresponding to this NDVI image
        looks OK.
        """
        rgb_filename = re.sub("NDVI","RGB",ndvi_filename)
        rgb_img = self.get_image(os.path.join(input_path, rgb_filename))

        img_ok = check_image_ok(rgb_img, 0.05)
        return img_ok


    def process_sub_image(self, ndvi_filepath, date_string, coords_string):
        """
        Calculate mean and standard deviation of NDVI in a sub-image,
        both with and without masking out non-vegetation pixels.
        """

        # open NDVI images
        ndvi_sub_image = self.get_image(ndvi_filepath)
        ndvi_image_array = pillow_to_numpy(ndvi_sub_image)
        bwndvi_sub_image = self.get_image(ndvi_filepath.replace('NDVI', 'BWNDVI'))
        bwndvi_image_array = pillow_to_numpy(bwndvi_sub_image)

        # get average NDVI across the whole image (in case there is no patterned veg)
        ndvi_mean = round(ndvi_image_array.mean(), 4)

        # use the BWDVI to mask the NDVI and calculate the average pixel value of veg pixels
        veg_mask = (bwndvi_image_array == 0)
        if veg_mask.sum() > 0:
            ndvi_veg_mean = ndvi_image_array[veg_mask].mean()
        else:
            ndvi_veg_mean = np.NaN

        # format coords
        coords = [round(float(c), 4) for c in coords_string.split("_")]

        # store results in a dict
        ndvi_result = {}
        ndvi_result['date'] = date_string
        ndvi_result['latitude'] = coords[1]
        ndvi_result['longitude'] = coords[0]
        ndvi_result['ndvi'] = ndvi_mean
        ndvi_result['ndvi_veg'] = ndvi_veg_mean

        return ndvi_result


    def process_single_date(self, date_string):
        """
        Each date will have a subdirectory called 'SPLIT' with ~400 NDVI
        sub-images.
        """
        # if we are given a list of date strings to process, and this isn't
        # one of them, skip it.
        if self.dates_to_process and not date_string in self.dates_to_process:
            print("{} will not process date {}".format(self.name, date_string))
            return True

        # see if there is already a ndvi.json file in
        # the output location - if so, skip
        output_location = os.path.join(self.output_location, date_string,
                                       *(self.output_location_subdirs))
        if (not self.replace_existing_files) and \
           self.check_for_existing_files(output_location, self.num_files_per_point):
            return True

        input_path = os.path.join(self.input_location, date_string,
                                  *(self.input_location_subdirs))
        all_input_files = self.list_directory(input_path, self.input_location_type)
        print("input path is {}".format(input_path))

        # list all the "NDVI" sub-images where RGB image passes quality check
        input_files = [filename for filename in all_input_files \
                       if "_NDVI" in filename and \
                       self.check_sub_image(filename, input_path)]

        if len(input_files) == 0:
            print("{}: No sub-images for date {}".format(self.name,
                                                         date_string))
            return
        else:
            print("{} found {} sub-images".format(self.name, len(input_files)))
        # if we only want a subset of sub-images, truncate the list here
        if self.n_sub_images > 0:
            input_files = input_files[:self.n_sub_images]

        ndvi_vals = []
        for ndvi_file in input_files:
            coords_string = find_coords_string(ndvi_file)
            ndvi_dict = self.process_sub_image(os.path.join(input_path, ndvi_file),
                                               date_string, coords_string)
            ndvi_vals.append(ndvi_dict)

        self.save_json(ndvi_vals, "ndvi_values.json",
                       output_location,
                       self.output_location_type)

        return True
