"""
Definitions:
===========

A PIPELINE is the whole analysis procedure for one set of coordinates.
It will likely consist of a couple of SEQUENCES - e.g.
one for vegetation data and one for weather data.

A SEQUENCE is composed of one or more MODULES, that each do specific tasks,
e.g. download data, process images, calculate quantities from image.

A special type of MODULE may be placed at the end of a PIPELINE to combine
the results of the different SEQUENCES into one output file.
"""

import os
import json
import subprocess

from pyveg.src.file_utils import save_json
try:
    from pyveg.src import azure_utils
except:
    print("Azure utils could not be imported - is Azure SDK installed?")

class Pipeline(object):
    """
    A Pipeline contains all the Sequences we want to run on a particular
    set of coordinates and a date range. e.g. there might be one Sequence
    for vegetation data and one for weather data.
    """

    def __init__(self, name):
        self.name = name
        self.sequences = []
        self.coords = None
        self.date_range = None
        self.output_location = None
        self.output_location_type = None
        self.is_configured = False


    def __iadd__(self, sequence):
        """
        Overload the '+=' operator, so we can add sequences
        directly to the pipeline.
        """
        sequence.parent = self
        self.__setattr__(sequence.name, sequence)
        self.sequences.append(sequence)
        return self


    def __repr__(self):
        """
        overload the builtin operator for printing the pipeline.
        """
        output = "\n[Pipeline]: {} \n".format(self.name)
        output += "=======================\n"
        output += "coordinates: {}\n".format(self.coords)
        output += "date_range:  {}\n".format(self.date_range)
        output += "output_location:  {}\n".format(self.output_location)
        output += "output_location_type:  {}\n".format(self.output_location_type)
        output += "\n ------- Sequences ----------\n\n"
        for s in self.sequences:
            output += s.__repr__()
        output += "=======================\n"
        return output


    def get(self, seq_name):
        """
        Return a sequence object when asked for by name.
        """
        for sequence in self.sequences:
            if sequence.name == seq_name:
                return sequence


    def configure(self):
        """
        Configure all the sequences in this pipeline.
        """
        for var in ["coords", "date_range", "output_location","output_location_type"]:
            if (not var in vars(self)) or ( not self.__getattribute__(var)):
                raise RuntimeError("{}: need to set {} before calling configure()"\
                                   .format(self.name, var))
        if self.output_location_type == "azure":
            container_name = azure_utils.sanitize_container_name(self.output_location)
            print("Create container {}".format(container_name))
            azure_utils.create_container(container_name)
            self.output_location = container_name

        for sequence in self.sequences:
            if not "coords" in vars(sequence):
                sequence.coords = self.coords
            if not "date_range" in vars(sequence):
                sequence.date_range = self.date_range
            sequence.configure()


        self.is_configured = True


    def run(self):
        """
        run all the sequences in this pipeline
        """
        for sequence in self.sequences:
            sequence.run()



class Sequence(object):
    """
    A Sequence is a collection of modules where the output of one module is
    typically the input to the next one.
    It will typically correspond to a particular data collection, e.g. for
    vegetation imagery, we might have one module to download the images,
    one to process them, and one to analyze the processed images.
    """

    def __init__(self, name):
        self.name = name
        self.modules = []
        self.depends_on = []
        self.parent = None
        self.output_location = None
        self.output_location_type = None
        self.is_configured = False


    def __iadd__(self, module):
        """
        overload the += operator so we can add modules directly to the sequence
        """
        module.parent = self
        self.modules.append(module)
        # if the module doesn't already have a name, or only the default one,
        # give it a name <sequence_name>_<class_name> here
        if (not module.name) or (module.name == module.__class__.__name__):
            module.name = "{}_{}".format(self.name, module.__class__.__name__)
        # add module name as an attribute of the sequence
        self.__setattr__(module.name, module)

        return self


    def set_output_location(self):
        if self.parent:
            self.output_location = os.path.join(self.parent.output_location,
                                           f'gee_{self.coords[0]}_{self.coords[1]}'\
                                           +"_"+self.name.replace('/', '-'))
            self.output_location_type = self.parent.output_location_type
        else:
            self.output_location = f'gee_{self.coords[0]}_{self.coords[1]}'\
                +"_"+self.name.replace('/', '-')
            self.output_location_type = "local"


    def set_config(self, config_dict):
        for k, v in config_dict.items():
            print("{}: setting {} to {}".format(self.name, k,v))
            self.__setattr__(k,v)


    def configure(self):

        if (not self.coords) or (not self.date_range):
            raise RuntimeError("{}: Need to set coords and date range before calling configure()"\
                               .format(self.name))
        if not self.output_location:
            self.set_output_location()
        # set the input location for each module to be the output of the previous one.
        for i, module in enumerate(self.modules):
            module.output_location = self.output_location
            module.output_location_type = self.output_location_type
            if i>0:
                module.input_location = self.modules[i-1].output_location
                module.input_location_type = self.modules[i-1].output_location_type
            module.coords = self.coords
            module.date_range = self.date_range
            module.configure()
        self.is_configured = True


    def run(self):
        for module in self.modules:
            module.run()


    def __repr__(self):
        if not self.is_configured:
            return "Sequence not configured\n"

        output = "\n    [Sequence]: {} \n".format(self.name)
        output += "    =======================\n"
        for k, v in vars(self).items():
            # exclude the things we don't want to print
            if k == "name" or k == "modules" or k == "parent" or isinstance(v, BaseModule):
                continue
            output += "    {}: {}\n".format(k,v)
        output += "\n    ------- Modules ----------\n\n"
        for m in self.modules:
            output += m.__repr__()
        output += "    =======================\n\n"
        return output


    def get(self, mod_name):
        """
        Return a module object when asked for by name.
        """
        for module in self.modules:
            if module.name == mod_name:
                return module


class BaseModule(object):
    """
    A "Module" is a building block of a sequence - takes some input, does something
    (e.g. Downloads from GEE, processes some images, ...) and produces some output.
    The working directory for all modules within a sequence will be given by the sequence -
    modules may write output to subdirectories of this (e.g. for different dates), but what
    we call "output_location" will be the base directory common to all modules, and will contain
    info about the image collection name, and the coordinates.
    """
    def __init__(self, name=None):
        if name:
            self.name = name
        else:
            self.name = self.__class__.__name__
        self.params = []
        self.parent = None
        self.is_configured = False
        self.is_finished = False


    def set_parameters(self, config_dict):
        for k, v in config_dict.items():
            print("{}: setting {} to {}".format(self.name,k,v))
            self.__setattr__(k, v)


    def configure(self, config_dict=None):
        """
        Order of preference for configuriation:
        1) config_dict
        2) values held by the parent Sequence
        3) default values
        So we set them in reverse order here, so higher priorities will override.
        """
        self.set_default_parameters()

        if self.parent:
            for param, param_type in self.params:
                if param in vars(self.parent):
                    self.__setattr__(param, self.parent.__getattribute__(param))
        if config_dict:
            self.set_parameters(config_dict)

        self.check_config()
        self.is_configured = True


    def check_config(self):
        """
        Loop through list of parameters, which will each be a tuple (name, [allowed_types])
        and check that the parameter exists, and is of the correct type.
        """
        for param in self.params:
            if not param[0] in vars(self):
                raise RuntimeError("{}: {} needs to be set."\
                    .format(self.name, param[0]))
            val = self.__getattribute__(param[0])
            type_ok = False
            for param_type in param[1]:
                if isinstance(val, param_type):
                    type_ok = True
                    break
            if not type_ok:
                raise TypeError("{}: {} should be {}, got {}:{}"\
                                   .format(self.name,param[0],
                                           param[1],
                                           val,
                                           type(val)))
        return True


    def set_default_parameters(self):
        pass


    def run(self):
        if not self.is_configured:
            raise RuntimeError("Module {} needs to be configured before running".format(self.name))


    def __repr__(self):
        if not self.is_configured:
            return"\n        Module not configured"

        output = "        [Module]: {} \n".format(self.name)
        output += "        =======================\n"
        for k, v in vars(self).items():
            # exclude the things we don't want to print
            if k == "name" or k == "parent" or k == "params":
                continue
            output += "        {}: {}\n".format(k,v)
        output += "        =======================\n\n"
        return output


    def copy_to_output_location(self, tmpdir, output_location, file_endings=[]):
        """
        Copy contents of a temporary directory to a specified output location.

        Parameters
        ==========
        tmpdir: str, location of temporary directory
        output_location: str, either path to a local directory (if self.output_location_type is "local")
                              or to Azure <container>/<blob_path> if self.output_location_type=="azure")
        file_endings: list of str, optional.  If given, only files with those endings will be copied.
        """
        if self.output_location_type == "local":
            os.makedirs(output_location, exist_ok=True)
            for root, dirs, files in os.walk(tmpdir):
                for filename in files:
                    if file_endings:
                        for ending in file_endings:
                            if filename.endswith(ending):
                                subprocess.run(["cp","-r",os.path.join(root, filename),
                                                os.path.join(output_location, filename)])
                    else:
                        subprocess.run(["cp","-r",os.path.join(root, filename),
                                        os.path.join(output_location, filename)])
        elif self.output_location_type == "azure":
            # first part of self.output_location should be the container name
            container_name = self.output_location.split("/")[0]
            azure_utils.write_files_to_blob(tmpdir,
                                            container_name,
                                            output_location,
                                            file_endings)


    def list_directory(self, directory_path, location_type):
        """
        List contents of a directory, either on local file system
        or Azure blob storage.
        """
        if location_type == "local":
            return os.listdir(directory_path)
        elif location_type == "azure":
            # first part of self.output_location should be the container name
            container_name = self.output_location.split("/")[0]
            return azure_utils.list_directory(directory_path, container_name)
        else:
            raise RuntimeError("Unknown location_type - must be 'local' or 'azure'")


    def save_json(self, data, filename, location, location_type):
        """
        Save json to local filesystem or blob storage depending on location_type
        """
        if location_type == "local":
            save_json( data, location, filename)
        elif location_type == "azure":
            # first part of self.output_location should be the container name
            container_name = self.output_location.split("/")[0]
            azure_utils.save_json(data, location, filename, container_name)
        else:
            raise RuntimeError("Unknown location_type - must be 'local' or 'azure'")


    def get_json(self, filepath, location_type):
        """
        Read a json file either local or blob storage.
        """
        if location_type == "local":
            return json.load(open(filepath))
        elif location_type == "azure":
            # first part of filepath  should be the container name
            container_name = filepath.split("/")[0]
            return azure_utils.read_json(filepath, container_name)
        else:
            raise RuntimeError("Unknown location_type - must be 'local' or 'azure'")


    def get_file(self, filename, location_type):
        """
        Just return the filename if location _type is "local".
        Otherwise return a tempfile with the contents of a blob if the location
        is "azure".
        """
        if location_type == "local":
            return filename
        elif location_type == "azure":
            # first part of self.output_location should be the container name
            container_name = self.output_location.split("/")[0]
            return azure_utils.get_blob_to_tempfile(filename, container_name)
        else:
            raise RuntimeError("Unknown location_type - must be 'local' or 'azure'")


    def check_for_existing_files(self, location, num_files_expected):
        """
        See if there are already num_files in the specified location
        """
        # if we haven't specified number of expected files per point it will be -1
        if num_files_expected < 0:
            return False

        if self.output_location_type == "local":
            os.makedirs(location, exist_ok=True)
        existing_files = self.list_directory(location, self.output_location_type)
        if len(existing_files) == num_files_expected:
            print("{}: Already found {} files in {} - skipping"\
                  .format(self.name, num_files_expected, location))
            return True
        return False


    def save_config(self, config_location):
        """
        Write out the configuration of this module as a json file.
        """
        config_dict = {}
        for param, _ in self.params:
            config_dict[param] = self.__getattribute__(param)
        output_config_dir = os.path.dirname(config_location)
        if output_config_dir and not os.path.exists(output_config_dir):
            os.makedirs(output_config_dir)

        with open(config_location, "w") as output_json:
            json.dump(config_dict, output_json)
        print("{} wrote config to {}".format(self.name, config_location))
