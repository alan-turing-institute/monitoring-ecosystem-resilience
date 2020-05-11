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
        self.output_dir = None
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
        output += "output_dir:  {}\n".format(self.output_dir)
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
        for var in ["coords", "date_range", "output_dir"]:
            if (not var in vars(self)) or ( not self.__getattribute__(var)):
                raise RuntimeError("{}: need to set {} before calling configure()"\
                                   .format(self.name, var))

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
        self.output_dir = None
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


    def set_output_dir(self):
        if self.parent:
            self.output_dir = os.path.join(self.parent.output_dir,
                                           f'gee_{self.coords[0]}_{self.coords[1]}'\
                                           +"_"+self.name.replace('/', '-'))
        else:
            self.output_dir = f'gee_{self.coords[0]}_{self.coords[1]}'\
                +"_"+self.name.replace('/', '-')


    def set_config(self, config_dict):
        for k, v in config_dict.items():
            print("{}: setting {} to {}".format(self.name, k,v))
            self.__setattr__(k,v)



    def configure(self):

        if (not self.coords) or (not self.date_range):
            raise RuntimeError("{}: Need to set coords and date range before calling configure()"\
                               .format(self.name))
        if not self.output_dir:
            self.set_output_dir()

        for i, module in enumerate(self.modules):
            module.output_dir = self.output_dir
            if i>0:
                module.input_dir = self.modules[i-1].output_dir
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
    we call "output_dir" will be the base directory common to all modules, and will contain
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


    def configure(self, config_dict=None):
        """
        Order of preference for configuring:
        1) configuration dictionary
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
            for k, v in config_dict.items():
                print("{}: setting {} to {}".format(self.name,k,v))
                self.__setattr__(k, v)

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
