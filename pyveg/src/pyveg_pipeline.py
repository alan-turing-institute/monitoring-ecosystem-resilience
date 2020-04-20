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


class Pipeline(object):

    def __init__(self, name):
        self.name = name
        self.sequences = []
        self.coords = None
        self.date_range = None
        self.output_basedir = None

    def __iadd__(self, sequence):
        sequence.parent = self
        self.sequences.append(sequence)
        return self


    def configure(self):
        for var in ["coords", "date_range", "output_dir"]:
            if not var in vars(self):
                raise RuntimeError("{}: need to set {} before calling configure()"\
                                   .format(self.name, var))

        for sequence in self.sequences:
            sequence.configure()

    def run(self):
        for sequence in self.sequences:
            sequence.run()



class Sequence(object):

    def __init__(self, name):
        self.name = name
        self.modules = []
        self.depends_on = []
        self.parent = None
        self.output_dir = None


    def __iadd__(self, module):
        module.parent = self
        self.modules.append(module)
        return self


    def configure(self):
        if self.parent and not self.output_dir:
            self.output_dir = os.path.join(self.parent.output_dir,
                                           self.name)
        for module in self.modules:
            module.configure()

    def run(self):
        for module in self.modules:
            module.run()



class BaseModule(object):

    def __init__(self, name):
        self.name = name
        self.params = []
        self.parent = None


    def configure(self, config_dict=None):
        self.set_default_parameters()
        if config_dict:
            for k, v in config_dict.items():
                print("{}: setting {} to {}".format(self.name,k,v))
                self.__setattr(k, v)
        if self.parent and not self.output_dir:
            self.output_dir = self.parent.output_dir
        self.check_config()


    def check_config(self):
        """
        Loop through list of parameters, which will each be a tuple (name, type)
        and check that the parameter exists, and is of the correct type.
        """
        for param in self.params:
            if not param[0] in vars(self):
                raise RuntimeError("{}: {} needs to be set."\
                    .format(self.name, param[0]))
            if not isinstance(self.__getattribute__(param[0]), param[1]):
                raise TypeError("{}: {} should be a {}"\
                                   .format(self.name,param[0],
                                                     param[1]))
        return True


    def set_default_parameters(self):
        pass


    def run(self):
        raise RuntimeError("This method needs to be implemented in concrete class")
