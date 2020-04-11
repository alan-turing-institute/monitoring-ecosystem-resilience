
class BaseModule(object):

    def __init__(self, name):
        self.name = name
        self.params = []


    def config_from_dict(self, config_dict):
        for k, v in config_dict.items():
            print("{}: setting {} to {}".format(self.name,k,v))
            self.__setattr(k, v)


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


    def run(self):
        raise RuntimeError("This method needs to be implemented in concrete class")
