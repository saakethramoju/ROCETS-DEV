class MissingConfigurationError(Exception):
    pass

class PortNotConnectedError(Exception):
    pass

#class NoInjectorError(Exception):
#    pass

class MissingConfigurationKeyError(KeyError):
    pass

class MissingConfigurationValueError(ValueError):
    pass