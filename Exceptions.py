class MissingConfigurationError(Exception):
    pass

class PortNotConnectedError(Exception):
    pass

#class NoInjectorError(Exception):
#    pass

class NoMatchingPortsError(Exception):
    pass

class PortTypeError(Exception):
    pass

class AmbiguousPortError(Exception):
    pass

class MissingConfigurationKeyError(KeyError):
    pass

class MissingConfigurationValueError(ValueError):
    pass

class MissingGuessError(Exception):
    pass

class MissingGuessKeyError(KeyError):
    pass


class MissingGuessValueError(ValueError):
    pass