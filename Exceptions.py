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

class MissingGuessError(Exception):
    pass

class MissingGuessKeyError(KeyError):
    pass


class MissingGuessValueError(ValueError):
    pass