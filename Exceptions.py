class MissingConfigurationError(Exception):
    pass

class MissingConfigurationKeyError(KeyError):
    pass

class MissingConfigurationValueError(ValueError):
    pass

class MissingPortError(Exception):
    pass

class PortTypeError(Exception):
    pass

class MissingFlowPortError(Exception):
    """Raised when no flow ports are available to connect."""
    pass

class FlowPortTypeError(Exception):
    """Raised when trying to connect incompatible flow port types (e.g., two InFlows)."""
    pass

class MissingMixtureRatioError(Exception):
    pass

class MissingMassConservationEquation(Exception):
    pass