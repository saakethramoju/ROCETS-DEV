class PortPermissionError(PermissionError):
    """Raised when a value is set on a port that doesn't allow writes."""
    pass

class PortNotFoundError(KeyError):
    """Raised when a port name lookup fails in a Component."""
    pass

class PortKeyError(AttributeError):
    """Raised when a port set call references a nonexistent attribute."""
    pass

class PortConnectionError(Exception):
    """Raised when ports are already connected and connection is attempted again."""
    pass
