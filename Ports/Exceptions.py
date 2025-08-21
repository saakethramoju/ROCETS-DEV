class FlowPortError(Exception):
    """Base class for all FlowPort-related errors."""
    pass


class AlreadyConnectedError(FlowPortError):
    """Raised when trying to connect a FlowPort that is already connected to another."""
    def __init__(self, port, connection):
        super().__init__(f"{port} is already connected to {connection}")


class InvalidConnectionError(FlowPortError):
    """Raised when attempting to connect incompatible port types (e.g. InFlow ↔ InFlow)."""
    def __init__(self, port_a, port_b):
        super().__init__(f"Cannot connect {port_a.__class__.__name__} to {port_b.__class__.__name__}")


class FluidTypeError(FlowPortError):
    """Raised when assigning an invalid fluid type (must be Cantera ThermoPhase)."""
    def __init__(self, value):
        super().__init__(f"Invalid fluid type: {type(value).__name__}. "
                         "Must be a Cantera ThermoPhase (e.g. ct.Solution, ct.Water).")


class ConnectionConflictError(FlowPortError):
    """Raised when both ports already have fluids but cannot resolve ownership."""
    def __init__(self, port_a, port_b):
        super().__init__(f"Fluid conflict: both {port_a.name} and {port_b.name} already "
                         f"have fluids, and neither is an OutFlow to take priority.")
