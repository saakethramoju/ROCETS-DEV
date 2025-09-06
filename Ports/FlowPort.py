import cantera as ct
from ambiance import Atmosphere
from .Exceptions import (
    AlreadyConnectedError,
    InvalidConnectionError,
    FluidTypeError,
    ConnectionConflictError,
)


class FlowPort:
    """
    Base class for a flow connection point carrying a Cantera fluid.
    """

    _id_counter = 0  # for default naming

    def __init__(self, name=None, fluid=None, parent=None):
        # Default fluid: Water at 293.15 K and 1 atm
        if fluid is None:
            fluid = ct.Water()
            atm = Atmosphere(h=0)
            P = atm.pressure[0]
            T = atm.temperature[0]
            fluid.TP = T, P

        self._fluid = fluid
        self._mass_flow = 0  # default

        self.parent = parent
        self.connection = None

        if name:
            self.name = name
        else:
            FlowPort._id_counter += 1
            self.name = f"FlowPort_{FlowPort._id_counter}"

    # -----------------------------
    # Fluid property
    # -----------------------------
    @property
    def fluid(self):
        return self._fluid

    @fluid.setter
    def fluid(self, value):
        if value is not None and not isinstance(value, ct.ThermoPhase):
            raise FluidTypeError(value)
        self._fluid = value
        # propagate update if connected
        if self.connection:
            self.connection._fluid = value

    # -----------------------------
    # Mass flow property
    # -----------------------------
    @property
    def mass_flow(self):
        """Mass flow rate [kg/s] through this port."""
        return self._mass_flow

    @mass_flow.setter
    def mass_flow(self, value):
        if value is not None and not isinstance(value, (int, float)):
            raise TypeError("mass_flow must be a number or None")
        self._mass_flow = value
        # propagate update if connected
        if self.connection:
            self.connection._mass_flow = value

    # -----------------------------
    # Connection handling
    # -----------------------------
    def connect(self, other):
        """Connect this port to another port (enforcing type compatibility and fluid sync)."""
        if self.connection is not None:
            raise AlreadyConnectedError(self, self.connection)
        if other.connection is not None:
            raise AlreadyConnectedError(other, other.connection)

        if not self._can_connect_to(other):
            raise InvalidConnectionError(self, other)

        # Handle fluid synchronization
        if self.fluid is None and other.fluid is not None:
            self.fluid = other.fluid
        elif other.fluid is None and self.fluid is not None:
            other.fluid = self.fluid
        elif self.fluid is not None and other.fluid is not None:
            # OutFlow takes priority
            if isinstance(self, OutFlow):
                other.fluid = self.fluid
            elif isinstance(other, OutFlow):
                self.fluid = other.fluid
            else:
                raise ConnectionConflictError(self, other)

        # Handle mass flow synchronization
        if self.mass_flow is None and other.mass_flow is not None:
            self.mass_flow = other.mass_flow
        elif other.mass_flow is None and self.mass_flow is not None:
            other.mass_flow = self.mass_flow

        # Finalize connection
        self.connection = other
        other.connection = self

    def _can_connect_to(self, other):
        return isinstance(other, FlowPort)

    # -----------------------------
    # String representations
    # -----------------------------
    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name}>"

    def __str__(self):
        parent_name = self.parent.__class__.__name__ if self.parent else "None"
        conn_name = self.connection.name if self.connection else "None"
        fluid_name = self.fluid.name if self.fluid else "None"
        return (
            f"{self.__class__.__name__} '{self.name}'\n"
            f"  Parent: {parent_name}\n"
            f"  Fluid: {fluid_name}\n"
            f"  Mass flow: {self.mass_flow if self.mass_flow is not None else 'None'} kg/s\n"
            f"  Connected to: {conn_name}"
        )


class InFlow(FlowPort):
    def _can_connect_to(self, other):
        return isinstance(other, OutFlow)


class OutFlow(FlowPort):
    def _can_connect_to(self, other):
        return isinstance(other, InFlow)
