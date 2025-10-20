from Fluids import Fluid
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
    _default_fluid = Fluid("Water", P=101325, T=288.15)
    _id_counter = 0
    _instances = []   # track all ports

    def __init__(self, name=None, fluid=None, parent=None):
        if fluid is None:
            self._fluid = FlowPort._default_fluid
            self._using_default = True
        else:
            self._fluid = fluid
            self._using_default = False

        self._mass_flow = 0
        self.parent = parent
        self.connection = None

        if name:
            self.name = name
        else:
            FlowPort._id_counter += 1
            self.name = f"FlowPort_{FlowPort._id_counter}"

        FlowPort._instances.append(self)

    # -----------------------------
    # Class-level default setter
    # -----------------------------
    @classmethod
    def set_default_fluid(cls, fluid: Fluid):
        """Update global default and propagate to ports still using default."""
        if not isinstance(fluid, Fluid):
            raise TypeError(f"Default fluid must be a Fluid instance, got {type(fluid)}")

        cls._default_fluid = fluid
        for port in cls._instances:
            if port._using_default:
                port._fluid = fluid


    @classmethod
    def get_default_fluid(cls):
        """Return the current default fluid."""
        return cls._default_fluid

    # -----------------------------
    # Fluid property
    # -----------------------------
    @property
    def fluid(self):
        return self._fluid

    @fluid.setter
    def fluid(self, value):
        if not isinstance(value, Fluid):
            raise TypeError(f"Expected Fluid, got {type(value)}")
        self._fluid = value
        self._using_default = False   # user explicitly set fluid
        if self.connection:
            self.connection._fluid = value
            self.connection._using_default = False


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
        fluid_name = ", ".join(self.fluid.species) if self.fluid else "None"
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
