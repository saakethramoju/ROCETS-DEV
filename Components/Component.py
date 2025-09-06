import difflib
from typing import List, Dict, Optional, TYPE_CHECKING, Union
from prettytable import PrettyTable
from Ports import InFlow, OutFlow
from .ComponentType import ComponentType
from .Value import State, Parameter, Substance


if TYPE_CHECKING:
    from System import System


class Component:
    """
    Base class for all components in the fluid network.
    """

    configuration_keys: List[str] = []
    state_keys: List[str] = []
    inflow_keys: List[str] = []
    outflow_keys: List[str] = [] 
    substance_keys: List[str] = [] 

    component_type: ComponentType = ComponentType.FLOW

    iteration_keys: List[str] = []

    def __init__(
        self,
        name: str,
        parent: Optional["System"] = None,
    ):
        self.name = name
        self.parent = parent
        self.inflows: List[InFlow] = []
        self.outflows: List[OutFlow] = []
        self._states: Dict[str, State] = {}
        self._parameters: Dict[str, Parameter] = {}
        self._substances: Dict[str, Substance] = {}

        self._initialize_default_ports()
        self._initialize_default_states()
        self._initialize_configuration()
        self._initialize_default_substances()


    # -----------------------------
    # Configuration
    # -----------------------------
    def _initialize_configuration(self):
        for key in self.configuration_keys:
            self.add_parameter(key)

    def add_parameter(self, name: str, initial=None):
        if name in self._parameters:
            raise ValueError(f"Parameter '{name}' already exists in {self.name}")
        self._parameters[name] = Parameter(name, initial)
        return self._parameters[name]
    
    @property
    def parameters(self):
        return list(self._parameters.keys())


    # -----------------------------
    # States
    # -----------------------------
    def _initialize_default_states(self):
        for key in self.state_keys:
            self.add_state(key)

    def add_state(self, name: str, initial=None):
        if name in self._states:
            raise ValueError(f"State '{name}' already exists in {self.name}")
        self._states[name] = State(name, initial)
        return self._states[name]

    @property
    def states(self):
        return list(self._states.keys())
    
    # -----------------------------
    # Substances
    # -----------------------------
    def _initialize_default_substances(self):
        for key in self.substance_keys:
            self.add_substance(key)

    def add_substance(self, name: str, initial=None):
        if name in self._substances:
            raise ValueError(f"Substance '{name}' already exists in {self.name}")
        self._substances[name] = Substance(name, initial)
        return self._substances[name]

    @property
    def substances(self):
        return list(self._substances.keys())

    # -----------------------------
    # Add inflow/outflow (factory style)
    # -----------------------------
    def add_inflow(self, name: Optional[str] = None, fluid=None) -> InFlow:
        port = InFlow(name=name, fluid=fluid, parent=self)
        self.inflows.append(port)
        return port

    def add_outflow(self, name: Optional[str] = None, fluid=None) -> OutFlow:
        port = OutFlow(name=name, fluid=fluid, parent=self)
        self.outflows.append(port)
        return port

    def _initialize_default_ports(self):
        """Create default ports from inflow_keys and outflow_keys."""
        for name in self.inflow_keys:
            self.add_inflow(name)
        for name in self.outflow_keys:
            self.add_outflow(name)

    # -----------------------------
    # Connect logic
    # -----------------------------
    def connect(
        self,
        other: Union["Component", str],
        other_comp: Optional["Component"] = None,
        other_port: Optional[str] = None,
    ):
        """
        Connect this component to another.

        Usage:
        - comp1.connect(comp2) -> fuzzy-match ports (or fallback to first available)
        - comp1.connect("Out1", comp2, "InA") -> connect specific ports with fuzzy matching
        """
        if isinstance(other, Component) and other_comp is None:
            # Case 1: bulk connect components with fuzzy port matching
            for outp in self.outflows:
                candidates = [p.name for p in other.inflows]
                if not candidates:
                    continue
                match = difflib.get_close_matches(outp.name, candidates, n=1, cutoff=0.6)
                if match:
                    other_port_obj = next(p for p in other.inflows if p.name == match[0])
                else:
                    # fallback: first available inflow
                    other_port_obj = other.inflows[0]
                outp.connect(other_port_obj)

            for inp in self.inflows:
                candidates = [p.name for p in other.outflows]
                if not candidates:
                    continue
                match = difflib.get_close_matches(inp.name, candidates, n=1, cutoff=0.6)
                if match:
                    other_port_obj = next(p for p in other.outflows if p.name == match[0])
                else:
                    # fallback: first available outflow
                    other_port_obj = other.outflows[0]
                inp.connect(other_port_obj)

        elif isinstance(other, str) and isinstance(other_comp, Component) and isinstance(other_port, str):
            # Case 2: connect explicit ports by fuzzy name
            my_port = self._fuzzy_find_port(other)
            other_port_obj = other_comp._fuzzy_find_port(other_port)

            if my_port is None:
                raise ValueError(f"No matching port found for '{other}' in {self.name}")
            if other_port_obj is None:
                raise ValueError(f"No matching port found for '{other_port}' in {other_comp.name}")

            my_port.connect(other_port_obj)

        else:
            raise TypeError("Invalid arguments for connect().")

    # -----------------------------
    # Fuzzy matching helpers
    # -----------------------------
    def _fuzzy_find_state(self, name: str) -> Optional[State]:
        candidates = list(self._states.keys())
        match = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
        if match:
            return self._states[match[0]]   # return the State object, not the string
        return None

    def _fuzzy_find_port(self, name: str):
        all_ports = self.inflows + self.outflows
        candidates = [p.name for p in all_ports]
        match = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
        if match:
            return next(p for p in all_ports if p.name == match[0])
        return None
        
    def _fuzzy_find_parameter(self, name: str) -> Optional[Parameter]:
        candidates = list(self._parameters.keys())
        match = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
        if match:
            return self._parameters[match[0]]
        return None

    def _fuzzy_find_substance(self, name: str) -> Optional[Substance]:   # NEW
        candidates = list(self._substances.keys())
        match = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
        if match:
            return self._substances[match[0]]
        return None

    # -----------------------------
    # Dict-like access: states + ports + parameters
    # -----------------------------

    def __getitem__(self, key: str):
        # States
        state = self._fuzzy_find_state(key)
        if state:
            return state
        # Parameters
        param = self._fuzzy_find_parameter(key)
        if param:
            return param
        # Substances
        substance = self._fuzzy_find_substance(key)
        if substance:
            return substance
        # Ports
        port = self._fuzzy_find_port(key)
        if port:
            return port
        raise KeyError(f"No state, parameter, substance, or port found for '{key}' in {self.name}")


    def __setitem__(self, key: str, value):
        # States
        state = self._fuzzy_find_state(key)
        if state:
            state.set(value)
            return
        # Parameters
        param = self._fuzzy_find_parameter(key)
        if param:
            param.set(value)
            return
        # Substances
        substance = self._fuzzy_find_substance(key)
        if substance:
            substance.set(value)
            return
        # Ports
        port = self._fuzzy_find_port(key)
        if port:
            if hasattr(value, "TP"):  # Cantera fluid
                port.fluid = value
            elif isinstance(value, (int, float)) or value is None:
                port.mass_flow = value
            else:
                raise TypeError("Value must be a Cantera fluid, a number (mass flow), or None.")
            return
        raise KeyError(f"No state, parameter, substance, or port found for '{key}' in {self.name}")

    # -----------------------------
    # Equality & hashing (needed for set operations in System)
    # -----------------------------
    def __eq__(self, other):
        return isinstance(other, Component) and id(self) == id(other)

    def __hash__(self):
        return id(self)

    # -----------------------------
    # Representations
    # -----------------------------
    def __repr__(self):
        return f"<Component {self.name} type={self.component_type.name}>"

    def __str__(self):
        parent_name = self.parent.name if self.parent else "None"

        table = PrettyTable()
        table.field_names = [
            "Port Name",
            "Direction",
            "Connected To",
            "Connected Port",
            "Fluid",
            "T [K]",
            "P [Pa]",
            "Q (quality)",
            "ṁ [kg/s]",
        ]

        for port in self.inflows + self.outflows:
            conn_comp = port.connection.parent.name if (port.connection and port.connection.parent) else "None"
            conn_port = port.connection.name if port.connection else "None"

            fluid = port.fluid
            fluid_name = fluid.name if fluid else "None"
            T = f"{fluid.T:.2f}" if fluid else "None"
            P = f"{fluid.P:.1f}" if fluid else "None"
            Q = getattr(fluid, "Q", None)
            Q = f"{Q:.2f}" if Q is not None else "N/A"

            m_dot = f"{port.mass_flow:.4f}" if port.mass_flow is not None else "None"

            table.add_row(
                [
                    port.name,
                    "InFlow" if isinstance(port, InFlow) else "OutFlow",
                    conn_comp,
                    conn_port,
                    fluid_name,
                    T,
                    P,
                    Q,
                    m_dot,
                ]
            )

        return (
            f"Component '{self.name}' (Type: {self.component_type.name}, Parent: {parent_name})\n"
            f"{table}"
        )

    def steady_state(self):
        return 0
    
    def transient(self):
        return 0