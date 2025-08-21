import difflib
from typing import List, Optional, TYPE_CHECKING, Union
from prettytable import PrettyTable
from Ports import InFlow, OutFlow
from .ComponentType import ComponentType

if TYPE_CHECKING:
    from ..Scrapped.System import System


class Component:
    """
    Base class for all components in the fluid network.
    """

    configuration_keys: List[str] = []

    def __init__(
        self,
        name: str,
        parent: Optional["System"] = None,
        type_: ComponentType = ComponentType.FLOW,
    ):
        self.name = name
        self.parent = parent
        self.inflows: List[InFlow] = []
        self.outflows: List[OutFlow] = []
        self.configuration: dict = {}
        self.type = type_

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
        - comp1.connect(comp2) -> fuzzy-match ports
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
                    outp.connect(other_port_obj)

            for inp in self.inflows:
                candidates = [p.name for p in other.outflows]
                if not candidates:
                    continue
                match = difflib.get_close_matches(inp.name, candidates, n=1, cutoff=0.6)
                if match:
                    other_port_obj = next(p for p in other.outflows if p.name == match[0])
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

    def _fuzzy_find_port(self, name: str):
        """Find a port (inflow or outflow) by fuzzy name matching."""
        all_ports = self.inflows + self.outflows
        candidates = [p.name for p in all_ports]
        match = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
        if match:
            return next(p for p in all_ports if p.name == match[0])
        return None

    # -----------------------------
    # Dict-like access to ports
    # -----------------------------
    def __getitem__(self, key: str):
        port = self._fuzzy_find_port(key)
        if port is None:
            raise KeyError(f"No matching port found for '{key}' in {self.name}")
        return port

    def __setitem__(self, key: str, value):
        port = self._fuzzy_find_port(key)
        if port is None:
            raise KeyError(f"No matching port found for '{key}' in {self.name}")
        # If you pass in a Cantera fluid, set fluid. Otherwise try setting mass_flow.
        if hasattr(value, "TP"):  # crude check if it's a Cantera fluid
            port.fluid = value
        elif isinstance(value, (int, float)) or value is None:
            port.mass_flow = value
        else:
            raise TypeError("Value must be a Cantera fluid, a number (mass flow), or None.")

    # -----------------------------
    # Representations
    # -----------------------------
    def __repr__(self):
        return f"<Component {self.name} type={self.type.name}>"

    def __str__(self):
        parent_name = self.parent.__class__.__name__ if self.parent else "None"

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
            "ṁ [kg/s]",   # <-- new column
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
            f"Component '{self.name}' (Type: {self.type.name}, Parent: {parent_name})\n"
            f"{table}"
        )
