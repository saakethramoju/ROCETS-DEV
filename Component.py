
import difflib
import re
from typing import List, Union
from prettytable import PrettyTable
from Exceptions import (PortNotFoundError, PortPermissionError, PortKeyError,
                        PortConnectionError)
from FlowPort import FlowPort, InFlow, OutFlow
from PropertyPort import PropertyPort, PropertyIn, PropertyOut
from Fluid import Fluid

class Component:
    def __init__(self, name: str):
        self.name = name
        self.ports: List[Union[FlowPort, PropertyPort]] = []
        self._inputs: dict[tuple[str, str], float] = {}  # (port_name, var_name) → input value
        self.system = None
        self.provides_inputs_only = False

    def residual(self, x):
        if self.provides_inputs_only:
            return []  # Safe to ignore
        if self.get_iteration_keys():
            raise NotImplementedError(
                f"{self.__class__.__name__}.residual(x) must be implemented for components with iteration variables."
            )
        return []


    def get_iteration_keys(self) -> list[tuple[str, str]]:
        return self.detect_iteration_variables()


    def get_iteration_vector(self) -> list[float]:
        """Return current values of iteration variables as a flat vector."""
        return [
            getattr(self[port_name], var)
            for (port_name, var) in self.get_iteration_keys()
        ]


    def set_iteration_vector(self, x: list[float]):
        """
        Set current values of iteration variables from a flat vector.
        Used by the solver to inject updated guesses.
        """
        keys = self.get_iteration_keys()
        if len(x) != len(keys):
            raise ValueError(
                f"{self.name}: Input vector length {len(x)} doesn't match "
                f"number of iteration variables {len(keys)}"
            )
        for (val, (port_name, var)) in zip(x, keys):
            setattr(self[port_name], var, val)


    def set_input(self, port_name: str, var_name: str, value: float):
        """
        Store an input value for a specific variable on a port.
        """
        self._inputs[(port_name, var_name)] = value

    def apply_inputs(self):
        """
        Apply only those inputs that match the component's iteration variables.
        """
        for port_name, var in self.detect_iteration_variables():
            key = (port_name, var)
            if key in self._inputs:
                port = self[port_name]
                setattr(port, var, self._inputs[key])


    def detect_iteration_variables(self):
        """
        Detect iteration variables on OutFlow ports based on the following rules:

        1. If component has only OutFlows:
        - If mass_flow is set and fluid is under-defined → iterate on T and P
        - If fluid is defined (fluid_name + 2 of T/P/X) but mass_flow is missing → iterate on mass_flow
        - If both are defined → no iteration variable

        2. If component has InFlows and OutFlows:
        - If OutFlow has no fluid and no mass_flow → iterate on T and P
        - If fluid is defined and mass_flow is missing → no iteration variable
        - If mass_flow is set but fluid is under-defined:
            - Only iterate on T and P if at least one other port (InFlow or OutFlow) is missing mass_flow
            - Otherwise, no iteration variable
        """
        if hasattr(self, "_iteration_variables") and self._iteration_variables:
            return self._iteration_variables

        self._iteration_variables = []

        inflows = [p for p in self.ports if isinstance(p, InFlow)]
        outflows = [p for p in self.ports if isinstance(p, OutFlow)]

        has_inflows = len(inflows) > 0
        has_outflows = len(outflows) > 0

        for port in outflows:
            name = port.name
            mass_flow = port.mass_flow
            fluid_name = port.fluid_name
            T, P, X = port.T, port.P, port.X

            state = {'T': T, 'P': P, 'X': X}
            defined_state = [k for k, v in state.items() if v is not None]
            num_state_vars = len(defined_state)
            fluid_defined = fluid_name is not None and num_state_vars >= 2

            # ------------------------------
            # CASE: component has only OutFlows
            # ------------------------------
            if has_outflows and not has_inflows:
                if mass_flow is not None and not fluid_defined:
                    # Rule 1a
                    for attr in ['T', 'P']:
                        if state[attr] is None:
                            self._iteration_variables.append((name, attr))
                elif fluid_defined and mass_flow is None:
                    # Rule 1b
                    self._iteration_variables.append((name, "mass_flow"))
                # else: both defined → Rule 2 → no iteration variables

            # ------------------------------
            # CASE: component has both InFlows and OutFlows
            # ------------------------------
            elif has_inflows and has_outflows:
                if mass_flow is None and not fluid_defined:
                    # Rule 3
                    for attr in ['T', 'P']:
                        if state[attr] is None:
                            self._iteration_variables.append((name, attr))
                elif fluid_defined and mass_flow is None:
                    # Rule 4 → nothing to iterate
                    continue
                elif mass_flow is not None and not fluid_defined:
                    # Rule 5 → check other ports for unset mass flow
                    others = [p for p in self.ports if p is not port and isinstance(p, FlowPort)]
                    missing_mdot_elsewhere = any(p.mass_flow is None for p in others)
                    if missing_mdot_elsewhere:
                        for attr in ['T', 'P']:
                            if state[attr] is None:
                                self._iteration_variables.append((name, attr))
                    # else: all other mass flows defined → nothing to iterate

        return self._iteration_variables
    
    def get_upstream_iterations(self, port_name=None):
        """
        If `port_name` is given:
            → Returns a list of iteration variable names (e.g. ['mass_flow', 'T']).

        If `port_name` is None:
            → Returns a dict mapping each InFlow name to a list of iteration variable names.

        Example:
            pipe.get_upstream_iterations("Source")
                → ['mass_flow']

            pipe.get_upstream_iterations()
                → {'Source': ['mass_flow']}
        """
        results = {}

        inflows = [p for p in self.ports if isinstance(p, InFlow)]

        for inflow in inflows:
            if port_name and inflow.name != port_name:
                continue

            upstream = inflow.connected_port
            if upstream and upstream.parent:
                try:
                    upstream_vars = [
                        var for pname, var in upstream.parent.detect_iteration_variables()
                        if pname == upstream.name
                    ]
                except Exception as e:
                    upstream_vars = [f"Error: {e}"]
            else:
                upstream_vars = []

            results[inflow.name] = upstream_vars

        if port_name:
            return results.get(port_name, [])
        return results

    def inlet_mass_flows(self) -> list[float]:
        return [
            port.mass_flow
            for port in self.ports
            if isinstance(port, InFlow) and port.mass_flow is not None
        ]

    def add_inflow(self, name: str) -> InFlow:
        port = InFlow(name, self)
        self.ports.append(port)
        return port

    def add_outflow(self, name: str) -> OutFlow:
        port = OutFlow(name, self)
        self.ports.append(port)
        return port

    def add_property_in(self, name: str) -> PropertyIn:
        port = PropertyIn(name, parent=self)
        self.ports.append(port)
        return port

    def add_property_out(self, name: str) -> PropertyOut:
        port = PropertyOut(name, parent=self)
        self.ports.append(port)
        return port

    def _normalize(self, name: str) -> str:
        name = name.lower()
        name = re.sub(r'[^a-z]', '', name)
        substitutions = {
            'ox': 'oxidizer',
            'fuel': 'fuel',
            'inflow': 'inlet',
            'in': 'inlet',
            'out': 'outlet',
            'exit': 'outlet',
            'exhaust': 'outlet'
        }
        for short, full in substitutions.items():
            name = name.replace(short, full)
        return name

    def get_port(self, name: str) -> Union[FlowPort, PropertyPort]:
        norm_name = self._normalize(name)
        norm_map = {self._normalize(p.name): p for p in self.ports}
        matches = difflib.get_close_matches(norm_name, norm_map.keys(), n=1, cutoff=0.6)
        if matches:
            return norm_map[matches[0]]
        raise PortNotFoundError(f"Port '{name}' not found in component '{self.name}'")

    def connect_ports(self, my_port_name: str, other: "Component", other_port_name: str):
        my_port = self.get_port(my_port_name)
        other_port = other.get_port(other_port_name)

        if isinstance(my_port, FlowPort) and isinstance(other_port, PropertyPort):
            raise PortConnectionError(f"Cannot connect FlowPort '{my_port.name}' to PropertyPort '{other_port.name}'")
        if isinstance(my_port, PropertyPort) and isinstance(other_port, FlowPort):
            raise PortConnectionError(f"Cannot connect PropertyPort '{my_port.name}' to FlowPort '{other_port.name}'")

        if (
            (isinstance(my_port, InFlow) and isinstance(other_port, InFlow)) or
            (isinstance(my_port, OutFlow) and isinstance(other_port, OutFlow)) or
            (isinstance(my_port, PropertyIn) and isinstance(other_port, PropertyIn)) or
            (isinstance(my_port, PropertyOut) and isinstance(other_port, PropertyOut))
        ):
            raise PortConnectionError(f"Cannot connect two input ports or two output ports: '{my_port.name}' ↔ '{other_port.name}'")

        my_port.connect(other_port)

    def __getitem__(self, port_name: str) -> Union[FlowPort, float]:
        port = self.get_port(port_name)
        if isinstance(port, PropertyPort):
            return port.value
        return port

    def __setitem__(self, port_name: str, value):
        port = self.get_port(port_name)

        if isinstance(port, PropertyOut):
            if not isinstance(value, (int, float)):
                raise ValueError(f"PropertyOut ports must be assigned a scalar, got: {type(value).__name__}")
            port.value = value
            return

        if isinstance(port, OutFlow):
            if isinstance(value, Fluid):
                port.fluid = value
                return
            if isinstance(value, dict):
                for key, val in value.items():
                    if key == "fluid_name":
                        port.fluid_name = val
                    elif hasattr(port, key):
                        setattr(port, key, val)
                    else:
                        raise PortKeyError(f"Port '{port.name}' has no attribute '{key}'")
                return

            raise ValueError("OutFlow values must be a dict or a Fluid object")

        raise PortPermissionError(f"Cannot set value on non-output port '{port.name}'")

    def connect(self, other: "Component", cutoff: float = 0.6):
        my_inflows = [p for p in self.ports if isinstance(p, InFlow) and p.connected_port is None]
        my_outflows = [p for p in self.ports if isinstance(p, OutFlow) and p.connected_port is None]
        other_inflows = [p for p in other.ports if isinstance(p, InFlow) and p.connected_port is None]
        other_outflows = [p for p in other.ports if isinstance(p, OutFlow) and p.connected_port is None]

        def normalize_map(port_list):
            return {self._normalize(p.name): p for p in port_list}

        other_out_dict = normalize_map(other_outflows)
        for my_port in my_inflows:
            matches = difflib.get_close_matches(self._normalize(my_port.name), other_out_dict.keys(), n=1, cutoff=cutoff)
            if matches:
                my_port.connect(other_out_dict[matches[0]])

        other_in_dict = normalize_map(other_inflows)
        for my_port in my_outflows:
            matches = difflib.get_close_matches(self._normalize(my_port.name), other_in_dict.keys(), n=1, cutoff=cutoff)
            if matches:
                my_port.connect(other_in_dict[matches[0]])

        # 🔁 Auto-refresh system membership
        if self.system and other.system is not self.system:
            self.system.add_component(other)
        elif other.system and self.system is not other.system:
            other.system.add_component(self)


    def connect_all(self, other: "Component", cutoff: float = 0.6):
        self.connect(other, cutoff=cutoff)

        def normalize(name: str) -> str:
            return self._normalize(name)

        def normalize_unconnected(port_list):
            return {
                normalize(p.name): p
                for p in port_list
                if isinstance(p, (PropertyIn, PropertyOut)) and not p.is_connected
            }

        my_inputs = [p for p in self.ports if isinstance(p, PropertyIn)]
        other_outputs = [p for p in other.ports if isinstance(p, PropertyOut)]
        other_out_dict = normalize_unconnected(other_outputs)

        for my_in in my_inputs:
            if my_in.is_connected:
                continue
            matches = difflib.get_close_matches(normalize(my_in.name), other_out_dict.keys(), n=1, cutoff=cutoff)
            if matches:
                my_in.connect(other_out_dict[matches[0]])

        my_outputs = [p for p in self.ports if isinstance(p, PropertyOut)]
        other_inputs = [p for p in other.ports if isinstance(p, PropertyIn)]
        other_in_dict = normalize_unconnected(other_inputs)

        for my_out in my_outputs:
            matches = difflib.get_close_matches(normalize(my_out.name), other_in_dict.keys(), n=1, cutoff=cutoff)
            for match in matches:
                other_in = other_in_dict[match]
                if not other_in.is_connected:
                    my_out.connect(other_in)

        # 🔁 Auto-refresh system membership
        if self.system and other.system is not self.system:
            self.system.add_component(other)
        elif other.system and self.system is not other.system:
            other.system.add_component(self)


    def __str__(self):
        def format_val(val):
            return f"{val:.3g}" if val is not None else "—"

        def build_flow_table(ports, label):
            table = PrettyTable()
            table.title = f"{label} Ports for {self.name}"
            table.field_names = [
                "Port", "Connected To", "Fluid", "Phase", "Mass Flow (kg/s)", "Pressure (Pa)",
                "Temperature (K)", "Quality (X)"
            ]
            for port in ports:
                conn = port.connected_port
                connected = f"{conn.name} [{conn.parent.name}]" if conn and conn.parent else "—"
                fluid = port.fluid
                fluid_name = fluid.name if fluid else "—"
                phase = fluid.phase if fluid and hasattr(fluid, "phase") else "—"
                table.add_row([
                    port.name, connected, fluid_name, phase,
                    format_val(port.mass_flow),
                    format_val(port.P),
                    format_val(port.T),
                    format_val(port.X)
                ])
            return table

        def build_property_table(ports, label):
            table = PrettyTable()
            table.title = f"{label} Property Ports for {self.name}"
            table.field_names = ["Port", "Connected To", "Value"]
            for port in ports:
                group = port._group() if hasattr(port, "_group") else []
                others = [p for p in group if p is not port]
                connections = [f"{p.name} [{p.parent.name}]" for p in others] or ["—"]
                value = format_val(port.value)
                for i, conn in enumerate(connections):
                    row = [port.name, conn, value] if i == 0 else ["", conn, ""]
                    table.add_row(row)
            return table

        inflows = [p for p in self.ports if isinstance(p, InFlow)]
        outflows = [p for p in self.ports if isinstance(p, OutFlow)]
        prop_ins = [p for p in self.ports if isinstance(p, PropertyIn)]
        prop_outs = [p for p in self.ports if isinstance(p, PropertyOut)]

        sections = [build_flow_table(inflows, "Inflow"), build_flow_table(outflows, "Outflow")]

        if prop_ins:
            sections.append(build_property_table(prop_ins, "Input"))
        if prop_outs:
            sections.append(build_property_table(prop_outs, "Output"))

        return f"\n========== Component: {self.name} ==========\n" + "\n".join(str(s) for s in sections)

if __name__ == "__main__":

    from Fluid import Fluid

    tca = Component("Heatsink")
    tca.add_inflow("Fuel Flow")
    tca.add_inflow("Oxidizer Flow")
    tca.add_property_in("Mixing Efficiency")

    injector = Component("Coax")
    injector.add_outflow("Fuel Flow")
    injector.add_outflow("Oxidizer Flow")
    injector.add_property_out("mixing efficiency")

    injector.connect_all(tca)
    print(injector)

    injector["fuel flow"].fluid = Fluid("Methane", T=100, P=1e6)
    injector["oxidizer flow"].fluid = Fluid("Oxygen", T=90, P=1e6)
    #injector["fuel flow"].P = 1e6
    #injector["fuel flow"].T = 100
    #injector["fuel flow"].fluid_name = "Methane"
    print(tca.detect_iteration_variables())
    print(injector["oxidizer flow"])
    print(injector["oxidizer flow"].fluid)