
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

    #injector["fuel flow"].fluid = Fluid("Methane", T=90, P=1e6)
    injector["fuel flow"].P = 1e6
    injector["fuel flow"].T = 100
    injector["fuel flow"].fluid_name = "Methane"
    print(tca)
