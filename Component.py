
import difflib
import re
from typing import List
from prettytable import PrettyTable
from Exceptions import (PortNotFoundError, PortPermissionError, PortKeyError)
from FlowPort import FlowPort, InFlow, OutFlow

class Component:
    def __init__(self, name: str):
        self.name = name
        self.ports: List[FlowPort] = []

    def add_inflow(self, name: str) -> InFlow:
        port = InFlow(name, self)
        self.ports.append(port)
        return port

    def add_outflow(self, name: str) -> OutFlow:
        port = OutFlow(name, self)
        self.ports.append(port)
        return port
        
    def inlet_mass_flows(self) -> list[float]:
        """Return a list of mass flows from all connected InFlow ports (skipping None)."""
        return [
            port.mass_flow
            for port in self.ports
            if isinstance(port, InFlow) and port.mass_flow is not None
        ]

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

    def get_port(self, name: str) -> FlowPort:
        norm_name = self._normalize(name)
        norm_map = {self._normalize(p.name): p for p in self.ports}
        matches = difflib.get_close_matches(norm_name, norm_map.keys(), n=1, cutoff=0.6)
        if matches:
            return norm_map[matches[0]]
        raise PortNotFoundError(f"Port '{name}' not found in component '{self.name}'")

    def connect_ports(self, my_port_name: str, other: "Component", other_port_name: str):
        my_port = self.get_port(my_port_name)
        other_port = other.get_port(other_port_name)
        my_port.connect(other_port)

    def __getitem__(self, port_name: str) -> FlowPort:
        return self.get_port(port_name)

    def __setitem__(self, port_name: str, value: dict):
        port = self.get_port(port_name)
        if not isinstance(port, OutFlow):
            raise PortPermissionError(f"Cannot set values on InFlow port '{port_name}'")
        if not isinstance(value, dict):
            raise ValueError("Value must be a dictionary of attributes to set")

        for key, val in value.items():
            if hasattr(port, key):
                setattr(port, key, val)
            else:
                raise PortKeyError(f"Port '{port.name}' has no attribute '{key}'")

    def connect(self, other: "Component", cutoff: float = 0.6):
        my_inflows = [p for p in self.ports if isinstance(p, InFlow) and p.connected_port is None]
        my_outflows = [p for p in self.ports if isinstance(p, OutFlow) and p.connected_port is None]
        other_inflows = [p for p in other.ports if isinstance(p, InFlow) and p.connected_port is None]
        other_outflows = [p for p in other.ports if isinstance(p, OutFlow) and p.connected_port is None]

        other_out_dict = {self._normalize(p.name): p for p in other_outflows}
        for my_port in my_inflows:
            matches = difflib.get_close_matches(self._normalize(my_port.name), other_out_dict.keys(), n=1, cutoff=cutoff)
            if matches:
                my_port.connect(other_out_dict[matches[0]])

        other_in_dict = {self._normalize(p.name): p for p in other_inflows}
        for my_port in my_outflows:
            matches = difflib.get_close_matches(self._normalize(my_port.name), other_in_dict.keys(), n=1, cutoff=cutoff)
            if matches:
                my_port.connect(other_in_dict[matches[0]])

    def __str__(self):
        def build_table(ports, label):
            table = PrettyTable()
            table.title = f"{label} Ports for {self.name}"
            table.field_names = ["Port", "Connected To"]
            table.align["Port"] = "l"
            table.align["Connected To"] = "l"
            for port in ports:
                conn = port.connected_port
                if conn:
                    connected = f"{conn.name} [{conn.parent.name}]"
                else:
                    connected = "—"
                table.add_row([port.name, connected])
            return table

        inflows = [p for p in self.ports if isinstance(p, InFlow)]
        outflows = [p for p in self.ports if isinstance(p, OutFlow)]

        inflow_table = build_table(inflows, "Inflow")
        outflow_table = build_table(outflows, "Outflow")

        return (
            f"\n========== Component: {self.name} ==========\n"
            f"{inflow_table}\n"
            f"{outflow_table}"
        )

if __name__ == "__main__":


    tca = Component("Heatsink")
    tca.add_inflow("Fuel Inflow")
    tca.add_inflow("Oxidizer Inflow")

    injector = Component("Coax")
    injector.add_outflow("Fuel Outflow")
    injector.add_outflow("Oxidizer Outflow")
    injector["Oxidizer outflow"].mass_flow = 3
    injector["fuel flow"].mass_flow = 1.5

    tca.connect(injector)
    print(tca)
    #injector["Oxidizer outflow"].mass_flow = 5
    print(sum(tca.inlet_mass_flows()))