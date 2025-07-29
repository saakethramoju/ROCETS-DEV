# Component.py
from __future__ import annotations
from typing import Dict, Iterator, Optional
from prettytable import PrettyTable
import difflib
from Ports import InFlow, OutFlow, FlowPort, PropertyIn, PropertyOut



class Component:

    configuration_keys = []
    
    def __init__(self, name: str):
        self.name = name
        self.inflows: Dict[str, InFlow] = {}
        self.outflows: Dict[str, OutFlow] = {}
        self.property_ins = {}
        self.property_outs = {}

    def add_property_in(self, name: str) -> PropertyIn:
        if name in self.property_ins:
            raise ValueError(f"PropertyIn '{name}' already exists on {self.name}.")
        port = PropertyIn(name, parent=self)
        self.property_ins[name] = port
        return port

    def add_property_out(self, name: str) -> PropertyOut:
        if name in self.property_outs:
            raise ValueError(f"PropertyOut '{name}' already exists on {self.name}.")
        port = PropertyOut(name, parent=self)
        self.property_outs[name] = port
        return port


    def add_inflow(self, port_name: str) -> InFlow:
        if port_name in self.inflows:
            raise ValueError(f"Inflow port '{port_name}' already exists on {self.name}.")
        port = InFlow(port_name, parent=self)  # ✅ pass parent
        self.inflows[port_name] = port
        return port

    def add_outflow(self, port_name: str) -> OutFlow:
        if port_name in self.outflows:
            raise ValueError(f"Outflow port '{port_name}' already exists on {self.name}.")
        port = OutFlow(port_name, parent=self)  # ✅ pass parent
        self.outflows[port_name] = port
        return port


    def connect_ports(self, my_port_name, other: Component, other_port_name):
        def fuzzy_lookup(name: str, port_dicts: list[dict]) -> Optional[object]:
            all_keys = {}
            for d in port_dicts:
                all_keys.update(d)
            match = difflib.get_close_matches(name.lower(), all_keys.keys(), n=1, cutoff=0.8)
            return all_keys.get(match[0]) if match else None

        my_port = fuzzy_lookup(
            my_port_name,
            [self.inflows, self.outflows, self.property_ins, self.property_outs],
        )
        their_port = fuzzy_lookup(
            other_port_name,
            [other.inflows, other.outflows, other.property_ins, other.property_outs],
        )

        if not my_port or not their_port:
            raise ValueError(
                f"Could not find ports '{my_port_name}' or '{other_port_name}'.\n"
                f"Available ports in self: {list(self.ports())}\n"
                f"Available ports in other: {list(other.ports())}"
            )

        my_port.connect(their_port)



    def __str__(self) -> str:
        inflow_table = PrettyTable()
        outflow_table = PrettyTable()
        inflow_table.field_names = outflow_table.field_names = [
            "Port", "Connected To", "Fluid", "T [K]", "P [Pa]", "X", "Mass Flow [kg/s]"
        ]

        for port_name, port in self.inflows.items():
            inflow_table.add_row([
                port_name,
                port.connected_port.name if port.connected_port else "-",
                port.fluid_name,
                port.T if port.T is not None else "-",
                port.P if port.P is not None else "-",
                port.X if port.X is not None else "-",
                port.mass_flow if port.mass_flow is not None else "-"
            ])

        for port_name, port in self.outflows.items():
            outflow_table.add_row([
                port_name,
                port.connected_port.name if port.connected_port else "-",
                port.fluid_name,
                port.T if port.T is not None else "-",
                port.P if port.P is not None else "-",
                port.X if port.X is not None else "-",
                port.mass_flow if port.mass_flow is not None else "-"
            ])

        title = f" COMPONENT: {self.name} "
        box_width = max(len(title), 30)
        border = "═" * box_width
        centered_title = title.center(box_width)

        header = f"╔{border}╗\n║{centered_title}║\n╚{border}╝"
        return f"{header}\n\nInlets:\n{inflow_table}\n\nOutlets:\n{outflow_table}"

    def __repr__(self) -> str:
        return f"<Component {self.name}>"

    def connect(self, other: "Component", print_summary: bool = False) -> None:
        """
        Automatically connects matching ports between two components by exact name.
        Only connects unconnected ports:
        - self.outflows → other.inflows
        - other.outflows → self.inflows
        """
        connections = []

        # Connect: self.outflows → other.inflows
        for name, my_port in self.outflows.items():
            if my_port.is_connected():
                continue
            if name in other.inflows:
                other_port = other.inflows[name]
                if not other_port.is_connected():
                    my_port.connect(other_port)
                    connections.append((f"{self.name}.{name}", f"{other.name}.{name}"))

        # Connect: other.outflows → self.inflows
        for name, their_port in other.outflows.items():
            if their_port.is_connected():
                continue
            if name in self.inflows:
                my_port = self.inflows[name]
                if not my_port.is_connected():
                    their_port.connect(my_port)
                    connections.append((f"{other.name}.{name}", f"{self.name}.{name}"))

        # Optional summary
        if print_summary and connections:
            print(f"[{self.name}] connected to [{other.name}]:")
            for src, dst in connections:
                print(f"  {src} → {dst}")

    def connect_all(self, other: "Component", print_summary: bool = False) -> None:
        self.connect(other, print_summary=print_summary)

        def norm(s): return s.lower()

        # Track what got connected
        summary = []

        # --- PropertyOut → PropertyIn ---
        for my_name, my_out in self.property_outs.items():
            if not isinstance(my_out, PropertyOut):
                continue
            for their_name, their_in in other.property_ins.items():
                if not isinstance(their_in, PropertyIn):
                    continue
                if their_in.is_connected():
                    continue
                if norm(my_name) == norm(their_name) or norm(my_name) in norm(their_name) or norm(their_name) in norm(my_name):
                    try:
                        my_out.connect(their_in)
                        summary.append((f"{self.name}.{my_name}", f"{other.name}.{their_name}"))
                        break
                    except Exception:
                        continue

        # --- PropertyOut (other) → PropertyIn (self) ---
        for their_name, their_out in other.property_outs.items():
            if not isinstance(their_out, PropertyOut):
                continue
            for my_name, my_in in self.property_ins.items():
                if not isinstance(my_in, PropertyIn):
                    continue
                if my_in.is_connected():
                    continue
                if norm(their_name) == norm(my_name) or norm(their_name) in norm(my_name) or norm(my_name) in norm(their_name):
                    try:
                        their_out.connect(my_in)
                        summary.append((f"{other.name}.{their_name}", f"{self.name}.{my_name}"))
                        break
                    except Exception:
                        continue

        if print_summary and summary:
            print(f"[{self.name}] connected property ports with [{other.name}]:")
            for src, dst in summary:
                print(f"  {src} → {dst}")


    # ---- dict-like access ----

    def _fuzzy_find_port_name(self, key: str) -> Optional[str]:
        all_keys = (
            list(self.inflows.keys())
            + list(self.outflows.keys())
            + list(self.property_ins.keys())
            + list(self.property_outs.keys())
        )
        matches = difflib.get_close_matches(key, all_keys, n=1, cutoff=0.8)
        return matches[0] if matches else None

    def __getitem__(self, key: str):
        match = self._fuzzy_find_port_name(key)
        if not match:
            raise KeyError(f"No port similar to '{key}' on component '{self.name}'")

        if match in self.inflows:
            return self.inflows[match]
        if match in self.outflows:
            return self.outflows[match]
        if match in self.property_ins:
            return self.property_ins[match].value
        if match in self.property_outs:
            return self.property_outs[match].value

        raise KeyError(f"No matching port found for '{key}' on component '{self.name}'")

    def __setitem__(self, key: str, value):
        match = self._fuzzy_find_port_name(key)
        if not match:
            raise KeyError(f"No port similar to '{key}' on component '{self.name}'")

        if match in self.property_ins:
            raise AttributeError("Cannot assign value to PropertyIn; must connect it to a PropertyOut.")
        elif match in self.property_outs:
            self.property_outs[match].value = value
        else:
            raise KeyError(f"Cannot assign value to non-property port '{key}'")

    def __contains__(self, key: str) -> bool:
        return key in self.inflows or key in self.outflows

    def keys(self) -> Iterator[str]:
        """All port names (inflows first, then outflows)."""
        yield from self.inflows.keys()
        yield from self.outflows.keys()

    def ports(self, include_properties=False) -> Dict[str, FlowPort]:
        """True if you want to include property ports"""
        base = {**self.outflows, **self.inflows}
        if include_properties:
            base.update(self.property_ins)
            base.update(self.property_outs)
        return base

