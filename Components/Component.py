# Component.py
from __future__ import annotations
from typing import Dict, Iterator, Optional, TYPE_CHECKING
from prettytable import PrettyTable
import difflib
from Ports import InFlow, OutFlow, FlowPort, PropertyIn, PropertyOut

if TYPE_CHECKING:
    from System import System 

class Component:

    configuration_keys = []
    
    def __init__(self, name: str):
        self.name = name
        self.inflows: Dict[str, InFlow] = {}
        self.outflows: Dict[str, OutFlow] = {}
        self.property_ins = {}
        self.property_outs = {}
        self._system: Optional["System"] = None
        self.configuration = {}

        for key in self.configuration_keys:
            self.configuration[key] = None

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
            
    def _maybe_expand_system(self, other: "Component"):
        if self.system:
            self.system.add_component(other)
        elif other.system:
            other.system.add_component(self)


    def connect_ports(self, my_port_name, other: Component, other_port_name):
        def fuzzy_lookup(name: str, port_dicts: list[dict]) -> Optional[object]:
            all_keys = {}
            for d in port_dicts:
                all_keys.update(d)
            match = difflib.get_close_matches(name.lower(), all_keys.keys(), n=1, cutoff=0.6)
            return all_keys.get(match[0]) if match else None

        my_port = fuzzy_lookup(
            my_port_name, [self.inflows, self.outflows, self.property_ins, self.property_outs]
        )
        their_port = fuzzy_lookup(
            other_port_name, [other.inflows, other.outflows, other.property_ins, other.property_outs]
        )

        if not my_port or not their_port:
            raise ValueError(
                f"Could not find ports '{my_port_name}' or '{other_port_name}'.\n"
                f"Available ports in self: {list(self.ports(include_properties=True).keys())}\n"
                f"Available ports in other: {list(other.ports(include_properties=True).keys())}"
            )

        # FlowPort: only Out → In
        if isinstance(my_port, OutFlow) and isinstance(their_port, InFlow):
            if not my_port.is_connected() and not their_port.is_connected():
                my_port.connect(their_port)
                self._maybe_expand_system(other)
        elif isinstance(my_port, InFlow) and isinstance(their_port, OutFlow):
            if not my_port.is_connected() and not their_port.is_connected():
                their_port.connect(my_port)
                self._maybe_expand_system(other)
        # PropertyPort: only Out → In
        elif isinstance(my_port, PropertyOut) and isinstance(their_port, PropertyIn):
            if not their_port.is_connected():
                my_port.connect(their_port)
                self._maybe_expand_system(other)
        elif isinstance(my_port, PropertyIn) and isinstance(their_port, PropertyOut):
            if not my_port.is_connected():
                their_port.connect(my_port)
                self._maybe_expand_system(other)
        else:
            raise TypeError(f"Incompatible port types: {type(my_port).__name__} ↔ {type(their_port).__name__}")


    def __str__(self) -> str:
        from prettytable import PrettyTable

        # ----- Flow Ports -----
        inflow_table = PrettyTable()
        outflow_table = PrettyTable()
        inflow_table.field_names = outflow_table.field_names = [
            "Port", "Connected To", "Fluid", "Phase", "T [K]", "P [Pa]", "X", "Mass Flow [kg/s]"
        ]

        for port_name, port in self.inflows.items():
            conn = f"{port.connected_port.parent.name}.{port.connected_port.name}" if port.connected_port else "-"
            phase = getattr(port.fluid, "phase", "-") if port.fluid else "-"
            inflow_table.add_row([
                port_name,
                conn,
                port.fluid_name or "-",
                phase,
                port.T if port.T is not None else "-",
                port.P if port.P is not None else "-",
                port.X if port.X is not None else "-",
                port.mass_flow if port.mass_flow is not None else "-"
            ])

        for port_name, port in self.outflows.items():
            conn = f"{port.connected_port.parent.name}.{port.connected_port.name}" if port.connected_port else "-"
            phase = getattr(port.fluid, "phase", "-") if port.fluid else "-"
            outflow_table.add_row([
                port_name,
                conn,
                port.fluid_name or "-",
                phase,
                port.T if port.T is not None else "-",
                port.P if port.P is not None else "-",
                port.X if port.X is not None else "-",
                port.mass_flow if port.mass_flow is not None else "-"
            ])

        # ----- Property Ports -----
        prop_in_table = PrettyTable()
        prop_out_table = PrettyTable()
        prop_in_table.field_names = prop_out_table.field_names = [
            "Port", "Connected To", "Value"
        ]

        for port_name, port in self.property_ins.items():
            conn = f"{port.connected_port.parent.name}.{port.connected_port.name}" if port.connected_port else "-"
            prop_in_table.add_row([
                port_name,
                conn,
                port.value if port.value is not None else "-"
            ])

        for port_name, port in self.property_outs.items():
            conn = f"{port.connected_port.parent.name}.{port.connected_port.name}" if port.connected_port else "-"
            prop_out_table.add_row([
                port_name,
                conn,
                port.value if port.value is not None else "-"
            ])

        # ----- Header Box -----
        title = f" COMPONENT: {self.name} "
        box_width = max(len(title), 30)
        border = "═" * box_width
        header = f"╔{border}╗\n║{title.center(box_width)}║\n╚{border}╝"

        return (
            f"{header}\n\n"
            f"Inlets:\n{inflow_table}\n\n"
            f"Outlets:\n{outflow_table}\n\n"
            f"Property Ins:\n{prop_in_table}\n\n"
            f"Property Outs:\n{prop_out_table}"
        )


    def __repr__(self) -> str:
        return f"<Component {self.name}>"



    def connect(self, other: "Component", print_summary: bool = False) -> None:
        connections = []

        # OutFlow (self) → InFlow (other)
        for name, my_port in self.outflows.items():
            if name in other.inflows:
                their_port = other.inflows[name]
                if not my_port.is_connected() and not their_port.is_connected():
                    my_port.connect(their_port)
                    connections.append((f"{self.name}.{name}", f"{other.name}.{name}"))

        # OutFlow (other) → InFlow (self)
        for name, their_port in other.outflows.items():
            if name in self.inflows:
                my_port = self.inflows[name]
                if not my_port.is_connected() and not their_port.is_connected():
                    their_port.connect(my_port)
                    connections.append((f"{other.name}.{name}", f"{self.name}.{name}"))

        if print_summary and connections:
            print(f"[{self.name}] connected to [{other.name}]:")
            for src, dst in connections:
                print(f"  {src} → {dst}")

        self._maybe_expand_system(other)



    def connect_all(self, other: "Component", print_summary: bool = False) -> None:
        self.connect(other, print_summary=print_summary)

        def norm(s): return s.lower()
        summary = []

        # PropertyOut (self) → PropertyIn (other)
        for my_name, my_out in self.property_outs.items():
            for their_name, their_in in other.property_ins.items():
                if not their_in.is_connected():
                    if norm(my_name) == norm(their_name) or norm(my_name) in norm(their_name) or norm(their_name) in norm(my_name):
                        try:
                            my_out.connect(their_in)
                            summary.append((f"{self.name}.{my_name}", f"{other.name}.{their_name}"))
                            break
                        except Exception:
                            continue

        self._maybe_expand_system(other)

        # PropertyOut (other) → PropertyIn (self)
        for their_name, their_out in other.property_outs.items():
            for my_name, my_in in self.property_ins.items():
                if not my_in.is_connected():
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

    @property
    def system(self) -> Optional["System"]:
        return self._system


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
        if match:
            if match in self.inflows:
                return self.inflows[match]
            if match in self.outflows:
                return self.outflows[match]
            if match in self.property_ins:
                return self.property_ins[match].value
            if match in self.property_outs:
                return self.property_outs[match].value

        # If no matching port, try configuration keys (case-insensitive match)
        config_key = self._fuzzy_find_config_key(key)
        if config_key:
            return self.configuration.get(config_key)

        raise KeyError(f"No port or configuration key similar to '{key}' on component '{self.name}'")


    def __setitem__(self, key: str, value):
        match = self._fuzzy_find_port_name(key)
        if match:
            if match in self.property_ins:
                raise AttributeError("Cannot assign value to PropertyIn; must connect it to a PropertyOut.")
            elif match in self.property_outs:
                self.property_outs[match].value = value
            else:
                raise KeyError(f"Cannot assign value to non-property port '{key}'")
            return

        # Fallback to configuration
        config_key = self._fuzzy_find_config_key(key)
        if config_key:
            self.configuration[config_key] = value
            return

        raise KeyError(f"No port or configuration key similar to '{key}' on component '{self.name}'")
        
    def _fuzzy_find_config_key(self, key: str) -> Optional[str]:
        matches = difflib.get_close_matches(key, self.configuration_keys, n=1, cutoff=0.8)
        return matches[0] if matches else None



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

    def evaluate(self):
        """Evaluate this component to update its outflows based on inputs."""
        return  # Overridden by subclasses
