from dataclasses import dataclass, field
from typing import Optional, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from Component import Component
class SharedPortValue:
    """
    Shared container for values shared across multiple connected ports.
    Automatically updates all connected ports when the value changes.
    Propagates iteration_variable status across all subscribers.
    """
    def __init__(self):
        self.value = None
        self.subscribers: List["BasePort"] = []
        self.name: Optional[str] = None  # Node name
        self.iteration_variable: bool = False

    def subscribe(self, port: "BasePort"):
        """Subscribe a port and sync iteration_variable status."""
        if port not in self.subscribers:
            self.subscribers.append(port)
            port._value = self
            if self.name is None:
                self.name = port.name
            # Propagate iteration variable flag
            if port.iteration_variable:
                self.iteration_variable = True
            elif self.iteration_variable:
                port.iteration_variable = True

    def broadcast(self, value: Any):
        """Broadcast a new value to all connected ports."""
        if self.value is not None and self.value != value:
            print(f"[Value Override!] {self.value} → {value}")
        self.value = value
        for port in self.subscribers:
            port.on_value_changed(value)

    def merge(self, other: "SharedPortValue"):
        """Merge another shared value node into this one."""
        if self is other:
            return

        if self.value is not None and other.value is not None and self.value != other.value:
            print(f"[Warning] Conflicting values during merge: keeping {other.value}, overwriting {self.value}")

        if other.value is not None:
            self.value = other.value

        for port in other.subscribers:
            self.subscribe(port)

        # Propagate iteration_variable across all ports
        if other.iteration_variable:
            self.iteration_variable = True
            for port in self.subscribers:
                port.iteration_variable = True

        # Prefer earliest set name
        if self.name is None:
            self.name = other.name

        self.broadcast(self.value)


class BasePort:
    """
    Base class for all input/output ports in a component.
    Tracks if a port (and its shared group) is an iteration variable.
    """
    def __init__(self, name: str, component: "Component", iteration_variable: bool = False):
        self.name = name
        self.component = component
        self._value = SharedPortValue()
        self._value.subscribe(self)
        self.connected_ports: List["BasePort"] = []

        if iteration_variable:
            self.iteration_variable = True  # uses property setter below
        else:
            self._iteration_variable = False

        self.guess_variable = False

    def connect(self, other: "BasePort"):
        """Connect this port to another, merging shared state."""
        self._value.merge(other._value)
        self.connected_ports.append(other)
        other.connected_ports.append(self)

    @property
    def value(self) -> Any:
        return self._value.value if self._value else None

    @value.setter
    def value(self, val: Any):
        self._value.broadcast(val)

    def on_value_changed(self, val: Any):
        pass

    def is_connected(self) -> bool:
        return bool(self.connected_ports)

    @property
    def node_name(self):
        return self._value.name if self._value else None

    @property
    def node_value(self):
        return self._value.value if self._value else None

    @property
    def iteration_variable(self) -> bool:
        return self._value.iteration_variable

    @iteration_variable.setter
    def iteration_variable(self, flag: bool):
        self._iteration_variable = flag
        self._value.iteration_variable = flag
        for port in self._value.subscribers:
            port._iteration_variable = flag

    def __repr__(self):
        if not self.connected_ports:
            return f"{self.component.name}:{self.name} → (unconnected)" if isinstance(self, OutputPort) \
                else f"(unconnected) → {self.component.name}:{self.name}"

        lines = []
        for p in self.connected_ports:
            if isinstance(self, OutputPort):
                lines.append(f"{self.component.name}:{self.name} → {p.component.name}:{p.name}")
            else:
                lines.append(f"{p.component.name}:{p.name} → {self.component.name}:{self.name}")
        return "\n".join(lines)


class InputPort(BasePort):
    pass


class OutputPort(BasePort):
    pass
