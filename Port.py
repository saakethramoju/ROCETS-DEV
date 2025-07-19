from dataclasses import dataclass, field
from typing import Optional, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from Component import Component

class SharedPortValue:
    def __init__(self):
        self.value = None
        self.subscribers = []

    def subscribe(self, port: "BasePort"):
        if port not in self.subscribers:
            self.subscribers.append(port)
            port._value = self

    def broadcast(self, value):
        if self.value is not None and self.value != value:
            print(f"[Warning] Overriding existing shared value: {self.value} → {value}")
        self.value = value
        for port in self.subscribers:
            port.on_value_changed(value)

    def merge(self, other: "SharedPortValue"):
        if self is other:
            return

        if self.value is not None and other.value is not None and self.value != other.value:
            print(f"[Warning] Conflicting values during merge: keeping {other.value}, overwriting {self.value}")

        if other.value is not None:
            self.value = other.value

        for port in other.subscribers:
            self.subscribe(port)

        self.broadcast(self.value)


class BasePort:
    def __init__(self, name: str, component: "Component"):
        self.name = name
        self.component = component
        self._value = SharedPortValue()
        self._value.subscribe(self)
        self.connected_ports: list["BasePort"] = []

    def connect(self, other: "BasePort"):
        # Always merge values, ensuring all ports share one SharedPortValue
        self._value.merge(other._value)
        self.connected_ports.append(other)
        other.connected_ports.append(self)

    @property
    def value(self):
        return self._value.value if self._value else None

    @value.setter
    def value(self, val):
        self._value.broadcast(val)

    def is_connected(self):
        return len(self.connected_ports) > 0

    def on_value_changed(self, val):
        pass  # hook for extension


class InputPort(BasePort):
    pass

class OutputPort(BasePort):
    pass
