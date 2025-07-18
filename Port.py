from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from Component import Component


@dataclass
class Port:
    name: str
    component: "Component"

    def is_connected(self) -> bool:
        raise NotImplementedError("Implemented in subclasses")

    def __repr__(self):
        raise NotImplementedError("Use __repr__ in InputPort or OutputPort")


@dataclass
class InputPort(Port):
    connected_ports: List["OutputPort"] = field(default_factory=list, init=False)

    @property
    def connected_components(self) -> List["Component"]:
        return [f"{p.component.name}" for p in self.connected_ports]

    def connect(self, output: "OutputPort"):
        if not isinstance(output, OutputPort):
            raise TypeError("InputPort must connect to OutputPort.")
        if output not in self.connected_ports:
            self.connected_ports.append(output)
        if self not in output.connected_ports:
            output.connected_ports.append(self)

    def is_connected(self) -> bool:
        return len(self.connected_ports) > 0

    def __repr__(self):
        if self.is_connected():
            conns = ", ".join(f"{p.component.name}.{p.name}" for p in self.connected_ports)
            return f"{self.component.name}.{self.name} ← {conns}"
        else:
            return f"{self.component.name}.{self.name} ← None"


@dataclass
class OutputPort(Port):
    connected_ports: List[InputPort] = field(default_factory=list, init=False)

    @property
    def connected_components(self) -> List["Component"]:
        return [f"{p.component.name}" for p in self.connected_ports]

    def connect(self, input_port: "InputPort"):
        if not isinstance(input_port, InputPort):
            raise TypeError("OutputPort must connect to InputPort.")
        input_port.connect(self)

    def is_connected(self) -> bool:
        return len(self.connected_ports) > 0

    def __repr__(self):
        if self.is_connected():
            conns = ", ".join(f"{p.component.name}.{p.name}" for p in self.connected_ports)
            return f"{self.component.name}.{self.name} → {conns}"
        else:
            return f"{self.component.name}.{self.name} → None"
