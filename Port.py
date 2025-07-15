from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class Port:
    name: str
    component: str
    connected_port: Optional["Port"] = field(default=None, init=False)
    value: Any = field(default=None, init=False)

    def connect(self, other: "Port"):
        raise NotImplementedError("Connect must be implemented in subclasses.")

    def get_connection(self):
        print(self.connected_port)

    def is_connected(self):
        if self.connected_port is not None:
            return True
        else:
            return False


class InputPort(Port):
    def connect(self, output: "OutputPort"):
        if not isinstance(output, OutputPort):
            raise TypeError("InputPort must connect to OutputPort.")
        self.connected_port = output
        output.connected_port = self

    def receive(self):
        if self.connected_port is not None:
            self.value = self.connected_port.value
        else:
            self.value = None
        return self.value



class OutputPort(Port):
    def connect(self, input: "InputPort"):
        if not isinstance(input, InputPort):
            raise TypeError("OutputPort must connect to InputPort.")
        input.connect(self)

    def transmit(self, data: Any):
        self.value = data

