from typing import Optional, TYPE_CHECKING
from prettytable import PrettyTable
if TYPE_CHECKING:
    from Component import Component


class PropertyPort:
    def __init__(self, name: str, *, parent: Optional["Component"] = None, direction: str):
        self.name = name
        self.parent = parent
        self._value = None
        self._direction = direction
        self._connections: list[PropertyPort] = []

    def connect(self, other: "PropertyPort"):
        if self.direction == "out" and other.direction == "out":
            raise ValueError("Cannot connect two output ports.")

        def get_output(port: PropertyPort):
            return next((p for p in port._connections if p.direction == "out"), None)

        if self.direction == "in" and other.direction == "in":
            self_out = get_output(self)
            other_out = get_output(other)

            if self_out and other_out and self_out != other_out:
                raise ValueError("Cannot connect inputs linked to different outputs.")
            elif self_out:
                self_out.connect(other)
            elif other_out:
                other_out.connect(self)
            else:
                raise ValueError("Cannot directly connect two unconnected input ports.")
            return

        inp, outp = (self, other) if self.direction == "in" else (other, self)

        existing_output = get_output(inp)
        if existing_output and existing_output != outp:
            raise ValueError(f"Input port '{inp.name}' is already connected to a different output.")

        if inp not in outp._connections:
            outp._connections.append(inp)
        inp._connections = [outp]

        shared = self._shared_value() or other._shared_value()
        self._propagate_value(shared)

    def _group(self) -> set["PropertyPort"]:
        visited = set()
        stack = [self]
        while stack:
            port = stack.pop()
            if port not in visited:
                visited.add(port)
                stack.extend(port._connections)
        return visited

    def _shared_value(self):
        for p in self._group():
            if p._value is not None:
                return p._value
        return None

    def _propagate_value(self, value):
        for p in self._group():
            p._value = value

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def value(self):
        return self._shared_value()

    @value.setter
    def value(self, val):
        raise NotImplementedError("Only output ports may set values.")

    @property
    def connected_ports(self):
        return self._connections

    @property
    def is_connected(self) -> bool:
        return bool(self._connections)

    def __str__(self):
        table = PrettyTable()
        table.title = f"{'Output' if self.direction == 'out' else 'Input'} Property Port: {self.name}"
        table.field_names = ["Field", "Value"]
        table.align["Field"] = "l"
        table.align["Value"] = "l"

        parent_name = self.parent.name if self.parent and hasattr(self.parent, "name") else "—"
        conn_names = "\n".join(f"{p.name}" + (f" [{p.parent.name}]" if p.parent else "") for p in self._connections) or "—"
        value = self.value if self.value is not None else "—"

        table.add_row(["Parent", parent_name])
        table.add_row(["Connected To", conn_names])
        table.add_row(["Value", value])

        return str(table)


class PropertyIn(PropertyPort):
    def __init__(self, name: str, parent: Optional["Component"] = None):
        super().__init__(name, parent=parent, direction="in")

    @PropertyPort.value.setter
    def value(self, val):
        raise AttributeError(f"Cannot set value on input property port '{self.name}' — it is read-only.")


class PropertyOut(PropertyPort):
    def __init__(self, name: str, parent: Optional["Component"] = None):
        super().__init__(name, parent=parent, direction="out")

    @PropertyPort.value.setter
    def value(self, val):
        self._propagate_value(val)


if __name__ == "__main__":

    out = PropertyOut("Setpoint")
    in1 = PropertyIn("ThrottleInput1")
    in2 = PropertyIn("ThrottleInput2")

    out.connect(in1)
    in1.connect(in2)
    out.value = 0.8
    #in1.value = 1

    print(out)
    print(in2)
    print(in1)
