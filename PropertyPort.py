from typing import Optional
from prettytable import PrettyTable


class PropertyPort:
    def __init__(self, name: str, parent: Optional[object] = None):
        self.name = name
        self.parent = parent
        self._value = None
        self._connections: list[PropertyPort] = []

    def connect(self, other: "PropertyPort"):
        # Prevent output-to-output connections
        if self.direction == "out" and other.direction == "out":
            raise ValueError("Cannot connect two output ports.")

        # Helper: get connected output for an input
        def get_connected_output(port: PropertyPort):
            for p in port._connections:
                if p.direction == "out":
                    return p
            return None

        if self.direction == "in" and other.direction == "in":
            self_out = get_connected_output(self)
            other_out = get_connected_output(other)

            if self_out and other_out and self_out != other_out:
                raise ValueError(f"Cannot connect two inputs connected to different outputs.")
            elif self_out:
                self_out.connect(other)
            elif other_out:
                other_out.connect(self)
            else:
                raise ValueError("Cannot connect two unconnected inputs directly.")

            return

        # Make sure inputs only connect to one output
        inp, outp = (self, other) if self.direction == "in" else (other, self)

        if inp._connections and any(p.direction == "out" for p in inp._connections):
            existing_output = get_connected_output(inp)
            if existing_output != outp:
                raise ValueError(f"Input port '{inp.name}' is already connected to a different output.")

        # Perform the connection
        if inp not in outp._connections:
            outp._connections.append(inp)
        inp._connections = [outp]  # Replace any previous input-to-input links

        # Share value
        shared_value = inp._shared_value() or outp._shared_value()
        self._propagate_value(shared_value)


    def _has_output_connection(self) -> bool:
        return any(p.direction == "out" or p._has_output_connection() for p in self._connections)

    def _is_connected_conflict(self, other: "PropertyPort") -> bool:
        if self.direction != "in":
            return False
        if not self._connections:
            return False
        existing_has_output = self._has_output_connection()
        other_has_output = other._has_output_connection()
        return existing_has_output and other_has_output

    def _shared_value(self):
        for port in self._group():
            if port._value is not None:
                return port._value
        return None

    def _propagate_value(self, value):
        for port in self._group():
            port._value = value

    def _group(self) -> set["PropertyPort"]:
        visited = set()
        stack = [self]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                stack.extend(current._connections)
        return visited

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
    def direction(self) -> str:
        raise NotImplementedError

    def __str__(self):
        table = PrettyTable()
        table.title = f"{'Output' if self.direction == 'out' else 'Input'} Property Port: {self.name}"
        table.field_names = ["Field", "Value"]
        table.align["Field"] = "l"
        table.align["Value"] = "l"

        parent_str = self.parent.name if self.parent and hasattr(self.parent, "name") else "—"
        conn_names = ", ".join(sorted(p.name for p in self._connections)) if self._connections else "—"
        value_display = self.value if self.value is not None else "—"

        table.add_row(["Parent", parent_str])
        table.add_row(["Connected To", conn_names])
        table.add_row(["Value", value_display])

        return str(table)




class PropertyIn(PropertyPort):
    @property
    def direction(self) -> str:
        return "in"





class PropertyOut(PropertyPort):
    @property
    def direction(self) -> str:
        return "out"

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
