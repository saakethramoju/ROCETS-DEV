
from typing import Dict, Any
from Port import InputPort, OutputPort
from Exceptions import (PortNotConnectedError, PortTypeError, AmbiguousPortError,
                        NoMatchingPortsError)
import re

class Component:
    def __init__(self, name: str):
        self.name: str = name
        self.inputs: Dict[str, InputPort] = {}
        self.outputs: Dict[str, OutputPort] = {}
        self.required_inputs: Dict[str, InputPort] = {}
        self.required_outputs: Dict[str, OutputPort] = {}

    def add_input(self, port_name: str, required: bool = False, guess_required: bool = False):
        port = InputPort(name=port_name, component=self)
        self.inputs[port_name] = port
        if required:
            self.required_inputs[port_name] = port
        if guess_required:
            if not hasattr(self, "_guess_required_inputs"):
                self._guess_required_inputs = set()
            self._guess_required_inputs.add(port_name)
        return port


    def add_output(self, port_name: str, required: bool = False):
        port = OutputPort(name=port_name, component=self)
        self.outputs[port_name] = port
        if required:
            self.required_outputs[port_name] = port
        return port

    def validate_required_connections(self):
        missing = []

        for name, port in self.required_inputs.items():
            if not port.is_connected():
                missing.append(f"Input: {name}")

        for name, port in self.required_outputs.items():
            if not port.is_connected():
                missing.append(f"Output: {name}")

        if missing:
            raise PortNotConnectedError(
                f"{self.name} is missing connections for: {', '.join(missing)}"
            )

    def connect(self, other: "Component", only_required: bool = False):
        """
        Automatically connect matching ports between self and other using fuzzy, case-insensitive matching.
        If only_required is True, only connects to required inputs/outputs of the target component.
        Raises NoMatchingPortsError if no connections are made.
        """
        other_inputs = other.required_inputs if only_required else other.inputs
        other_outputs = other.required_outputs if only_required else other.outputs

        connection_made = False

        for self_port_name, self_port in {**self.inputs, **self.outputs}.items():
            # Determine direction
            if isinstance(self_port, OutputPort):
                search_ports = other_inputs
            elif isinstance(self_port, InputPort):
                search_ports = other_outputs
            else:
                continue  # Unexpected port type

            matches = [
                (name, port) for name, port in search_ports.items()
                if self._normalize(name) == self._normalize(self_port_name)
            ]

            if len(matches) == 1:
                other_port_name, other_port = matches[0]
                if isinstance(self_port, OutputPort) and isinstance(other_port, InputPort):
                    self_port.connect(other_port)
                    print(f"[Connected] {self.name}: {self_port_name} → {other.name}: {other_port_name}")
                    connection_made = True
                elif isinstance(self_port, InputPort) and isinstance(other_port, OutputPort):
                    other_port.connect(self_port)
                    print(f"[Connected] {other.name}: {other_port_name} → {self.name}: {self_port_name}")
                    connection_made = True
                else:
                    raise PortTypeError(
                        f"Invalid connection between {self.name}.{self_port_name} and {other.name}.{other_port_name}"
                    )
            elif len(matches) > 1:
                raise AmbiguousPortError(
                    f"Ambiguous port match for '{self_port_name}' between {self.name} and {other.name}: multiple similar targets"
                )

        if not connection_made:
            raise NoMatchingPortsError(
                f"No ports could be connected between {self.name} and {other.name}."
            )


    def manual_connect(self, output_name: str, input_comp: "Component", input_name: str):
        out = self.outputs.get(output_name)
        inp = input_comp.inputs.get(input_name)
        if out and inp:
            out.connect(inp)
            print(f"[Connected] {self.name}: {output_name} → {input_comp.name}: {input_name}")

    def connect_all_necessary_ports(self, other: "Component"):
        self_outputs_lower = {self._normalize(name): port for name, port in self.outputs.items()}
        other_inputs_required_lower = {
            self._normalize(name): (name, port) for name, port in other.required_inputs.items()
        }

        unmatched_ports = []

        for norm_name, (orig_name, input_port) in other_inputs_required_lower.items():
            output_port = self_outputs_lower.get(norm_name)
            if output_port:
                output_port.connect(input_port)
                print(f"[Connected] {self.name}: {output_port.name} → {other.name}: {orig_name}")
            else:
                unmatched_ports.append(orig_name)

        for port_name in unmatched_ports:
            print(f"[Warning] Could not connect required input '{port_name}' in {other.name} from any output in {self.name}.")

    def _normalize(self, name: str):
        # Remove anything in parentheses, strip spaces, and lower-case
        name = re.sub(r"\(.*?\)", "", name)
        return name.strip().lower()

    def _resolve_port(self, port_name: str):
        norm = self._normalize(port_name)
        for name, port in {**self.inputs, **self.outputs}.items():
            if self._normalize(name) == norm:
                return port
        raise KeyError(f"Port '{port_name}' not found in {self.name}.")

    def __getitem__(self, port_name: str):
        return self._resolve_port(port_name).value

    def __setitem__(self, port_name: str, value: Any):
        self._resolve_port(port_name).value = value

    def __repr__(self):
        def format_input(name, port):
            marker = "*" if name in self.required_inputs else " "
            if port.is_connected():
                conns = ", ".join(f"{p.name} in {p.component.name}" for p in port.connected_ports)
                return f"{marker}{name} ← {conns}"
            else:
                return f"{marker}{name} ← None"

        def format_output(name, port):
            marker = "*" if name in self.required_outputs else " "
            if port.is_connected():
                conns = ", ".join(f"{p.name} in {p.component.name}" for p in port.connected_ports)
                return f"{marker}{name} → {conns}"
            else:
                return f"{marker}{name} → None"

        inputs = "\n    ".join(format_input(name, port) for name, port in self.inputs.items()) or "    None"
        outputs = "\n    ".join(format_output(name, port) for name, port in self.outputs.items()) or "    None"

        return f"* {self.name}\n  Inputs:\n    {inputs}\n  Outputs:\n    {outputs}"



if __name__ == "__main__":

    '''
    # Create two components
    injector = Component("Injector")
    chamber = Component("Chamber")

    # Define ports
    injector.add_output("Chamber Pressure (psia)")
    chamber.add_input("Chamber Pressure (psia)", required=True)

    # Option 1: Set value on OUTPUT before connecting
    injector["Chamber Pressure (psia)"] = 500
    injector.connect(chamber)
    print("\nAfter setting injector's output value and connecting:")
    print(f"Injector port value: {injector['Chamber Pressure (psia)']}")
    print(f"Chamber port value: {chamber['Chamber Pressure (psia)']}")

    # Option 2: Reset, set input value first
    print("\nNow reversing the value direction...")

    # Reset ports manually for clean test (simulate new components)
    injector = Component("Injector")
    chamber = Component("Chamber")
    injector.add_output("Chamber Pressure (psia)")
    chamber.add_input("Chamber Pressure (psia)", required=True)

    chamber["Chamber Pressure (psia)"] = 600
    chamber.connect(injector)  # This time connecting from input side
    print("\nAfter setting chamber's input value and connecting:")
    print(f"Injector port value: {injector['Chamber Pressure (psia)']}")
    print(f"Chamber port value: {chamber['Chamber Pressure (psia)']}")

    # Option 3: Both sides set
    print("\nNow testing override...")
    injector = Component("Injector")
    chamber = Component("Chamber")
    injector.add_output("Chamber Pressure (psia)")
    chamber.add_input("Chamber Pressure (psia)", required=True)

    injector["Chamber Pressure (psia)"] = 700
    chamber["Chamber Pressure (psia)"] = 800
    injector.connect(chamber)
    print("\nAfter both sides had values and connecting (override expected):")
    print(f"Injector port value: {injector['Chamber Pressure (psia)']}")
    print(f"Chamber port value: {chamber['Chamber Pressure (psia)']} \n")'''



    print("Testing shared value bus")

    injector = Component("Injector")
    chamber = Component("Chamber")
    sensor = Component("Sensor")

    injector.add_output("Flow (kg/s)")
    chamber.add_input("Flow", required=True)
    sensor.add_input("Flow", required=True)

    print(injector)
    print(chamber)
    print(sensor)

    print(chamber['Flow'])

    injector["Flow"] = 100
    injector.connect(chamber)
    sensor.connect(injector)

    print(chamber["flow"])
    print(sensor['flow'])

    sensor["flow"] = 25

    print(chamber["Flow"])
    print(injector['flow'])




