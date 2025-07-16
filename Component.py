from dataclasses import dataclass
from Port import InputPort, OutputPort

class Component:
    def __init__(self, name: str):
        self.name = name
        self.inputs = {}
        self.outputs = {}

    def add_input(self, port_name: str):
        port = InputPort(name=port_name, component=self.name)
        self.inputs[port_name] = port
        return port

    def add_output(self, port_name: str):
        port = OutputPort(name=port_name, component=self.name)
        self.outputs[port_name] = port
        return port

    def receive(self, port_name: str):
        return self.inputs[port_name].receive()

    def transmit(self, port_name: str, value):
        self.outputs[port_name].transmit(value)

    def connect_to_input(self, other: "Component"):
        for out_name, out_port in self.outputs.items():
            match = other.inputs.get(out_name)
            if match:
                out_port.connect(match)
                print(f"[Connected] {self.name}: {out_name} → {other.name}: {match.name}")
            else:
                print(f"[Warning] No input port for output '{out_name}' in '{other.name}'")

    def manual_connect(self, output_port: str, input_component: "Component", input_port: str):
        out = self.outputs.get(output_port)
        inp = input_component.inputs.get(input_port)

        if out and inp:
            out.connect(inp)
            print(f"[Connected] {self.name}: {output_port} → {input_component.name}: {input_port}")
        else:
            print(f"[Error] Could not connect {output_port} to {input_port}")

    def __repr__(self):
        def port_str(port, arrow, default="None"):
            return f"{port.name} {arrow} {port.connected_port.component} as {port.connected_port.name}" if port.connected_port else f"{port.name} {arrow} {default}"

        inputs_str = "\n    ".join(port_str(p, "←") for p in self.inputs.values()) or "    None"
        outputs_str = "\n    ".join(port_str(p, "→") for p in self.outputs.values()) or "    None"

        return f"* {self.name}\n  Inputs:\n    {inputs_str}\n  Outputs:\n    {outputs_str}"


if __name__ == "__main__":

    A = Component("Tank")
    B = Component("Engine")
    C = Component("COPV")

    TankOut = A.add_output("tank Data")
    TankIn = A.add_input("Pressure Data")
    EngineIn = B.add_input("Tank Data")
    COPVOut = C.add_output("COPV Data")

    A.connect_to_input(B)
    C.manual_connect("COPV Data", A, "Pressure Data")

    print(A)
    print(B)
