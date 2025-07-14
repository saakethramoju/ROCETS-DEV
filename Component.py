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

    def transmit(self, port_name: str, value: any):
        self.outputs[port_name].transmit(value)

    def connect_to_input(self, other: "Component"):
        # Build a lowercase map of the other component's input ports
        input_map = {name.lower(): port for name, port in other.inputs.items()}

        for out_name, out_port in self.outputs.items():
            matched_in_port = input_map.get(out_name.lower())

            if matched_in_port:
                out_port.connect(matched_in_port)
                print(f"[Connected] {self.name}.{out_name} → {other.name}.{matched_in_port.name}")
            else:
                print(f"[Warning] No input port found for output '{out_name}' in component '{other.name}'")

    def manual_connect(self, output_port: str, input_component: "Component", input_port: str):
        if output_port not in self.outputs:
            print(f"[Error] Output port '{output_port}' not found in component '{self.name}'")
            return

        if input_port not in input_component.inputs:
            print(f"[Error] Input port '{input_port}' not found in component '{input_component.name}'")
            return

        self.outputs[output_port].connect(input_component.inputs[input_port])
        #print(f"[Connected] {self.name}.{output_port} → {input_component.name}.{input_port}")


    def __repr__(self):
        inputs_lines = "\n    ".join(
            f"{p.name} ← {p.connected_port.component} as {p.connected_port.name}"
            if p.connected_port else f"{p.name} ← None"
            for p in self.inputs.values()
        ) or "    None"

        outputs_lines = "\n    ".join(
            f"{p.name} → {p.connected_port.component} as {p.connected_port.name}"
            if p.connected_port else f"{p.name} → None"
            for p in self.outputs.values()
        ) or "    None"

        return (
            f"* {self.name}\n"
            f"  Inputs:\n    {inputs_lines}\n"
            f"  Outputs:\n    {outputs_lines}"
        )

    
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

