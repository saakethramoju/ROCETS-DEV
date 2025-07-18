from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from Port import InputPort, OutputPort

class Component:
    def __init__(self, name: str):
        self.name = name
        self.inputs = {}
        self.outputs = {}


    def add_input(self, port_name: str):
        port = InputPort(name=port_name, component=self)
        self.inputs[port_name] = port
        return port

    def add_output(self, port_name: str):
        port = OutputPort(name=port_name, component=self)
        self.outputs[port_name] = port
        return port


    def connect_to_input(self, other: "Component"):
        for out_name, out_port in self.outputs.items():
            inp_port = other.inputs.get(out_name)
            if inp_port:
                out_port.connect(inp_port)
                print(f"[Connected] {self.name}: {out_name} → {other.name}: {out_name}")
        self.on_connect()

    def manual_connect(self, output_name: str, input_comp: "Component", input_name: str):
        out = self.outputs.get(output_name)
        inp = input_comp.inputs.get(input_name)
        if out and inp:
            out.connect(inp)
            print(f"[Connected] {self.name}: {output_name} → {input_comp.name}: {input_name}")
        self.on_connect()

    #def on_connect(self):
    #    return

    def __repr__(self):
        def format_input(name, port):
            if port.is_connected():
                conns = ", ".join(f"{p.name} in {p.component.name}" for p in port.connected_ports)
                return f"{name} ← {conns}"
            else:
                return f"{name} ← None"

        def format_output(name, port):
            if port.is_connected():
                conns = ", ".join(f"{p.name} in {p.component.name}" for p in port.connected_ports)
                return f"{name} → {conns}"
            else:
                return f"{name} → None"

        inputs = "\n    ".join(format_input(name, port) for name, port in self.inputs.items()) or "    None"
        outputs = "\n    ".join(format_output(name, port) for name, port in self.outputs.items()) or "    None"

        return f"* {self.name}\n  Inputs:\n    {inputs}\n  Outputs:\n    {outputs}"


if __name__ == "__main__":

    tca = Component("Heatsink")
    injector = Component("Coax")

    inj_out = injector.add_output("TCA Data")
    tca_in = tca.add_input("Injector Data")
    inj_out_2 = injector.add_output("Extra Data")

    injector.manual_connect("TCA Data", tca, "Injector Data")
    injector.manual_connect("Extra Data", tca, "Injector Data")

    #print(tca)
    #print(injector)

    inj_out.component.pressure = 6
    tca_in.component.pressure = 10

    print(inj_out.component.pressure)
    #print(tca.pressure)
    #print(injector.pressure)


