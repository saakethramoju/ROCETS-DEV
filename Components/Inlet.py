from Components import Component
from Ports import OutFlow

class Inlet(Component):
    def __init__(self, name="Inlet", port_name="Source"):
        super().__init__(name)
        self.add_outflow(port_name)
        self.provides_inputs_only = True

    def residual(self, x):
        # No residuals — boundary condition only
        return []

    def iteration_vars(self):
        out = self.ports[0]
        if out.fluid is not None and out.fluid.is_defined():
            return [(out.name, "mass_flow")]
        else:
            return [(out.name, "T"), (out.name, "P")]

