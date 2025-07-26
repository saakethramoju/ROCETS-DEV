from Components import Component
from Ports import InFlow

class Outlet(Component):
    def __init__(self, name="Outlet", port_name="Drain"):
        super().__init__(name)
        self.add_inflow(port_name)
        self.provides_inputs_only = True

    def residual(self, x):
        return []

    def iteration_vars(self):
        inflow = self.ports[0]  # Only one inflow by design

        has_fluid = inflow.fluid is not None and inflow.fluid.is_defined()
        has_mass_flow = inflow.mass_flow is not None

        # If one is defined but not the other → do not iterate
        if has_fluid != has_mass_flow:
            return []

        # If both are missing → solve for T and P
        if not has_fluid and not has_mass_flow:
            return [(inflow.name, "T"), (inflow.name, "P")]

        # If both are defined → downstream is fully constrained
        return []