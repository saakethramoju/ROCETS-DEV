from Components import Component
from Ports import InFlow

class Outlet(Component):
    def __init__(self, name="Outlet", port_name="Drain"):
        super().__init__(name)
        self.add_inflow(port_name)
        self.provides_inputs_only = True

    def residual(self, x):
        return []
