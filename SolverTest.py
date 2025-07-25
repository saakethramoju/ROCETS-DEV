from Component import Component
from FlowPort import OutFlow, InFlow
from Fluid import Fluid
import numpy as np
from scipy.optimize import root


class PipeTest(Component):
    def __init__(self, name, Cd, A):
        super().__init__(name)
        self.Cd = Cd  # discharge coefficient
        self.A = A    # cross-sectional area [m^2]
        self._initialize_default_ports()

    def _initialize_default_ports(self):
        self.add_inflow("Source")
        self.add_outflow("Drain")

    def residual(self):
        """
        Dispatches to the correct residual function based on upstream iteration variable.
        """
        upstream_vars = self.get_upstream_iterations("Source")

        if "mass_flow" in upstream_vars:
            return self.pipe01()
        else:
            raise NotImplementedError(
                f"No residual function implemented for upstream iteration variable(s): {upstream_vars}"
            )


    def pipe01(self):
        """
        Residual function for pipe flow solver.
        x[0] is the guessed mass flow rate at the inlet (Source).
        """
        source = self["Source"]
        drain = self["Drain"]

        # Get fluid states
        fluid1 = source.fluid
        fluid2 = drain.fluid

        if fluid1 is None or fluid2 is None:
            raise ValueError("Both inlet and outlet ports must have fluid_name, P, T set.")

        rho1 = fluid1.density
        rho2 = fluid2.density

        v1 = source.mass_flow / (rho1 * self.Cd * self.A)
        v2 = np.sqrt((2 / rho2) * (source.P - drain.P + 0.5 * rho1 * v1**2))

        return [rho2 * v2 * self.Cd * self.A - source.mass_flow]

    def solve(self, initial_guess=1.0):
        """
        Solve for mass flow rate using `scipy.optimize.root`.
        """
        source = self["Source"]
        drain = self["Drain"]

        # Propagate fluid name to outlet if not yet defined
        if drain.fluid_name is None:
            drain.fluid_name = source.fluid_name

        # Trigger fluid object creation on drain (if not yet done)
        _ = drain.fluid  # access to force fluid build if T/P are ready

        result = root(self.residual, [initial_guess])
        if not result.success:
            raise RuntimeError(f"PipeTest '{self.name}' failed to converge: {result.message}")

        # Set solved mass flow back
        return result.x[0]


pipe = PipeTest("pipe", Cd=0.6, A=1e-3)

# Define inlet and outlet
inlet = Component("Inlet")
inlet.add_outflow("Source")
inlet["Source"].fluid = Fluid("Water", T=300, P=1e6)  # No mass flow yet
#inlet["Source"].mass_flow = 5

outlet = Component("Outlet")
outlet.add_inflow("Drain")
outlet["Drain"].T = 300
outlet["Drain"].P = 101325
#outlet["Drain"].mass_flow = 5

# Connect components
pipe.connect(inlet)
pipe.connect(outlet)

print(inlet.detect_iteration_variables())
print(pipe.detect_iteration_variables())
print(outlet.detect_iteration_variables())

print(pipe.get_upstream_iterations())
print(pipe.residual())
# Solve for mass flow
#mdot_solution = pipe.solve()
#inlet["Source"].mass_flow = mdot_solution
#print(f"Solved mass flow: {mdot_solution:.4f} kg/s")
