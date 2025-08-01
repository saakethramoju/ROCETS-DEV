import numpy as np
from Fluids import Fluid
from Components import (
    Component, MassFlowOutlet, MassFlowInlet,
    FluidStateInlet, FluidStateOutlet, Inlet, Outlet
)


class IncompressibleLine(Component):
    """
    Represents an incompressible pipe-like component.
    Solves flow through a duct using Bernoulli-based approximations, assuming:
    - No heat transfer -> adiabatic (for now)
    - No work interactions -> frictionless (for now)
    - Incompressible fluid behavior
    """

    configuration_keys = [
        "Discharge Coefficient",
        "Cross-Sectional Area (sq. m.)"
    ]

    def __init__(self, name):
        super().__init__(name)
        self._initialize_default_ports()
        self.configuration = {}

    def _initialize_default_ports(self):
        """
        Sets up the default inlet and outlet ports.
        """
        # Inflows:
        self.add_inflow("Source")

        # Outflows:
        self.add_outflow("Drain")

        # Property Outs:
        self.add_property_out("Mass Flow (kg/s)")

    '''
    def get_additional_iteration_variables(self) -> list[tuple[str, Any]]:
        #if not self["Source"].is_boundary(Inlet):
        #    return [(f"{self.name}:Mass Flow (kg/s)", self["Source"].mass_flow)]
        return []

    def set_additional_iteration_variable(self, label: str, value: float):
        expected = f"{self.name}:Mass Flow (kg/s)"
        if label == expected:
            self["Source"].mass_flow = value
    '''

    def evaluate(self):
        """
        Selects the appropriate flow-solving method based on available boundary conditions.
        Also propagates mass flow to connected fluid-state boundaries.
        """
        source = self["Source"]
        drain = self["Drain"]

        source_inlet = source.is_boundary(MassFlowInlet)
        drain_outlet = drain.is_boundary(MassFlowOutlet)

        # Check for conflicting mass flow boundary conditions
        if source_inlet and drain_outlet and source.mass_flow != drain.mass_flow:
            raise Exception(f"[{self.name}] Conflicting mass flow boundary conditions!")

        # Dispatch to appropriate solver
        if source_inlet:
            result = self.pipe2()
        elif drain_outlet:
            result = self.pipe3()
        else:
            result = self.pipe1()

        # Propagate mass flow to FluidState boundaries if connected
        if source.is_boundary(FluidStateInlet, Inlet) and source.connected_port:
            source.connected_port.mass_flow = source.mass_flow
        if drain.is_boundary(FluidStateOutlet, Outlet) and drain.connected_port:
            drain.connected_port.mass_flow = drain.mass_flow

        self["Mass Flow (kg/s)"] = result

        return result

    def _align_fluids(self):
        """
        Ensures that the downstream fluid object matches the upstream type.
        If different, replaces the drain fluid with a new Fluid of the same type.
        """
        source = self["Source"]
        drain = self["Drain"]
        fluid1 = source.fluid
        fluid2 = drain.fluid

        if fluid2.name != fluid1.name:
            drain.fluid = Fluid(fluid1.name, P=fluid2.P, T=fluid2.T)

    def pipe1(self):
        """
        Internal → internal connection.
        Solves for mass flow rate using upstream and downstream pressures.
        """
        #print("PIPE1")
        self._align_fluids()

        source = self["Source"]
        drain = self["Drain"]
        f1, f2 = source.fluid, drain.fluid

        Cd = self["Discharge Coefficient"]
        A = self["Cross-Sectional Area (sq. m.)"]
        rho = np.mean([f1.density, f2.density])
        dP = f1.P - f2.P

        mdot = np.sign(dP) * Cd * A * np.sqrt(2 * rho * abs(dP))

        source.mass_flow = drain.mass_flow = mdot
        return mdot

    def pipe2(self):
        """
        MassFlowInlet boundary present.
        Solves for upstream pressure and temperature given downstream state and mdot.
        """
        #print("PIPE2")
        self._align_fluids()

        source = self["Source"]
        drain = self["Drain"]
        f1, f2 = source.fluid, drain.fluid

        Cd = self["Discharge Coefficient"]
        A = self["Cross-Sectional Area (sq. m.)"]
        mdot = source.mass_flow
        rho = f2.density
        P2 = f2.P
        h2 = f2.enthalpy

        sign = np.sign(mdot)
        P1 = (mdot / (Cd * A))**2 / (2 * rho) + sign * P2
        T1 = Fluid.get_temperature_from_Ph(f1.name, P1, h2)
        """
        if isinstance(f1, Mixture):
            T1 = Mixture.get_temperature_from_Ph(f1.components, P1, h2, f1.fraction_type, debug=True)
        else:
            T1 = Fluid.get_temperature_from_Ph(f1.name, P1, h2, debug=True)"""


        source.P = P1
        source.T = T1
        drain.mass_flow = mdot

        return mdot

    def pipe3(self):
        """
        MassFlowOutlet boundary present.
        Solves for downstream pressure and temperature given upstream state and mdot.
        """
        #print("PIPE3")
        self._align_fluids()

        source = self["Source"]
        drain = self["Drain"]
        f1, f2 = source.fluid, drain.fluid

        Cd = self["Discharge Coefficient"]
        A = self["Cross-Sectional Area (sq. m.)"]
        mdot = drain.mass_flow
        rho = f1.density
        P1 = f1.P
        h1 = f1.enthalpy

        sign = np.sign(mdot)
        P2 = P1 - sign * (mdot / (Cd * A))**2 / (2 * rho)
        T2 = Fluid.get_temperature_from_Ph(f1.name, P2, h1)
        """
        if isinstance(f1, Mixture):
            T2 = Mixture.get_temperature_from_Ph(f1.components, P2, h1, f1.fraction_type, debug=True)
        else:
            T2 = Fluid.get_temperature_from_Ph(f1.name, P2, h1, debug=True)"""


        drain.P = P2
        drain.T = T2
        source.mass_flow = mdot

        return mdot
