from Fluids import Fluid, Mixture, Propellant
from Ports import InFlow, OutFlow
from Components import Component, MassFlowOutlet, MassFlowInlet, FluidStateInlet, FluidStateOutlet, Inlet, Outlet
from System import System

from typing import Any
import numpy as np
from scipy.optimize import root_scalar, minimize_scalar



'''
mix = Mixture({"Methane": 0.6, "Ethane": 0.4}, fraction_type="mole", T=300, P=101325)


inlet = Component("Inlet")
inlet.add_outflow("Source")
inlet.add_outflow("new source")
outlet = FluidStateOutlet("Outlet", "Source")
#inlet.connect(outlet, print_summary=True)
#inlet.connect_ports("source", outlet, "source")
inlet.connect_all(outlet, print_summary=True)

inlet["source"].fluid = mix
print(inlet["source"])
print(inlet["source"].node)

print(inlet["source"].mass_fractions)
print(inlet["source"].mole_fractions)

# Set new mole fractions
inlet["source"].mole_fractions = {"Methane": 0.8, "Ethane": 0.2}

print(outlet["source"].fluid_name)

inlet["source"].mass_fractions = {"Methane": 0.5, "Ethane": 0.5}

print(outlet["source"].fluid_name)

inlet["source"].mole_fractions = {"Methane": 0.5, "Ethane": 0.5}

inlet["source"].T = 400
inlet["source"].P = 2e6

inlet["source"].fluid.set_state(P=2e6, X=0.5)

print(outlet["source"].fluid_name)
print(outlet["source"].node.mass_fractions)
print(outlet["source"].node.mole_fractions)

print(inlet)
print(outlet)
print(outlet["source"].node)
print(inlet["source"].node)
'''



class Pipe(Component):

    configuration_keys = [
        "Discharge Coefficient",
        "Cross-Sectional Area (sq. m.)"
    ]

    def __init__(self, name):
        super().__init__(name)
        self._initialize_default_ports()
        self.configuration = {}

    def _initialize_default_ports(self):
        self.add_inflow("Source")
        self.add_outflow("Drain")

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
        result = self.pipe1()
        # Propagate mass flow to connected boundaries
        if self["Source"].is_boundary(FluidStateInlet) and self["Source"].connected_port:
            self["Source"].connected_port.mass_flow = self["Source"].mass_flow
        if self["Drain"].is_boundary(FluidStateOutlet) and self["Drain"].connected_port:
            self["Drain"].connected_port.mass_flow = self["Drain"].mass_flow
        return result
        
    
    def pipe1(self):
        fluid1 = self["Source"].fluid
        fluid2 = self["Drain"].fluid

        if fluid2.name != fluid1.name:
            self["Drain"].fluid = Fluid(fluid1.name, P=fluid2.P, T=fluid2.T)
            fluid2 = self["Drain"].fluid

        Cd  = self["Discharge Coefficient"]
        A   = self["Cross-Sectional Area (sq. m.)"]

        P1 = fluid1.P
        P2 = fluid2.P

        rho = np.mean([fluid1.density, fluid2.density])
        mdot = np.sign(P1 - P2) * Cd * A * np.sqrt(2*rho*(np.abs(P1 - P2)))
        # use this equation when using Cd directly. If trying use a friction factor
        # momentum balance will use v1 and v2 to calculate mdot

        self["Source"].mass_flow = mdot
        self["Drain"].mass_flow = mdot

        return f"Fluid out: {fluid2.name}, Mass Flow (kg/s): {mdot:.3f}"

    #def pipe2(self)




runline1 = Pipe("Runline1")
runline2 = Pipe("Runline2")

runline2.connect_ports("Source", runline1, "Drain")

inlet = FluidStateInlet("Inlet", "Fluid Out")
inlet.connect_ports("Fluid Out", runline1, "Source")

outlet = FluidStateOutlet("Outlet", "Fluid In")
runline2.connect_ports("drain", outlet, "Fluid in")

vespula = System("Vespula")
vespula.add_component(runline2)

#vespula.generate_configuration_template()
vespula.load_configuration("Vespula_Configuration.yaml")
#vespula.generate_input_template()
vespula.load_inputs("Vespula_Inputs.yaml")
#vespula.evaluate(True)
vespula.solve(verbose=True)
print(inlet)
print(runline1)
print(runline2)
print(outlet)
