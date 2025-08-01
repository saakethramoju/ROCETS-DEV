from Fluids import Fluid, Mixture, Propellant
from Ports import InFlow, OutFlow
from Components import Component, MassFlowOutlet, MassFlowInlet, FluidStateInlet, FluidStateOutlet, Inlet, Outlet, IncompressibleLine, Sensor
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


runline1 = IncompressibleLine("Runline1")
runline2 = IncompressibleLine("Runline2")
inlet = FluidStateInlet("Inlet", "Fluid Out")
#inlet = MassFlowInlet("Inlet", "Fluid Out")
outlet = FluidStateOutlet("Outlet", "Fluid In")
flow_meter1 = Sensor("Runline1 Flow Meter")
flow_meter2 = Sensor("Runline2 Flow Meter")

vespula = System("Vespula")
vespula.add_component(runline2)

runline2.connect_ports("Source", runline1, "Drain")
inlet.connect_ports("Fluid Out", runline1, "Source")
runline2.connect_ports("drain", outlet, "Fluid in")
#runline1.print_properties()
flow_meter1.connect_ports("Value", runline1, "Mass Flow")
flow_meter2.connect_ports("Value", runline2, "Mass Flow")

#vespula.generate_configuration_template()
vespula.load_configuration("Vespula_Configuration.yaml")
#vespula.generate_input_template()
vespula.load_inputs("Vespula_Inputs.yaml")
#vespula.load_inputs("Vespula_Inputs_1.yaml")
#vespula.evaluate(True)
vespula.solve(verbose=True)


print(inlet)
print(runline1)
print(runline2)
print(outlet)
print(flow_meter1)
print(flow_meter2)

vespula.export()


