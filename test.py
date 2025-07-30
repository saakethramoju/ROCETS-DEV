from Fluids import Fluid, Mixture, Propellant
from Ports import InFlow, OutFlow
from Components import Component, MassFlowOutlet, MassFlowInlet, FluidStateInlet, FluidStateOutlet, Inlet, Outlet
from System import System
import numpy as np
'''

inlet = Component("Inlet")
inlet.add_outflow("Source")
outlet = FluidStateOutlet("Outlet", "Source")

inlet.connect(outlet, print_summary=True)

print(inlet["source"])
inlet["source"].fluid = Fluid("Oxygen", P=2e6, X=0.4)
print(inlet["source"])
print(outlet["source"])
outlet["source"].P = 1e6
#outlet["source"].T = 200
print(inlet["source"])
print(outlet["source"])
inlet["source"].fluid.set_state(T=100, X=0.4)
print(inlet["source"])
print(outlet["source"].fluid)'''


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


system = System("Vespula")

system.add_component(inlet)

print(system)

thing = Component("Thing")
thing.add_inflow("new source")
thing.connect(inlet)
print(system)

#new_system = System("Subscale")
#new_system.add_component(inlet)'''




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

    def evaluate(self):
        if self["Source"].is_boundary(FluidStateInlet):
            result = self.pipe1()
            # Propagate mass flow to connected boundaries
            if self["Source"].connected_port:
                self["Source"].connected_port.mass_flow = self["Source"].mass_flow
            if self["Drain"].is_boundary(Outlet) and self["Drain"].connected_port:
                self["Drain"].connected_port.mass_flow = self["Drain"].mass_flow
            return result
        
    def pipe1(self):
        fluid1 = self["Source"].fluid
        fluid2 = self["Drain"].fluid

        dp = self["Source"].P - self["Drain"].P
        rho = fluid1.density
        Cd = self["Discharge Coefficient"]
        A = self["Cross-Sectional Area (sq. m.)"]

        mdot = Cd*A*np.sqrt(2*rho*dp)

        self["Source"].mass_flow = mdot
        self["Drain"].mass_flow = mdot
        self["Drain"].fluid = Fluid(fluid1.name, P=fluid2.P, T=fluid2.T)

        return f"Fluid out: {fluid2.name}, Mass Flow (kg/s): {mdot:.3f}"



pipe = Pipe("Runline")

inlet = FluidStateInlet("Inlet", "Fluid Out")
inlet.connect_ports("Fluid Out", pipe, "Source")

outlet = FluidStateOutlet("Outlet", "Fluid In")
pipe.connect_ports("drain", outlet, "Fluid in")

vespula = System("Vespula")
vespula.add_component(pipe)
#vespula.generate_configuration_template()
vespula.load_configuration("Vespula_Configuration.yaml")
#vespula.generate_input_template()
vespula.load_inputs("Vespula_Inputs.yaml")
vespula.evaluate(True)
print(inlet)
print(pipe)
print(outlet)




'''
pipe = Pipe("Runline")
sys = System("Sys")
sys.add_component(pipe)
#sys.generate_configuration_template()
sys.load_configuration("Sys_Configuration.yaml")
#sys.generate_input_template()
sys.load_inputs("Sys_Inputs.yaml")
print(pipe.evaluate())
'''

'''
pipe = Pipe("Runline")
sys = System("Sys")
sys.add_component(pipe)
outlet = FluidStateOutlet("Outlet", "Drain")
outlet.connect(pipe)
sys.generate_input_template()
#sys.load_inputs("Sys_Inputs.yaml")
print(pipe)'''


#f = Fluid("Water", T = 200, P = 101325)
#print(f)