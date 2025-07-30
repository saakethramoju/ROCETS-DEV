from Fluids import Fluid, Mixture, Propellant
from Ports import InFlow, OutFlow
from Components import Component, MassFlowOutlet, MassFlowInlet, FluidStateInlet, FluidStateOutlet
from System import System
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

pipe = Pipe("Runline")
#print(pipe["dishcarge coefficient"])
inlet = FluidStateInlet("Inlet", "Fluid Out")
inlet.connect_ports("Fluid Out", pipe, "Source")

vespula = System("Vespula")
vespula.add_component(pipe)

print(vespula)
#vespula.generate_configuration_template()
vespula.load_configuration("Vespula_Configuration.yaml")

print(pipe["dishcarge coefficient"])