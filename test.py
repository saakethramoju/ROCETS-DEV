from Fluids import Fluid, Mixture, Propellant
from Ports import InFlow, OutFlow
from Components import Component, MassFlowOutlet, MassFlowInlet, FluidStateInlet, FluidStateOutlet
'''
f = Fluid("Water", P=2e6, X = 0.5)
print(f)

a = OutFlow("Source")
b = InFlow("Drain")

a.fluid = f
print(a.P)
a.P = 1e6
print(a.P)
print(a.T)
print(a.X)
a.fluid = Fluid('Methane', P=2e6, T = 100)
print(a.P)
print(a.T)
print(a.X)

b.fluid = Fluid('Water', P=2e6, T = 100)
a.connect(b)

print(b.fluid)
b.T= 200
#b.X= 0.3
print(a)
print(a.node)
a.node.T = 400
#a.node.X = 0.4
print(a)
print(b)

b.fluid.set_state(P=3e6, X=0.4)
print(a)
print(b)
print(b.node)
print(a.node)

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

mix = Mixture({"Methane": 0.6, "Ethane": 0.4}, fraction_type="mole", T=300, P=101325)


#print(mix) # slow by nature
#mix.set_state(P=2e6, X = 0.4)
#print("---------------------------------")
#print(mix)

'''
inlet = Component("Inlet")
inlet.add_outflow("Source")
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
print(inlet["source"].node)'''



#print(outlet["source"].fluid)
'''

print(inlet["source"].mass_fractions)
print(inlet["source"].mole_fractions)

print(inlet["source"].fluid)

# Or set new mass fractions instead
inlet["source"].mass_fractions = {"Methane": 0.5, "Ethane": 0.5}

print(inlet["source"].mass_fractions)
print(inlet["source"].mole_fractions)


print(inlet["source"].node.mass_fractions)

print(inlet["source"].fluid)'''


fuel = Propellant("RP-1", T=300, P=101325)
print(fuel)

