from Fluid import Fluid
from Ports import InFlow, OutFlow
from Components import Component, MassFlowOutlet, MassFlowInlet, FluidStateInlet, FluidStateOutlet

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
print(outlet["source"].fluid)