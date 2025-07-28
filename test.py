from Ports import InFlow, OutFlow
from Fluid import Fluid
from Components import Component

inlet = InFlow("Inlet")
outlet = OutFlow("Outlet")


# Connect BEFORE defining state
outlet.connect(inlet)
print(inlet.node)  # node_0, all None

# Change via property setters → node + other port update
inlet.T = 300
inlet.P = 101325
print(inlet.node)

# Push a full Fluid in one shot → syncs everywhere
outlet.fluid = Fluid("Water", T=350, P=101325)
print(inlet.T, inlet.P)      # 350, 101325  (synced from node)
print(outlet.T, outlet.P)    # 350, 101325
print(inlet.node)            # node updated

# Mass flow
inlet.mass_flow = 0.2
print(inlet.mass_flow)
print(outlet.node.inlet_mass_flow)


a = Component("Heater")
a.add_inflow("inlet")
a.add_outflow("outlet")
boop = a.add_property_out("Mixing Efficiency")

b = Component("Pump")
b.add_inflow("outlet")
b.add_outflow("drain")
b.add_property_in("Mixing Efficiency")

c = Component("Thing")
beep = c.add_property_in("Mixing Efficiency")

a.connect_all(b, print_summary=True)



