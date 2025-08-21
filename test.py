from Components import Component

comp = Component("Tank1")

inflow = comp.add_inflow("In1")
outflow = comp.add_outflow("Out1")

test = Component("Pipe")
source = test.add_inflow("Out1")


#source.connect(outflow)
comp.connect("Out1", test, "out1")

outflow.fluid.TQ = 300, 0.5

print(comp)
print(source.mass_flow)

test["out1"] = 1
print(test)
print(comp)