from Components import Component
import numpy as np
from Fluids import Fluid


class Pipe(Component):

    configuration_keys = ["Discharge Coefficient",
                          "Cross-sectional Area (m^2)",
                          "Length (m)"]
    state_keys = ["Mass Flow (kg/s)"]
    inflow_keys = ["Source"]
    outflow_keys = ["Drain"]
    substance_keys = ["Fluid"]
    
    def steady(self):

        Cd = self["Discharge Coefficient"]()
        A = self["Cross-sectional Area (m^2)"]()
        self["Fluid"](self["Source"].fluid)

        R = 1 / (2 * (Cd * A) ** 2)
        rho = self["Fluid"]().density

        dp = self["Source"].fluid.pressure - self["Drain"].fluid.pressure
        mdot = np.sign(dp) * np.sqrt(rho * np.abs(dp) / R)
        self["Mass Flow (kg/s)"](mdot)

        return 0
    
        


pipe = Pipe("runline")
pipe["Discharge Coefficient"](0.8)
pipe["Cross-sectional Area (m^2)"] = 5e-4
pipe["Length (m)"] = 1
pipe["Source"] = Fluid("RP-1", P=2e5, T=300)
pipe["Drain"] = Fluid("nDodecane", P=101325, T=300)

pipe.steady()

print(pipe)