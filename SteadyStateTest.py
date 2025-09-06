from Components import Component, ComponentType
import Globals
from System import System
import cantera as ct
import numpy as np
from scipy.integrate import LSODA
import matplotlib.pyplot as plt
import Constants

class Pipe(Component):

    configuration_keys = ["Discharge Coefficient",
                          "Cross-sectional Area (m^2)",
                          "Length (m)"]
    state_keys = ["Mass Flow (kg/s)"]
    inflow_keys = ["Source"]
    outflow_keys = ["Drain"]
    fluid_keys = ["Fluid"]


    def flow(self):

        self["Fluid"](self["Source"].fluid)

        Cd = pipe["Discharge Coefficient"]()
        A = pipe["Cross-sectional Area (m^2)"]()
        R = 1 / (2 * (Cd * A) ** 2)
        rho = self["Source"].fluid.density_mass

        dp = self["Source"].fluid.P - self["Drain"].fluid.P
        mdot = np.sign(dp) * np.sqrt(rho * np.abs(dp) / R)
        self["Mass Flow (kg/s)"](mdot)

        f = self["Source"].fluid
        f.HP = self["Drain"].fluid.enthalpy_mass, self["Drain"].fluid.P
        self["Drain"].fluid = f

        return 0


class Tank(Component):

    configuration_keys = ["Cross-sectional Area (m^2)",
                          "Fluid Mass (kg)"]
    state_keys = ["Effective Pressure (Pa)",
                  "Effective Enthalpy (J/kg)",
                  "Ullage Pressure (Pa)",
                  "Fluid Height (m)",
                  "Fluid Volume (m^3)"]
    outflow_keys = ["Drain"]
    fluid_keys = ["Ullage",
                  "Bulk"] # take input for fluid only if Junction

    component_type = ComponentType.SOURCE




    def prime(self):

        self["Ullage Pressure (Pa)"](self["Ullage"]().P)

        P = self["Ullage Pressure (Pa)"]()
        rho = self["Ullage"]().density_mass
        V = self["Fluid Mass (kg)"]() / rho
        h = V / self["Cross-sectional Area (m^2)"]()
        P_eff = P + rho*Constants.g*h
        T = self["Bulk"]().T

        self["Bulk"]().TP = T, P_eff
        self["Effective Pressure (Pa)"](P_eff)
        self["Effective Enthalpy (J/kg)"](self ["Drain"].fluid.enthalpy_mass)
        self["Fluid Height (m)"](h)
        self["Fluid Volume (m^3)"](V)

        self["Drain"].fluid = self["Bulk"]()

        return 0




tank = Tank("Tank")
pipe = Pipe("Line")
tank.connect("Drain", pipe, "Source")


tank["Bulk"] = ct.Solution("nDodecane_Reitz.yaml")
tank["Ullage"] = ct.Solution("air.yaml")
tank["Fluid Mass (kg)"] = 1e3 
tank["Cross-sectional Area (sq. m.)"] = 0.1

pipe["Discharge Coefficient"] = 0.8
pipe["Cross-sectional Area (sq. m.)"] = 5.1e-4 
pipe["Length (m)"] = 0.5 
pipe["Drain"].fluid = ct.Solution("air.yaml")

tank.prime()

print(tank)
print(pipe)
print(tank["Ullage"]().report())
print(tank["Bulk"][100].report())

from CoolProp.CoolProp import PhaseSI
print(PhaseSI("T", 298.15, "P", 101325, "Water"))
print(PhaseSI("T", 298.15, "P", 101325, "Air"))