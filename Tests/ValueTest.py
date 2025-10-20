from Components import Component, ComponentType
import numpy as np
import Constants
from Fluids import Fluid

class Pipe(Component):

    configuration_keys = ["Discharge Coefficient",
                          "Cross-sectional Area (m^2)",
                          "Length (m)"]
    state_keys = ["Mass Flow (kg/s)"]
    inflow_keys = ["Source"]
    outflow_keys = ["Drain"]
    substance_keys = ["Fluid"]


    def steady_state(self):

        self["Fluid"](self["Source"].fluid)

        Cd = self["Discharge Coefficient"]()
        A = self["Cross-sectional Area (m^2)"]() # m^2

        R = 1 / (2 * (Cd * A) ** 2)
        rho1 = self["Source"].fluid.density_mass
        rho2 = self["Drain"].fluid.density_mass
        rho = 0.5 * (rho1 + rho2)

        dp = self["Source"].fluid.P - self["Drain"].fluid.P
        mdot = np.sign(dp) * np.sqrt(rho * np.abs(dp) / R)
        self["Mass Flow (kg/s)"](mdot)

        f = self["Source"].fluid
        f.HP = self["Drain"].fluid.enthalpy_mass, self["Drain"].fluid.P
        self["Drain"].fluid = f

        return 0


    def transient(self, damping_strong=2.0, eps=200.0):

        self["Fluid"](self["Source"].fluid)

        Cd = self["Discharge Coefficient"]()
        A = self["Cross-sectional Area (m^2)"]() # m^2
        l = self["Length (m)"]()  # m
        mdot_old = self["Mass Flow (kg/s)"]()

        L = l / A
        R = 1 / (2 * (Cd * A) ** 2)

        rho1 = self["Source"].fluid.density_mass
        rho2 = self["Drain"].fluid.density_mass
        rho = 0.5 * (rho1 + rho2)
        dp = self["Source"].fluid.P - self["Drain"].fluid.P

        drive = np.sign(dp) * (np.abs(dp) - R * (mdot_old**2) / rho) * (1 / L)

        # adaptive damping that gets VERY strong when |dp| < eps
        damp_coeff = damping_strong / (1 + (np.abs(dp) / eps))
        decay = -damp_coeff * mdot_old

        return drive + decay
    

class Tank(Component):

    configuration_keys = ["Cross-sectional Area (m^2)",
                          "Fluid Mass (kg)"]
    state_keys = ["Effective Pressure (Pa)",
                  "Effective Enthalpy (J/kg)",
                  "Ullage Pressure (Pa)",
                  "Fluid Height (m)",
                  "Fluid Volume (m^3)"]
    outflow_keys = ["Drain"]
    substance_keys = ["Ullage",
                  "Bulk"] # take input for fluid only if Junction

    component_type = ComponentType.SOURCE

    def steady_state(self):

        self["Ullage Pressure (Pa)"](self["Ullage"]().P)

        P = self["Ullage Pressure (Pa)"]()
        rho = self["Ullage"]().density_mass
        V = self["Fluid Mass (kg)"]() / rho
        h = V / self["Cross-sectional Area (m^2)"]()
        P_eff = P + rho*Constants.g*h
        T = self["Bulk"]().T

        self["Bulk"]().TP = T, P_eff
        self["Drain"].fluid = self["Bulk"]()
        self["Effective Pressure (Pa)"](P_eff)
        self["Effective Enthalpy (J/kg)"](self ["Drain"].fluid.enthalpy_mass)
        self["Fluid Height (m)"](h)
        self["Fluid Volume (m^3)"](V)
        return 0


class Volume(Component): 

    configuration_keys = ["Volume (m^3)"]
    state_keys = ["Mass (kg)",
                  "Pressure (Pa)",
                  "Enthalpy (J/kg)"]
    inflow_keys = ["In"]
    outflow_keys = ["Out"]
    fluid_keys = ["Fluid"]

    component_type = ComponentType.JUNCTION

    iteration_keys = ["Pressure (Pa)",
                      "Enthalpy (J/kg)"]

    def steady_state(self):
        self["Fluid"]()



tank = Tank("Tank")
pipe = Pipe("Line")

tank["Bulk"][5] = Fluid("Water", P=101325, T=288.15)


print("=== Parameter Tests ===")
# --- Constant ---
pipe["Discharge Coefficient"] = 0.8
print("Constant value:", pipe["Discharge Coefficient"].value)
print("Cd[0.0]:", pipe["Discharge Coefficient"][0.0])
print("Cd[10.0]:", pipe["Discharge Coefficient"][10.0])
print("History:", pipe["Discharge Coefficient"].history)

# Update at a new time (promotes automatically)
pipe["Discharge Coefficient"][5.0] = 0.75
print("Cd at 5.0:", pipe["Discharge Coefficient"][5.0])
print("History after promotion:", pipe["Discharge Coefficient"].history)


# --- Time series via list of tuples ---
pipe["Cross-sectional Area (sq. m.)"] = [
    (0.0, 5.1e-4),
    (0.5, 4.9e-4),
    (1.0, 4.5e-4),
]
print("A[0.3]:", pipe["Cross-sectional Area (sq. m.)"][0.3])
print("A[0.75]:", pipe["Cross-sectional Area (sq. m.)"][0.75])
print("History:", pipe["Cross-sectional Area (sq. m.)"].history)

# Insert new point
pipe["Cross-sectional Area (sq. m.)"][0.75] = 4.7e-4
print("After insertion:", pipe["Cross-sectional Area (sq. m.)"][0.75])
print("Updated history:", pipe["Cross-sectional Area (sq. m.)"].history)


# --- Time series via arrays ---
t_vals = np.array([0.0, 0.5, 1.0])
cd_vals = np.array([0.8, 0.7, 0.5])
pipe["Discharge Coefficient"] = (t_vals, cd_vals)
print("Cd[0.3]:", pipe["Discharge Coefficient"][0.3])
print("Cd[0.9]:", pipe["Discharge Coefficient"][0.9])
print("History:", pipe["Discharge Coefficient"].history)


print("\n=== State Tests ===")
# --- Constant ---
pipe["Mass Flow (kg/s)"] = 0.0
print("Initial value:", pipe["Mass Flow (kg/s)"].value)
print("At t=2.0 (still constant):", pipe["Mass Flow (kg/s)"][2.0])
print("History:", pipe["Mass Flow (kg/s)"].history)

# Promote to time series
pipe["Mass Flow (kg/s)"][2.0] = 0.1
print("At t=2.0 after promotion:", pipe["Mass Flow (kg/s)"][2.0])
print("History after promotion:", pipe["Mass Flow (kg/s)"].history)

# --- Time series via arrays ---
t_vals = np.array([0.0, 0.5, 1.0])
v_vals = np.array([0.0, 0.05, 0.1])
pipe["Mass Flow (kg/s)"] = (t_vals, v_vals)
print("At t=0.3:", pipe["Mass Flow (kg/s)"][0.3])
print("At t=0.9:", pipe["Mass Flow (kg/s)"][0.9])
print("History:", pipe["Mass Flow (kg/s)"].history)

# Insert mid-series
pipe["Mass Flow (kg/s)"][0.75] = 0.08
print("After insertion at 0.75:", pipe["Mass Flow (kg/s)"][0.75])
print("Updated history:", pipe["Mass Flow (kg/s)"].history)