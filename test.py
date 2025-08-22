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
                          "Cross-sectional Area (sq. m.)",
                          "Length (m)"]
    state_keys = ["Mass Flow (kg/s)"]
    inflow_keys = ["Source"]
    outflow_keys = ["Drain"]

    '''def __init__(self, name, parent=None, type_=ComponentType.FLOW):
        super().__init__(name, parent, type_)'''

    def steady_state(self):

        Cd = pipe["Discharge Coefficient"]()
        A = pipe["Cross-sectional Area (sq. m.)"]() # m^2

        R = 1 / (2 * (Cd * A) ** 2)
        rho1 = self["Source"].fluid.density_mass
        rho2 = self["Drain"].fluid.density_mass
        rho = 0.5 * (rho1 + rho2)

        dp = self["Source"].fluid.P - self["Drain"].fluid.P
        return np.sign(dp) * np.sqrt(rho * np.abs(dp) / R)


    def transient(self, damping_strong=2.0, eps=200.0):

        Cd = pipe["Discharge Coefficient"]()
        A = pipe["Cross-sectional Area (sq. m.)"]() # m^2
        l = pipe["Length (m)"]()  # m
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

    configuration_keys = ["Cross-sectional Area (sq. m.)",
                          "Length (m)"]
    state_keys = ["Pressure (Pa)",
                  "Enthalpy (J/kg)",
                  "Fluid Mass (kg)"]
    outflow_keys = ["Drain"]


tank = Tank("Tank")

pipe = Pipe("Line")
pipe["Source"].fluid.TP = 300, 2e5
pipe["Discharge Coefficient"] = 0.8
#pipe["Discharge Coefficient"] = ([0, 0.5, 0.8], [0.5, 0.6, 0.8])
pipe["Cross-sectional Area (sq. m.)"] = 5.1e-4 
#t = np.linspace(0, 1, 100)
#A = 0.001 * t +  5.1e-4
#pipe["Cross-sectional Area (sq. m.)"] = (t, A)
pipe["Length (m)"] = 0.5 


t_start = 0
dt = 0.01
t_end = 1

def rhs(t, y):
    return pipe.transient()

# Initialize state
pipe["Mass Flow (kg/s)"] = 0.0   # constant initialization
solver = LSODA(rhs, t0=t_start, y0=[pipe["Mass Flow (kg/s)"].value],
               t_bound=t_end, max_step=dt)

Globals.reset_time()
while solver.status == 'running':
    Globals.set_time(solver.t) # update global simulation time
    solver.step()
    pipe["Mass Flow (kg/s)"][solver.t] = solver.y[0] # record value at solver.t

# Extract times and values from State
times, mdot_vals = pipe["Mass Flow (kg/s)"].history

plt.plot(times, mdot_vals, label="Transient mdot")
plt.title("Transient Mass Flow (kg/s)")
plt.ylabel("Mass Flow (kg/s)")
plt.xlabel("Time (s)")
plt.legend()
#plt.show()


'''
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
print("Updated history:", pipe["Mass Flow (kg/s)"].history)'''