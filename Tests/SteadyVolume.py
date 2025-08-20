import numpy as np
from scipy.optimize import root 
from scipy.integrate import solve_ivp
import cantera as ct
import matplotlib.pyplot as plt
from ambiance import Atmosphere

P = 2e5    # Pa
T = 300    # K

atm = Atmosphere(h=0)
Pamb = atm.pressure[0]
Tamb = atm.temperature[0]

mdot0 = 0   # kg/s
M0 = 1e3    # kg

fluid = ct.Water()

Cd = 0.8
A = 5.1e-4 # m^2
R = 1 / (2 * (Cd*A)**2)

def pipe(fluid, P1, T1, P2, T2, R):
    fluid.TP = T1, P1
    rho1 = fluid.density_mass
    fluid.TP = T2, P2
    rho2 = fluid.density_mass
    rho = 0.5*(rho1 + rho2)

    dp = P1 - P2
    return np.sign(dp) * np.sqrt(rho * np.abs(dp) / R )

mdot = pipe(fluid, P, T, Pamb, Tamb, R)

print(f"Steady-state mass flow: {mdot:.3f} kg/s")


