import numpy as np
from scipy.optimize import least_squares
import cantera as ct
import matplotlib.pyplot as plt

t_start = 0
dt = 0.01
t_end = 1


# Known states
P1, h1 = 200000, -15858020.188392682  # Pa, J/kg
P2, h2 = 101325, -15865846.944355397  # Pa, J/kg
fluid = ct.Water()

# Geom / discharge
A1 = 5.1e-4   # m^2
A2 = 5.0e-4   # m^2
Cd1 = 0.8     # fixed
Cd2 = 0.7     # fixed
l1 = 0.5      # m
l2 = 0.75     # m



def safe_set_HP(fluid, h, P):
    """Try setting (h, P). Return True if valid, else False."""
    try:
        fluid.HP = h, P
        return True
    except Exception:
        return False

def mass_flow_ss(fluid, P1, h1, P2, h2, R):
    if not safe_set_HP(fluid, h1, P1):
        return 1e6  # penalty so that the solver doesn't explore that region again!!!!
    rho1 = fluid.density_mass

    if not safe_set_HP(fluid, h2, P2):
        return 1e6  # penalty
    rho2 = fluid.density_mass

    rho = 0.5 * (rho1 + rho2)
    dp = P1 - P2
    return np.sign(dp) * np.sqrt(max(rho * np.abs(dp) / R, 0.0))


def mass_flow(fluid, P1, h1, P2, h2, R, L, mdot_old):
    if not safe_set_HP(fluid, h1, P1):
        return 1e6 
    rho1 = fluid.density_mass

    if not safe_set_HP(fluid, h2, P2):
        return 1e6  # penalty
    rho2 = fluid.density_mass

    rho = 0.5*(rho1 + rho2)

    dp = P1 - P2
    dmdt = np.sign(dp) * (np.abs(dp) - R*(mdot_old**2)/rho) * (1/L)
    
    return dmdt

T = np.arange(t_start, t_end + dt, dt)
mdot = np.zeros(len(T))
