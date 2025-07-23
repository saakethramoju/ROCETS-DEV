import cantera as ct
import numpy as np
from scipy.optimize import root

def pipe01(mdot_guess_array, Cd, A, P1, T1, P2, T2):
    """Mass flow residual function for root-finding (mdot iteration)"""
    mdot_guess = mdot_guess_array[0]

    # Inlet
    water1 = ct.Solution("liquidvapor.yaml", "water")
    water1.TP = T1, P1
    rho1 = water1.density
    v1 = mdot_guess / (rho1 * Cd * A)

    # Outlet
    water2 = ct.Solution("liquidvapor.yaml", "water")
    water2.TP = T2, P2
    rho2 = water2.density
    v2 = np.sqrt((2 / rho2) * (P1 - P2 + 0.5 * rho1 * v1**2))

    # Residual: want rho2 * v2 * A * Cd = mdot
    residual = rho2 * v2 * Cd * A - mdot_guess
    return [residual]


# Parameters
Cd = 0.6
A = 1e-3  # m²
P1 = 2e5  # Pa
T1 = 300  # K
P2 = 101325  # Pa
T2 = 300  # K

# Solve using scipy.root
sol = root(pipe01, x0=[1.0], args=(Cd, A, P1, T1, P2, T2))

print(f"Success: {sol.success}")
print(f"mdot: {sol.x[0]:.6f} kg/s")
