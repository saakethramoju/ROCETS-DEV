import numpy as np
from scipy.optimize import least_squares
import cantera as ct
import matplotlib.pyplot as plt

# Known states
P1, h1 = 200000, -15858020.188392682  # Pa, J/kg (300K)
P2, h2 = 101325, -15865846.944355397  # Pa, J/kg (298.15 K)
fluid = ct.Water()

# Geom / discharge
A1 = 5.1e-4   # m^2
A2 = 5.0e-4   # m^2
Cd2 = 0.3     # fixed
target_mdot = 0.1


def safe_set_HP(fluid, h, P):
    """Try setting (h, P). Return True if valid, else False."""
    try:
        fluid.HP = h, P
        return True
    except Exception:
        return False


def mass_flow(fluid, P1, h1, P2, h2, R):
    if not safe_set_HP(fluid, h1, P1):
        return 1e6  # penalty so that the solver doesn't explore that region again!!!!
    rho1 = fluid.density_mass

    if not safe_set_HP(fluid, h2, P2):
        return 1e6  # penalty
    rho2 = fluid.density_mass

    rho = 0.5 * (rho1 + rho2)
    dp = P1 - P2
    return np.sign(dp) * np.sqrt(max(rho * np.abs(dp) / R, 0.0))


def residual(x):
    P, h, Cd1 = x
    R1 = 1 / (2 * (Cd1 * A1) ** 2)
    R2 = 1 / (2 * (Cd2 * A2) ** 2)

    mdot1 = mass_flow(fluid, P1, h1, P, h, R1)
    mdot2 = mass_flow(fluid, P, h, P2, h2, R2)

    # If invalid state -> big penalty
    if abs(mdot1) > 1e5 or abs(mdot2) > 1e5:
        return [1e3, 1e3, 1e3]

    mass = (mdot1 - mdot2) / target_mdot
    energy = (h1 * mdot1 - h2 * mdot2) / (target_mdot * 1e5)
    flow = (mdot1 - target_mdot) / target_mdot

    return [mass, energy, flow]


# Initial guesses
P_guess = 1.5e5
T_guess = 300  # midpoint
fluid.TP = T_guess, P_guess
h_guess = fluid.enthalpy_mass
Cd_guess = 0.8

# Bounds: [P, h, Cd]
lb = [1e3, -2.0e7, 0.01]
ub = [3e8, 0, 0.99]

sol = least_squares(
    residual,
    [P_guess, h_guess, Cd_guess],
    bounds=(lb, ub),
    max_nfev=500,   # allow more iterations
    method="trf",   # more robust
    loss="soft_l1"  # robust to bad steps
)

# Get final state safely
if safe_set_HP(fluid, sol.x[1], sol.x[0]):
    print(f"Converged: {sol.success}, message: {sol.message}")
    print(f"Pressure: {fluid.P:.2f} Pa")
    print(f"Temperature: {fluid.T:.2f} K")
    print(f"Cd1: {sol.x[2]:.3f}")
else:
    print("Solution ended in invalid (h, P) state")
