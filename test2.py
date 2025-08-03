import numpy as np
from scipy.optimize import root_scalar
from Fluids import Fluid  # Your Fluid class
import matplotlib.pyplot as plt

# Constants
dt = 0.1
t_end = 3
T = 298.15
V = 1.0  # m³
step = 0

P1 = 3e5      # inlet pressure
P2 = 101325   # outlet pressure
Cd = 0.8
A = 0.8e-5    # m²

# Initial condition
P_guess = 101325.0
f = Fluid(T=T, P=P_guess)
rho = f.density
M = rho * V

# Data storage
t_vals = [0]
P_vals = [P_guess]
M_vals = [M]

def mdot(P_up, P_down, rho, Cd, A):
    if P_up <= P_down:
        return 0.0
    return Cd * A * np.sqrt(2 * rho * (P_up - P_down))

# Time stepping
while t_vals[-1] < t_end + dt:
    M_prev = M  # known mass at tⁿ

    def residual(P):
        f = Fluid(T=T, P=P)
        rho = f.density
        m_in = mdot(P1, P, rho, Cd, A)
        m_out = mdot(P, P2, rho, Cd, A)
        if step == 0:
            return m_out
        M_est = rho * V
        return M_est - M_prev - dt * (m_in - m_out)

    sol = root_scalar(residual, bracket=[1e4, 5e5], method='brentq')
    if not sol.converged:
        raise RuntimeError(f"Root solve failed at t = {t_vals[-1]}s")

    P_next = sol.root
    f = Fluid(T=T, P=P_next)
    rho = f.density
    M = rho * V

    m_in = mdot(P1, P_next, rho, Cd, A)
    m_out = mdot(P_next, P2, rho, Cd, A)

    print(f"t = {t_vals[-1]:.1f} s | P = {P_next:.2f} Pa | M = {M:.4f} kg, mdot1 = {m_in:.2f}, mdot2 = {m_out:.2f}")

    # Save data
    t_vals.append(t_vals[-1] + dt)
    P_vals.append(P_next)
    M_vals.append(M)
    step += 1
