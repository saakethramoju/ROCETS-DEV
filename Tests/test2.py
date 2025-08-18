import numpy as np
from scipy.optimize import root   # bracketed solver
import cantera as ct
import matplotlib.pyplot as plt

Cd = 0.8
A = 5.1e-4  # m^2
l = 0.5     # m

P1 = 2e5      # Pa
T1 = 298.15   # K
P2 = 101325   # Pa
T2 = 298.15   # K

w = ct.Water()
w.TP = T1, P1; rho1 = w.density_mass
w.TP = T2, P2; rho2 = w.density_mass


def rho_at(P):
    w.TP = T1, P
    return w.density_mass

def mdot_ss(Pu, rho_u, Pd, rho_d, Cd, A):
    """Quasi-steady orifice-like relation with sign from ΔP."""
    rho = 0.5*(rho_u + rho_d)
    dP = Pu - Pd
    return np.sign(dP) * Cd * A * np.sqrt(2.0 * rho * abs(dP))

def mdot_transient(mdot_old, dt, Pu, rho_u, Pd, rho_d, R, Z):
    rho = 0.5*(rho_u + rho_d)
    dP = Pu - Pd
    dmdot_dt = np.sign(dP) * (np.abs(dP) - R*(mdot_old**2)/rho) * (1/Z)
    mdot = dmdot_dt*dt + mdot_old
    return mdot

def residual(P_mid):
    # ensure scalar pressure for Cantera
    P = P_mid[0]
    w = ct.Water()
    w.TP = T1, P
    rho_mid = w.density_mass

    mdot1 = mdot_ss(P1, rho1, P, rho_mid, Cd, A)  # upstream leg
    mdot2 = mdot_ss(P, rho_mid, P2, rho2, Cd, A)  # downstream leg
    return mdot1 - mdot2

P_guess = 101325
sol = root(residual, P_guess)
#print(sol.x)
P_mid = sol.x[0]

w.TP = T1, P_mid
rho_mid = w.density_mass
mdot1 = mdot_ss(P1, rho1, P_mid, rho_mid, Cd, A)
mdot2 = mdot_ss(P_mid, rho_mid, P2, rho2, Cd, A)

"""print(f"P_mid = {P_mid:.2f} Pa")
print(f"mdot   = {mdot1:.6f} kg/s")
print(f"mdot   = {mdot2:.6f} kg/s")"""

# Transient params
R = 1.0 / (2.0 * (Cd*A)**2)
Z = l / A

mdot1 = 0.0
mdot2 = 0.0
t = 0.0
dt = 0.01
t_end = 0.5
P_guess = 101325

P_hist = []
M1_hist, M2_hist, T_hist = [], [], []

while t <= t_end + 1e-12:

    def residual_tr(P):
        rho_mid = rho_at(P)
        m1 = mdot_transient(mdot1, dt, P1, rho1, P, rho_mid, R, Z)
        m2 = mdot_transient(mdot2, dt, P, rho_mid, P2, rho2, R, Z)
        return m1 - m2

    # Keep P between the physical bounds
    sol = root(residual_tr, P_guess)
    #if not sol.converged:
    #    raise RuntimeError(f"Pressure solve failed at t={t:.3f}s")

    P_mid = sol.x[0]
    rho_mid = rho_at(P_mid)
    P_guess = P_mid
    # Advance mdots
    mdot1 = mdot_transient(mdot1, dt, P1, rho1, P_mid, rho_mid, R, Z)
    mdot2 = mdot_transient(mdot2, dt, P_mid, rho_mid, P2, rho2, R, Z)

    # Store history
    P_hist.append(P_mid)
    M1_hist.append(mdot1)
    M2_hist.append(mdot2)
    T_hist.append(t)

    print(f"t = {t:3f} s, P_mid = {P_mid:.1f}, mdot1 = {mdot1:.2f}, mdot2 = {mdot2:.2f}")

    t += dt

plt.plot(T_hist, M1_hist)
plt.show()