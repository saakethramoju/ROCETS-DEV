import numpy as np
from scipy.optimize import root 
from scipy.integrate import solve_ivp
import cantera as ct
import matplotlib.pyplot as plt

t_start = 0
dt = 0.01
t_end = 1

P1, T1 = 150000, 300 # Pa, K
P2, T2 = 101325, 298.15 # Pa, K
fluid = ct.Water()

mdot0 = 0
Cd = 0.8
A = 5.1e-4 # m^2
l = 1 # m

R = 1 / (2 * (Cd*A)**2)
L = l / A


def pipe_ss(fluid, P1, T1, P2, T2, R):
    fluid.TP = T1, P1
    rho1 = fluid.density_mass
    fluid.TP = T2, P2
    rho2 = fluid.density_mass
    rho = 0.5*(rho1 + rho2)

    dp = P1 - P2
    return np.sign(dp) * np.sqrt(rho * np.abs(dp) / R )

def pipe(fluid, P1, T1, P2, T2, R, L, mdot_old, dt):
    fluid.TP = T1, P1
    rho1 = fluid.density_mass
    fluid.TP = T2, P2
    rho2 = fluid.density_mass
    rho = 0.5*(rho1 + rho2)

    dp = P1 - P2
    dmdt = np.sign(dp) * (np.abs(dp) - R*(mdot_old**2)/rho) * (1/L)
    
    return dmdt


T = np.arange(t_start, t_end + dt, dt)
mdot = np.zeros(len(T))
mdot_ss = pipe_ss(fluid, P1, T1, P2, T2, R)


def rhs(t, y):
    return pipe(fluid, P1, T1, P2, T2, R, L, y[0], dt)

sol = solve_ivp(rhs, [t_start, t_end], [mdot0], method="LSODA", t_eval=T)#, rtol=1e-6, atol=1e-9, max_step=1e-3)

plt.plot(sol.t, sol.y[0], label="Transient mdot")
plt.axhline(mdot_ss, color="r", linestyle="--", label="Steady-state")
plt.title(f"Transient Mass Flow (kg/s), Cd = {Cd}, A = {A} m^2")
plt.ylabel("Mass Flow (kg/s)")
plt.xlabel("Time (s)")
plt.legend()
plt.show()
