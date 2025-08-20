import numpy as np
from scipy.integrate import LSODA
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


def pipe(fluid, P1, T1, P2, T2, R, L, mdot_old):
    fluid.TP = T1, P1
    rho1 = fluid.density_mass
    fluid.TP = T2, P2
    rho2 = fluid.density_mass
    rho = 0.5*(rho1 + rho2)

    dp = P1 - P2
    dmdt = np.sign(dp) * (np.abs(dp) - R*(mdot_old**2)/rho) * (1/L)
    
    return dmdt


def rhs(t, y):
    return [pipe(fluid, P1, T1, P2, T2, R, L, y[0])]


solver = LSODA(rhs, t0=t_start, y0=[mdot0], t_bound=t_end, max_step=dt)

t_vals = [solver.t]
mdot_vals = [solver.y[0]]

while solver.status == 'running':
    solver.step()
    t_vals.append(solver.t)
    mdot_vals.append(solver.y[0])


t_vals = np.array(t_vals)
mdot_vals = np.array(mdot_vals)

mdot_ss = pipe_ss(fluid, P1, T1, P2, T2, R)

plt.plot(t_vals, mdot_vals, label="Transient mdot")
plt.axhline(mdot_ss, color="r", linestyle="--", label="Steady-state")
plt.title(f"Transient Mass Flow (kg/s), Cd = {Cd}, A = {A} m^2")
plt.ylabel("Mass Flow (kg/s)")
plt.xlabel("Time (s)")
plt.legend()
plt.show()
