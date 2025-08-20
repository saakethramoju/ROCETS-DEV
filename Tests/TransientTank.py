import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from ambiance import Atmosphere
from scipy.integrate import LSODA

t_start = 0
dt = 0.01
t_end = 500

# Tank initial conditions
P = 101325    # Pa (designated base pressure)
T = 300    # K
M0 = 1e5   # kg
fluid = ct.Water()

fluid.TP = T, P
V0 = M0 / fluid.density_mass   # initial tank volume

# Pipe
mdot0 = 0.0   # kg/s
Cd = 0.8
A = 5.1e-4    # m^2
l = 0.5       # m

# Tank geometry for head pressure
A_tank = 0.1   # m², cross-sectional area of tank
g = 9.81       # m/s²
h0 = V0 / A_tank   # initial fluid height

# Ambient
atm = Atmosphere(h=0)
Pamb = atm.pressure[0]
Tamb = atm.temperature[0]

def pipe(fluid, P1, T1, P2, T2, R, L, mdot_old, damping_strong=2.0, eps=200.0):
    """
    Pipe dynamics with adaptive stabilization.
    - Allows backflow (sign(dp)).
    - Strong decay near Δp ~ 0 to kill oscillations.
    """
    fluid.TP = T1, P1
    rho1 = fluid.density_mass
    fluid.TP = T2, P2
    rho2 = fluid.density_mass
    rho = 0.5 * (rho1 + rho2)

    dp = P1 - P2

    # "spring-like" driving force from pressure difference
    drive = np.sign(dp) * (np.abs(dp) - R * (mdot_old**2) / rho) * (1 / L)

    # adaptive damping that gets VERY strong when |dp| < eps
    damp_coeff = damping_strong / (1 + (np.abs(dp) / eps))
    decay = -damp_coeff * mdot_old

    return drive + decay


def tank_fluid_mass(mdot_out):
    return -mdot_out

def tank_fluid_volume(mdot_out, rho_out):
    return -mdot_out / rho_out

# --- smooth switch ---
def smooth_switch(x, x_scale):
    #return 0.5 * (1 + np.tanh((x - x_scale) / x_scale))
    return 1

def rhs(t, y):
    M, V, mdot = y
    rho_tank = max(M / V, 1e-9)   # avoid NaN
    h = V / A_tank
    P_tank = P + rho_tank * g * h   # pressure = base + hydro head

    R = 1 / (2 * (Cd * A) ** 2)
    L = l / A
    dmdot = pipe(fluid, P_tank, T, Pamb, Tamb, R, L, mdot)

    dMdt = tank_fluid_mass(mdot)
    dVdt = tank_fluid_volume(mdot, rho_tank)

    # fade dynamics smoothly as M -> 0 (scale to 2% of initial mass)
    fade = smooth_switch(M, M0 * 0.02)
    return [dMdt * fade, dVdt * fade, dmdot * fade]


# --- LSODA manual stepper ---
solver = LSODA(rhs, t0=t_start, y0=[M0, V0, mdot0], t_bound=t_end, max_step=dt)

t_vals = [solver.t]
M_vals = [solver.y[0]]
V_vals = [solver.y[1]]
mdot_vals = [solver.y[2]]
P_vals = []
h_frac_vals = []

# initial values
rho_tank = M0 / V0
h = V0 / A_tank
P_vals.append(P + rho_tank * g * h)
h_frac_vals.append(100 * h / h0)

while solver.status == 'running':
    solver.step()
    t_vals.append(solver.t)
    M_vals.append(solver.y[0])
    V_vals.append(solver.y[1])
    mdot_vals.append(solver.y[2])

    rho_tank = solver.y[0] / solver.y[1]
    h = solver.y[1] / A_tank
    P_tank = P + rho_tank * g * h

    P_vals.append(P_tank)
    h_frac_vals.append(100 * h / h0)

# convert to arrays
t_vals = np.array(t_vals)
M_vals = np.array(M_vals)
V_vals = np.array(V_vals)
mdot_vals = np.array(mdot_vals)
P_vals = np.array(P_vals)
h_frac_vals = np.array(h_frac_vals)

fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)

axs = axs.flatten()  # flatten 2D grid into 1D array

axs[0].plot(t_vals, M_vals, label="Tank Mass")
axs[0].set_ylabel("Mass (kg)")
axs[0].set_title("Tank Mass vs Time")
axs[0].legend()

axs[1].plot(t_vals, V_vals, label="Tank Volume", color="orange")
axs[1].set_ylabel("Volume (m³)")
axs[1].set_title("Tank Volume vs Time")
axs[1].legend()

axs[2].plot(t_vals, mdot_vals, label="Pipe Mass Flow", color="green")
axs[2].set_ylabel("ṁ (kg/s)")
axs[2].set_title("Pipe Mass Flow vs Time")
axs[2].legend()

axs[3].plot(t_vals, P_vals, label="Tank Pressure", color="red")
axs[3].set_ylabel("Pressure (Pa)")
axs[3].set_title("Tank Pressure vs Time")
axs[3].legend()

axs[4].plot(t_vals, h_frac_vals, label="Tank Height (%)", color="purple")
axs[4].set_ylabel("Height (%)")
axs[4].set_xlabel("Time (s)")
axs[4].set_title("Tank Height as % of Initial")
axs[4].legend()

# Hide last subplot (unused)
axs[5].axis("off")

plt.tight_layout()
plt.show()
