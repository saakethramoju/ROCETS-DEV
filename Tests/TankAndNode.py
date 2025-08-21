import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from ambiance import Atmosphere
from scipy.integrate import LSODA

# -----------------------------
# Simulation settings
# -----------------------------
t_start = 0
dt = 0.1
t_end = 10

# Tank initial conditions
P_tank = 101325    # Pa
T_tank = 300       # K
M_tank0 = 1e3      # kg
fluid = ct.Water()

fluid.TP = T_tank, P_tank
V_tank0 = M_tank0 / fluid.density_mass   # initial tank volume

A_tank = 0.1   # m² cross-section
g = 9.81
h0 = V_tank0 / A_tank   # initial fluid height

# Ambient
atm = Atmosphere(h=0)
Pamb = atm.pressure[0]
Tamb = atm.temperature[0]

# Pipe 1
mdot1_0 = 0.0
Cd1 = 0.8
A1 = 5.1e-4
l1 = 0.5

# Pipe 2
mdot2_0 = 0.0
Cd2 = 0.9
A2 = 5.1e-4
l2 = 0.3

# Node
M_node0 = 0.01  # small seed mass to avoid div/0
T_node = 300
l = 0.1
A = 5.1e-4
V_node = A * l

# -----------------------------
# Flow model with damping
# -----------------------------
def pipe(fluid, P1, T1, P2, T2, R, L, mdot_old, damping_strong=2.0, eps=200.0):
    fluid.TP = T1, P1
    rho1 = fluid.density_mass
    fluid.TP = T2, P2
    rho2 = fluid.density_mass
    rho = 0.5 * (rho1 + rho2)

    dp = P1 - P2
    drive = np.sign(dp) * (np.abs(dp) - R * (mdot_old**2) / rho) * (1 / L)

    damp_coeff = damping_strong / (1 + (np.abs(dp) / eps))
    decay = -damp_coeff * mdot_old
    return drive + decay

def tank_fluid_mass(mdot_out): return -mdot_out
def tank_fluid_volume(mdot_out, rho_out): return -mdot_out / rho_out
def node_mass(mdot_in, mdot_out): return mdot_in - mdot_out

# -----------------------------
# RHS system
# -----------------------------
def rhs(t, y):
    M_tank, V_tank, mdot1, M_node, mdot2 = y
    rho_tank = max(M_tank / V_tank, 1e-9)
    h = V_tank / A_tank
    P_head = P_tank + rho_tank * g * h

    # pipe 1
    R1 = 1 / (2 * (Cd1 * A1) ** 2)
    L1 = l1 / A1
    dmdt1 = pipe(fluid, P_head, T_tank, P_node_guess, T_node, R1, L1, mdot1)

    dM_tankdt = tank_fluid_mass(mdot1)
    dV_tankdt = tank_fluid_volume(mdot1, rho_tank)

    # node properties
    rho_node = max(M_node / V_node, 1e-9)
    fluid.TD = T_node, rho_node
    P_node = fluid.P

    # pipe 2
    R2 = 1 / (2 * (Cd2 * A2) ** 2)
    L2 = l2 / A2
    dmdt2 = pipe(fluid, P_node, T_node, Pamb, Tamb, R2, L2, mdot2)

    dM_nodedt = node_mass(mdot1, mdot2)
    return [dM_tankdt, dV_tankdt, dmdt1, dM_nodedt, dmdt2]

# -----------------------------
# Initial state
# -----------------------------
P_node_guess = 101325  # just to initialize pipe1 at t=0
y0 = [M_tank0, V_tank0, mdot1_0, M_node0, mdot2_0]

# -----------------------------
# Integrate with LSODA (manual stepping)
# -----------------------------
solver = LSODA(rhs, t0=t_start, y0=y0, t_bound=t_end, max_step=dt)

# histories
t_hist, M_tank_hist, V_tank_hist, mdot1_hist, M_node_hist, mdot2_hist, P_node_hist = ([] for _ in range(7))

while solver.status == "running":
    solver.step()
    t_hist.append(solver.t)
    M_tank_hist.append(solver.y[0])
    V_tank_hist.append(solver.y[1])
    mdot1_hist.append(solver.y[2])
    M_node_hist.append(solver.y[3])
    mdot2_hist.append(solver.y[4])

    # compute node pressure for storage
    rho_node = max(solver.y[3] / V_node, 1e-9)
    fluid.TD = T_node, rho_node
    P_node_hist.append(fluid.P)

# convert to arrays
t_hist = np.array(t_hist)
M_tank_hist = np.array(M_tank_hist)
V_tank_hist = np.array(V_tank_hist)
mdot1_hist = np.array(mdot1_hist)
M_node_hist = np.array(M_node_hist)
mdot2_hist = np.array(mdot2_hist)
P_node_hist = np.array(P_node_hist)

# -----------------------------
# Plotting
# -----------------------------
fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
axs = axs.flatten()

axs[0].plot(t_hist, M_tank_hist, label="Tank Mass")
axs[0].set_ylabel("Mass (kg)")
axs[0].set_title("Tank Mass vs Time")
axs[0].legend()

axs[1].plot(t_hist, V_tank_hist, label="Tank Volume", color="orange")
axs[1].set_ylabel("Volume (m³)")
axs[1].set_title("Tank Volume vs Time")
axs[1].legend()

axs[2].plot(t_hist, mdot1_hist, label="Pipe 1 Flow", color="green")
axs[2].set_ylabel("ṁ1 (kg/s)")
axs[2].set_title("Pipe 1 Mass Flow vs Time")
axs[2].legend()

axs[3].plot(t_hist, M_node_hist, label="Node Mass", color="purple")
axs[3].set_ylabel("Mass (kg)")
axs[3].set_title("Node Mass vs Time")
axs[3].legend()

axs[4].plot(t_hist, mdot2_hist, label="Pipe 2 Flow", color="red")
axs[4].set_ylabel("ṁ2 (kg/s)")
axs[4].set_title("Pipe 2 Mass Flow vs Time")
axs[4].set_xlabel("Time (s)")
axs[4].legend()

axs[5].plot(t_hist, P_node_hist, label="Node Pressure", color="brown")
axs[5].set_ylabel("Pressure (Pa)")
axs[5].set_title("Node Pressure vs Time")
axs[5].set_xlabel("Time (s)")
axs[5].legend()

plt.tight_layout()
plt.show()
