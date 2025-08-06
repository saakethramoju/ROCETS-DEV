import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import cantera as ct

# --- Simulation settings ---
V = 1e-1  # control volume [m³]
Cd = 0.8
A = 0.8e-5  # flow area [m²]
t_end = 10

# --- Boundary conditions ---
P1 = 2e5     # inlet pressure [Pa]
h1 = -15865755.52454401
P2 = 101325  # outlet pressure [Pa]
h2 = -15865846.944355397

# --- Create inlet/outlet fluid states ---
w1 = ct.Water(); w1.HP = h1, P1
w2 = ct.Water(); w2.HP = h2, P2
rho_in = w1.density_mass
rho_out = w2.density_mass
h_in = w1.enthalpy_mass

# --- Flow functions ---
def mdot(P_up, P_down, rho, Cd, A):
    return Cd * A * np.sqrt(2 * rho * max(P_up - P_down, 0))

def edot(mdot, h):
    return mdot * h

# --- STEP 1: Solve initial state where mdot_out = 0 and edot_out = 0 ---
def initial_residual(x):
    P, h = x
    try:
        w = ct.Water()
        w.HP = h, P
        rho = w.density_mass
        h_local = w.enthalpy_mass
        m_out = mdot(P, P2, 0.5 * (rho + rho_out), Cd, A)
        e_out = edot(m_out, h_local)
        return [m_out, e_out]
    except Exception:
        return [1e9, 1e9]

sol_init = root(initial_residual, [P2, h2], method="lm")
if not sol_init.success:
    raise RuntimeError("Initial state solver failed.")

P0, h0 = sol_init.x

# Compute initial mass and energy
w = ct.Water(); w.HP = h0, P0
rho0 = w.density_mass
u0 = w.int_energy_mass
M0 = rho0 * V
U0 = rho0 * u0 * V

# --- Global variables to store last good P and h ---
P_last, h_last = P0, h0

# Containers for mass flow rates
m_in_vals, m_out_vals, time_vals = [], [], []

# --- STEP 2: Define ODEs for dM/dt and dU/dt ---
def rhs(t, Y):
    global P_last, h_last
    M, U = Y

    def invert_state(x):
        P, h = x
        try:
            w = ct.Water()
            w.HP = h, P
            rho = w.density_mass
            u = w.int_energy_mass
            return [rho * V - M, rho * u * V - U]
        except Exception:
            return [1e9, 1e9]

    sol = root(invert_state, [P_last, h_last], method='lm')
    if not sol.success:
        raise RuntimeError("State inversion failed during integration")

    P, h = sol.x
    P_last, h_last = P, h

    w = ct.Water(); w.HP = h, P
    rho = w.density_mass
    h_local = w.enthalpy_mass

    m_in = mdot(P1, P, 0.5 * (rho_in + rho), Cd, A)
    m_out = mdot(P, P2, 0.5 * (rho + rho_out), Cd, A)

    m_in_vals.append(m_in)
    m_out_vals.append(m_out)
    time_vals.append(t)

    e_in = edot(m_in, h_in)
    e_out = edot(m_out, h_local)

    dMdt = m_in - m_out
    dUdt = e_in - e_out

    return [dMdt, dUdt]

# --- STEP 3: Integrate in time using solve_ivp ---
t_eval = np.linspace(0, t_end, 300)
sol = solve_ivp(rhs, [0, t_end], [M0, U0], method='BDF', t_eval=t_eval)

# --- STEP 4: Post-process: get P, h from M, U at each step ---
P_vals, h_vals = [], []

for M, U in zip(sol.y[0], sol.y[1]):
    def invert_state(x):
        P, h = x
        try:
            w = ct.Water()
            w.HP = h, P
            rho = w.density_mass
            u = w.int_energy_mass
            return [rho * V - M, rho * u * V - U]
        except Exception:
            return [1e9, 1e9]

    sol_i = root(invert_state, [P_last, h_last], method='lm')
    if sol_i.success:
        P, h = sol_i.x
        P_last, h_last = P, h
    else:
        P, h = np.nan, np.nan
    P_vals.append(P)
    h_vals.append(h)

# --- STEP 5: Plotting ---
fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axs[0].plot(sol.t, P_vals, label='Pressure [Pa]')
axs[0].set_ylabel("Pressure [Pa]")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(sol.t, h_vals, label='Enthalpy [J/kg]')
axs[1].set_ylabel("Enthalpy [J/kg]")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(time_vals, m_in_vals, label='Inlet mdot [kg/s]')
axs[2].plot(time_vals, m_out_vals, label='Outlet mdot [kg/s]')
axs[2].set_ylabel("Mass Flow Rate [kg/s]")
axs[2].set_xlabel("Time [s]")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
