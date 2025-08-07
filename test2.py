import numpy as np
from scipy.optimize import root
#from Fluids import Fluid
import matplotlib.pyplot as plt
import cantera as ct

# Simulation settings
dt = 0.01
t_end = 2
step = 0
Cd = 0.8
A = 0.8e-5  # m²

V = 1e-4 # m³

# Boundary conditions
P1 = 2e5     # inlet pressure [Pa]
h1 = -15865755.52454401
P2 = 101325     # outlet pressure [Pa]
h2 = -15865846.944355397

w1 = ct.Water()
w1.HP = h1, P1

w2 = ct.Water()
w2.HP = h2, P2

rho_in = w1.density_mass
rho_out = w2.density_mass
h_in = w1.enthalpy_mass

# Initial guess
P_guess = 2e5
h_guess = -1.3e7

# Initial state
water = ct.Water()
rho = water.density_mass
u = water.int_energy_mass
M = rho * V
U = M * u

# Data storage for plotting
t_vals = []
P_vals = []
m_in_vals = []
m_out_vals = []

# Mass flow rate
def mdot(P_up, P_down, rho, Cd, A):
    return Cd * A * np.sqrt(2 * rho * max(P_up - P_down, 0))

# Energy flow rate
def edot(mdot, h):
    return mdot * h

t = 0.0
while t < t_end + dt / 2:


    def residual(x):
        P, h = x
        try:
            # Fluid states
            w = ct.Water()
            w.HP = h, P

            # Properties
            rho = w.density_mass
            u = w.int_energy_mass
            h = w.enthalpy_mass

            # Flow rates (averaged densities)
            mdot_in = mdot(P1, P, 0.5 * (rho_in + rho), Cd, A)
            mdot_out = mdot(P, P2, 0.5 * (rho + rho_out), Cd, A)
            edot_in_val = edot(mdot_in, h_in)
            edot_out_val = edot(mdot_out, h)

            # Conservation residuals
            mass_residual = (rho * V - M) / dt + (mdot_out - mdot_in)
            energy_residual = (rho * u * V - U) / dt + (edot_out_val - edot_in_val)

            # Scaling
            mass_scale = abs(mdot_in + mdot_out) / V + 1e-6
            energy_scale = abs(edot_in_val + edot_out_val) / V + 1e-6

            return [mass_residual / mass_scale, energy_residual / energy_scale]

        except Exception:
            return [1e9, 1e9]

    sol = root(residual, [P_guess, h_guess], method='lm')
    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")
    
    P_guess, h_guess = sol.x
    

    w = ct.Water()
    w.HP = h_guess, P_guess
    rho = w.density_mass
    u = w.int_energy_mass
    h = w.enthalpy_mass


    m_in = mdot(P1, P_guess, 0.5 * (rho_in + rho), Cd, A)
    m_out = mdot(P_guess, P2, 0.5 * (rho + rho_out), Cd, A)
    edot_in_val = edot(m_in, h_in)
    edot_out_val = edot(m_out, h)

    # Update mass and energy by integration (conservation)
    M_new = M + dt * (m_in - m_out)
    U_new = U + dt * (edot_in_val - edot_out_val)
    dMdt = (M_new - M) / dt

    # Print current state
    print(f"t = {t:.3f} s | P = {P_guess:.2f} Pa | h = {h_guess:.4f} J/kg | dM/dt = {dMdt:.4f} kg/s | mdot_in = {m_in:.4f} | mdot_out = {m_out:.4f}")

    # Store for plotting
    t_vals.append(t)
    P_vals.append(P_guess)
    m_in_vals.append(m_in)
    m_out_vals.append(m_out)

    M = M_new
    U = U_new
    t += dt
    step += 1


def compute_steady_state_time(time, signals, threshold=1e-5, window=10):
    """
    Compute the time when any signal reaches steady state.
    
    Parameters:
    - time: list or array of time points
    - signals: list of signals (each a list of values over time)
    - threshold: max allowed rate of change (absolute)
    - window: how many steps in a row must meet the threshold
    
    Returns:
    - steady_time: time when steady state is reached
    - signal_index: index of the signal that first reached it
    """
    signals = np.array(signals)
    time = np.array(time)
    dt = np.diff(time)

    for i, y in enumerate(signals):
        dy = np.diff(y)
        rate = np.abs(dy / dt)

        # Find where the rate stays below the threshold for `window` steps
        for j in range(len(rate) - window):
            if np.all(rate[j:j+window] < threshold):
                return time[j + window], i  # return time and signal index
    
    return None, None  # steady state not reached



# Plotting results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t_vals, P_vals, label='Pressure [Pa]')
plt.ylabel("Pressure [Pa]")
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_vals, m_in_vals, label='Inlet mdot [kg/s]')
plt.plot(t_vals, m_out_vals, label='Outlet mdot [kg/s]')
plt.ylabel("Mass Flow Rate [kg/s]")
plt.xlabel("Time [s]")
plt.grid()
plt.legend()

plt.tight_layout()

t, _ = compute_steady_state_time(t_vals, [m_in_vals])
print(f"time: {t:.2f}")

plt.show()

