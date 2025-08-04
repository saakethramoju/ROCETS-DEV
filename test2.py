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
A = 0.8e-3    # m²

# Initial condition
P_guess = 101325.0
f = Fluid(T=T, P=P_guess)
rho = f.density
M = rho * V

# Data storage
t_vals = [0]
P_vals = [P_guess]
M_vals = [M]
mdot_in_vals = []
mdot_out_vals = []

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
        M_est = rho * V
        return M_est - M_prev - dt * (m_in - m_out)

    sol = root_scalar(residual, bracket=[1e4, 5e5], method='brentq')
    if not sol.converged:
        raise RuntimeError(f"Root solve failed at t = {t_vals[-1]}s")

    P_next = sol.root
    f = Fluid(T=T, P=P_next)
    rho = f.density
    M = rho * V
    print(f.internal_energy)

    m_in = mdot(P1, P_next, rho, Cd, A)
    m_out = mdot(P_next, P2, rho, Cd, A)

    print(f"t = {t_vals[-1]:.1f} s | P = {P_next:.2f} Pa | M = {M:.4f} kg, mdot1 = {m_in:.2f}, mdot2 = {m_out:.2f}")

    # Save data
    mdot_in_vals.append(m_in)
    mdot_out_vals.append(m_out)
    t_vals.append(t_vals[-1] + dt)
    P_vals.append(P_next)
    M_vals.append(M)
    step += 1



def get_steady_state_info(t, y, tol=0.02):
    """
    Estimates steady-state value and settling time.

    Parameters:
    - t: time array
    - y: value array (e.g., pressure or mdot)
    - tol: tolerance band for settling (default: 2%)

    Returns:
    - steady_value: estimated steady-state value
    - settling_time: time when y enters and stays within tol band
    """
    y = np.array(y)
    t = np.array(t)
    steady_value = y[-1]
    tol_band = tol * abs(steady_value)

    # Check where the signal stays within the tolerance band
    out_of_band = np.abs(y - steady_value) > tol_band
    if np.any(out_of_band):
        last_out_of_band_idx = np.max(np.where(out_of_band))
        if last_out_of_band_idx + 1 < len(t):
            settling_time = t[last_out_of_band_idx + 1]
        else:
            settling_time = None  # Never settled within tolerance
    else:
        settling_time = t[0]  # Already settled from the beginning

    return steady_value, settling_time



# Remove final time step to match mdot arrays
t_plot = t_vals[:-1]
P_plot = P_vals[:-1]

# Plotting
fig, ax1 = plt.subplots()

# Mass flow rate plot (primary y-axis)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Mass Flow Rate [kg/s]')
ax1.plot(t_plot, mdot_in_vals, label='Inlet mdot', color='tab:blue')
ax1.plot(t_plot, mdot_out_vals, label='Outlet mdot', color='tab:orange')
ax1.legend(loc='upper left')
ax1.grid(True)

# Pressure plot (secondary y-axis)
ax2 = ax1.twinx()
ax2.set_ylabel('Pressure [Pa]')
ax2.plot(t_plot, P_plot, label='Pressure', color='tab:red', linestyle='--')
ax2.legend(loc='upper right')

plt.title('Mass Flow Rates and Pressure Over Time')
plt.tight_layout()

steady_pressure, settling_time = get_steady_state_info(t_plot, P_plot)
print(f"Steady-state pressure: {steady_pressure:.2f} Pa")
print(f"Settling time: {settling_time:.2f} s")

#plt.show()
