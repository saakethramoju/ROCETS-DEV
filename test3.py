import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp
#from Fluids import Fluid
import matplotlib.pyplot as plt
import cantera as ct

# Simulation settings
dt = 0.001
t_end = 10
step = 0
Cd = 0.8
A = 0.8e-6  # m²

V = 1 # m³

# Boundary conditions
P1 = 110000     # inlet pressure [Pa]
T1 = 298.15
P2 = 101325     # outlet pressure [Pa]
T2 = 280

w1 = ct.Water()
w1.TP = T1, P1

w2 = ct.Water()
w2.TP = T2, P2

rho_in = w1.density_mass
rho_out = w2.density_mass
h_in = w1.enthalpy_mass
h_out = w2.enthalpy_mass

# Initial values
P0 = 101325
T0 = 300

# Initial state
water = ct.Water()
water.TP = T0, P0
rho = water.density_mass
u = water.int_energy_mass
M = rho * V
U = M * u


def mdot(P_up, P_down, rho, Cd, A):
    return Cd * A * np.sqrt(2 * rho * max(P_up - P_down, 0))

def edot(mdot, h):
    return mdot * h


def partials(w, dP=1, dT=0.1):
    P, T = w.P, w.T

    # Central differences for dρ/dP, dρ/dT
    w1 = ct.Water(); w1.TP = T, P + dP
    w2 = ct.Water(); w2.TP = T, P - dP
    drho_dP = (w1.density_mass - w2.density_mass) / (2 * dP)
    du_dP = (w1.int_energy_mass - w2.int_energy_mass) / (2 * dP)

    w3 = ct.Water(); w3.TP = T + dT, P
    w4 = ct.Water(); w4.TP = T - dT, P
    drho_dT = (w3.density_mass - w4.density_mass) / (2 * dT)
    du_dT = (w3.int_energy_mass - w4.int_energy_mass) / (2 * dT)

    rho = w.density_mass
    u = w.int_energy_mass

    return {
        "drho_dP": drho_dP,
        "drho_dT": drho_dT,
        "du_dP": du_dP,
        "du_dT": du_dT,
        "rho": rho,
        "u": u
    }


def mass_energy_to_state_derivatives(partials, dMdt, dUdt, V):
    """
    Convert dM/dt and dU/dt into dP/dt and dT/dt using thermodynamic partials.

    Parameters:
    - partials: dict from the `partials()` function
    - dMdt: time derivative of mass [kg/s]
    - dUdt: time derivative of internal energy [J/s]
    - V: control volume [m³]

    Returns:
    - dPdt: time derivative of pressure [Pa/s]
    - dTdt: time derivative of temperature [K/s]
    """
    drho_dP = partials["drho_dP"]
    drho_dT = partials["drho_dT"]
    du_dP   = partials["du_dP"]
    du_dT   = partials["du_dT"]
    rho     = partials["rho"]
    u       = partials["u"]

    # Construct Jacobian matrix J = d(M, U)/d(P, T)
    J = np.array([
        [drho_dP * V, drho_dT * V],
        [(drho_dP * u + rho * du_dP) * V, (drho_dT * u + rho * du_dT) * V]
    ])

    rhs = np.array([dMdt, dUdt])

    # Solve for [dP/dt, dT/dt]
    dPdt, dTdt = np.linalg.solve(J, rhs)

    return dPdt, dTdt

def dstate_dt(t, y):
    P, T = y

    # Set state
    w = ct.Water()
    T_safe = np.clip(T, 274, 1000)   # restrict to [0°C, 1000 K]
    P_safe = max(P, 1e3)                # avoid vacuum/negative pressure

    try:
        w.TP = T_safe, P_safe
    except Exception as e:
        print(f"[t={t:.3f}s] Invalid TP set: T={T}, P={P}. Clipped to T={T_safe}, P={P_safe}")
        raise e

    rho = w.density_mass
    #u = w.int_energy_mass
    h = w.enthalpy_mass

    # Flow properties
    rho_avg_in = 0.5 * (rho + rho_in)
    rho_avg_out = 0.5 * (rho + rho_out)

    m_in = mdot(P1, P, rho_avg_in, Cd, A)
    m_out = mdot(P, P2, rho_avg_out, Cd, A)

    e_in = edot(m_in, h_in)
    e_out = edot(m_out, h_out)

    dMdt = m_in - m_out
    dUdt = e_in - e_out

    # Thermo partials
    partial = partials(w)

    # Compute dP/dt and dT/dt
    dPdt, dTdt = mass_energy_to_state_derivatives(partial, dMdt, dUdt, V)

    # Print diagnostics
    print(f"t = {t:.3f} s | P = {P:.2f} Pa | T = {T:.2f} K | "
          f"mdot_in = {m_in:.5f} kg/s | mdot_out = {m_out:.5f} kg/s | "
          #f"dM/dt = {dMdt:.5f} kg/s | dU/dt = {dUdt:.2f} W")
          f"edot_in = {e_in:.3f} W/s | edot_out = {e_out:.3f} W/s")
    
    return [dPdt, dTdt]


# Solve from t=0 to t_end
mdot_in_vals = []
mdot_out_vals = []
edot_in_vals = []
edot_out_vals = []

sol = solve_ivp(
    dstate_dt,
    [0, t_end],
    [P0, T0],
    method='LSODA',
    t_eval=np.linspace(0, t_end, int(t_end / dt) + 1),
    dense_output=True,
    rtol=1e-3,
    atol=1e-6
)

# Extract results
t_vals = sol.t
P_vals, T_vals = sol.y

# Recompute flow values at t_eval
mdot_in_vals = []
mdot_out_vals = []
edot_in_vals = []
edot_out_vals = []

for t, P, T in zip(sol.t, sol.y[0], sol.y[1]):
    w = ct.Water()
    T_safe = np.clip(T, 274, 1000)
    P_safe = max(P, 1e3)

    try:
        w.TP = T_safe, P_safe
    except Exception as e:
        print(f"[postprocess] t={t:.3f}s: Invalid TP set — T={T}, P={P} → clipped to T={T_safe}, P={P_safe}")
        raise e


    rho_avg_in = 0.5 * (rho + rho_in)
    rho_avg_out = 0.5 * (rho + rho_out)

    m_in = mdot(P1, P, rho_avg_in, Cd, A)
    m_out = mdot(P, P2, rho_avg_out, Cd, A)

    e_in = edot(m_in, h_in)
    e_out = edot(m_out, h_out)

    mdot_in_vals.append(m_in)
    mdot_out_vals.append(m_out)
    edot_in_vals.append(e_in)
    edot_out_vals.append(e_out)


# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t_vals, P_vals, label='Pressure [Pa]')
plt.ylabel("Pressure [Pa]")
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_vals, T_vals, label='Temperature [K]', color='orange')
plt.ylabel("Temperature [K]")
plt.xlabel("Time [s]")
plt.grid()
plt.legend()

plt.tight_layout()

# Plot mass flow rates
plt.figure(figsize=(12, 4))
plt.plot(t_vals, mdot_in_vals, label='mdot_in [kg/s]')
plt.plot(t_vals, mdot_out_vals, label='mdot_out [kg/s]')
plt.ylabel("Mass Flow Rate [kg/s]")
plt.xlabel("Time [s]")
plt.title("Mass Flow Rates")
plt.grid()
plt.legend()

# Plot energy flow rates
plt.figure(figsize=(12, 4))
plt.plot(t_vals, edot_in_vals, label='edot_in [W]')
plt.plot(t_vals, edot_out_vals, label='edot_out [W]')
plt.ylabel("Energy Flow Rate [W]")
plt.xlabel("Time [s]")
plt.title("Energy Flow Rates")
plt.grid()
plt.legend()

plt.show()


