import numpy as np
from scipy.optimize import root
from Fluids import Fluid
import matplotlib.pyplot as plt

# Constants
dt = 0.01
t_end = 3
T = 298.15
V = 1.0  # m³
step = 0

P1 = 200000     # inlet pressure
P2 = 101325     # outlet pressure
Cd = 0.8
A = 0.8e-5      # m²

P_guess = 101325.0
f = Fluid(T=T, P=P_guess)
rho = f.density
u = f.internal_energy
M = rho * V
M_prev = 0
U = M * u
U_prev = 0

# Storage for plotting — now empty until data is computed
t_vals = []
P_vals = []
m_in_vals = []
m_out_vals = []

def mdot(P_up, P_down, rho, Cd, A):
    if P_up <= P_down:
        return 0.0
    return Cd * A * np.sqrt(2 * rho * (P_up - P_down))

#def edot()

# Main loop
t = 0.0
while t < t_end + dt / 2:  # include last step robustly

    def residual(P):
        P_scalar = P.item() if isinstance(P, np.ndarray) else P
        try:
            f = Fluid(T=T, P=P_scalar)
            rho = f.density
            mdot_in = mdot(P1, P_scalar, rho, Cd, A)
            mdot_out = mdot(P_scalar, P2, rho, Cd, A)
            if step == 0:
                res = mdot_out
            else:
                res = (rho * V - M) / dt + mdot_out - mdot_in
            #print(f"[DEBUG] residual(P={P_scalar:.2f}) = {res:.4e}")
            return res
        except Exception as e:
            #print(f"[ERROR] residual(P={P_scalar:.2f}) -> EXCEPTION: {e}")
            return 1e9  # big value to repel solver



    sol = root(residual, P_guess, method = 'lm') # lm handles flat regions better
    #sol = root(residual, P_guess, method='hybr', options={'xtol': 1e-6, 'maxfev': 200}) # set solution tolerance and max iterations if needed. 
    # default tolerance is 1.49012e-8 which is very narrow
    if not sol.success:
        raise RuntimeError(f"{sol.message}")

    P_guess = sol.x[0]
    f = Fluid(T=T, P=P_guess)
    rho = f.density

    m_in = mdot(P1, P_guess, rho, Cd, A)
    m_out = mdot(P_guess, P2, rho, Cd, A)

    M_prev = M
    M = rho * V
    dMdt = (M - M_prev) / dt

    #print(f"t = {t:.3f} s | P = {P_guess:.2f} Pa | dM/dt = {dMdt:.4f} kg/s | mdot_in = {m_in:.4f} | mdot_out = {m_out:.4f}")

    # Store values only after computation
    t_vals.append(t)
    P_vals.append(P_guess)
    m_in_vals.append(m_in)
    m_out_vals.append(m_out)

    # Advance time
    t += dt
    step += 1

# Plotting
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
plt.show()
