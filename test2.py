import numpy as np
from scipy.optimize import root
from Fluids import Fluid  # Adjust import to match your folder structure

# Constants
fluid_name = "Water"
P = 101325  # Pa
T0 = 300    # K
m = 1.0     # kg
Q_dot = 1000.0  # W (J/s) - constant heat input
dt = 0.5    # time step in seconds
n_steps = 20

# Store time history
times = [0.0]
temperatures = [T0]
enthalpies = [Fluid(fluid_name, T=T0, P=P).enthalpy]

T_prev = T0
h_prev = enthalpies[-1]

for step in range(1, n_steps + 1):
    time = step * dt

    def residual(T_next):
        try:
            fluid_next = Fluid(fluid_name, T=T_next, P=P)
            h_next = fluid_next.enthalpy
            dU = m * (h_next - h_prev)
            return dU - Q_dot * dt  # backward Euler: U[n+1] - U[n] = Q·dt
        except Exception:
            return 1e9  # large residual if T causes CoolProp failure

    sol = root(lambda x: residual(x[0]), [T_prev])

    if not sol.success:
        print(f"[✗] Step {step} failed: {sol.message}")
        break

    T_next = sol.x[0]
    fluid_next = Fluid(fluid_name, T=T_next, P=P)
    h_next = fluid_next.enthalpy

    # Store results
    times.append(time)
    temperatures.append(T_next)
    enthalpies.append(h_next)

    # Prepare for next step
    T_prev = T_next
    h_prev = h_next

    print(f"[✓] t = {time:.2f}s | T = {T_next:.2f} K | h = {h_next:.2f} J/kg")

# Optional: plot
try:
    import matplotlib.pyplot as plt
    plt.plot(times, temperatures, marker="o")
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [K]")
    plt.title("Transient Fluid Heating (Backward Euler)")
    plt.grid(True)
    plt.show()
except ImportError:
    print("Install matplotlib to visualize results.")
