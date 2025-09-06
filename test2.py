from Components import Component, ComponentType, Balance
import Globals
from System import System
import cantera as ct
import numpy as np
from scipy.integrate import LSODA
from scipy.optimize import root
import matplotlib.pyplot as plt
import Constants

# ----------------------------------------------------------------------
# Component definitions
# ----------------------------------------------------------------------
class Pipe(Component):

    configuration_keys = ["Discharge Coefficient",
                          "Cross-sectional Area (m^2)",
                          "Length (m)"]
    state_keys = ["Mass Flow (kg/s)"]
    inflow_keys = ["Source"]
    outflow_keys = ["Drain"]
    fluid_keys = ["Fluid"]

    def steady_state(self):
        self["Fluid"](self["Source"].fluid)

        Cd = self["Discharge Coefficient"]()
        A = self["Cross-sectional Area (m^2)"]()

        R = 1 / (2 * (Cd * A) ** 2)
        rho1 = self["Source"].fluid.density_mass
        rho2 = self["Drain"].fluid.density_mass
        rho = 0.5 * (rho1 + rho2)

        dp = self["Source"].fluid.P - self["Drain"].fluid.P
        mdot = np.sign(dp) * np.sqrt(rho * np.abs(dp) / R)
        self["Mass Flow (kg/s)"](mdot)
        return mdot

    def transient(self, damping_strong=2.0, eps=200.0):
        self["Fluid"](self["Source"].fluid)

        Cd = self["Discharge Coefficient"]()
        A = self["Cross-sectional Area (m^2)"]()
        l = self["Length (m)"]()
        mdot_old = self["Mass Flow (kg/s)"]()

        L = l / A
        R = 1 / (2 * (Cd * A) ** 2)

        rho1 = self["Source"].fluid.density_mass
        rho2 = self["Drain"].fluid.density_mass
        rho = 0.5 * (rho1 + rho2)
        dp = self["Source"].fluid.P - self["Drain"].fluid.P

        drive = np.sign(dp) * (np.abs(dp) - R * (mdot_old**2) / rho) * (1 / L)

        damp_coeff = damping_strong / (1 + (np.abs(dp) / eps))
        decay = -damp_coeff * mdot_old

        return drive + decay
    

class Tank(Component):

    configuration_keys = ["Cross-sectional Area (m^2)",
                          "Fluid Mass (kg)"]
    state_keys = ["Effective Pressure (Pa)",
                  "Effective Enthalpy (J/kg)",
                  "Ullage Pressure (Pa)",
                  "Fluid Height (m)",
                  "Fluid Volume (m^3)"]
    outflow_keys = ["Drain"]
    fluid_keys = ["Ullage", "Bulk"]

    component_type = ComponentType.JUNCTION

    def steady_state(self):
        self["Ullage Pressure (Pa)"](self["Ullage"]().P)

        P = self["Ullage Pressure (Pa)"]()
        rho = self["Ullage"]().density_mass
        V = self["Fluid Mass (kg)"]() / rho
        h = V / self["Cross-sectional Area (m^2)"]()
        P_eff = P + rho*Constants.g*h
        T = self["Bulk"]().T

        self["Bulk"]().TP = T, P_eff
        self["Drain"].fluid = self["Bulk"]()
        self["Effective Pressure (Pa)"](P_eff)
        self["Effective Enthalpy (J/kg)"](self["Drain"].fluid.enthalpy_mass)
        self["Fluid Height (m)"](h)
        self["Fluid Volume (m^3)"](V)
        return P_eff


# ----------------------------------------------------------------------
# Balance solving
# ----------------------------------------------------------------------
def solve_balances(balances, components, mode="steady", t=None):
    if not balances:
        return None
    if t is None:
        t = Globals.get_time()

    x0 = [bal.independent[t] for bal in balances]

    def F(z):
        for bal, val in zip(balances, z):
            bal.set_independent(val, t)
        for comp in components:
            if mode == "steady":
                comp.steady_state()
            elif mode == "transient":
                comp.transient()
        return [bal.residual(t) for bal in balances]

    sol = root(F, x0)
    return sol

def run_steady_state(components, balances, t=0.0):
    """Solve for steady-state trim point (derivatives = 0, balances closed)."""
    Globals.set_time(t)

    # collect states + balance unknowns
    x0 = [comp[k]() for comp in components for k in comp.state_keys] \
       + [bal.independent[t] for bal in balances]

    def F(z):
        idx = 0
        # assign trial states
        for comp in components:
            for k in comp.state_keys:
                comp[k](z[idx]); idx += 1
        # assign algebraics
        for bal in balances:
            bal.set_independent(z[idx], t); idx += 1

        # residuals: derivatives → should be 0 at steady-state
        ode_resids = []
        for comp in components:
            derivs = comp.transient()
            if np.ndim(derivs) == 0:
                ode_resids.append(derivs)
            else:
                ode_resids.extend(derivs)

        balance_resids = [bal.residual(t) for bal in balances]
        return ode_resids + balance_resids

    sol = root(F, x0)
    if not sol.success:
        raise RuntimeError(f"Steady solve failed: {sol.fun}")
    return sol


def run_transient(components, balances, t_start, t_end, dt):
    """Run transient integration with balances closed each step."""

    state_vars = [(comp[k], comp) for comp in components for k in comp.state_keys]
    alg_vars   = [bal.independent for bal in balances]

    def rhs(t, y):
        Globals.set_time(t)
        for (v, comp), val in zip(state_vars, y):
            v(val)

        if balances:
            def coupled_resids(vars):
                trial_states = vars[:len(state_vars)]
                trial_algs   = vars[len(state_vars):]

                for (v, comp), val in zip(state_vars, trial_states):
                    v(val)
                for v, val in zip(alg_vars, trial_algs):
                    v(val)

                ode_resids = []
                for (v, comp), trial_val in zip(state_vars, trial_states):
                    deriv = comp.transient()
                    ode_resids.append((trial_val - v())/dt - deriv)

                balance_resids = [bal.residual(t) for bal in balances]
                return ode_resids + balance_resids

            x0 = [v() for (v, comp) in state_vars] + [v() for v in alg_vars]
            sol = root(coupled_resids, x0)
            if not sol.success:
                raise RuntimeError(f"DAE solve failed at t={t}, resids={sol.fun}")
            trial_states = sol.x[:len(state_vars)]
            trial_algs   = sol.x[len(state_vars):]
            for (v, comp), val in zip(state_vars, trial_states):
                v(val)
            for v, val in zip(alg_vars, trial_algs):
                v(val)

        return [comp.transient() for (v, comp) in state_vars]

    solver = LSODA(rhs, t0=t_start, y0=[v() for (v, comp) in state_vars],
                   t_bound=t_end, max_step=dt)

    Globals.reset_time()
    while solver.status == 'running':
        solver.step()
        Globals.set_time(solver.t)
        for (v, comp), val in zip(state_vars, solver.y):
            v[solver.t] = val

    return solver



# ----------------------------------------------------------------------
# Setup + Run Example
# ----------------------------------------------------------------------
tank = Tank("Tank")
pipe = Pipe("Line")

# fluids
w = ct.Water(); w.TP = 300, 101325
air = ct.Solution("air.yaml"); air.TP = 300, 101325
tank["Bulk"] = w
tank["Ullage"] = air
tank["Fluid Mass (kg)"] = 1e3
tank["Cross-sectional Area (sq. m.)"] = 0.1

pipe["Source"].fluid.TP = 300, 2e5
pipe["Discharge Coefficient"] = 0.8
pipe["Cross-sectional Area (sq. m.)"] = 5.1e-4 
pipe["Length (m)"] = 0.5 
pipe["Mass Flow (kg/s)"] = 0.0

# balance example
b1 = Balance(pipe["Discharge Coefficient"], pipe["Mass Flow (kg/s)"], 5)
#balances = [b1]
balances = []

# steady trim
steady_sol = run_steady_state([pipe], balances)
print("Steady solved:", pipe["Mass Flow (kg/s)"](), pipe["Discharge Coefficient"]())

# transient
solver = run_transient([pipe], balances, t_start=0, t_end=1, dt=0.01)

# plot
times, mdot_vals = pipe["Mass Flow (kg/s)"].history
plt.plot(times, mdot_vals, label="Transient mdot")
plt.legend(); plt.show()


'''
print("=== Parameter Tests ===")
# --- Constant ---
pipe["Discharge Coefficient"] = 0.8
print("Constant value:", pipe["Discharge Coefficient"].value)
print("Cd[0.0]:", pipe["Discharge Coefficient"][0.0])
print("Cd[10.0]:", pipe["Discharge Coefficient"][10.0])
print("History:", pipe["Discharge Coefficient"].history)

# Update at a new time (promotes automatically)
pipe["Discharge Coefficient"][5.0] = 0.75
print("Cd at 5.0:", pipe["Discharge Coefficient"][5.0])
print("History after promotion:", pipe["Discharge Coefficient"].history)


# --- Time series via list of tuples ---
pipe["Cross-sectional Area (sq. m.)"] = [
    (0.0, 5.1e-4),
    (0.5, 4.9e-4),
    (1.0, 4.5e-4),
]
print("A[0.3]:", pipe["Cross-sectional Area (sq. m.)"][0.3])
print("A[0.75]:", pipe["Cross-sectional Area (sq. m.)"][0.75])
print("History:", pipe["Cross-sectional Area (sq. m.)"].history)

# Insert new point
pipe["Cross-sectional Area (sq. m.)"][0.75] = 4.7e-4
print("After insertion:", pipe["Cross-sectional Area (sq. m.)"][0.75])
print("Updated history:", pipe["Cross-sectional Area (sq. m.)"].history)


# --- Time series via arrays ---
t_vals = np.array([0.0, 0.5, 1.0])
cd_vals = np.array([0.8, 0.7, 0.5])
pipe["Discharge Coefficient"] = (t_vals, cd_vals)
print("Cd[0.3]:", pipe["Discharge Coefficient"][0.3])
print("Cd[0.9]:", pipe["Discharge Coefficient"][0.9])
print("History:", pipe["Discharge Coefficient"].history)


print("\n=== State Tests ===")
# --- Constant ---
pipe["Mass Flow (kg/s)"] = 0.0
print("Initial value:", pipe["Mass Flow (kg/s)"].value)
print("At t=2.0 (still constant):", pipe["Mass Flow (kg/s)"][2.0])
print("History:", pipe["Mass Flow (kg/s)"].history)

# Promote to time series
pipe["Mass Flow (kg/s)"][2.0] = 0.1
print("At t=2.0 after promotion:", pipe["Mass Flow (kg/s)"][2.0])
print("History after promotion:", pipe["Mass Flow (kg/s)"].history)

# --- Time series via arrays ---
t_vals = np.array([0.0, 0.5, 1.0])
v_vals = np.array([0.0, 0.05, 0.1])
pipe["Mass Flow (kg/s)"] = (t_vals, v_vals)
print("At t=0.3:", pipe["Mass Flow (kg/s)"][0.3])
print("At t=0.9:", pipe["Mass Flow (kg/s)"][0.9])
print("History:", pipe["Mass Flow (kg/s)"].history)

# Insert mid-series
pipe["Mass Flow (kg/s)"][0.75] = 0.08
print("After insertion at 0.75:", pipe["Mass Flow (kg/s)"][0.75])
print("Updated history:", pipe["Mass Flow (kg/s)"].history)'''

'''from Components import Value

Cd = Value.Parameter("Cd", constant=0.8)
mdot = Value.State("Mass Flow (kg/s)", constant=2.0)

# Default: enforce mdot == 5
b1 = Balance(Cd, mdot, 5.0)

# Custom: enforce Cd * mdot == 5
b2 = Balance(Cd, mdot, 5.0, residual_fn=lambda indep, x, y: indep * x - y)

print(b1.residual())  # 2.0 - 5.0 = -3.0
print(b2.residual())  # 0.8 * 2.0 - 5.0 = -3.4'''
