from Fluids import Fluid
from scipy.optimize import root_scalar

P = 101325  # 1 atm

# Step 1: Choose a target enthalpy (in J/kg)
target_enthalpy = 104860  # e.g. 100,000 J/kg

# Step 2: Determine valid temperature bounds for your fluid
Tmin = Fluid(P=P, T=300).min_temperature + 1.0  # small buffer to avoid edge errors
Tmax = Fluid(P=P, T=300).max_temperature - 1.0

# Step 3: Define residual function
def residual(T):
    try:
        f = Fluid(T=T, P=P)
        return f.enthalpy - target_enthalpy
    except Exception:
        return float('nan')

# Optional: check sign at bracket ends to ensure root exists
if residual(Tmin) * residual(Tmax) > 0:
    raise ValueError("No sign change in bracket — target enthalpy may be out of range.")

# Step 4: Solve
sol = root_scalar(residual, bracket=[Tmin, Tmax], method='brentq')

# Step 5: Output
if sol.converged:
    print(f"Temperature = {sol.root:.4f} K for enthalpy = {target_enthalpy} J/kg")
else:
    print("Root finding failed to converge.")
