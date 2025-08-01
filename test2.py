from Fluids import Fluid, Mixture
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


mix = Mixture({"Oxygen": 0.21, "Nitrogen": 0.79}, fraction_type="mass", T=300, P=101325)
print(mix)
print(mix.min_temperature, mix.max_temperature)
#print(Mixture.get_temperature_from_Ph({"Methane": 0.5, "Ethane": 0.5}, P=101325, target_h=104860, fraction_type="mole"))

import CoolProp.CoolProp as CP
from CoolProp import AbstractState

# Set backend and components
backend = "HEOS"
components = ["Oxygen", "Nitrogen"]

# Define mass fractions: 21% O₂, 79% N₂ by mass
mass_fractions = [0.21, 0.79]

# Create AbstractState instance correctly
state = AbstractState(backend, "&".join(components))

# Set mass fractions
state.set_mass_fractions(mass_fractions)

# Set thermodynamic state
T = 298.15     # K
P = 101325     # Pa
state.update(CP.PT_INPUTS, P, T)

# Map CoolProp phase integer to string
phase_map = {
    CP.iphase_liquid: "Liquid",
    CP.iphase_gas: "Gas",
    CP.iphase_twophase: "Two-Phase",
    CP.iphase_supercritical: "Supercritical",
    CP.iphase_supercritical_gas: "Supercritical Gas",
    CP.iphase_supercritical_liquid: "Supercritical Liquid",
    CP.iphase_critical_point: "Critical Point",
    CP.iphase_unknown: "Unknown",
    CP.iphase_not_imposed: "Not Imposed"
}

# Print results
#print("Temperature:", T, "K")
#print("Pressure:", P, "Pa")
#print("Mass fractions:", dict(zip(components, mass_fractions)))
#print("Phase (enum):", state.phase())
#print("Phase (string):", phase_map.get(state.phase(), "Unknown"))
