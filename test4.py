import cantera as ct
from CoolProp.CoolProp import PropsSI

# Standard temperature and pressure
T = 298.15  # K
P = 2e5  # Pa


water = ct.Water()
water.TP = T, P
h = water.enthalpy_mass  # J/kg

print(h)