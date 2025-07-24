from Component import Component
from FlowPort import OutFlow, InFlow
from Fluid import Fluid
from scipy.optimize import root
import numpy as np

def pipe01(mdot_guess_array, Cd, A, P1, T1, P2, T2):
    """Mass flow residual function for root-finding (mdot iteration)"""
    mdot_guess = mdot_guess_array[0]

    # Inlet
    water1 = Fluid("Water", P=2e5, T=300)
    rho1 = water1.density
    v1 = mdot_guess / (rho1 * Cd * A)

    # Outlet
    water2 = Fluid("Water", P=101325, T=300)
    rho2 = water2.density
    v2 = np.sqrt((2 / rho2) * (P1 - P2 + 0.5 * rho1 * v1**2))

    # Residual: want rho2 * v2 * A * Cd = mdot
    residual = rho2 * v2 * Cd * A - mdot_guess
    return [residual]


# Parameters
Cd = 0.6
A = 1e-3  # m²
P1 = 2e5  # Pa
T1 = 300  # K
P2 = 101325  # Pa
T2 = 300  # K

# Solve using scipy.root
#sol = root(pipe01, x0=[1.0], args=(Cd, A, P1, T1, P2, T2))

#print(f"Success: {sol.success}")
#print(f"mdot: {sol.x[0]:.6f} kg/s")

class PipeTest(Component):
    def __init__(self, name, Cd, A):
        super().__init__(name)
        self.Cd = Cd
        self.A = A
        
        self._initialize_default_ports()

    def _initialize_default_ports(self):

        # inlet flows:
        self.add_inflow("Inlet")

        # outlet flows:
        self.add_outflow("Outlet")

    def pipe01(self):

        mdot1 = self["Inlet"].mass_flow
        fluid1 = self["Inlet"].fluid
        return fluid1
    

pipe = PipeTest("pipe", 0.6, 1e-3)
inlet = Component("Inlet")
inlet.add_outflow("Outflow")
inlet["Outflow"].fluid = Fluid()

