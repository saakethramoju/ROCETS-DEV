import pandas as pd
from .Component import Component


class Inlet(Component):
    """Boundary condition with a single OutFlow port."""

    def __init__(self, name: str, port_name: str):
        super().__init__(name)
        self.outflow = self.add_outflow(port_name)

    def __str__(self):
        p = self.outflow
        return (
            f"[Inlet: {self.name}]\n"
            f"  Port: {p.name}\n"
            f"  Connected To: {p.connected_port.name if p.connected_port else '-'}\n"
            f"  Fluid: {p.fluid_name if p.fluid else '-'}\n"
            f"  T [K]: {p.T if p.T is not None else '-'}\n"
            f"  P [Pa]: {p.P if p.P is not None else '-'}\n"
            f"  X: {p.X if p.X is not None else '-'}\n"
            f"  Mass Flow [kg/s]: {p.mass_flow if p.mass_flow is not None else '-'}"
        )
    
    def get_results_dataframe(self):
        p = self.outflow
        if not p:
            return None
        return pd.DataFrame([{
            "Port": p.name,
            "Fluid": str(p.fluid_name) if p.fluid else "-",
            "T [K]": p.T,
            "P [Pa]": p.P,
            "X": p.X,
            "Mass Flow [kg/s]": p.mass_flow,
        }])


class Outlet(Component):
    """Boundary condition with a single InFlow port."""

    def __init__(self, name: str, port_name: str):
        super().__init__(name)
        self.inflow = self.add_inflow(port_name)

    def __str__(self):
        p = self.inflow  # ✅ fixed (was incorrectly using self.outflow)
        return (
            f"[Outlet: {self.name}]\n"
            f"  Port: {p.name}\n"
            f"  Connected To: {p.connected_port.name if p.connected_port else '-'}\n"
            f"  Fluid: {p.fluid_name if p.fluid else '-'}\n"
            f"  T [K]: {p.T if p.T is not None else '-'}\n"
            f"  P [Pa]: {p.P if p.P is not None else '-'}\n"
            f"  X: {p.X if p.X is not None else '-'}\n"
            f"  Mass Flow [kg/s]: {p.mass_flow if p.mass_flow is not None else '-'}"
        )
    
    
    def get_results_dataframe(self):
        p = self.inflow
        if not p:
            return None
        return pd.DataFrame([{
            "Port": p.name,
            "Fluid": str(p.fluid_name) if p.fluid else "-",
            "T [K]": p.T,
            "P [Pa]": p.P,
            "X": p.X,
            "Mass Flow [kg/s]": p.mass_flow,
        }])



# Subclasses for explicit boundary types

class FluidStateInlet(Inlet):
    """Inlet with prescribed fluid state (T, P, optionally X)."""
    pass


class MassFlowInlet(Inlet):
    """Inlet with prescribed mass flow rate."""
    pass


class FluidStateOutlet(Outlet):
    """Outlet with prescribed fluid state (typically P, maybe T/X)."""
    pass


class MassFlowOutlet(Outlet):
    """Outlet with fixed mass flow draw rate."""
    pass
