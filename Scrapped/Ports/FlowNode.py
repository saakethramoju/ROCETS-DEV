from __future__ import annotations
from typing import Optional, Iterable, List, TYPE_CHECKING
from Fluids import BaseFluid, Mixture

if TYPE_CHECKING:
    from .FlowPort import FlowPort
    from Scrapped.System import System

class FlowNode:
    """
    Shared fluid state between two ports.
    Carries one BaseFluid object that obeys two-input constraint.
    """
    _counter = 0

    def __init__(self, fluid_a: Optional[BaseFluid] = None, fluid_b: Optional[BaseFluid] = None) -> None:
        self.name = f"node_{FlowNode._counter}"
        FlowNode._counter += 1

        self._ports: List[FlowPort] = []
        self._fluid: Optional[BaseFluid] = self._resolve_fluid(fluid_a, fluid_b)
        self._system: Optional["System"] = None

        # Transient storage state
        self.V: float = 1.0  # default volume [m^3], should be configurable per case
        self.M: float = 0.0  # total mass in node [kg]
        self.U: float = 0.0  # total internal energy [J]

        # Store previous state for time-stepping
        self.M_prev: float = 0.0
        self.U_prev: float = 0.0



    def is_boundary_node(self) -> bool:
        """
        Returns True if the node is connected to a boundary component.
        A boundary node is one where any connected port's parent is an inlet or outlet boundary class.
        """
        inlet_boundary_types = {"MassFlowInlet", "FluidStateInlet", "Inlet"}
        outlet_boundary_types = {"MassFlowOutlet", "FluidStateOutlet", "Outlet"}
        boundary_classes = inlet_boundary_types | outlet_boundary_types

        for port in self._ports:
            if port.parent and port.parent.__class__.__name__ in boundary_classes:
                return True
        return False

    def _resolve_fluid(self, a: Optional[BaseFluid], b: Optional[BaseFluid]) -> Optional[BaseFluid]:
        if a and b:
            return b  # Prioritize second
        return a or b

    # -------- Register ports --------

    def register_ports(self, *ports: FlowPort) -> None:
        for p in ports:
            if p not in self._ports:
                self._ports.append(p)
                p._node = self
                p._fluid = self._fluid  # Share fluid reference

    # -------- Fluid interface --------

    @property
    def fluid(self) -> Optional[BaseFluid]:
        return self._fluid

    @fluid.setter
    def fluid(self, f: BaseFluid) -> None:
        self._fluid = f
        for p in self._ports:
            p.fluid = f

    @property
    def fluid_name(self) -> Optional[str]:
        return self._fluid.name if self._fluid else None

    @property
    def T(self) -> Optional[float]:
        return self._fluid.T if self._fluid else None

    @T.setter
    def T(self, value: float) -> None:
        if not self._fluid:
            raise ValueError("No fluid available to set T.")
        self._fluid.T = value

    @property
    def P(self) -> Optional[float]:
        return self._fluid.P if self._fluid else None

    @P.setter
    def P(self, value: float) -> None:
        if not self._fluid:
            raise ValueError("No fluid available to set P.")
        self._fluid.P = value

    @property
    def X(self) -> Optional[float]:
        return self._fluid.X if self._fluid else None

    @X.setter
    def X(self, value: float) -> None:
        if not self._fluid:
            raise ValueError("No fluid available to set X.")
        self._fluid.X = value

    # -------- Adoption logic --------

    def set_from_fluid(self, fluid: BaseFluid) -> None:
        if self._fluid and self._fluid.name == fluid.name:
            return  # Already compatible
        self.fluid = fluid

    def adopt_from_port(self, port: FlowPort) -> None:
        if port.fluid:
            self.fluid = port.fluid

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, sys):
        if self._system is sys:
            return
        self._system = sys
        if hasattr(sys, "nodes") and self not in sys.nodes:
            sys.nodes.append(self) 


    # -------- Mass flow helpers --------

    @property
    def port_mass_flows(self) -> List[Optional[float]]:
        return [p.mass_flow for p in self._ports]

    @property
    def inlet_mass_flow(self) -> Optional[float]:
        return self._get_port_mass_flow("InFlow")

    @property
    def outlet_mass_flow(self) -> Optional[float]:
        return self._get_port_mass_flow("OutFlow")

    def _get_port_mass_flow(self, role: str) -> Optional[float]:
        for p in self._ports:
            if p.__class__.__name__ == role:
                return p.mass_flow
        return None

    @property
    def port_energy_flows(self) -> List[Optional[float]]:
        return [p.energy_flow for p in self._ports]
        
    @property
    def inlet_energy_flow(self) -> Optional[float]:
        return self._get_port_energy_flow("InFlow")

    @property
    def outlet_energy_flow(self) -> Optional[float]:
        return self._get_port_energy_flow("OutFlow")

    def _get_port_energy_flow(self, role: str) -> Optional[float]:
        for p in self._ports:
            if p.__class__.__name__ == role:
                return p.energy_flow
        return None


    # -------- Repr / Str --------

    def __repr__(self) -> str:
        return f"<FlowNode {self.name} | fluid={self.fluid_name}, T={self.T}, P={self.P}, X={self.X}>"

    def __str__(self) -> str:
        phase = getattr(self._fluid, "phase", None)
        return (
            f"FlowNode('{self.name}')\n"
            f"  Fluid     = {self.fluid_name}\n"
            f"  Phase     = {phase if phase is not None else '-'}\n"
            f"  T         = {self.T} K\n"
            f"  P         = {self.P} Pa\n"
            f"  X         = {self.X}"
        )

    # -------- Mixture-specific support --------

    @property
    def mole_fractions(self) -> dict[str, float]:
        if isinstance(self._fluid, Mixture):
            return self._fluid.mole_fractions
        raise AttributeError("Mole fractions only exist for mixture fluids.")

    @mole_fractions.setter
    def mole_fractions(self, new_fractions: dict[str, float]) -> None:
        if isinstance(self._fluid, Mixture):
            self._fluid.set_mole_fractions(new_fractions)
            for p in self._ports:
                p.fluid = self._fluid
        else:
            raise AttributeError("Cannot set mole fractions on non-mixture fluid.")

    @property
    def mass_fractions(self) -> dict[str, float]:
        if isinstance(self._fluid, Mixture):
            return self._fluid.mass_fractions
        raise AttributeError("Mass fractions only exist for mixture fluids.")

    @mass_fractions.setter
    def mass_fractions(self, new_fractions: dict[str, float]) -> None:
        if isinstance(self._fluid, Mixture):
            self._fluid.set_mass_fractions(new_fractions)
            for p in self._ports:
                p.fluid = self._fluid
        else:
            raise AttributeError("Cannot set mass fractions on non-mixture fluid.")


    def mass_residual(self, analysis_type: str = "steady-state") -> Optional[float]:
        """
        Compute mass conservation residual:
            steady-state: m_dot_out - m_dot_in
            transient: not yet implemented
        """
        if len(self._ports) != 2:
            return None  # ill-formed node
        

        a, b = self._ports

        if a.mass_flow is None or b.mass_flow is None:
            return None

        if a.__class__.__name__ == "InFlow":
            m_out, m_in = a.mass_flow, b.mass_flow
        elif b.__class__.__name__ == "InFlow":
            m_out, m_in = b.mass_flow, a.mass_flow
        else:
            return None  # malformed node (shouldn’t happen)
        

        if analysis_type == "steady-state":
            return m_out - m_in


        elif analysis_type == "transient":
            dMdt = (self.M - self.M_prev) / self.system.dt
            return m_out + dMdt - m_in

        return None  # invalid analysis type
    

    def energy_residual(self, analysis_type: str = "steady-state") -> Optional[float]:
        """
        Compute energy conservation residual:
            steady-state: e_out - e_in
        """
        if len(self._ports) != 2:
            return None  # ill-formed node

        a, b = self._ports

        if a.energy_flow is None or b.energy_flow is None:
            return None

        if a.__class__.__name__ == "InFlow":
            e_out, e_in = a.energy_flow, b.energy_flow
        elif b.__class__.__name__ == "InFlow":
            e_out, e_in = b.energy_flow, a.energy_flow
        else:   
            return None  # malformed node
        
        
        if analysis_type == "steady-state":
            return e_out - e_in

        elif analysis_type == "transient":
            dUdt = (self.U - self.U_prev) / self.system.dt
            return e_out + dUdt - e_in

        return None



    def residual(self, analysis_type: str = "steady-state") -> list[float]:
        """
        Returns a list of residuals associated with this node for the given analysis type.
        Includes mass and energy residuals if defined.
        """
        residuals = []

        mass = self.mass_residual(analysis_type=analysis_type)
        if mass is not None:
            residuals.append(mass)

        energy = self.energy_residual(analysis_type=analysis_type)
        if energy is not None:
            residuals.append(energy)
            

        return residuals



    def timestep(self):
        """
        Update node's mass and energy storage based on inflow/outflow over dt.
        Only applies to non-boundary nodes.
        """
        if self.is_boundary_node():
            return  # Skip storage update for boundaries

        # Validate mass flow
        if self.inlet_mass_flow is None or self.outlet_mass_flow is None:
            raise RuntimeError(f"{self.name}: Missing mass flow rate.")

        if self.inlet_energy_flow is None or self.outlet_energy_flow is None:
            raise RuntimeError(f"{self.name}: Missing energy flow rate.")

        # Save previous state
        self.M_prev = self.M
        self.U_prev = self.U

        # Update storage using conservation equations
        dm = self.inlet_mass_flow - self.outlet_mass_flow
        dU = self.inlet_energy_flow - self.outlet_energy_flow

        self.M += dm * self.system.dt
        self.U += dU * self.system.dt

        print(self.M, self.M_prev)

