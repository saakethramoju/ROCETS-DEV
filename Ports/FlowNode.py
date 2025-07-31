from __future__ import annotations
from typing import Optional, Iterable, List, TYPE_CHECKING
from Fluids import BaseFluid, Mixture

if TYPE_CHECKING:
    from .FlowPort import FlowPort


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
        if analysis_type == "steady-state":
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

            return m_out - m_in

        elif analysis_type == "transient":
            return None  # not implemented

        return None  # invalid analysis type
    


    def energy_residual(self, analysis_type: str = "steady-state") -> Optional[float]:
        """
        Compute energy conservation residual:
            steady-state: m_out*(h_out + ½v_out²) - m_in*(h_in + ½v_in²)
            transient: not yet implemented
        """
        if analysis_type == "steady-state":
            if len(self._ports) != 2:
                return None  # ill-formed node

            a, b = self._ports

            if a.mass_flow is None or b.mass_flow is None:
                return None
            if a.fluid is None or b.fluid is None:
                return None

            if a.__class__.__name__ == "InFlow":
                out_port, in_port = a, b
            elif b.__class__.__name__ == "InFlow":
                out_port, in_port = b, a
            else:
                return None  # malformed node

            try:
                m_in = in_port.mass_flow
                m_out = out_port.mass_flow

                rho_in = in_port.fluid.density
                rho_out = out_port.fluid.density

                v_in = m_in / rho_in # assume A = 1
                v_out = m_out / rho_out


                h_in = in_port.fluid.enthalpy
                h_out = out_port.fluid.enthalpy


                return m_out * (h_out + 0.5 * v_out**2) - m_in * (h_in + 0.5 * v_in**2)

            except (AttributeError, ZeroDivisionError):
                return None

        elif analysis_type == "transient":
            return None  # not implemented

        return None  # invalid analysis type



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
