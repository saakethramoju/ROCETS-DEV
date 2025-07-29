from __future__ import annotations
from typing import Optional, Iterable, List, TYPE_CHECKING
from Fluid import Fluid

if TYPE_CHECKING:
    from .FlowPort import FlowPort


class FlowNode:
    """
    Shared fluid state between two ports.
    Carries one Fluid object that obeys two-input constraint.
    """
    _counter = 0

    def __init__(self, fluid_a: Optional[Fluid] = None, fluid_b: Optional[Fluid] = None) -> None:
        self.name = f"node_{FlowNode._counter}"
        FlowNode._counter += 1

        self._ports: List[FlowPort] = []
        self._fluid: Optional[Fluid] = None

        # Decide which fluid to adopt
        self._fluid = self._resolve_fluid(fluid_a, fluid_b)

    def _resolve_fluid(self, a: Optional[Fluid], b: Optional[Fluid]) -> Optional[Fluid]:
        if a and b:
            # Prioritize b (typically OutFlow), replace a
            return b
        return a or b

    # -------- Register ports --------

    def register_ports(self, *ports: Iterable[FlowPort]) -> None:
        for p in ports:
            if p not in self._ports:
                self._ports.append(p)
                p._node = self
                p._fluid = self._fluid  # Share fluid reference

    # -------- Fluid interface --------

    @property
    def fluid(self) -> Optional[Fluid]:
        return self._fluid

    @fluid.setter
    def fluid(self, f: Fluid) -> None:
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

    def set_from_fluid(self, fluid: Fluid) -> None:
        self.fluid = fluid  # will push to all ports

    def adopt_from_port(self, port: FlowPort) -> None:
        if not port.fluid:
            return
        self.fluid = port.fluid  # Shared across all

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
        return (
            f"FlowNode('{self.name}')\n"
            f"  Fluid     = {self.fluid_name}\n"
            f"  T         = {self.T} K\n"
            f"  P         = {self.P} Pa\n"
            f"  X         = {self.X}"
        )