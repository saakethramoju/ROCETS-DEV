from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from .FlowNode import FlowNode
from Fluids import BaseFluid, Mixture, Fluid

if TYPE_CHECKING:
    from Components.Component import Component


class FlowPort(ABC):
    def __init__(self, name: str, fluid: Optional[BaseFluid] = None, parent: Optional["Component"] = None) -> None:
        if fluid is None:
            # Provide a default fluid like water
            fluid = Fluid("Water", T=298.15, P=101325)  # You can tweak this fluid choice
        self.name = name
        self.parent = parent
        self._fluid: Optional[BaseFluid] = fluid
        self._node: Optional[FlowNode] = None
        self._connected_port: Optional["FlowPort"] = None
        self._m_dot: Optional[float] = None
        self._e_dot: Optional[float] = None

    @property
    def fluid(self) -> Optional[BaseFluid]:
        return self._fluid

    @fluid.setter
    def fluid(self, f: BaseFluid) -> None:
        if self._fluid is f:
            return  # ✅ Prevent recursion loop
        self._fluid = f
        if self._node:
            self._node.set_from_fluid(f)

    @property
    def fluid_name(self) -> Optional[str]:
        return getattr(self._fluid, "name", None)

    @property
    def T(self) -> Optional[float]:
        return self._fluid.T if self._fluid else None

    @T.setter
    def T(self, value: float) -> None:
        if not self._fluid:
            raise AttributeError("No fluid set to update temperature.")
        self._fluid.T = value

    @property
    def P(self) -> Optional[float]:
        return self._fluid.P if self._fluid else None

    @P.setter
    def P(self, value: float) -> None:
        if not self._fluid:
            raise AttributeError("No fluid set to update pressure.")
        self._fluid.P = value

    @property
    def X(self) -> Optional[float]:
        return self._fluid.X if self._fluid else None

    @X.setter
    def X(self, value: float) -> None:
        if not self._fluid:
            raise AttributeError("No fluid set to update quality.")
        self._fluid.X = value

    @property
    def mass_flow(self) -> Optional[float]:
        return self._m_dot

    @mass_flow.setter
    def mass_flow(self, value: Optional[float]) -> None:
        self._m_dot = value

    @property
    def energy_flow(self) -> Optional[float]:
        return self._e_dot

    @energy_flow.setter
    def energy_flow(self, value: Optional[float]) -> None:
        self._e_dot = value

    @property
    def connected_port(self) -> Optional["FlowPort"]:
        return self._connected_port

    @property
    def node(self) -> Optional[FlowNode]:
        return self._node

    def is_connected(self) -> bool:
        return self._node is not None

    def disconnect(self) -> None:
        if self._connected_port:
            self._connected_port._connected_port = None
            self._connected_port._node = None
        self._connected_port = None
        self._node = None

    @staticmethod
    def _ensure_free(a: "FlowPort", b: "FlowPort") -> None:
        if a.is_connected() or b.is_connected():
            raise RuntimeError("One or both ports are already connected.")

    @abstractmethod
    def connect(self, other: "FlowPort") -> None:
        ...

    def __repr__(self) -> str:
        role = self.__class__.__name__
        return f"<{role} {self.name} | fluid={self.fluid_name}, T={self.T}, P={self.P}, X={self.X}, m_dot={self.mass_flow}, e_dot={self.energy_flow}>"

    def __str__(self) -> str:
        role = self.__class__.__name__
        phase = getattr(self._fluid, "phase", None)

        return (
            f"{role}('{self.name}')\n"
            f"  Fluid     = {self.fluid_name}\n"
            f"  Phase     = {phase if phase is not None else '-'}\n"
            f"  T         = {self.T} K\n"
            f"  P         = {self.P} Pa\n"
            f"  X         = {self.X}\n"
            f"  Mass Flow = {self.mass_flow} kg/s\n"
            f"  Energy Flow = {self.energy_flow} W\n"
            f"  Connected to = {self._connected_port.name if self._connected_port else 'None'}\n"
            f"  Node         = {self._node.name if self._node else 'None'}"
        )


    @property
    def mole_fractions(self) -> dict[str, float]:
        if isinstance(self._fluid, Mixture):
            return self._fluid.mole_fractions
        raise AttributeError("Mole fractions only exist for mixture fluids.")

    @mole_fractions.setter
    def mole_fractions(self, new_fractions: dict[str, float]) -> None:
        if isinstance(self._fluid, Mixture):
            self._fluid.set_mole_fractions(new_fractions)
            self.fluid = self._fluid  # ✅ Force sync
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
            self.fluid = self._fluid  # ✅ Force sync and notify node
        else:
            raise AttributeError("Cannot set mass fractions on non-mixture fluid.")

    def set_state(self, *, T=None, P=None, X=None):
        if not self._fluid:
            raise AttributeError("No fluid to set state on.")
        self._fluid.set_state(T=T, P=P, X=X)


    def is_boundary(self, *boundary_classes: type) -> bool:
        """
        Return True if this port is:
        1) Not connected, or
        2) Connected to a port whose parent is exactly one of the given boundary classes (not subclasses).
        
        Parameters:
            *boundary_classes: Variable number of class references to check against.
            
        Returns:
            bool: True if unconnected or connected to the exact specified boundary types.
        """
        if self.connected_port is None:
            return True

        parent = self.connected_port.parent
        if parent is None:
            return False

        return any(type(parent) is cls for cls in boundary_classes)




class InFlow(FlowPort):
    def connect(self, other: FlowPort) -> None:
        if not isinstance(other, OutFlow):
            raise TypeError("InFlow can only connect to an OutFlow.")
        self._ensure_free(self, other)

        shared_fluid = other.fluid or self.fluid
        node = FlowNode(shared_fluid)

        self._node = other._node = node
        self._connected_port = other
        other._connected_port = self

        self.fluid = shared_fluid
        other.fluid = shared_fluid
        node.register_ports(self, other)

        # ✅ System detection
        parent_systems = [
            p.parent.system for p in (self, other)
            if p.parent and hasattr(p.parent, "system") and p.parent.system is not None
        ]
        if parent_systems:
            node.system = parent_systems[0]


class OutFlow(FlowPort):
    def connect(self, other: FlowPort) -> None:
        if not isinstance(other, InFlow):
            raise TypeError("OutFlow can only connect to an InFlow.")
        self._ensure_free(self, other)

        shared_fluid = self.fluid or other.fluid
        node = FlowNode(shared_fluid)

        self._node = other._node = node
        self._connected_port = other
        other._connected_port = self

        self.fluid = shared_fluid
        other.fluid = shared_fluid
        node.register_ports(self, other)

        # ✅ System detection
        parent_systems = [
            p.parent.system for p in (self, other)
            if p.parent and hasattr(p.parent, "system") and p.parent.system is not None
        ]
        if parent_systems:
            node.system = parent_systems[0]
