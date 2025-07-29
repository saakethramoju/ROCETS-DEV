from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from .FlowNode import FlowNode
from Fluids import Fluid

if TYPE_CHECKING:
    from Components.Component import Component


class FlowPort(ABC):
    def __init__(self, name: str, fluid: Optional[Fluid] = None, parent: Optional["Component"] = None) -> None:
        self.name = name
        self.parent = parent
        self._fluid: Optional[Fluid] = fluid
        self._node: Optional[FlowNode] = None
        self._connected_port: Optional["FlowPort"] = None
        self._m_dot: Optional[float] = None

    @property
    def fluid(self) -> Optional[Fluid]:
        return self._fluid

    @fluid.setter
    def fluid(self, f: Fluid) -> None:
        if self._fluid is f:
            return  # ✅ Prevent recursion loop
        self._fluid = f
        if self._node:
            self._node.set_from_fluid(f)

    @property
    def fluid_name(self) -> Optional[str]:
        return self._fluid.name if self._fluid else None

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
        return f"<{role} {self.name} | fluid={self.fluid_name}, T={self.T}, P={self.P}, X={self.X}, m_dot={self.mass_flow}>"

    def __str__(self) -> str:
        role = self.__class__.__name__
        return (
            f"{role}('{self.name}')\n"
            f"  Fluid     = {self.fluid_name}\n"
            f"  T         = {self.T} K\n"
            f"  P         = {self.P} Pa\n"
            f"  X         = {self.X}\n"
            f"  Mass Flow = {self.mass_flow} kg/s\n"
            f"  Connected to = {self._connected_port.name if self._connected_port else 'None'}\n"
            f"  Node         = {self._node.name if self._node else 'None'}"
        )
    
class InFlow(FlowPort):
    def connect(self, other: FlowPort) -> None:
        if not isinstance(other, OutFlow):
            raise TypeError("InFlow can only connect to an OutFlow.")
        self._ensure_free(self, other)

        # Let the node be created with None if both ports are fluidless
        shared_fluid = other.fluid or self.fluid
        node = FlowNode(shared_fluid)

        self._node = other._node = node
        self._connected_port = other
        other._connected_port = self

        self.fluid = shared_fluid  # may be None
        other.fluid = shared_fluid
        node.register_ports(self, other)



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
