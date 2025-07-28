from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from .FlowNode import FlowNode
from Fluid import Fluid
if TYPE_CHECKING:
    from Components.Component import Component


class FlowPort(ABC):
    def __init__(self, name: str, fluid_name: str = "Water", parent: Optional["Component"] = None) -> None:
        self.name = name  # just "inlet", "outlet", etc.
        self.parent = parent  # <- New attribute to track parent Component
        self._fluid_name = fluid_name

        self._fluid: Optional[Fluid] = None
        self._node: Optional[FlowNode] = None
        self._connected_port: Optional["FlowPort"] = None

        self._m_dot: Optional[float] = None

        self._T: Optional[float] = None
        self._P: Optional[float] = None
        self._X: Optional[float] = None

        self._suspend_sync: bool = False

    # --------------- fluid build / sync helpers ---------------
    def _build_fluid_if_ready(self) -> None:
        if self._fluid is None and self._has_two_state_vars():
            self._fluid = Fluid(self._fluid_name, T=self._T, P=self._P, X=self._X)

    def _update_fluid(self) -> None:
        if self._has_two_state_vars():
            self._fluid = Fluid(self._fluid_name, T=self._T, P=self._P, X=self._X)

    def _has_two_state_vars(self) -> bool:
        return sum(v is not None for v in [self._T, self._P, self._X]) >= 2

    def _apply_node_state(self, *, T, P, X, fluid_name) -> None:
        self._suspend_sync = True
        try:
            if fluid_name is not None:
                self._fluid_name = fluid_name
            self._T = T
            self._P = P
            self._X = X
            self._update_fluid()
        finally:
            self._suspend_sync = False

    def _tell_node_to_update_from_self(self) -> None:
        if self._node and not self._suspend_sync:
            self._node.adopt_from_port(self)
    # --------------- public API ---------------

    @property
    def T(self) -> Optional[float]:
        return self._node.T if self._node else (self._fluid.temperature if self._fluid else self._T)

    @T.setter
    def T(self, value: float) -> None:
        self._T = value
        if self._node:
            self._node.set_T(value)
        else:
            if self._fluid:
                self._update_fluid()
            else:
                self._build_fluid_if_ready()

    @property
    def P(self) -> Optional[float]:
        return self._node.P if self._node else (self._fluid.pressure if self._fluid else self._P)

    @P.setter
    def P(self, value: float) -> None:
        self._P = value
        if self._node:
            self._node.set_P(value)
        else:
            if self._fluid:
                self._update_fluid()
            else:
                self._build_fluid_if_ready()

    @property
    def X(self) -> Optional[float]:
        return self._node.X if self._node else (self._fluid.quality if self._fluid else self._X)

    @X.setter
    def X(self, value: float) -> None:
        self._X = value
        if self._node:
            self._node.set_X(value)
        else:
            if self._fluid:
                self._update_fluid()
            else:
                self._build_fluid_if_ready()

    @property
    def fluid(self) -> Optional[Fluid]:
        return self._fluid

    @fluid.setter
    def fluid(self, f: Fluid) -> None:
        self._fluid = f
        self._fluid_name = f.name
        self._T = f.temperature
        self._P = f.pressure
        self._X = f.quality
        if self._node:
            self._node.set_from_fluid(f)  # push to node & other port
        else:
            # local only
            self._update_fluid()

    @property
    def mass_flow(self) -> Optional[float]:
        return self._m_dot

    @mass_flow.setter
    def mass_flow(self, value: Optional[float]) -> None:
        self._m_dot = value

    @property
    def fluid_name(self) -> Optional[str]:
        return self._fluid_name

    @fluid_name.setter
    def fluid_name(self, name: str) -> None:
        self._fluid_name = name
        if self._fluid:
            # Rebuild the fluid using existing T/P/X but new name
            self._update_fluid()
        else:
            self._build_fluid_if_ready()
        if self._node:
            self._node.set_state(
                fluid_name=name,
                T=self._T,
                P=self._P,
                X=self._X,
            )


    @property
    def node(self) -> Optional[FlowNode]:
        return self._node

    @property
    def connected_port(self) -> Optional["FlowPort"]:
        return self._connected_port

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

    # --------------- repr/str ---------------

    def __repr__(self) -> str:
        role = self.__class__.__name__
        return f"<{role} {self.name} | T={self.T}, P={self.P}, X={self.X}, m_dot={self.mass_flow}>"

    def __str__(self) -> str:
        role = self.__class__.__name__
        return (
            f"{role}('{self.name}')\n"
            f"  T = {self.T} K\n"
            f"  P = {self.P} Pa\n"
            f"  X = {self.X}\n"
            f"  m_dot = {self.mass_flow} kg/s\n"
            f"  Connected to = {self._connected_port.name if self._connected_port else 'None'}\n"
            f"  Node = {self._node.name if self._node else 'None'}"
        )


class InFlow(FlowPort):
    def __init__(self, name: str, fluid_name: str = "Water", parent: Optional["Component"] = None):
        super().__init__(name, fluid_name, parent=parent)

    def connect(self, other: FlowPort) -> None:
        if not isinstance(other, OutFlow):
            raise TypeError("InFlow can only connect to an OutFlow.")
        self._ensure_free(self, other)
        node = FlowNode(self._fluid, other._fluid)
        self._node = other._node = node
        self._connected_port = other
        other._connected_port = self
        node.register_ports(self, other)
        node.adopt_from_port(self)
        node.adopt_from_port(other)


class OutFlow(FlowPort):
    def __init__(self, name: str, fluid_name: str = "Water", parent: Optional["Component"] = None):
        super().__init__(name, fluid_name, parent=parent)

    def connect(self, other: FlowPort) -> None:
        if not isinstance(other, InFlow):
            raise TypeError("OutFlow can only connect to an InFlow.")
        self._ensure_free(self, other)
        node = FlowNode(self._fluid, other._fluid)
        self._node = other._node = node
        self._connected_port = other
        other._connected_port = self
        node.register_ports(self, other)
        node.adopt_from_port(self)
        node.adopt_from_port(other)

#from Components import Component