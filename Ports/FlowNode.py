from __future__ import annotations
from typing import Optional, Iterable, List
from Fluid import Fluid


class FlowNode:
    """
    Shared thermodynamic state (T, P, X, fluid_name) between exactly two FlowPorts.
    Mass flow is NOT stored here, but accessible from connected ports.
    """
    _counter = 0

    def __init__(self, fluid_a: Optional[Fluid] = None, fluid_b: Optional[Fluid] = None) -> None:
        self.name = f"node_{FlowNode._counter}"
        FlowNode._counter += 1

        self._fluid_name: Optional[str] = None
        self._T: Optional[float] = None
        self._P: Optional[float] = None
        self._X: Optional[float] = None

        self._ports: List[FlowPort] = []

        if fluid_a and fluid_b:
            if fluid_a.name != fluid_b.name:
                raise ValueError("Connected fluids must be of the same type.")
            self._fluid_name = fluid_a.name
            self._T = self._coalesce(fluid_a.temperature, fluid_b.temperature)
            self._P = self._coalesce(fluid_a.pressure, fluid_b.pressure)
            self._X = self._coalesce_nullable(fluid_a.quality, fluid_b.quality)

    # -------- registration --------

    def register_ports(self, *ports: Iterable[FlowPort]) -> None:
        for p in ports:
            if p not in self._ports:
                self._ports.append(p)

    # -------- getters --------

    @property
    def T(self) -> Optional[float]:
        return self._T

    @property
    def P(self) -> Optional[float]:
        return self._P

    @property
    def X(self) -> Optional[float]:
        return self._X

    @property
    def fluid_name(self) -> Optional[str]:
        return self._fluid_name

    @property
    def port_mass_flows(self) -> List[Optional[float]]:
        return [p.mass_flow for p in self._ports]

    @property
    def inlet_mass_flow(self) -> Optional[float]:
        return self._get_port_mass_flow_by_type("in")

    @property
    def outlet_mass_flow(self) -> Optional[float]:
        return self._get_port_mass_flow_by_type("out")

    def _get_port_mass_flow_by_type(self, kind: str) -> Optional[float]:
        for p in self._ports:
            if kind == "in" and p.__class__.__name__ == "InFlow":
                return p.mass_flow
            if kind == "out" and p.__class__.__name__ == "OutFlow":
                return p.mass_flow
        return None

    # -------- setters (authoritative, push to ports) --------

    def set_T(self, T: float) -> None:
        self._T = T
        self._push_to_ports()

    def set_P(self, P: float) -> None:
        self._P = P
        self._push_to_ports()

    def set_X(self, X: float) -> None:
        self._X = X
        self._push_to_ports()

    def set_state(
        self,
        *,
        T: Optional[float] = None,
        P: Optional[float] = None,
        X: Optional[float] = None,
        fluid_name: Optional[str] = None,
    ) -> None:
        if fluid_name is not None:
            self._fluid_name = fluid_name
        if T is not None:
            self._T = T
        if P is not None:
            self._P = P
        if X is not None:
            self._X = X
        self._push_to_ports()

    def set_from_fluid(self, fluid: Fluid) -> None:
        self.set_state(
            T=fluid.temperature,
            P=fluid.pressure,
            X=fluid.quality,
            fluid_name=fluid.name,
        )

    def adopt_from_port(self, port: FlowPort) -> None:
        self.set_state(
            T=port.T if port.T is not None else self._T,
            P=port.P if port.P is not None else self._P,
            X=port.X if port.X is not None else self._X,
            fluid_name=port._fluid_name or self._fluid_name,
        )

    # -------- internals --------

    def _push_to_ports(self) -> None:
        for p in self._ports:
            p._apply_node_state(
                T=self._T,
                P=self._P,
                X=self._X,
                fluid_name=self._fluid_name,
            )

    @staticmethod
    def _coalesce(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is not None and b is not None:
            return 0.5 * (a + b)
        return a if a is not None else b

    @staticmethod
    def _coalesce_nullable(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return FlowNode._coalesce(a, b)

    # -------- repr/str --------

    def __repr__(self) -> str:
        return f"<FlowNode {self.name} | T={self.T}, P={self.P}, X={self.X}>"

    def __str__(self) -> str:
        return (
            f"FlowNode('{self.name}')\n"
            f"  T = {self.T} K\n"
            f"  P = {self.P} Pa\n"
            f"  X = {self.X}"
        )


# ✅ Delayed import to resolve circular dependency
from .FlowPort import FlowPort
