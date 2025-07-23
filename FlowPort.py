
from typing import TYPE_CHECKING
from Exceptions import (PortPermissionError, PortConnectionError)

if TYPE_CHECKING:
    from Component import Component
class FlowPort:
    def __init__(self, name: str, parent: "Component" = None):
        self.name = name
        self.parent = parent
        self._P = None
        self._T = None
        self._mass_flow = None
        self.fluid = None
        self.connected_port: "FlowPort" = None

    def connect(self, other: "FlowPort"):
        if self.connected_port or other.connected_port:
            raise PortConnectionError(
                f"Ports '{self.name}' and/or '{other.name}' are already connected."
            )
        self.connected_port = other
        other.connected_port = self

    @property
    def P(self):
        if self._P is not None:
            return self._P
        if self.connected_port and self.connected_port is not self:
            return self.connected_port._P  # Access _P directly to avoid recursion
        return None

    @P.setter
    def P(self, value):
        if isinstance(self, OutFlow):
            self._P = value
        else:
            raise PortPermissionError(f"Cannot set pressure on InFlow port '{self.name}'")

    @property
    def T(self):
        if self._T is not None:
            return self._T
        if self.connected_port and self.connected_port is not self:
            return self.connected_port._T
        return None

    @T.setter
    def T(self, value):
        if isinstance(self, OutFlow):
            self._T = value
        else:
            raise PortPermissionError(f"Cannot set temperature on InFlow port '{self.name}'")

    @property
    def mass_flow(self):
        if self._mass_flow is not None:
            return self._mass_flow
        if self.connected_port and self.connected_port is not self:
            return self.connected_port._mass_flow
        return None

    @mass_flow.setter
    def mass_flow(self, value):
        if isinstance(self, OutFlow):
            self._mass_flow = value
        else:
            raise PortPermissionError(f"Cannot set mass flow on InFlow port '{self.name}'")
            
    @property
    def is_connected(self) -> bool:
        """Return True if this port is connected to another."""
        return self.connected_port is not None


    def __str__(self):
        parent_name = self.parent.name if self.parent else "None"
        direction = "OutFlow" if isinstance(self, OutFlow) else "InFlow"
        conn = self.connected_port
        conn_str = f"{conn.name} [{conn.parent.name}]" if conn and conn.parent else "—"
        status = "Connected" if conn else "Unconnected"
        try:
            pressure = f"{self.P:.3g}" if self.P is not None else "—"
            temperature = f"{self.T:.3g}" if self.T is not None else "—"
            mass_flow = f"{self.mass_flow:.3g}" if self.mass_flow is not None else "—"
        except RecursionError:
            pressure = temperature = mass_flow = "ERR"
        return (
            f"{direction} '{self.name}' of {parent_name}:\n"
            f"  Connected To: {conn_str} ({status})\n"
            f"  Pressure    : {pressure} Pa\n"
            f"  Temperature : {temperature} K\n"
            f"  Mass Flow   : {mass_flow} kg/s"
        )



class InFlow(FlowPort):
    pass

class OutFlow(FlowPort):
    pass