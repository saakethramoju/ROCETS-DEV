from typing import TYPE_CHECKING, Optional
from Fluid import Fluid
from Exceptions import PortPermissionError, PortConnectionError

if TYPE_CHECKING:
    from Components.Component import Component
class FlowPort:
    def __init__(self, name: str, parent: "Component" = None):
        self.name = name
        self.parent = parent
        self._P = None
        self._T = None
        self._X = None
        self._mass_flow = None
        self._fluid: Optional[Fluid] = None
        self._fluid_name: Optional[str] = "Water"  # ✅ Default fluid
        self.connected_port: Optional["FlowPort"] = None

        self._update_fluid()  # ✅ Attempt fluid generation on init

    def connect(self, other: "FlowPort"):
        if self.connected_port or other.connected_port:
            raise PortConnectionError(f"Ports '{self.name}' and/or '{other.name}' are already connected.")
        self.connected_port = other
        other.connected_port = self

    def _update_fluid(self):
        """Attempt to construct a Fluid object from available values."""
        name = self._fluid_name or (self.connected_port._fluid_name if self.connected_port else None)
        if not name:
            return

        T = self.T
        P = self.P
        X = self.X

        try:
            if T is not None and P is not None and X is not None:
                self._fluid = Fluid(name, T=T, X=X)
            elif T is not None and P is not None:
                self._fluid = Fluid(name, T=T, P=P)
            elif P is not None and X is not None:
                self._fluid = Fluid(name, P=P, X=X)
            elif T is not None and X is not None:
                self._fluid = Fluid(name, T=T, X=X)
            else:
                self._fluid = None
                return

            self._T = self._fluid.temperature
            self._P = self._fluid.pressure
            self._X = self._fluid.quality
            self._fluid_name = self._fluid.name

        except Exception as e:
            raise ValueError(
                f"Failed to update fluid on port '{self.name}' with "
                f"fluid_name={name}, T={T}, P={P}, X={X}: {e}"
            )

    @property
    def P(self):
        if self._P is not None:
            return self._P
        elif self.connected_port and self.connected_port._P is not None:
            return self.connected_port._P
        elif self.connected_port and self.connected_port._fluid:
            return self.connected_port._fluid.pressure
        elif self._fluid:
            return self._fluid.pressure
        return None

    @P.setter
    def P(self, value):
        self._P = value
        self._update_fluid()
        if self.connected_port and self.connected_port._P is None:
            self.connected_port._P = value
            self.connected_port._update_fluid()

    @property
    def T(self):
        if self._T is not None:
            return self._T
        elif self.connected_port and self.connected_port._T is not None:
            return self.connected_port._T
        elif self.connected_port and self.connected_port._fluid:
            return self.connected_port._fluid.temperature
        elif self._fluid:
            return self._fluid.temperature
        return None

    @T.setter
    def T(self, value):
        self._T = value
        self._update_fluid()
        if self.connected_port and self.connected_port._T is None:
            self.connected_port._T = value
            self.connected_port._update_fluid()

    @property
    def X(self):
        if self._X is not None:
            return self._X
        elif self.connected_port and self.connected_port._X is not None:
            return self.connected_port._X
        elif self.connected_port and self.connected_port._fluid:
            return self.connected_port._fluid.quality
        elif self._fluid:
            return self._fluid.quality
        return None

    @X.setter
    def X(self, value):
        self._X = max(0.0, min(1.0, value)) if value is not None else None
        self._update_fluid()
        if self.connected_port and self.connected_port._X is None:
            self.connected_port._X = self._X
            self.connected_port._update_fluid()

    @property
    def mass_flow(self):
        if self._mass_flow is not None:
            return self._mass_flow
        elif self.connected_port and self.connected_port._mass_flow is not None:
            return self.connected_port._mass_flow
        return None

    @mass_flow.setter
    def mass_flow(self, value):
        if isinstance(self, OutFlow):
            self._mass_flow = value
        else:
            raise PortPermissionError(f"Cannot set mass flow on InFlow port '{self.name}'")

    @property
    def fluid(self) -> Optional[Fluid]:
        return self._fluid if self._fluid is not None else (
            self.connected_port._fluid if self.connected_port else None
        )

    @fluid.setter
    def fluid(self, value: Fluid):
        self._fluid = value
        self._fluid_name = value.name

        self._P = getattr(value, "P", None) or getattr(value, "pressure", None)
        self._T = getattr(value, "T", None) or getattr(value, "temperature", None)

        x_val = getattr(value, "X", None) or getattr(value, "quality", None)
        self._X = max(0.0, min(1.0, x_val)) if x_val is not None else None

        self._update_fluid()

    @property
    def fluid_name(self) -> Optional[str]:
        return self._fluid_name or (
            self.connected_port._fluid_name if self.connected_port else None
        )

    @fluid_name.setter
    def fluid_name(self, name: str):
        self._fluid_name = name
        self._update_fluid()
        if self.connected_port and not self.connected_port._fluid_name:
            self.connected_port._fluid_name = name
            self.connected_port._update_fluid()

    @property
    def is_connected(self) -> bool:
        return self.connected_port is not None

    def __str__(self):
        parent_name = self.parent.name if self.parent else "None"
        direction = "OutFlow" if isinstance(self, OutFlow) else "InFlow"
        conn = self.connected_port
        conn_str = f"{conn.name} [{conn.parent.name}]" if conn and conn.parent else "—"
        fluid_obj = self.fluid
        fluid_name = fluid_obj.name if fluid_obj else "—"
        phase = fluid_obj.phase if fluid_obj else "—"
        pressure = f"{self.P:.3g}" if self.P is not None else "—"
        temperature = f"{self.T:.3g}" if self.T is not None else "—"
        quality = f"{self.X:.3g}" if self.X is not None else "—"
        mass_flow = f"{self.mass_flow:.3g}" if self.mass_flow is not None else "—"

        return (
            f"{direction} '{self.name}' of {parent_name}:\n"
            f"  Connected To  : {conn_str} ({'Connected' if conn else 'Unconnected'})\n"
            f"  Fluid         : {fluid_name}\n"
            f"  Phase         : {phase}\n"
            f"  Pressure      : {pressure} Pa\n"
            f"  Temperature   : {temperature} K\n"
            f"  Quality       : {quality}\n"
            f"  Mass Flow     : {mass_flow} kg/s"
        )

class InFlow(FlowPort): pass
class OutFlow(FlowPort): pass
