from abc import ABC, abstractmethod
from typing import Any, Optional, List


class PropertyPort(ABC):
    def __init__(self, name: str, parent: Optional[object] = None) -> None:
        self.name = name
        self.parent = parent  # Reference to owning Component

    @abstractmethod
    def is_connected(self) -> bool:
        ...

    @abstractmethod
    def disconnect(self) -> None:
        ...

    @abstractmethod
    def connect(self, other: "PropertyPort") -> None:
        ...


class PropertyIn(PropertyPort):
    def __init__(self, name: str, parent: Optional[object] = None) -> None:
        super().__init__(name, parent)
        self._connected_out: Optional["PropertyOut"] = None

    @property
    def value(self) -> Any:
        """Returns the value from the connected PropertyOut, or None if unconnected."""
        if self._connected_out:
            return self._connected_out.value
        return None

    @value.setter
    def value(self, val: Any) -> None:
        raise AttributeError("Cannot set value on PropertyIn; it's received from a connected PropertyOut.")

    @property
    def connected_port(self) -> Optional["PropertyOut"]:
        return self._connected_out

    def is_connected(self) -> bool:
        return self._connected_out is not None

    def disconnect(self) -> None:
        """Disconnect from any connected PropertyOut."""
        if self._connected_out:
            if self in self._connected_out._connected_ins:
                self._connected_out._connected_ins.remove(self)
            self._connected_out = None

    def connect(self, other: PropertyPort) -> None:
        if isinstance(other, PropertyIn):
            # Allow chaining only if one is already connected
            if self._connected_out:
                other.connect(self._connected_out)
            elif other._connected_out:
                self.connect(other._connected_out)
            else:
                raise ValueError(
                    "Cannot connect two PropertyIns unless one is already connected to a PropertyOut."
                )
            return

        if not isinstance(other, PropertyOut):
            raise TypeError("PropertyIn can only connect to a PropertyOut.")

        if self._connected_out is other:
            return  # Already connected

        if self._connected_out:
            raise RuntimeError(f"{self.full_name} is already connected to a different PropertyOut.")

        self._connected_out = other
        if self not in other._connected_ins:
            other._connected_ins.append(self)

    @property
    def full_name(self) -> str:
        return f"{self.parent.name}.{self.name}" if self.parent else self.name

    @property
    def source(self) -> Optional["PropertyOut"]:
        """Public accessor for the connected PropertyOut."""
        return self._connected_out

    def __repr__(self) -> str:
        return f"[PropertyIn] {self.full_name} | value={self.value}"


class PropertyOut(PropertyPort):
    def __init__(self, name: str, parent: Optional[object] = None) -> None:
        super().__init__(name, parent)
        self._value: Optional[Any] = None
        self._connected_ins: List[PropertyIn] = []

    @property
    def value(self) -> Any:
        """The output value this port provides to connected inputs."""
        return self._value

    @value.setter
    def value(self, val: Any) -> None:
        self._value = val

    @property
    def connected_ports(self) -> List["PropertyIn"]:
        """Alias for the list of connected PropertyIns."""
        return self._connected_ins

    @property
    def targets(self) -> List["PropertyIn"]:
        """Alias for connected_ports, clearer in some contexts."""
        return self._connected_ins

    def is_connected(self) -> bool:
        return bool(self._connected_ins)

    def disconnect(self) -> None:
        """Disconnect from all connected PropertyIns."""
        for p in self._connected_ins:
            p._connected_out = None
        self._connected_ins.clear()

    def connect(self, other: PropertyPort) -> None:
        if not isinstance(other, PropertyIn):
            raise TypeError("PropertyOut can only connect to PropertyIn.")
        other.connect(self)

    @property
    def full_name(self) -> str:
        return f"{self.parent.name}.{self.name}" if self.parent else self.name

    def __repr__(self) -> str:
        return f"[PropertyOut] {self.full_name} | value={self.value}"
