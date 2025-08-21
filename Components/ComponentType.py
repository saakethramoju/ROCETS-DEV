# Components/ComponentType.py

from enum import Enum, auto

class ComponentType(Enum):
    """
    Enumeration of different component types in the fluid network.
    """

    FLOW = auto()
    JUNCTION = auto()

    def __str__(self):
        return self.name
