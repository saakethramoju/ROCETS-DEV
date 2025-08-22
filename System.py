from typing import Set
from Components import Component


class System:
    """
    Represents a collection of connected components forming a fluid system.
    """

    def __init__(self, name: str):
        self.name = name
        self.components: Set[Component] = set()  # use a set internally

    def add_component(self, component: Component):
        """
        Add a component (and any connected components) into this system.
        """
        to_add: Set[Component] = set()
        self._gather_connected_components(component, to_add)

        # Assign them to this system
        for comp in to_add:
            comp.parent = self
        self.components.update(to_add)

    def _gather_connected_components(self, component: Component, collected: Set[Component]):
        """
        Recursively gather all connected components to ensure system completeness.
        """
        if component in collected:
            return
        collected.add(component)

        # Walk through inflows and outflows
        for port in component.inflows + component.outflows:
            if port.connection is not None:
                other_comp = port.connection.parent  # <-- other component
                if other_comp is not None:
                    self._gather_connected_components(other_comp, collected)

    def __repr__(self):
        return f"<System {self.name}, {len(self.components)} components>"

    def __str__(self):
        comp_list = "\n".join(f"  - {comp.name} ({comp.type.name})" for comp in self.components)
        return f"System '{self.name}' with {len(self.components)} components:\n{comp_list}"
