# Sensor.py
from Components import Component

class Sensor(Component):
    """
    A simple sensor component that accepts a single property input.

    This can be used to tap into any scalar property (e.g., temperature, pressure, etc.)
    and monitor or record it downstream in a model.
    """

    def __init__(self, name: str, input_name: str = "Value"):
        """
        Initialize the sensor with a single PropertyIn port.

        Parameters:
            name (str): Name of the sensor component.
            input_name (str): Name of the property input port (default: "Signal").
        """
        super().__init__(name)
        self.input_port = self.add_property_in(input_name)

    def __str__(self):
        return f"{self.name}: {self.input_port.value}"