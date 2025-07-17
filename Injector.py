from Component import Component
from typing import Optional
from Exceptions import (PortNotConnectedError, MissingConfigurationError, MissingConfigurationKeyError,
                        MissingConfigurationValueError)

class Injector(Component):
    def __init__(self, name: str, config: Optional[dict] = None):
        super().__init__(name)
        self.configuration = config
        self._initialize_defaults()
        self._initialize_necessary_ports()
        self._initialize_extra_ports()

        if config:
            self.set_config(config)

    def _initialize_defaults(self):
        self.chamber_pressure = None

    def _initialize_necessary_ports(self):
        # Inputs
        self.fuel_data = self.add_input("Fuel Data")
        self.oxidizer_data = self.add_input("Oxidizer Data")

        # Outputs
        self.tca_data = self.add_output("TCA Data")


    def _initialize_extra_ports(self):
        # Inputs
        # EMPTY

        # Outputs
        # EMPTY
        pass

    def set_config(self, config: dict):
        self.configuration = config
        get = config.get

        self.geometry = get("Geometry")  # PLACEHOLDER, REPLACE WITH ACTUAL GEOMETRY LATER

        has_fuel_stiffness = "Fuel Stiffness (%)" in config
        has_ox_stiffness = "Oxidizer Stiffness (%)" in config

        self.given_stiffness_data = has_fuel_stiffness or has_ox_stiffness

        self.fuel_stiffness = get("Fuel Stiffness (%)") if has_fuel_stiffness else None
        self.oxidizer_stiffness = get("Oxidizer Stiffness (%)") if has_ox_stiffness else None



    def validate_config(self):
        if not self.configuration:
            raise MissingConfigurationError(f"No configuration provided for {self.name}")
        config = self.configuration

        if not self.given_stiffness_data:
            if "Geometry" not in config:
                raise MissingConfigurationKeyError("Missing required key: 'Geometry' when stiffness data is not provided")
            if config["Geometry"] is None:
                raise MissingConfigurationValueError("Configuration value for 'Geometry' cannot be None when stiffness data is not provided")
        else:
            if "Fuel Stiffness (%)" not in config:
                raise MissingConfigurationKeyError("Missing required key: 'Fuel Stiffness (%)' when stiffness data is provided")
            if config["Fuel Stiffness (%)"] is None:
                raise MissingConfigurationValueError("Configuration value for 'Fuel Stiffness (%)' cannot be None when stiffness data is provided")

            if "Oxidizer Stiffness (%)" not in config:
                raise MissingConfigurationKeyError("Missing required key: 'Oxidizer Stiffness (%)' when stiffness data is provided")
            if config["Oxidizer Stiffness (%)"] is None:
                raise MissingConfigurationValueError("Configuration value for 'Oxidizer Stiffness (%)' cannot be None when stiffness data is provided")



    def check_fuel_connection(self):
        if not self.fuel_data.is_connected():
            raise PortNotConnectedError(f"Input port not connected: {self.fuel_data.name} in {self.name}")
        
    def check_oxidizer_connection(self):
        if not self.fuel_data.is_connected():
            raise PortNotConnectedError(f"Input port not connected: {self.oxidizer_data.name} in {self.name}")
        
    def check_oxidizer_connection(self):
        if not self.tca_data.is_connected():
            raise PortNotConnectedError(f"Output port not connected: {self.tca_data.name} in {self.name}")
        


if __name__ == "__main__":

    from TCA import TCA


    coax = Injector("Coax")
    tca = TCA("Heatsink")


    coax_config = {"Fuel Stiffness (%)": 30,
              "Oxidizer Stiffness (%)": 15}
    coax.set_config(coax_config)

    tca_config = {'Number of Points Contour': 400,
              "Nozzle Type": "Bell",
              "Throat Radius (in)": 1.2,
              "Chamber Length (in)": 7.9,
              "Contraction Ratio": 2,
              "Convergence Half-Angle (°)": 37.5,
              "Convergence Radius Factor": 1,
              "Lead-in Radius Factor": 1,
              "Lead-out Radius Factor": 0.5,
              "Expansion Ratio": 100,
              "Divergence Half-Angle (°)": 15,
              "Divergence Entrance Angle (°)": 22,
              "Divergence Exit Angle (°)": 10,
              "Percent Bell (%)": 80,
              "Exit Pressure (psia)": 10,
              "Ambient Pressure (psia)": 14.7,
              "Combustor Area": "Finite"}
    tca.set_config(tca_config)

    coax.manual_connect("TCA Data", tca, "Injector Data")

    print(coax)
    print(tca)