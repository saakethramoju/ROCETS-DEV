import numpy as np
from scipy.interpolate import interp1d
from Core import Component
from Constants import Constants as cs
from rocketcea.cea_obj_w_units import CEA_Obj
from Exceptions import (MissingConfigurationKeyError, MissingConfigurationValueError,
                        MissingMixtureRatioError)

class TCA(Component):

    required_config_keys = [
        "Contour Points",
        "Nozzle Type",
        "Throat Radius (in)",
        "Chamber Length (in)",
        "Contraction Ratio",
        "Convergence Half-Angle (°)",
        "Convergence Radius Factor",
        "Lead-in Radius Factor",
        "Lead-out Radius Factor",
        "Expansion Ratio",
        "Ambient Pressure (psia)",
        "Combustor Area"
    ]

    optional_config_keys = [
        "Divergence Half-Angle (°)",  # conical
        "Divergence Entrance Angle (°)",
        "Divergence Exit Angle (°)",
        "Percent Bell (%)"  # bell
    ]

    config_keys = required_config_keys + optional_config_keys

    def __init__(self, name):
        super().__init__(name)

        self._initialize_default_ports()

    def _initialize_default_ports(self):

        # Inputs:
        self.add_input("Chamber Pressure (psia)")
        self.add_input("Mixture Ratio")
        self.add_input("Fuel Temperature (K)")
        self.add_input("Oxidizer Temperature (K)")
        self.add_input("Oxidizer")
        self.add_input("Fuel")

        # Outputs:

    def validate_config(self, return_missing=False):
        super().validate_config()  # validates required keys + values, including "Nozzle Type"
        nozzle_type = self.configuration.get("Nozzle Type", "").strip().lower()
        if nozzle_type == "bell":
            expected = [
                "Divergence Entrance Angle (°)",
                "Divergence Exit Angle (°)",
                "Percent Bell (%)"
            ]
        elif nozzle_type == "conical":
            expected = ["Divergence Half-Angle (°)"]
        else:
            print(f"Warning: Unknown nozzle type '{nozzle_type}' in TCA '{self.name}'")
            expected = []
        missing_keys = []
        missing_values = []
        for key in expected:
            if key not in self.configuration:
                missing_keys.append(key)
            elif self.configuration[key] in [None, "", "—"]:
                missing_values.append(key)
        if missing_keys:
            raise MissingConfigurationKeyError(
                f"TCA '{self.name}' is missing required config keys for nozzle type '{nozzle_type}': {', '.join(missing_keys)}"
            )
        if missing_values:
            raise MissingConfigurationValueError(
                f"TCA '{self.name}' has required config keys with missing values for nozzle type '{nozzle_type}': {', '.join(missing_values)}"
            )
        if return_missing:
            return {"missing_keys": missing_keys, "missing_values": missing_values}

    def _get_mixture_ratio(self):
        val = self["Mixture Ratio"]
        if val is None:
            raise MissingMixtureRatioError("Please specficy a valid mixture ratio!")
        return val
    
    def throat_area(self):
        return np.pi * (self["Throat Radius (in)"] ** 2)

    def generate_cea(self):
        """Create a CEA Object and calculate Rayleigh-corrected chamber pressure."""
        Pc = self["Chamber Pressure (psia)"]
        mr = self._get_mixture_ratio()
        if self['Combustor Area'].lower() == 'finite':
            cea = CEA_Obj(
                oxName=self["Oxidizer"], fuelName=self["Fuel"], temperature_units='degK',
                cstar_units='m/sec', specific_heat_units='kJ/kg degK',
                sonic_velocity_units='m/s', enthalpy_units='J/kg',
                density_units='kg/m^3', fac_CR=self["Contraction Ratio"]
            )
            self.add_system_key("Chamber Pressure Rayleigh (psia)", None)
            self["Chamber Pressure Rayleigh (psia)"] = Pc / cea.get_Pinj_over_Pcomb(Pc, mr, self["Contraction Ratio"])
        else:
            cea = CEA_Obj(
                oxName=self["Oxidizer"], fuelName=self["Fuel"], temperature_units='degK',
                cstar_units='m/sec', specific_heat_units='kJ/kg degK',
                sonic_velocity_units='m/s', enthalpy_units='J/kg',
                density_units='kg/m^3'
            )
            self.add_system_key("Chamber Pressure Rayleigh (psia)", None)
            self["Chamber Pressure Rayleigh (psia)"] = self["Chamber Pressure (psia)"]

        self.cea = cea
        return cea
    
    def mass_conservation_equation(self):
        mr = self["Mixture Ratio"]
        At = self.throat_area()
        cea = self.generate_cea()
        _, Tt, _ = cea.get_Temperatures(self["Chamber Pressure Rayleigh (psia)"], mr)
        mwt, gammat = cea.get_Throat_MolWt_gamma(self["Chamber Pressure Rayleigh (psia)"], mr)
        return (At * self["Chamber Pressure Rayleigh (psia)"] * 4.44822) * np.sqrt(mwt * gammat / (cs.R * Tt))