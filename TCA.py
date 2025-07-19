from typing import Optional
import numpy as np
from scipy.interpolate import interp1d
from Component import Component
from Injector import Injector
from Constants import Constants as cs
from rocketcea.cea_obj_w_units import CEA_Obj
import re
from Exceptions import (MissingConfigurationError, PortNotConnectedError, MissingConfigurationKeyError,
                        MissingConfigurationValueError, MissingGuessError, MissingGuessKeyError,
                        MissingGuessValueError)

class TCA(Component):
    def __init__(self, name: str, config: Optional[dict] = None, guess: Optional[dict] = None):
        super().__init__(name)
        self.configuration = config
        self.guess = guess

        self._initialize_default_ports()

        if config:
            self.set_config(config)



    def _initialize_default_ports(self):
        self.add_input("Chamber Pressure (psia)", required=True, guess_required=True)
        self.add_input("Mixture Ratio", required=True, guess_required=True)
        self.add_input("Fuel Temperature (K)", required=True, guess_required=True)
        self.add_input("Oxidizer Temperature (K)", required=True, guess_required=True)
        self.add_input("Mass Flow Rate (kg/s)", required=True) # No guess required
        self.add_input("Oxidizer", required=True)  # No guess required
        self.add_input("Fuel", required=True)      # No guess required


    def set_config(self, config: dict):
        self.configuration = config
        get = config.get

        # Normalize all keys in config
        normalized_config = {
            self._normalize_key(k): (k, v) for k, v in config.items()
        }

        def lookup(key, condition=True):
            norm = self._normalize_key(key)
            original, val = normalized_config.get(norm, (None, None))
            return val if condition else None

        nozzle_type = lookup("Nozzle Type", True)
        combustor_area = lookup("Combustor Area", True)

        self.contour_points    = lookup("Number of Points Contour")
        self.nozzle_type       = nozzle_type.lower() if nozzle_type else ""
        self.throat_radius     = lookup("Throat Radius (in)")
        self.chamber_length    = lookup("Chamber Length (in)")
        self.contraction_ratio = lookup("Contraction Ratio")
        self.theta_c           = lookup("Convergence Half-Angle (°)")
        self.rc_rt             = lookup("Convergence Radius Factor")
        self.rtc_rt            = lookup("Lead-in Radius Factor")
        self.rtd_rt            = lookup("Lead-out Radius Factor")
        self.expansion_ratio   = lookup("Expansion Ratio")
        self.exit_pressure     = lookup("Exit Pressure (psia)")
        self.ambient_pressure  = lookup("Ambient Pressure (psia)")
        self.alpha             = lookup("Divergence Half-Angle (°)", self.nozzle_type == "conical")
        self.theta_n           = lookup("Divergence Entrance Angle (°)", self.nozzle_type == "bell")
        self.theta_e           = lookup("Divergence Exit Angle (°)", self.nozzle_type == "bell")
        self.percent_bell      = lookup("Percent Bell (%)", self.nozzle_type == "bell")
        self.combustor_area    = combustor_area

        return True


    def validate_config(self):
        if not self.configuration:
            raise MissingConfigurationError(f"No configuration provided for {self.name}")

        config = self.configuration
        normalized_config = {
            self._normalize_key(k): (k, v) for k, v in config.items()
        }

        required_keys = [
            "Number of Points Contour",
            "Nozzle Type",
            "Throat Radius (in)",
            "Chamber Length (in)",
            "Contraction Ratio",
            "Convergence Half-Angle (°)",
            "Convergence Radius Factor",
            "Lead-in Radius Factor",
            "Lead-out Radius Factor",
            "Expansion Ratio",
            "Exit Pressure (psia)",
            "Ambient Pressure (psia)",
            "Combustor Area"
        ]

        for key in required_keys:
            norm = self._normalize_key(key)
            if norm not in normalized_config:
                raise MissingConfigurationKeyError(f"Missing required configuration key: '{key}'")
            _, value = normalized_config[norm]
            if value is None:
                raise MissingConfigurationValueError(f"Configuration value for '{key}' cannot be None")

        nozzle_type = self.nozzle_type.lower() if hasattr(self, "nozzle_type") else config.get("Nozzle Type", "").lower()

        if nozzle_type == "conical":
            key = "Divergence Half-Angle (°)"
            norm = self._normalize_key(key)
            if norm not in normalized_config or normalized_config[norm][1] is None:
                raise MissingConfigurationKeyError(f"Missing or None value for required conical nozzle key: '{key}'")

        elif nozzle_type == "bell":
            for key in ["Divergence Entrance Angle (°)", "Divergence Exit Angle (°)", "Percent Bell (%)"]:
                norm = self._normalize_key(key)
                if norm not in normalized_config:
                    raise MissingConfigurationKeyError(f"Missing required bell nozzle configuration key: '{key}'")
                _, value = normalized_config[norm]
                if value is None:
                    raise MissingConfigurationValueError(f"Configuration value for '{key}' cannot be None")

        return True

    


    def _normalize_key(self, key: str) -> str:
        key = re.sub(r"\(.*?\)", "", key)  # remove units like (psia)
        return key.strip().lower()

    def set_guess(self, guess: dict):
        self.guess = guess
        for guess_key, value in guess.items():
            norm_guess_key = self._normalize_key(guess_key)
            for port_name, port in {**self.inputs, **self.outputs}.items():
                if self._normalize_key(port_name) == norm_guess_key:
                    self[port_name] = value
                    break
        return True



    def validate_guess(self):
        if not self.guess:
            raise MissingGuessError(f"Initial guesses not provided for {self.name}")

        guess = self.guess
        required = getattr(self, "_guess_required_inputs", set())

        normalized_guess_keys = {
            self._normalize_key(k): k for k in guess.keys()
        }

        for required_key in required:
            norm_key = self._normalize_key(required_key)
            if norm_key not in normalized_guess_keys:
                raise MissingGuessKeyError(f"Missing required initial guess key: '{required_key}'")
            original_key = normalized_guess_keys[norm_key]
            if guess[original_key] is None:
                raise MissingGuessValueError(f"Initial guess value for '{original_key}' cannot be None")

        return True



    def generate_cea(self):
        Pc = self["Chamber Pressure (psia)"]
        if self.combustor_area.lower() == 'finite':
            cea = CEA_Obj(oxName=self["Oxidizer"], fuelName=self["Fuel"], temperature_units='degK', 
                 cstar_units='m/sec', specific_heat_units='kJ/kg degK', 
                 sonic_velocity_units='m/s', enthalpy_units='J/kg', 
                 density_units='kg/m^3', fac_CR=self.contraction_ratio)
            
            self.chamber_pressure_rayleigh = Pc * (1 / cea.get_Pinj_over_Pcomb(Pc, self['Mixture Ratio'], self.contraction_ratio))
        
        else:
            cea = CEA_Obj(oxName=self["Oxidizer"], fuelName=self["Fuel"], temperature_units='degK', 
                 cstar_units='m/sec', specific_heat_units='kJ/kg degK', 
                 sonic_velocity_units='m/s', enthalpy_units='J/kg', 
                 density_units='kg/m^3')
            self.chamber_pressure_rayleigh = self.chamber_pressure

        return cea

    def get_mdot(self):
        At = self.throat_area()
        cea = self.generate_cea()
        _, Tt, _ = cea.get_Temperatures(self.chamber_pressure_rayleigh, self["Mixture Ratio"])
        mwt, gammat = cea.get_Throat_MolWt_gamma(self.chamber_pressure_rayleigh, self["Mixture Ratio"])
        mdot = (At * self.chamber_pressure_rayleigh * 4.44822) * np.sqrt(mwt * gammat / (cs.R * Tt))
        return mdot

    def generate_chamber_geometry(self, points=100):
        self.validate_config()
        z = np.linspace(0, self.chamber_length, points)
        r = np.full_like(z, np.sqrt(self.contraction_ratio) * self.throat_radius)
        return self.resample_curve(np.vstack((z, r)), self.contour_points)

    def generate_converging_geometry(self, points=100):
        self.validate_config()
        R_t, eps_c = self.throat_radius, self.contraction_ratio
        Rc, Rtc = self.rc_rt * R_t, self.rtc_rt * R_t
        theta_c = np.radians(self.theta_c)
        Lc = self.chamber_length

        # Entrance arc
        t1 = np.linspace(np.pi/2, np.pi/2 - theta_c, points)
        z1 = Rc * np.cos(t1) + Lc
        r1 = Rc * np.sin(t1) + R_t * np.sqrt(eps_c) - Rc

        # Linear section
        y3 = R_t * (np.sqrt(eps_c) - 1) - Rc * (1 - np.cos(theta_c)) - Rtc * (1 - np.cos(theta_c))
        x3 = y3 / np.tan(theta_c)
        z2 = np.linspace(z1[-1], z1[-1] + x3, points)
        r2 = -np.tan(theta_c) * (z2 - z1[-1]) + r1[-1]

        # Throat arc
        t2 = np.linspace(np.pi + np.pi/2 - theta_c, 3 * np.pi / 2, points)
        h, k = z2[-1] + Rtc * np.sin(theta_c), R_t + Rtc
        z3 = Rtc * np.cos(t2) + h
        r3 = Rtc * np.sin(t2) + k

        z = np.hstack([z1, z2, z3])
        r = np.hstack([r1, r2, r3])
        return self.resample_curve(np.vstack((z, r)), self.contour_points)

    def generate_nozzle_geometry(self, points=100):
        throat = self.generate_converging_geometry(points)
        z_throat, r_throat = throat[:, -1]
        R_t, eps = self.throat_radius, self.expansion_ratio
        Rtd = self.rtd_rt * R_t
        type_ = self.nozzle_type
        alpha = np.radians(self.alpha or 0)
        theta_n = np.radians(self.theta_n or 0)
        theta_e = np.radians(self.theta_e or 0)
        percent_bell = self.percent_bell or 80

        # Entrance arc
        angle = alpha if type_ == "conical" else theta_n
        t1 = np.linspace(3 * np.pi / 2, 3 * np.pi / 2 + angle, points)
        z1 = Rtd * np.cos(t1) + z_throat
        r1 = Rtd * np.sin(t1) + R_t + Rtd

        # Diverging section
        if type_ == "conical":
            z2 = np.linspace(z1[-1], z1[-1] + (np.sqrt(eps) * R_t - r1[-1]) / np.tan(alpha), points)
            r2 = np.tan(alpha) * (z2 - z1[-1]) + r1[-1]
        else:
            N, E = [z1[-1], r1[-1]], [(percent_bell / 100) * ((np.sqrt(eps) * R_t - r1[-1]) / np.tan(np.radians(15))) + throat[0, 0], R_t * np.sqrt(eps)]
            m1, m2 = np.tan(theta_n), np.tan(theta_e)
            Qx = (m1 * N[0] - N[1] - m2 * E[0] + E[1]) / (m1 - m2)
            Qy = m1 * (Qx - N[0]) + N[1]
            t = np.linspace(0, 1, points)
            z2 = (1 - t)**2 * N[0] + 2 * (1 - t) * t * Qx + t**2 * E[0]
            r2 = (1 - t)**2 * N[1] + 2 * (1 - t) * t * Qy + t**2 * E[1]

        z = np.hstack([z1, z2])
        r = np.hstack([r1, r2])
        return self.resample_curve(np.vstack((z, r)), self.contour_points)

    def generate_geometry(self, points=100):
        self.chamber_geometry = self.generate_chamber_geometry(points)
        self.converging_geometry = self.generate_converging_geometry(points)
        self.nozzle_geometry = self.generate_nozzle_geometry(points)
        full = np.hstack([self.chamber_geometry, self.converging_geometry, self.nozzle_geometry])
        return self.resample_curve(full, self.contour_points)

    def resample_curve(self, curve, n):
        curve = curve.T
        arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
        fx = interp1d(arc, curve[:, 0])
        fy = interp1d(arc, curve[:, 1])
        s_new = np.linspace(0, arc[-1], n)
        return np.vstack((fx(s_new), fy(s_new)))

    def throat_area(self):
        self.validate_config()
        return np.pi * self.throat_radius**2 # in^2

    def injector_area(self):
        return self.throat_area() * self.contraction_ratio

    def chamber_volume(self):
        geometry = self.generate_chamber_geometry()
        y = geometry[1, :]
        x = geometry[0, :]
        dy = np.diff(y)
        dx = np.diff(x)
        return np.pi * np.sum((y[:-1]**2 + y[1:]**2) / 2 * dx)
    
    def converging_volume(self):
        geometry = self.generate_converging_geometry()
        y = geometry[1, :]
        x = geometry[0, :]
        dy = np.diff(y)
        dx = np.diff(x)
        return np.pi * np.sum((y[:-1]**2 + y[1:]**2) / 2 * dx)
    
    def chamber_surface_area(self):
        geometry = self.generate_chamber_geometry()
        y = geometry[1, :]
        x = geometry[0, :]
        dy = np.diff(y)
        dx = np.diff(x)
        ds = np.sqrt(dx**2 + dy**2) 
        return 2 * np.pi * np.sum((y[:-1] + y[1:]) / 2 * ds)
    
    def converging_surface_area(self):
        geometry = self.generate_converging_geometry()
        y = geometry[1, :]
        x = geometry[0, :]
        dy = np.diff(y)
        dx = np.diff(x)
        ds = np.sqrt(dx**2 + dy**2) 
        return 2 * np.pi * np.sum((y[:-1] + y[1:]) / 2 * ds)
    
    def L_star(self):
        return (self.chamber_volume() + self.converging_volume()) / (self.throat_area())
    



if __name__ == "__main__":


    tca = TCA("Heatsink")
    injector = Component("Coax")
    thermocouple = Component("OITC")

    injector.add_output("Chamber Pressure (psia)", required=True)
    injector.add_output("Mixture Ratio", required=True)
    injector.add_output("Fuel Temperature (K)", required=True)
    injector.add_output("Oxidizer Temperature (K)", required=True)
    injector.add_output("Oxidizer", required=True)
    injector.add_output("Fuel", required=True)
    injector.add_output('Mass Flow Rate (kg/s)', required=True)
    injector.add_output("Oxidizer Manifold Temperature (K)")

    thermocouple.add_input("Temperature Reading")
    injector.manual_connect("Oxidizer Manifold Temperature (K)", thermocouple, "Temperature Reading") # just to check in manual still works

    injector["Chamber Pressure (psia)"] = None
    injector["mixture ratio"] = None
    injector["Fuel Temperature"] = None
    injector["Oxidizer temperature"] = None
    injector["Oxidizer "] = 'LOX'
    injector['Fuel'] = 'RP-1'
    injector['Mass Flow Rate'] = 5
    tca.connect(injector)

    print(tca)
    print(injector)


    config = {'Number of Points Contour': 400,
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
    
    guess = {'Chamber Pressure (psia)': 300,
             'Mixture Ratio': 2,
             "Fuel Temperature (K)": 298.15,
             "Oxidizer Temperature (K)": 90}
    

    tca.set_config(config)
    tca.validate_config()

    tca.set_guess(guess)
    tca.validate_guess()

    # print(tca.get_mdot())


    