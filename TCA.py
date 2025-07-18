from typing import Optional
import numpy as np
from scipy.interpolate import interp1d
from Component import Component
from Injector import Injector
from Constants import Constants as cs
from rocketcea.cea_obj_w_units import CEA_Obj
from Exceptions import (MissingConfigurationError, PortNotConnectedError, MissingConfigurationKeyError,
                        MissingConfigurationValueError, MissingGuessError, MissingGuessKeyError,
                        MissingGuessValueError)
#import matplotlib.pyplot as plt

class TCA(Component):
    def __init__(self, name: str, config: Optional[dict] = None, guess: Optional[dict] = None):
        super().__init__(name)
        self.configuration = config
        self.guess = guess
        self._initialize_defaults()
        self._initialize_necessary_ports()
        self._initialize_extra_ports()

        if config:
            self.set_config(config)

    #def on_connect(self):
    #    print("test")

    def _initialize_defaults(self):
        self.fuel = self.oxidizer = None
        self.steady_state_converged = True
        self._simlation_start = False
        #self.steady_state_converged = True
        #self.steady_state_solved = False
        #self.transient_solved = False

        self.chamber_pressure = self.mixture_ratio = None
        self.fuel_temperature = self.oxidizer_temperature = None

    def _initialize_necessary_ports(self):
        # Inputs
        self.injector_data = self.add_input("Injector Data")

        # Outputs
        # EMPTY

    def _initialize_extra_ports(self):
        # Inputs
        # EMPTY

        # Outputs
        # EMPTY
        pass

    def set_injector_data(self):
        self.check_injector_connection()

        self.fuel = self.injector_data.connected_ports[0].component.fuel
        self.oxidizer = self.injector_data.connected_ports[0].component.fuel

        return True


    def set_guess(self, guess: dict):
        self.guess = guess
        get = guess.get

        self.chamber_pressure     = get("Chamber Pressure (psia)")
        self.mixture_ratio        = get("Mixture Ratio")
        self.fuel_temperature     = get("Fuel Temperature (K)")
        self.oxidizer_temperature = get("Oxidizer Temperature (K)")

        self._simlation_start = True

        return True
    

    def validate_guess(self):
        if not self.guess:
            raise MissingGuessError(f"Initial guesses not provided for {self.name}")
        
        guess = self.guess

        required_keys = [
            'Chamber Pressure (psia)',
            'Mixture Ratio',
            'Fuel Temperature (K)',
            'Oxidizer Temperature (K)'
        ]
        for key in required_keys:
            if key not in guess:
                raise MissingGuessKeyError(f"Missing required initial guess key: '{key}'")
            if guess[key] is None:
                raise MissingGuessValueError(f"Initial guess value for '{key}' cannot be None")
            
        return True
            


    def set_config(self, config: dict):
        self.configuration = config
        get = config.get
        nozzle_type = get("Nozzle Type", "").lower()
        combustor_area = get("Combustor Area", "").lower()

        def opt(key, condition=True):
            return get(key) if condition else None

        self.contour_points    = get("Number of Points Contour")
        self.nozzle_type       = nozzle_type
        self.throat_radius     = get("Throat Radius (in)")
        self.chamber_length    = get("Chamber Length (in)")
        self.contraction_ratio = get("Contraction Ratio")
        self.theta_c           = get("Convergence Half-Angle (°)")
        self.rc_rt             = get("Convergence Radius Factor")
        self.rtc_rt            = get("Lead-in Radius Factor")
        self.rtd_rt            = get("Lead-out Radius Factor")
        self.expansion_ratio   = get("Expansion Ratio")
        self.exit_pressure     = get("Exit Pressure (psia)")
        self.ambient_pressure  = get("Ambient Pressure (psia)")
        self.alpha             = opt("Divergence Half-Angle (°)", nozzle_type == "conical")
        self.theta_n           = opt("Divergence Entrance Angle (°)", nozzle_type == "bell")
        self.theta_e           = opt("Divergence Exit Angle (°)", nozzle_type == "bell")
        self.percent_bell      = opt("Percent Bell (%)", nozzle_type == "bell")
        self.combustor_area    = combustor_area


    def validate_config(self):
        if not self.configuration:
            raise MissingConfigurationError(f"No configuration provided for {self.name}")
        config = self.configuration

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

        # Check unconditional keys
        for key in required_keys:
            if key not in config:
                raise MissingConfigurationKeyError(f"Missing required configuration key: '{key}'")
            if config[key] is None:
                raise MissingConfigurationValueError(f"Configuration value for '{key}' cannot be None")

        nozzle_type = config.get("Nozzle Type", "").lower()

        # Conditional keys based on nozzle type
        if nozzle_type == "conical":
            if "Divergence Half-Angle (°)" not in config or config["Divergence Half-Angle (°)"] is None:
                raise MissingConfigurationKeyError("Missing or None value for required conical nozzle key: 'Divergence Half-Angle (°)'")
        elif nozzle_type == "bell":
            for key in ["Divergence Entrance Angle (°)", "Divergence Exit Angle (°)", "Percent Bell (%)"]:
                if key not in config:
                    raise MissingConfigurationKeyError(f"Missing required bell nozzle configuration key: '{key}'")
                if config[key] is None:
                    raise MissingConfigurationValueError(f"Configuration value for '{key}' cannot be None")
                
        return True


    def check_injector_connection(self):
        if not self.injector_data.is_connected():
            raise PortNotConnectedError(f"Injector input port must be connected: {self.injector_data.name} in {self.name}")
        return True
    

    def generate_cea(self):
        if self.combustor_area.lower() == 'finite':
            cea = CEA_Obj(oxName=self.oxidizer, fuelName=self.fuel, temperature_units='degK', 
                 cstar_units='m/sec', specific_heat_units='kJ/kg degK', 
                 sonic_velocity_units='m/s', enthalpy_units='J/kg', 
                 density_units='kg/m^3', fac_CR=self.contraction_ratio)
            
            self.chamber_pressure_rayleigh = self.chamber_pressure * (1 / cea.get_Pinj_over_Pcomb(self.chamber_pressure, self.mixture_ratio, self.contraction_ratio))
        
        else:
            cea = CEA_Obj(oxName=self.oxidizer, fuelName=self.fuel, temperature_units='degK', 
                 cstar_units='m/sec', specific_heat_units='kJ/kg degK', 
                 sonic_velocity_units='m/s', enthalpy_units='J/kg', 
                 density_units='kg/m^3')
            self.chamber_pressure_rayleigh = self.chamber_pressure

        return cea
    

    def get_mdot(self):
        At = self.throat_area()
        cea = self.generate_cea()
        Tt = cea.get_Temperatures(self.chamber_pressure_rayleigh, self.mixture_ratio, self.expansion_ratio)
        mwt, gammat = cea.get_Throat_MolWt_gamma(self.chamber_pressure_rayleigh, self.mixture_ratio, self.expansion_ratio)
        mdot = (At * self.chamber_pressure_rayleigh * 6894.76) * np.sqrt(mwt * gammat / (cs.R * Tt))
        return mdot


    '''
    def set_engine_parameters(self):
        self.check_injector_connection()
        self.chamber_pressure = self.injector_data.value["Chamber Pressure (psia)"]
        self.fuel = self.injector_data.value["Fuel"]
        self.oxidizer = self.injector_data.value["Oxidizer"]
        self.mdot = self.injector_data.value["Fuel Mass Flow Rate (kg/s)"] + self.injector_data.value["Oxidizer Mass Flow Rate (kg/s)"]
        self.mixture_ratio = self.injector_data.value["Oxidizer Mass Flow Rate (kg/s)"] / self.injector_data.value["Fuel Mass Flow Rate (kg/s)"]
        self.oxidizer_temperature = self.injector_data.value["Oxidizer Temperature (K)"]
        self.fuel_temperature = self.injector_data.value["Fuel Temperature (K)"]

    
    def ODE(self):
        self.set_engine_parameters()
        self.validate_config()

        if self.combustor_area.lower() == 'finite':
            ode = CEA_Obj(oxName=self.oxidizer, fuelName=self.fuel, temperature_units='degK', 
                 cstar_units='m/sec', specific_heat_units='kJ/kg degK', 
                 sonic_velocity_units='m/s', enthalpy_units='J/kg', 
                 density_units='kg/m^3', fac_CR=self.contraction_ratio)
            self.chamber_pressure_rayleigh = self.chamber_pressure * (1 / ode.get_Pinj_over_Pcomb(self.chamber_pressure, self.mixture_ratio, self.contraction_ratio))
        
        else:
            ode = CEA_Obj(oxName=self.oxidizer, fuelName=self.fuel, temperature_units='degK', 
                 cstar_units='m/sec', specific_heat_units='kJ/kg degK', 
                 sonic_velocity_units='m/s', enthalpy_units='J/kg', 
                 density_units='kg/m^3')
            self.chamber_pressure_rayleigh = None
        
        # Thrust coefficient
        self.thrust_coefficient_ideal, _, _ = ode.get_PambCf(self.ambient_pressure, self.chamber_pressure, self.mixture_ratio, self.expansion_ratio)
        # Characteristic velocity
        self.cstar_ideal = ode.get_Cstar(self.chamber_pressure, self.mixture_ratio)
        # Specific impulse
        self.isp_ideal, _ = ode.estimate_Ambient_Isp(self.chamber_pressure, self.mixture_ratio, self.expansion_ratio, self.ambient_pressure)
        self.isp_vaccuum_ideal = ode.get_Isp(self.chamber_pressure, self.mixture_ratio, self.expansion_ratio)
        self.thrust_coefficient_vaccum_ideal = self.isp_vaccuum_ideal * cs.g / self.cstar_ideal'''


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

    injector.add_output("TCA Inputs")
    injector.manual_connect("TCA Inputs", tca, "Injector Data")
    injector.fuel = 'RP-1'
    injector.oxidizer = 'LOX'

    #print(tca)


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
    
    guess = {'Chamber Pressure (psia)': 400,
             'Mixture Ratio': 2,
             "Fuel Temperature (K)": 298.15,
             "Oxidizer Temperature (K)": 90}
    
    tca.set_guess(guess)
    tca.set_config(config)
    tca.validate_guess()

    #print(tca.injector_data.connected_ports[0].component.fuel)

    #tca.get_mdot()



    