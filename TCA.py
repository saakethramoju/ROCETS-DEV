from Component import Component
from typing import Optional
import numpy as np

class TCA(Component):
    def __init__(self, name: str, config: Optional[dict] = None):
    #def __init__(self, name: str):
        super().__init__(name=name)

        self.injector_data = self.add_input("Injector Data")

        self.chamber_pressure = None
        self.mixture_ratio = None
        self.mdot = None
        self.fuel = None
        self.oxidizer = None
        self.fuel_temperature = None
        self.oxidizer_temperature = None

        if config is not None:
            #self.flow_model = config['Flow Model']
            self.contour_points = config['Number of Points Contour']
            self.nozzle_type = config['Nozzle Type']
            self.throat_radius = config['Throat Radius (in)']
            self.chamber_length = config['Chamber Length (in)']
            self.contraction_ratio = config['Contraction Ratio']
            self.theta_c = config['Convergence Half-Angle (°)']
            self.rc_rt = config['Convergence Radius Factor']
            self.rtc_rt = config['Lead-in Radius Factor']
            self.rtd_rt = config['Lead-out Radius Factor']
            self.expansion_ratio = config['Expansion Ratio']

            if self.nozzle_type.lower() == 'conical':
                self.alpha = config['Divergence Half-Angle (°)']
            else:
                self.alpha = None

            if self.nozzle_type.lower() == 'bell':
                self.theta_n = config['Divergence Entrance Angle (°)']
                self.theta_e = config['Divergence Exit Angle (°)']
                self.percent_bell = config['Percent Bell (%)']
            else:
                self.theta_n = None
                self.theta_e = None
                self.percent_bell = None

            self.exit_pressure = config['Exit Pressure (psia)']


    def update(self):
        if self.injector_data.is_connected():
            self.chamber_pressure = self.injector_data["Chamber Pressure (psia)"]
            self.mdot = self.injector_data["Fuel Mass Flow Rate (kg/s)"] + self.injector_data["Oxidizer Mass Flow Rate (kg/s)"]
            self.mixture_ratio = self.injector_data["Oxidizer Mass Flow Rate (kg/s)"] / self.injector_data["Fuel Mass Flow Rate (kg/s)"]
            self.fuel = self.injector_data["Fuel"]
            self.oxidizer = self.injector_data["Oxidizer"]
            self.fuel_temperature = self.injector_data["Fuel Temperature (K)"]
            self.oxidizer_temperature = self.injector_data["Oxidizer Temperature (K)"]
        else:
            print("[Warning] Input port is not connected")
    

    def generate_geometry(self):

        Rc = self.rc_rt * self.throat_radius
        Rtc = self.rtc_rt * self.throat_radius
        theta_c = np.deg2rad(self.theta_c)
        Rtd = self.rtd_rt * self.throat_radius

        # Chamber cylinder section
        f1z = np.linspace(0, self.chamber_length, 400)
        f1r = np.ones(len(f1z)) * np.sqrt((self.contraction_ratio)) * self.throat_radius

        # Converging entrance arc
        t = np.linspace(np.pi/2, np.pi/2 - theta_c, 400)
        f2z = Rc*np.cos(t) + self.chamber_length
        f2r = Rc*np.sinc(t) + self.throat_radius*np.sqrt(self.contraction_ratio) - Rc
        inds = np.argsort(f2z)
        f2z = f2z[inds]
        f2r = f2r[inds]

        # Converging linear section


if __name__ == "__main__":
    # Instantiate components with lowercase names
    tca = TCA("Heatsink")
    injector = Component("Coax")

    #print(tca)
    #print(injector)

    injector_output = injector.add_output("TCA Inputs")
    pluh = injector.add_output("leakage")
    injector.manual_connect("TCA Inputs", tca, "Injector Data")


    print(tca)
    print(injector)

    data = {"Chamber Pressure (psia)": 400,
            "Fuel Mass Flow Rate (kg/s)": 1.5,
            "Oxidizer Mass Flow Rate (kg/s)": 3,
            "Fuel Temperature (K)": 295.15,
            "Oxidizer Temperature (K)": 90,
            "Fuel": "RP-1",
            "Oxidizer": "LOX"}
    injector_output.transmit(data)
    tca.receive("Injector Data")

    #print(tca.injector_data.value)