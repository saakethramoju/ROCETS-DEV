from Component import Component
from typing import Optional
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from typing import Optional
import numpy as np
from scipy.interpolate import interp1d
from Component import Component

class TCA(Component):
    def __init__(self, name: str, config: Optional[dict] = None):
        super().__init__(name=name)
        self.injector_data = self.add_input("Injector Data")
        self.configuration = config
        self._initialize_defaults()

        if config:
            self.set_config(config)

    def _initialize_defaults(self):
        self.chamber_pressure = None
        self.mixture_ratio = None
        self.mdot = None
        self.fuel = None
        self.oxidizer = None
        self.fuel_temperature = None
        self.oxidizer_temperature = None

    def set_config(self, config: dict):
        self.configuration = config
        get = config.get  # shortcut

        self.contour_points   = get('Number of Points Contour')
        self.nozzle_type      = get('Nozzle Type', '').lower()
        self.throat_radius    = get('Throat Radius (in)')
        self.chamber_length   = get('Chamber Length (in)')
        self.contraction_ratio= get('Contraction Ratio')
        self.theta_c          = get('Convergence Half-Angle (°)')
        self.rc_rt            = get('Convergence Radius Factor')
        self.rtc_rt           = get('Lead-in Radius Factor')
        self.rtd_rt           = get('Lead-out Radius Factor')
        self.expansion_ratio  = get('Expansion Ratio')
        self.exit_pressure    = get('Exit Pressure (psia)')

        # Optional parameters
        self.alpha            = get('Divergence Half-Angle (°)') if self.nozzle_type == 'conical' else None
        self.theta_n          = get('Divergence Entrance Angle (°)') if self.nozzle_type == 'bell' else None
        self.theta_e          = get('Divergence Exit Angle (°)') if self.nozzle_type == 'bell' else None
        self.percent_bell     = get('Percent Bell (%)') if self.nozzle_type == 'bell' else None

    def generate_chamber_geometry(self, points: int = 100):
        R_t = self.throat_radius
        eps_c = self.contraction_ratio
        Lc = self.chamber_length

        z = np.linspace(0, Lc, points)
        r = np.ones_like(z) * np.sqrt(eps_c) * R_t
        return self.resample_curve(np.vstack((z, r)), self.contour_points)



    def generate_converging_geometry(self, points: int = 100):
        R_t = self.throat_radius
        Rc, Rtc = self.rc_rt * R_t, self.rtc_rt * R_t
        theta_c = np.deg2rad(self.theta_c)
        eps_c = self.contraction_ratio
        Lc = self.chamber_length

        # Entrance arc
        t1 = np.linspace(np.pi/2, np.pi/2 - theta_c, points)
        z1 = Rc * np.cos(t1) + Lc
        r1 = Rc * np.sin(t1) + R_t * np.sqrt(eps_c) - Rc
        z1, r1 = z1[np.argsort(z1)], r1[np.argsort(z1)]

        # Linear section
        y3 = R_t * (np.sqrt(eps_c) - 1) - (Rc - Rc * np.cos(theta_c)) - (Rtc - Rtc * np.cos(theta_c))
        x3 = y3 / np.tan(theta_c)
        z2 = np.linspace(z1[-1], z1[-1] + x3, points)
        r2 = -np.tan(theta_c) * (z2 - z1[-1]) + r1[-1]

        # Throat arc
        t2 = np.linspace(np.pi + np.pi/2 - theta_c, 3 * np.pi / 2, points)
        h = z2[-1] + Rtc * np.sin(theta_c)
        k = R_t + Rtc
        z3 = Rtc * np.cos(t2) + h
        r3 = Rtc * np.sin(t2) + k
        z3, r3 = z3[np.argsort(z3)], r3[np.argsort(z3)]

        z = np.hstack((z1, z2, z3))
        r = np.hstack((r1, r2, r3))
        return self.resample_curve(np.vstack((z, r)), self.contour_points)


    def generate_nozzle_geometry(self, points: int = 100):
        R_t = self.throat_radius
        Rtd = self.rtd_rt * R_t
        alpha = np.deg2rad(self.alpha) if self.alpha else 0
        theta_n = np.deg2rad(self.theta_n) if self.theta_n else 0
        theta_e = np.deg2rad(self.theta_e) if self.theta_e else 0
        eps = self.expansion_ratio
        nozzle_type = self.nozzle_type.lower()
        percent_bell = self.percent_bell or 80

        # Get throat exit point from converging geometry
        throat_geom = self.generate_converging_geometry(points)
        z_throat = throat_geom[0, -1]
        r_throat = throat_geom[1, -1]

        # Entrance arc
        angle = alpha if nozzle_type == 'conical' else theta_n
        t1 = np.linspace(3 * np.pi / 2, 3 * np.pi / 2 + angle, points)
        k = R_t + Rtd
        z1 = Rtd * np.cos(t1) + z_throat
        r1 = Rtd * np.sin(t1) + k
        z1, r1 = z1[np.argsort(z1)], r1[np.argsort(z1)]

        # Exit section
        if nozzle_type == 'conical':
            z2 = np.linspace(z1[-1], z1[-1] + (np.sqrt(eps) * R_t - r1[-1]) / np.tan(alpha), points)
            r2 = np.tan(alpha) * (z2 - z1[-1]) + r1[-1]
        else:
            N = [z1[-1], r1[-1]]
            conical_length = (np.sqrt(eps) * R_t - r1[-1]) / np.tan(np.deg2rad(15))
            E = [(percent_bell / 100) * conical_length + throat_geom[0, 0], R_t * np.sqrt(eps)]
            m1, m2 = np.tan(theta_n), np.tan(theta_e)
            Qx = (m1 * N[0] - N[1] - m2 * E[0] + E[1]) / (m1 - m2)
            Qy = m1 * (Qx - N[0]) + N[1]
            Q = [Qx, Qy]
            t = np.linspace(0, 1, points)
            z2 = ((1 - t)**2) * N[0] + 2 * (1 - t) * t * Q[0] + (t**2) * E[0]
            r2 = ((1 - t)**2) * N[1] + 2 * (1 - t) * t * Q[1] + (t**2) * E[1]

        z = np.hstack((z1, z2))
        r = np.hstack((r1, r2))
        return self.resample_curve(np.vstack((z, r)), self.contour_points)

    

    def generate_geometry(self, points: int = 100):
        if not self.configuration:
            raise ValueError("Cannot generate geometry: no configuration has been given")

        chamber = self.generate_chamber_geometry(points)
        converging = self.generate_converging_geometry(points)
        nozzle = self.generate_nozzle_geometry(points)

        self.chamber_geometry = chamber
        self.converging_geometry = converging
        self.nozzle_geometry = nozzle

        full_geometry = np.hstack((chamber, converging, nozzle))
        return self.resample_curve(full_geometry, self.contour_points)


    def resample_curve(self, curve, n):
        curve = curve.T
        deltas = np.diff(curve, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        arc_length = np.concatenate([[0], np.cumsum(segment_lengths)])

        fx = interp1d(arc_length, curve[:, 0], kind='linear')
        fy = interp1d(arc_length, curve[:, 1], kind='linear')
        new_s = np.linspace(0, arc_length[-1], n)
        new_x = fx(new_s)
        new_y = fy(new_s)
        return np.vstack((new_x, new_y))

    
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
              "Exit Pressure (psia)": 10}
    
    tca.set_config(config)
    #geometry = tca.generate_geometry()
    #plt.plot(geometry[0, :], geometry[1, :])
    #plt.show()
    #print(geometry.shape)