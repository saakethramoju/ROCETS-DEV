from Component import Component
#from Port import Port

class Engine(Component):
    def __init__(self, name: str, config: dict):
        super().__init__(name=name)

        self.flow_model = config['Flow Model']
        self.nozzle_shape = config['Nozzle Shape']
        self.throat_radius = config['Throat Radius (in)']
        self.chamber_length = config['Chamber Length (in)']
        self.contraction_ratio = config['Contraction Ratio']
        self.theta_c = config['Convergence Half-Angle (°)']
        self.rc_rt = config['Convergence Radius Factor']
        self.rtc_rt = config['Lead-in Radius Factor']
        self.rtd_rt = config['Lead-out Radius Factor']

        if self.nozzle_shape.lower() == 'conical':
            self.alpha = config['Divergence Half-Angle (°)']
        else:
            self.alpha = None

        if self.nozzle_shape.lower() == 'bell':
            self.theta_n = config['Divergence Entrance Angle (°)']
            self.theta_e = config['Divergence Exit Angle (°)']
            self.percent_bell = config['Percent Bell (%)']
        else:
            self.theta_n = None
            self.theta_e = None
            self.percent_bell = None