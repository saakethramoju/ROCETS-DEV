from typing import Optional
from Component import Component

class Injector(Component):
    required_keys = [
        "Oxidizer Stiffness",
        "Fuel Stiffness"
    ]

    def __init__(self, name: str, config: Optional[dict] = None, guess: Optional[dict] = None):
        super().__init__(name, config=config, guess=guess)

        self._initialize_default_ports()

        if config:
            self.set_config(config)

        self.assign_guess_variables(priority=[

        ])


    # ─────────────────────────────────────────────────────────────────────────────
    # PORT SETUP / CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────────────

    def _initialize_default_ports(self):
        """Define all input ports expected for the TCA model."""

        # INPUTS
        #self.add_input()

        # OUTPUTS:
        self.add_output("Chamber Pressure (psia)", required=True)
        self.add_output("Mixture Ratio", required=True)
        self.add_output("Fuel Temperature (K)", required=True)
        self.add_output("Oxidizer Temperature (K)", required=True)
        self.add_output("Fuel", required=True)
        self.add_output("Oxidizer", required=True)
        self.add_output("Injector Mass Flow Rate (kg/s)", required=True)


    def residuals(self):
        """
        For now, Injector provides no residuals.
        Can be expanded to include pressure drop or mass flow constraints.
        """
        return []

    def on_steady_state_solve(self, solution):
        """
        Optional hook after solver success. Can be used to compute diagnostics.
        """
        pass


if __name__ == "__main__":

    from TCA import TCA

    injector = Injector("Coax")
    tca = TCA("Heatsink")

    injector.connect(tca)

    print(injector.get_iteration_variables())