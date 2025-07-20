import yaml
import os
from scipy.optimize import root
from Exceptions import SteadyStateSolveError, GuessResidualMismatchError
import numpy as np


class System:
    def __init__(self, name: str):
        self.name = name
        self.components = []
        self.connections = []  # (from_component, from_port, to_component, to_port)

    def add_component(self, component):
        """Add a component and recursively include all connected components."""
        to_visit = [component]
        seen = set()

        while to_visit:
            current = to_visit.pop()
            if current in seen:
                continue
            seen.add(current)
            self.components.append(current)

            # Traverse all ports for this component
            for port in list(current.inputs.values()) + list(current.outputs.values()):
                for connected_port in port.connected_ports:
                    other_component = connected_port.component
                    if other_component not in seen and other_component not in self.components:
                        to_visit.append(other_component)


    def connect(self, from_component, from_port, to_component, to_port):
        self.connections.append((from_component, from_port, to_component, to_port))
        from_component.manual_connect(from_port, to_component, to_port)

    def get_residuals(self):
        residuals = []
        seen_nodes = set()
        for component in self.components:
            component.validate_all(seen_nodes=seen_nodes)
            residuals.extend(component.residuals())
        return residuals


    def get_guess_vector(self):
        seen = set()
        guess = []

        for component in self.components:
            for port in {**component.inputs, **component.outputs}.values():
                shared = port._value
                if shared in seen:
                    continue
                seen.add(shared)
                if shared.iteration_variable and shared.guess_variable:
                    guess.append(shared.value)

        return np.array(guess)


    def set_guess_vector(self, x):
        seen = set()
        i = 0

        for component in self.components:
            for port in {**component.inputs, **component.outputs}.values():
                shared = port._value
                if shared in seen:
                    continue
                seen.add(shared)
                if shared.iteration_variable and shared.guess_variable:
                    shared.broadcast(x[i])
                    i += 1


    def residual_function(self, x):
        self.set_guess_vector(x)
        return self.get_residuals()
        

    def collect_config_keys(self) -> dict:
        """
        Collect all configuration keys (required + optional) for each component.
        The output is a dictionary like:
        {"Heatsink [TCA]": {"Nozzle Type": None, ...}, ...}
        """
        config_map = {}

        for component in self.components:
            typename = type(component).__name__
            label = f"{component.name} [{typename}]"

            # Grab required_keys and all_config_keys if they exist
            required_keys = getattr(component.__class__, "required_keys", [])
            all_keys = getattr(component.__class__, "all_config_keys", required_keys)

            config_map[label] = {k: None for k in all_keys}

        return config_map
    

    def write_configuration_template(self, file_name: str = None):
        """
        Collect configuration keys from all components in the system and write them to a YAML template file.
        If a file with the same name already exists, a unique numbered file name is generated.
        """
        config_dict = self.collect_config_keys()

        if file_name is None:
            file_name = f"{self.name} Configuration.yaml"

        # Generate a unique file name if it already exists
        base, ext = os.path.splitext(file_name)
        counter = 1
        while os.path.exists(file_name):
            file_name = f"{base}_{counter}{ext}"
            counter += 1

        # Dump YAML as string to edit before writing
        yaml_string = yaml.dump(
            config_dict,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False
        )

        # Replace ": null" with just ":"
        cleaned_yaml = yaml_string.replace(": null", ":")

        # Ensure clear spacing between component blocks
        cleaned_yaml = "\n\n".join(block.strip() for block in cleaned_yaml.split("\n\n"))

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(cleaned_yaml)

        print(f"[✓] Configuration template written to: {os.path.abspath(file_name)}")

            
    def read_configuration(self, file_path: str) -> dict:
        """
        Load a YAML file containing config values and assign them to the appropriate components.
        Returns a dictionary of component names to their config values.
        """
        import yaml

        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        for component in self.components:
            typename = type(component).__name__
            label = f"{component.name} [{typename}]"
            component_config = config_data.get(label, {})

            # Convert blank strings or None to None explicitly
            parsed_config = {
                k: (None if v in ("", None) else v) for k, v in component_config.items()
            }

            component.set_config(parsed_config)

        return config_data
        
    def collect_guess_keys(self) -> dict:
        """
        Collect unique guess variable keys based on shared nodes, not per-component.
        Output format:
        {
            "Heatsink [TCA]": {"Chamber Pressure (psia)": None, ...},
            ...
        }
        The component that owns the first port in a shared group is responsible for writing the guess.
        """
        seen_nodes = set()
        guess_map = {}

        for component in self.components:
            typename = type(component).__name__
            label = f"{component.name} [{typename}]"

            for port in {**component.inputs, **component.outputs}.values():
                shared = port._value
                if shared in seen_nodes:
                    continue
                seen_nodes.add(shared)

                if shared.iteration_variable and shared.guess_variable:
                    if label not in guess_map:
                        guess_map[label] = {}
                    guess_map[label][port.name] = None  # Use the port name from this component's perspective

        return guess_map

    def write_guess_template(self, file_name: str = None):
        """
        Collect all guess variable keys from components and write them to a YAML file.
        If the file already exists, a unique filename is generated to avoid overwriting.
        """
        guess_dict = self.collect_guess_keys()

        if file_name is None:
            file_name = f"{self.name} Guess.yaml"

        # Generate a unique file name if it already exists
        base, ext = os.path.splitext(file_name)
        counter = 1
        while os.path.exists(file_name):
            file_name = f"{base}_{counter}{ext}"
            counter += 1

        # Dump YAML as string to edit before writing
        yaml_string = yaml.dump(
            guess_dict,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False
        )

        # Replace ": null" with just ":"
        cleaned_yaml = yaml_string.replace(": null", ":")
        cleaned_yaml = "\n\n".join(block.strip() for block in cleaned_yaml.split("\n\n"))

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(cleaned_yaml)

        print(f"[✓] Guess template written to: {os.path.abspath(file_name)}")


    def read_guess(self, filepath: str) -> dict:
        """Read guess YAML and assign to appropriate components."""
        import yaml
        with open(filepath, 'r') as f:
            raw_data = yaml.safe_load(f)

        # Normalize component names from YAML
        component_map = {
            self._normalize(f"{c.name} [{c.__class__.__name__}]"): c
            for c in self.components
        }

        for yaml_name, guess_block in raw_data.items():
            norm_name = self._normalize(yaml_name)
            component = component_map.get(norm_name)
            if component is None:
                print(f"[Warning] No matching component found for guess block: '{yaml_name}'")
                continue

            # Assign guess to component
            print(f"[Guess Assignment] Setting guess for {component.name} → {guess_block}")
            component.set_guess(guess_block)

        return raw_data


    

    def solve(self):
        from scipy.optimize import root
        from Exceptions import SteadyStateSolveError, GuessResidualMismatchError

        # Get all unique guess nodes
        guess_nodes = []
        seen = set()
        for c in self.components:
            for port in list(c.inputs.values()) + list(c.outputs.values()):
                node = port._value
                if node.guess_variable and node.iteration_variable and node not in seen:
                    seen.add(node)
                    guess_nodes.append(node)

        x0 = [n.value for n in guess_nodes]
        r0 = self.get_residuals()

        if len(x0) != len(r0):
            raise GuessResidualMismatchError(len(x0), len(r0))

        solution = root(self.residual_function, x0)
        if solution.success:
            # Apply solution to shared nodes
            for i, node in enumerate(guess_nodes):
                node.broadcast(solution.x[i])
            for c in self.components:
                c.on_steady_state_solve(solution)
        else:
            raise SteadyStateSolveError(
                component_name=self.name,
                message=solution.message,
                guess_vars=[n.name for n in guess_nodes]
            )


        if len(x0) != len(r0):
            raise GuessResidualMismatchError(len(x0), len(r0))

        solution = root(self.residual_function, x0)
        if solution.success:
            self.set_guess_vector(solution.x)
            for c in self.components:
                c.on_steady_state_solve(solution)
        else:
            raise SteadyStateSolveError(
                component_name=self.name,
                message=solution.message,
                guess_vars=[g for c in self.components for g in c.get_guess_variables()]
            )
        return solution

        
    def _normalize(self, name: str) -> str:
        """
        Normalize a component or port name for consistent lookup:
        - Remove parenthetical units
        - Convert to lowercase
        - Strip leading/trailing whitespace
        """
        import re
        return re.sub(r"\(.*?\)", "", name).strip().lower()



    def __repr__(self):
        return f"System: {self.name} → {len(self.components)} Components"


if __name__ == "__main__":

    from TCA import TCA
    from Component import Component


    EngineSystem = System("Vespula")
    tca = TCA("Heatsink")
    injector = Component("Coax")
    injector.add_output("Chamber Pressure (psia)", required=True)
    injector.add_output("Mixture Ratio", required=True)
    injector.add_output("Fuel Temperature (K)", required=True)
    injector.add_output("Oxidizer Temperature (K)", required=True)
    injector.add_output("Oxidizer", required=True)
    injector.add_output("Fuel", required=True)
    injector.add_output('Injector Mass Flow Rate (kg/s)', required=True)
    tca.connect(injector)

    #injector["Chamber Pressure (psia)"] = 400
    injector["mixture ratio"] = 2.3
    injector["Fuel Temperature"] = 298.15
    injector["Oxidizer temperature"] = 90
    injector["Oxidizer "] = 'LOX'
    injector['Fuel'] = 'RP-1'
    injector['injector Mass Flow Rate'] = 5

    EngineSystem.add_component(tca)
    #EngineSystem.add_component()
    print(EngineSystem)

    #tca.set_guess_variables(["mixture ratio"])
    #EngineSystem.write_configuration_template()
    #EngineSystem.write_guess_template()
    x = EngineSystem.read_configuration("/Users/saakethramramoju/Desktop/ROCETS DEV/Vespula Configuration.yaml")
    y = EngineSystem.read_guess("/Users/saakethramramoju/Desktop/ROCETS DEV/Vespula Guess.yaml")
    #print(type(y["Heatsink [TCA]"]['Mixture Ratio']))

    
    EngineSystem.solve()