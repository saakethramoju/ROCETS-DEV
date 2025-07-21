import os
import yaml
from Core import InputPort, OutputPort
from prettytable import PrettyTable

class System:
    def __init__(self, name):
        self.name = name
        self.components = []

    def add_component(self, component):
        visited = set()

        def _recursive_add(comp):
            if comp in visited:
                return
            visited.add(comp)
            if comp.system is not self:
                comp.system = self
                self.components.append(comp)
            for port in comp.ports:
                if port.node:
                    for connected_port in port.node.ports:
                        if connected_port.owner is not comp:
                            _recursive_add(connected_port.owner)

        _recursive_add(component)


    def generate_configuration_template(self, file_name: str = None):
        """
        Generate a YAML configuration template for all components in the system.
        Lists required and optional keys for each component as separate YAML sections.
        """
        config_dict = {}

        for component in self.components:
            component_dict = {}

            for key in getattr(component, "required_config_keys", []):
                component_dict[key] = None

            for key in getattr(component, "optional_config_keys", []):
                component_dict[key] = None

            config_dict[component.name] = component_dict

        if file_name is None:
            file_name = f"{self.name} Configuration.yaml"

        # Ensure file name is unique if it exists
        base, ext = os.path.splitext(file_name)
        counter = 1
        while os.path.exists(file_name):
            file_name = f"{base}_{counter}{ext}"
            counter += 1

        # Dump YAML to string for cleanup
        yaml_string = yaml.dump(
            config_dict,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False
        )

        # Replace ": null" with just ":" for cleaner appearance
        cleaned_yaml = yaml_string.replace(": null", ":")

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(f"# Configuration Template for System: {self.name}\n")
            f.write("# Fill in required values. Optional values may be left blank.\n\n")
            f.write(cleaned_yaml)

        print(f"[✓] Configuration template written to: {os.path.abspath(file_name)}")


    def read_configuration(self, file_path: str):
        """
        Read a YAML file and apply configuration to all matching components in the system.
        
        Each top-level key should match a component's name.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Configuration file '{file_path}' not found.")

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                config_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML file '{file_path}': {e}")

        if not isinstance(config_data, dict):
            raise ValueError("Top-level YAML structure must be a dictionary of components.")

        unmatched = []

        for comp_name, config in config_data.items():
            match = next((c for c in self.components if c.name == comp_name), None)
            if match:
                match.set_config(config)
                print(f"[✓] Configuration applied to component: {comp_name}")
            else:
                unmatched.append(comp_name)

        if unmatched:
            print(f"[!] Warning: Could not find component(s) in system: {', '.join(unmatched)}")


    def print_nodes(self):
        """Pretty print all nodes in the system with their connected component and port counts."""
        all_nodes = set()

        for component in self.components:
            for port in component.ports:
                if port.node:
                    all_nodes.add(port.node)

        table = PrettyTable()
        table.title = f"Nodes in System: {self.name}"
        table.field_names = ["Node Name", "# Components", "# Ports"]
        table.align["Node Name"] = "l"

        for node in sorted(all_nodes, key=lambda n: n.name):
            components = {p.owner for p in node.ports}
            table.add_row([node.name, len(components), len(node.ports)])

        print(table)


    def __str__(self):
        from prettytable import PrettyTable
        table = PrettyTable()
        table.title = f"System: {self.name}"
        table.field_names = ["Component", "# Inputs", "# Outputs"]
        table.align["Component"] = "l"

        for comp in self.components:
            inputs = sum(isinstance(p, InputPort) for p in comp.ports)
            outputs = sum(isinstance(p, OutputPort) for p in comp.ports)
            table.add_row([comp.name, inputs, outputs])

        return str(table)




if __name__ == "__main__":

    from TCA import TCA
    from Core import Component

    vespula = System("Vespula")
    tca = TCA("Heatsink")
    injector = Component("Coax")
    #sensor = Component("PT")
    #sensor.add_input("chamber pressure")
    injector.add_output("Chamber Pressure")
    injector.add_output("Mixture Ratio")
    injector.add_output("Fuel Temperature (K)")
    injector.add_output("Oxidizer Temperature")
    injector.add_output("Oxidizer")
    injector.add_output("Fuel")

    vespula.add_component(tca)
    tca.connect(injector)
    print(vespula)

    #vespula.generate_configuration_template()
    vespula.read_configuration("/Users/saakethramramoju/Desktop/ROCETS DEV/Vespula Configuration.yaml")
    tca.print_config_summary()
    #injector.print_config_summary()

    vespula.print_nodes()
    injector["CHamber pressure"] = 300
    tca["mixture ratio"] = 2
    injector["fuel"] = 'RP-1'
    injector["oxidizer"] = 'LOX'
    print(tca)
    print(tca.mass_conservation_equation())