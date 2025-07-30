# System.py
from Components import Component
import os
import yaml

class System:
    def __init__(self, name: str):
        self.name = name
        self.components: list[Component] = []

    def __str__(self) -> str:
        title = f" SYSTEM: {self.name} "
        box_width = len(title)
        border = "═" * box_width
        header = f"╔{border}╗\n║{title}║\n╚{border}╝"

        # Components list
        comp_lines = [f"• {comp.name}" for comp in self.components]
        comp_summary = f"{len(self.components)} components:\n" + "\n".join(comp_lines)

        # Connections list
        connections = []

        for comp in self.components:
            for out_port in comp.outflows.values():
                if out_port.connected_port:
                    tgt = out_port.connected_port
                    connections.append(f"{comp.name}.{out_port.name} → {tgt.parent.name}.{tgt.name}")
            for out_prop in comp.property_outs.values():
                if out_prop.connected_port:
                    tgt = out_prop.connected_port
                    connections.append(f"{comp.name}.{out_prop.name} ⇢ {tgt.parent.name}.{tgt.name}")

        conn_summary = "Connections:\n" + ("\n".join(connections) if connections else "(none)")

        return f"{header}\n\n{comp_summary}\n\n{conn_summary}"


    def add_component(self, root_component: Component):
        visited = set()
        queue = [root_component]

        while queue:
            comp = queue.pop()
            if comp in visited:
                continue
            visited.add(comp)

            # Enforce exclusive system membership
            if comp.system is not None and comp.system is not self:
                raise ValueError(
                    f"Component '{comp.name}' already belongs to system '{comp.system.name}', "
                    f"cannot add to system '{self.name}'."
                )

            if comp not in self.components:
                self.components.append(comp)
                comp._system = self  # claim ownership

            # Enqueue all connected components
            for port in list(comp.inflows.values()) + list(comp.outflows.values()) \
                    + list(comp.property_ins.values()) + list(comp.property_outs.values()):
                if port.connected_port:
                    queue.append(port.connected_port.parent)



    def generate_configuration_template(self, filename: str = None):
        if filename is None:
            base = f"{self.name}_Configuration.yaml"
            filename = base
            i = 1
            while os.path.exists(filename):
                filename = f"{self.name}_Configuration_{i}.yaml"
                i += 1

        data = {}

        for comp in self.components:
            if not comp.configuration_keys:
                continue

            comp_label = comp.name.replace("_", " ").title()
            data[comp_label] = {key: None for key in comp.configuration_keys}

        yaml_string = yaml.dump(
            data,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False
        )
        cleaned_yaml = yaml_string.replace(": null", ":")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Configuration Template for System: {self.name}\n")
            f.write("# Fill in values for each component’s configuration.\n\n")
            f.write(cleaned_yaml)

        print(f"[✓] Configuration template written to: {os.path.abspath(filename)}")


    def load_configuration(self, filename: str):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Configuration file not found: {filename}")

        with open(filename, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # Map component names to instances for direct lookup
        component_map = {comp.name.replace("_", " ").title(): comp for comp in self.components}

        for comp_label, config in config_data.items():
            if comp_label not in component_map:
                print(f"[!] Skipping unknown component '{comp_label}'")
                continue

            comp = component_map[comp_label]
            for key, value in config.items():
                if key not in comp.configuration_keys:
                    print(f"[!] Skipping unknown config key '{key}' in component '{comp.name}'")
                    continue
                comp.configuration[key] = value

        print(f"[✓] Configuration loaded from: {os.path.abspath(filename)}")
