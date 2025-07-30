# System.py
import os
import yaml
from collections import defaultdict
from Components import Component
from Fluids import Fluid


class System:
    def __init__(self, name: str):
        self.name = name
        self.components: list[Component] = []

    def evaluate(self, verbose: bool = False):
        for comp in self.components:
            if verbose:
                print(f"Evaluating: {comp.name}")
            comp.evaluate()


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

    def generate_input_template(system, filename: str = None):
        import os

        friendly_names = {
            "fluid_name": "Fluid",
            "mass_flow": "Mass Flow (kg/s)",
            "T": "Temperature (K)",
            "P": "Pressure (Pa)",
            "X": "Vapor Quality (0–1)",
        }

        if filename is None:
            base = f"{system.name}_Inputs.yaml"
            filename = base
            i = 1
            while os.path.exists(filename):
                filename = f"{system.name}_Inputs_{i}.yaml"
                i += 1

        inlet_boundary_types = {"MassFlowInlet", "FluidStateInlet", "Inlet"}
        outlet_boundary_types = {"MassFlowOutlet", "FluidStateOutlet", "Outlet"}

        lines = []
        lines.append(f"# Node Input Template for System: {system.name}")
        lines.append("# Provide boundary condition inputs and initial guesses for internal nodes.\n")

        comp_blocks = {}
        visited_nodes = set()

        for comp in system.components:
            for port in comp.ports().values():
                node = port.node
                if node and node not in visited_nodes:
                    visited_nodes.add(node)
                    connected_ports = node._ports
                    boundary = False
                    input_vars = []
                    owner_label = None

                    for p in connected_ports:
                        parent_class = p.parent.__class__.__name__ if p.parent else ""
                        if parent_class in inlet_boundary_types:
                            boundary = True
                            input_vars = ["fluid_name"] + ["T", "P"] if "FluidState" in parent_class else ["fluid_name", "mass_flow"]
                            owner_label = p.parent.name.replace("_", " ").title()
                            break
                        elif parent_class in outlet_boundary_types:
                            boundary = True
                            input_vars = ["T", "P"] if "FluidState" in parent_class else ["mass_flow"]
                            owner_label = p.parent.name.replace("_", " ").title()
                            break
                        elif p.connected_port is None:
                            boundary = True
                            if p.__class__.__name__ == "InFlow":
                                input_vars = ["fluid_name", "T", "P"]
                            else:
                                input_vars = ["T", "P"]
                            owner_label = p.parent.name.replace("_", " ").title()
                            break

                    if not boundary:
                        input_vars = ["T", "P"]
                        outflow = next((p for p in connected_ports if p.__class__.__name__ == "OutFlow"), connected_ports[0])
                        owner_label = outflow.parent.name.replace("_", " ").title()

                    node_label = node.name.replace("_", " ").title()
                    if owner_label not in comp_blocks:
                        comp_blocks[owner_label] = {"has_input": False, "nodes": []}
                    if boundary:
                        comp_blocks[owner_label]["has_input"] = True
                    comp_blocks[owner_label]["nodes"].append((node_label, input_vars, boundary))

        for comp in system.components:
            for port in comp.ports().values():
                if port.node is not None or port.connected_port is not None:
                    continue
                pseudo_label = f"{comp.name.replace('_', ' ').title()}:{port.name.replace('_', ' ').title()}"
                owner_label = comp.name.replace("_", " ").title()
                if port.__class__.__name__ == "InFlow":
                    input_vars = ["fluid_name", "T", "P"]
                else:
                    input_vars = ["T", "P"]

                if owner_label not in comp_blocks:
                    comp_blocks[owner_label] = {"has_input": False, "nodes": []}
                comp_blocks[owner_label]["has_input"] = True
                comp_blocks[owner_label]["nodes"].append((pseudo_label, input_vars, True))

        for comp_label, info in reversed(list(comp_blocks.items())):
            suffix = "  # Input" if info["has_input"] else ""
            lines.append(f"{comp_label}:{suffix}")
            for node_label, var_list, _ in info["nodes"]:
                lines.append(f"  {node_label}:")
                for var in var_list:
                    label = friendly_names.get(var, var)
                    lines.append(f"    {label}:")
            lines.append("")

        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"[✓] Node input template written to: {os.path.abspath(filename)}")


    def load_inputs(self, filename: str):

        with open(filename, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            print(f"[!] No data found in {filename}.")
            return

        label_to_var = {
            "Fluid": "fluid_name",
            "Temperature (K)": "T",
            "Pressure (Pa)": "P",
            "Mass Flow (kg/s)": "mass_flow",
            "Vapor Quality (0–1)": "X",
        }

        node_lookup = {
            node.name.replace("_", " ").title(): node
            for comp in self.components
            for port in comp.ports().values()
            if port.node
            for node in [port.node]
        }

        pseudo_lookup = {
            f"{comp.name.replace('_', ' ').title()}:{port.name.replace('_', ' ').title()}": port
            for comp in self.components
            for port in comp.ports().values()
            if port.node is None and port.connected_port is None
        }

        for comp_label, nodes in raw.items():
            if not isinstance(nodes, dict):
                continue
            for node_label, entries in nodes.items():
                if not isinstance(entries, dict):
                    continue

                if node_label in node_lookup:
                    target = node_lookup[node_label]
                    is_node = True
                elif node_label in pseudo_lookup:
                    target = pseudo_lookup[node_label]
                    is_node = False
                else:
                    print(f"[!] Entry '{node_label}' not found — skipping.")
                    continue

                temp_store = {"fluid_name": None, "T": None, "P": None, "X": None, "mass_flow": None}

                # First pass: collect everything
                for label, value in entries.items():
                    if value is None:
                        continue
                    clean_label = label.split(" #")[0].strip()
                    var = label_to_var.get(clean_label)
                    if not var:
                        print(f"[!] Unknown variable '{label}' — skipping.")
                        continue
                    temp_store[var] = value

                # Attempt to convert values
                for key in ["T", "P", "X", "mass_flow"]:
                    val = temp_store[key]
                    if isinstance(val, str):
                        try:
                            temp_store[key] = float(val)
                        except ValueError:
                            print(f"[!] Could not convert '{val}' to float for {key} — skipping.")
                            temp_store[key] = None

                # Try to construct the Fluid object if we have enough info
                fluid_name = temp_store["fluid_name"]
                state_args = {k: temp_store[k] for k in ("T", "P", "X") if temp_store[k] is not None}

                if fluid_name and len(state_args) >= 2:
                    try:
                        fluid = Fluid(fluid_name, **state_args)
                    except Exception as e:
                        print(f"[!] Failed to create Fluid({fluid_name}): {e}")
                        fluid = None
                else:
                    fluid = None

                # Apply to target
                try:
                    if is_node:
                        if fluid:
                            for port in target._ports:
                                port.fluid = fluid
                        if temp_store["mass_flow"] is not None:
                            for port in target._ports:
                                port.mass_flow = temp_store["mass_flow"]
                        for k in ("T", "P", "X"):
                            if temp_store[k] is not None:
                                setattr(target, k, temp_store[k])
                    else:
                        if fluid:
                            target.fluid = fluid
                        for k in ("T", "P", "X", "mass_flow"):
                            if temp_store[k] is not None:
                                setattr(target, k, temp_store[k])
                except Exception as e:
                    print(f"[!] Failed to set inputs on '{node_label}': {e}")

        print(f"[✓] Loaded node inputs from: {os.path.abspath(filename)}")
