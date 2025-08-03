# System.py
import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import root
from Components import Component
from Fluids import Fluid, Mixture
from Ports import FlowNode


class System:
    def __init__(self, name: str):
        self.name = name
        self.components: list[Component] = []
        self.nodes: list[FlowNode] = []
        self.t = np.array([0]) # time array
        self.dt = None # time step
        self.step = 0 # step number
        self.t_end = 0 # end time

    def solve(self, analysis_type: str = "steady-state", method: str = "hybr", tol: float = 1e-6, verbose=False, dt=None, t_end=None):
        """
        Solves the system using either steady-state or transient analysis.

        Parameters:
            analysis_type: 'steady-state' or 'transient'
            method: root solver method for steady-state or transient solves
            tol: numerical solver tolerance
            verbose: whether to print iteration status

        Raises:
            RuntimeError if a step fails to converge.
        """


        if analysis_type == "steady-state":
            self.evaluate(verbose=False)

            if len(self.get_state_vector()) < len(self.get_residual_vector(analysis_type=analysis_type)):
                method = 'lm'

            def residual_func(x):
                self.set_state_vector(x)
                self.evaluate(verbose=False)
                return self.get_residual_vector(analysis_type=analysis_type)

            x0 = self.get_state_vector()
            result = root(residual_func, x0, method=method, tol=tol)

            if result.success:
                self.set_state_vector(result.x)
                if verbose:
                    print(f"[✓] Converged in {result.nfev} evaluations.")
            else:
                raise RuntimeError(f"[✗] Solver failed: {result.message}")

        elif analysis_type == "transient":
            if dt is not None:
                self.dt = dt
            if t_end is not None:
                self.t_end = t_end

            if self.dt is None or self.t_end is None:
                raise ValueError("Both dt and t_end must be set for transient solving.")

            while self.t[-1] < self.t_end:
                print(f"\n[⏱] Time Step {self.step} — t = {self.t[-1]:.3f} s")
                # Call root solver for this timestep
                def residual_func(x):
                    self.set_state_vector(x)
                    self.evaluate(verbose=False)
                    return self.get_residual_vector(analysis_type="transient")

                x0 = self.get_state_vector()
                result = root(residual_func, x0, method=method, tol=tol)

                if result.success:
                    self.set_state_vector(result.x)
                    if verbose:
                        print(f"[✓] Converged in {result.nfev} evaluations.")
                else:
                    raise RuntimeError(f"[✗] Solver failed: {result.message}")

                # Advance time AFTER solving this timestep
                self.timestep()


    def timestep(self):
        """
        Perform one transient time step:
        - Solve for system state (T/P) with transient residuals.
        - Advance node storage (M, U).
        - Advance system time.
        """

        self.step += 1
        self.t = np.append(self.t, self.t[-1] + self.dt)
        for node in self.nodes:
            node.timestep()
        for comp in self.components:
            if hasattr(comp, "record"):
                comp.record()




    def evaluate(self, verbose: bool = False):
        """Evaluate all components in upstream-to-downstream order."""
        sorted_comps = self.topological_sort_components()

        for comp in sorted_comps:
            if verbose:
                print(f"Evaluating: {comp.name}")
            comp.evaluate()


    def get_state_vector(self) -> list[float]:
        x = []
        visited_nodes = set()

        for comp in self.components:
            for port in comp.ports().values():
                node = port.node
                if node and not node.is_boundary_node() and node not in visited_nodes:
                    x.append(node.T)
                    x.append(node.P)
                    visited_nodes.add(node)

        for comp in self.components:
            for label, value in comp.get_additional_iteration_variables():
                x.append(value)

        return x


    def set_state_vector(self, x: list[float]):
        i = 0
        visited_nodes = set()

        for comp in self.components:
            for port in comp.ports().values():
                node = port.node
                if node and not node.is_boundary_node() and node not in visited_nodes:
                    node.T = x[i]
                    node.P = x[i + 1]
                    i += 2
                    visited_nodes.add(node)

        for comp in self.components:
            for label, _ in comp.get_additional_iteration_variables():
                comp.set_additional_iteration_variable(label, x[i])
                i += 1



    def _all_flow_nodes(self) -> set[FlowNode]:
        return {port.node for comp in self.components for port in comp.ports().values() if port.node}


    def get_residual_vector(self, analysis_type: str = "steady-state") -> list[float]:
        """
        Gather all residuals from non-boundary nodes and components for the specified analysis type.

        Returns:
            List of floats representing system-wide residuals.
        """
        residuals = []
        visited_nodes = set()

        # Collect residuals from nodes
        for comp in self.components:
            for port in comp.ports().values():
                node = port.node
                if node and node not in visited_nodes and not node.is_boundary_node():
                    visited_nodes.add(node)
                    node_res = node.residual(analysis_type=analysis_type)
                    if node_res is not None:
                        if isinstance(node_res, list):
                            residuals.extend(node_res)
                        else:
                            residuals.append(node_res)

        # Collect residuals from components
        for comp in self.components:
            comp_res = comp.residual(analysis_type=analysis_type)
            if comp_res is not None:
                if isinstance(comp_res, list):
                    residuals.extend(comp_res)
                else:
                    residuals.append(comp_res)

        return residuals

    

    def topological_sort_components(self) -> list[Component]:
        from collections import defaultdict, deque

        # Build dependency graph: A → B if A feeds into B
        graph = defaultdict(set)
        in_degree = {comp: 0 for comp in self.components}

        for comp in self.components:
            for port in comp.ports().values():
                other = port.connected_port
                if other and other.parent and other.parent is not comp:
                    if port.__class__.__name__ == "OutFlow":
                        src = comp
                        dst = other.parent
                        graph[src].add(dst)
                        in_degree[dst] += 1

        # Kahn's algorithm for topological sort
        queue = deque([c for c in self.components if in_degree[c] == 0])
        sorted_comps = []

        while queue:
            node = queue.popleft()
            sorted_comps.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_comps) != len(self.components):
            raise RuntimeError("Cycle detected in component graph.")

        return sorted_comps



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
                for tgt in out_prop.connected_ports:
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

                # Flow ports and PropertyIn: single connection
                if hasattr(port, "connected_port") and port.connected_port:
                    queue.append(port.connected_port.parent)

                # PropertyOut: multiple connections
                elif hasattr(port, "connected_ports"):
                    for target in port.connected_ports:
                        queue.append(target.parent)



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
                if "fluid_name" in var_list:
                    lines.append(f"    # Specify binary mixture below if needed")
                    lines.append(f"    Constituents:")
                    lines.append(f"      Constituent 1:   # Replace with name of substance")
                    lines.append(f"      Constituent 2:   # Replace with name of substance")
                    lines.append(f"    Fraction Type: mole  # Define with mole or mass fractions")
            lines.append("")

        extra_vars_by_comp = {}
        for comp in system.components:
            for label, _ in comp.get_additional_iteration_variables():
                comp_name, param_label = label.split(":", 1)
                if comp_name not in extra_vars_by_comp:
                    extra_vars_by_comp[comp_name] = []
                extra_vars_by_comp[comp_name].append(param_label)

        for comp_name, var_list in extra_vars_by_comp.items():
            inserted = False
            for i, line in enumerate(lines):
                if line.strip().startswith(f"{comp_name}:"):
                    insert_at = i + 1
                    while insert_at < len(lines) and lines[insert_at].startswith("  "):
                        insert_at += 1
                    for var in var_list:
                        lines.insert(insert_at, f"  {friendly_names.get(var, var)}:")
                        insert_at += 1
                    inserted = True
                    break
            if not inserted:
                lines.append(f"{comp_name}:  # Initial Guess")
                for var in var_list:
                    lines.append(f"  {friendly_names.get(var, var)}:")
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

        pseudo_lookup = {}
        for comp in self.components:
            comp_label = comp.name.replace("_", " ").title()
            for port in comp.ports().values():
                if port.node is None:
                    key_full = f"{comp_label}:{port.name.replace('_', ' ').title()}"
                    key_simple = f"{comp_label}"
                    pseudo_lookup[key_full] = port
                    if len(comp.ports()) == 1:
                        pseudo_lookup[key_simple] = port

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
                    print(f"[!] Entry '{node_label}' under '{comp_label}' not found — skipping.")
                    continue

                temp_store = {"fluid_name": None, "T": None, "P": None, "X": None, "mass_flow": None}
                components = None
                fraction_type = "mole"

                for label, value in entries.items():
                    if value is None:
                        continue
                    clean_label = label.split(" #")[0].strip()
                    if clean_label == "Constituents" and isinstance(value, dict):
                        components = value
                    elif clean_label == "Fraction Type" and isinstance(value, str):
                        fraction_type = value.lower()
                    else:
                        var = label_to_var.get(clean_label)
                        if var:
                            temp_store[var] = value

                for key in ("T", "P", "X", "mass_flow"):
                    if isinstance(temp_store[key], str):
                        try:
                            temp_store[key] = float(temp_store[key])
                        except ValueError:
                            print(f"[!] Could not convert '{temp_store[key]}' to float for {key} — skipping.")
                            temp_store[key] = None

                fluid_name = temp_store["fluid_name"]
                T, P, X = temp_store["T"], temp_store["P"], temp_store["X"]

                try:
                    fluid_obj = None
                    if fluid_name:
                        fluid_obj = Fluid(fluid_name, T=T or 298.15, P=P or 101325)
                    elif components:
                        clean_components = {k: float(v) for k, v in components.items()}
                        if len(clean_components) == 1:
                            name = next(iter(clean_components))
                            fluid_obj = Fluid(name, T=T or 298.15, P=P or 101325)
                        else:
                            total_frac = sum(clean_components.values())
                            if abs(total_frac - 1.0) > 1e-6:
                                raise ValueError(
                                    f"Fractions for mixture at '{node_label}' under '{comp_label}' must sum to 1.0 "
                                    f"(currently {total_frac:.4f})."
                                )
                            fluid_obj = Mixture(
                                clean_components,
                                fraction_type=fraction_type,
                                T=T if T is not None else 298.15,
                                P=P if P is not None else 101325,
                                X=X,
                            )
                    if is_node:
                        if fluid_obj:
                            target.fluid = fluid_obj
                        if temp_store["mass_flow"] is not None:
                            for port in target._ports:
                                port.mass_flow = temp_store["mass_flow"]
                                if port.connected_port:
                                    port.connected_port.mass_flow = temp_store["mass_flow"]
                        for k in ("T", "P", "X"):
                            if temp_store[k] is not None:
                                setattr(target, k, temp_store[k])
                    else:
                        if fluid_obj:
                            target.fluid = fluid_obj
                            if target.connected_port:
                                target.connected_port.fluid = fluid_obj
                        if temp_store["mass_flow"] is not None:
                            target.mass_flow = temp_store["mass_flow"]
                            if target.connected_port:
                                target.connected_port.mass_flow = temp_store["mass_flow"]
                        for k in ("T", "P", "X"):
                            if temp_store[k] is not None:
                                setattr(target, k, temp_store[k])

                except Exception as e:
                    raise Exception(f"[!] Failed to set inputs on '{node_label}': {e}")

        for comp in self.components:
            expected_vars = comp.get_additional_iteration_variables()
            if not expected_vars:
                continue
            comp_label = comp.name.replace("_", " ").title()
            if comp_label not in raw:
                continue
            for label, _ in expected_vars:
                _, var_name = label.split(":", 1)
                field_value = raw[comp_label].get(var_name)
                if field_value is not None:
                    try:
                        comp.set_additional_iteration_variable(label, float(field_value))
                    except Exception:
                        print(f"[!] Failed to parse value for {label}.")

        print(f"[✓] Loaded node inputs from: {os.path.abspath(filename)}")


    def export(self, filepath=None):


        # Default file path
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{self.name}_Results_{timestamp}"
            filepath = base + ".xlsx"
            counter = 1
            while os.path.exists(filepath):
                filepath = f"{base}_{counter}.xlsx"
                counter += 1

        writer = pd.ExcelWriter(filepath, engine="openpyxl")

        # Detect transient run (more than one time step)
        is_transient = hasattr(self, "t") and hasattr(self.t, "__len__") and len(self.t) > 1

        sensor_rows = []

        for comp in self.components:
            name = comp.name.replace("_", " ").title()[:31]

            # Handle sensors separately
            if comp.__class__.__name__.lower() == "sensor":
                df = getattr(comp, "get_steady_state_dataframe", lambda: None)()
                if df is not None and not df.empty:
                    sensor_rows.append(df)
                continue

            # Choose appropriate result method
            if is_transient:
                df = getattr(comp, "get_transient_dataframe", lambda: None)()
            else:
                df = getattr(comp, "get_steady_state_dataframe", lambda: None)()

            if df is not None and not df.empty:
                df = df.copy()
                df.insert(0, "Component", comp.name)
                sheet_name = name
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Combine all sensors into one sheet
        if sensor_rows:
            sensor_df = pd.concat(sensor_rows, ignore_index=True)
            sensor_df.to_excel(writer, sheet_name="Sensors", index=False)

        writer.close()
        print(f"[✓] Results exported to: {os.path.abspath(filepath)}")
