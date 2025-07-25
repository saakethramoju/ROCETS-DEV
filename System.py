from FlowPort import InFlow, OutFlow
from PropertyPort import PropertyIn, PropertyOut
from Component import Component
import os
import yaml
import difflib

class System:
    def __init__(self, name):
        self.name = name
        self.components = []
        self.component_dict = {}

    def __str__(self):
        out = f"System: {self.name}"
        for comp in self.components:
            out += f"\n- {comp.name}"
        return out

    def add_component(self, root_component):
        """
        Add a component and recursively add any connected components.
        """
        visited = set(id(c) for c in self.components)
        to_visit = [root_component]

        while to_visit:
            current = to_visit.pop()
            cid = id(current)
            if cid in visited:
                continue

            current.system = self
            self.components.append(current)
            self.component_dict[current.name] = current
            visited.add(cid)

            for port in current.ports:
                neighbors = []

                if isinstance(port, (InFlow, OutFlow)) and port.connected_port:
                    neighbors.append(port.connected_port)
                elif isinstance(port, (PropertyIn, PropertyOut)):
                    group = port._group()
                    neighbors.extend(p for p in group if p is not port)

                for neighbor in neighbors:
                    if neighbor.parent is not None:
                        to_visit.append(neighbor.parent)

    def show_connections(self):
        print(f"\n System: {self.name}")
        print("─" * (10 + len(self.name)))

        for comp in sorted(self.components, key=lambda c: c.name.lower()):
            inflow_conns = []
            outflow_conns = []
            prop_conns = []

            for port in comp.ports:
                if isinstance(port, (InFlow, OutFlow)) and port.connected_port:
                    src_comp = port.connected_port.parent.name
                    src_port = port.connected_port.name
                    dst_comp = comp.name
                    dst_port = port.name

                    direction = "←" if isinstance(port, InFlow) else "→"
                    label = f"{src_comp}.{src_port} {direction} {dst_comp}.{dst_port}"

                    if isinstance(port, InFlow):
                        inflow_conns.append(label)
                    else:
                        outflow_conns.append(label)

                elif isinstance(port, (PropertyIn, PropertyOut)):
                    for p in port._group():
                        if p is not port and p.parent is not comp:
                            src = f"{p.parent.name}.{p.name}"
                            dst = f"{comp.name}.{port.name}"
                            prop_conns.append(f"{src} ⇄ {dst}")

            if inflow_conns or outflow_conns or prop_conns:
                print(f"\n* {comp.name}")
                if inflow_conns:
                    print("   Inflows:")
                    for line in sorted(inflow_conns):
                        print(f"    - {line}")
                if outflow_conns:
                    print("   Outflows:")
                    for line in sorted(outflow_conns):
                        print(f"    - {line}")
                if prop_conns:
                    print("   Properties:")
                    for line in sorted(set(prop_conns)):
                        print(f"    - {line}")

    def generate_input_template(self, filename: str = None):
        friendly_names = {
            "mass_flow": "Mass Flow (kg/s)",
            "T": "Temperature (K)",
            "P": "Pressure (Pa)",
            "X": "Vapor Quality (0–1)",
        }

        if filename is None:
            base = f"{self.name}_Inputs.yaml"
            filename = base
            i = 1
            while os.path.exists(filename):
                filename = f"{self.name}_Inputs_{i}.yaml"
                i += 1

        data = {}

        for comp in self.components:
            comp_label = comp.name.replace("_", " ").title()
            iters = comp.detect_iteration_variables()

            for port_name, var_key in iters:
                port_label = port_name.replace("_", " ").title()
                var_label = friendly_names.get(var_key, var_key)

                comp_block = data.setdefault(comp_label, {})
                port_block = comp_block.setdefault(port_label, {})
                port_block[var_label] = None

        yaml_string = yaml.dump(
            data,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False
        )
        cleaned_yaml = yaml_string.replace(": null", ":")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Input Template for Initial Guesses: {self.name}\n")
            f.write("# Fill in desired initial guesses for iteration variables.\n\n")
            f.write(cleaned_yaml)

        print(f"[✓] Input template written to: {os.path.abspath(filename)}")

    def load_inputs(self, filename: str):
        """
        Load user-provided inputs from a YAML file
        and assign them to the appropriate components.
        """
        with open(filename, "r") as f:
            data = yaml.safe_load(f)

        if not data:
            print("[!] YAML file is empty or malformed.")
            return

        reverse_friendly = {
            "Mass Flow (kg/s)": "mass_flow",
            "Temperature (K)": "T",
            "Pressure (Pa)": "P",
            "Vapor Quality (0–1)": "X",
        }

        for comp_label, port_block in data.items():
            component = self.get_component_by_label(comp_label)
            if not component:
                print(f"[!] Warning: No matching component for label '{comp_label}'")
                continue

            for port_label, inputs in port_block.items():
                for friendly_key, value in inputs.items():
                    if value is None:
                        continue

                    try:
                        var_key = reverse_friendly.get(friendly_key, friendly_key)
                        numeric_value = float(value)
                        component.set_input(port_label, var_key, numeric_value)
                    except ValueError:
                        print(f"[!] Skipping invalid value for {comp_label} > {port_label} > {friendly_key}: {value}")
                    except Exception as e:
                        print(f"[!] Error setting input: {e}")

        print(f"[✓] Loaded inputs from {filename}")


    def get_component_by_label(self, label: str):
        if label in self.component_dict:
            return self.component_dict[label]

        normalized_label = label.replace(" ", "").lower()
        for comp_name, comp in self.component_dict.items():
            if comp_name.replace("_", "").lower() == normalized_label:
                return comp

        close = difflib.get_close_matches(label, self.component_dict.keys(), n=1)
        if close:
            return self.component_dict[close[0]]
        return None
            
    def apply_inputs(self):
        """
        Apply stored inputs to all components in the system.
        """
        for comp in self.components:
            comp.apply_inputs()





    def get_iteration_keys(self) -> list[tuple[Component, str, str]]:
        """
        Return a list of (component, port_name, var_name) tuples
        representing all iteration variables across the system.
        """
        keys = []
        for comp in self.components:
            for port_name, var in comp.get_iteration_keys():
                keys.append((comp, port_name, var))
        return keys
        

    def get_iteration_vector(self) -> list[float]:
        """
        Return the flat vector of current iteration values from all components.
        """
        return [
            getattr(comp[port_name], var)
            for (comp, port_name, var) in self.get_iteration_keys()
        ]

    def set_iteration_vector(self, x: list[float]):
        """
        Apply a flat vector of iteration values to the correct ports
        across all components in the system.
        """
        keys = self.get_iteration_keys()
        if len(x) != len(keys):
            raise ValueError(
                f"[System] Input vector length {len(x)} doesn't match total "
                f"iteration variables ({len(keys)})"
            )
        for val, (comp, port_name, var) in zip(x, keys):
            setattr(comp[port_name], var, val)

    def residual(self, x):
        self.set_iteration_vector(x)
        res = []
        for comp in self.components:
            try:
                r = comp.residual(comp.get_iteration_vector())
                res.extend(r)
            except Exception as e:
                print(f"[!] Residual error in {comp.name}: {e}")
                raise
        print("[System] Residual vector:", [f"{r:.3e}" for r in res])
        return res



    def solve(self, method="hybr", tol=1e-6, verbose=True):
        """
        Solve all components in the system using scipy.optimize.root.

        Automatically:
        - Builds the full iteration vector
        - Applies it to the system
        - Computes residuals from each component
        - Updates ports after convergence

        Parameters:
            method: str
                Solver method (e.g., 'hybr', 'lm', etc.)
            tol: float
                Convergence tolerance
            verbose: bool
                Print status and result

        Returns:
            OptimizeResult from scipy.optimize.root
        """
        x0 = self.get_iteration_vector()

        def residual(x):
            self.set_iteration_vector(x)
            return self.residual(x)

        result = root(residual, x0, method=method, tol=tol)

        # Apply final solution to all ports once
        self.set_iteration_vector(result.x)

        if verbose:
            print(f"\n[✓] Solver finished for system '{self.name}'")
            print("Status:", result.message)
            print("Success:", result.success)
            print("Final x:", result.x)

        return result




if __name__ == "__main__":

    from Component import Component
    from Fluid import Fluid
    import numpy as np
    from scipy.optimize import root



    class PipeTest(Component):
        def __init__(self, name, Cd, A):
            super().__init__(name)
            self.Cd = Cd  # discharge coefficient
            self.A = A    # cross-sectional area [m^2]
            self._initialize_default_ports()

        def _initialize_default_ports(self):
            self.add_inflow("Source")
            self.add_outflow("Drain")

        def residual(self, x):
            """
            General residual method called by System.
            Updates the iteration variables using x and returns residual vector.
            """
            self.set_iteration_vector(x)
            upstream_vars = self.get_upstream_iterations("Source")

            if "mass_flow" in upstream_vars:
                return self.pipe01()
            else:
                raise NotImplementedError(
                    f"No residual function implemented for upstream iteration variable(s): {upstream_vars}"
                )

        def pipe01(self):
            """
            Residual function for pipe flow solver.
            Assumes mass flow is the iteration variable.
            """
            source = self["Source"]
            drain = self["Drain"]

            # Ensure both fluid states are defined
            if source.fluid is None or drain.fluid is None:
                raise ValueError("Both Source and Drain ports must have fluid_name, T, P (or X) set.")

            rho1 = source.fluid.density
            P1 = source.P
            #h1 = source.fluid.enthalpy

            rho2 = drain.fluid.density
            P2 = drain.P
            #h2 = drain.fluid.enthalpy


            mdot1 = source.mass_flow
            if rho1 == 0 or rho2 == 0:
                raise ZeroDivisionError("Density is zero, check fluid state inputs.")

            #v1 = mdot / (rho1 * self.Cd * self.A)
            #v2 = np.sqrt(2 * (h1 - h2 + 0.5 * (v1**2)))
            mdot2 = self.Cd*self.A*np.sqrt(2*rho2*(P1 - P2))

            residual = mdot2 - mdot1
            return [residual]



    pipe = PipeTest("pipe", Cd=0.6, A=8e-5)

    # Define inlet and outlet
    inlet = Component("Inlet")
    inlet.add_outflow("Source")
    inlet["Source"].fluid = Fluid("Water", T=298, P=3e6)  # No mass flow yet
    #inlet["Source"].mass_flow = 5

    outlet = Component("Outlet")
    outlet.add_inflow("Drain")
    outlet["Drain"].T = 298
    outlet["Drain"].P = 101325
    #outlet["Drain"].mass_flow = 5

    inlet.provides_inputs_only = True
    #outlet.provides_inputs_only = True

    # Connect components
    pipe.connect(inlet)
    pipe.connect(outlet)

    #print(inlet.detect_iteration_variables())
    #print(pipe.detect_iteration_variables())
    #print(outlet.detect_iteration_variables())

    #print(pipe.get_upstream_iterations())
    #inlet.set_guess("Source", "mass_flow", 5.0)

    #print(inlet["Source"].mass_flow)
    #inlet.apply_guesses()
    #print(inlet["Source"].mass_flow) 
    #print(pipe.residual())

    EngineSystem = System("Vespula")
    EngineSystem.add_component(pipe)
    #EngineSystem.show_connections()
    #EngineSystem.generate_input_template()
    EngineSystem.load_inputs("/Users/saakethramramoju/Desktop/ROCETS DEV/Vespula_Inputs.yaml")
    EngineSystem.apply_inputs()

    
    #print(inlet)
    #print(pipe)
    #print(outlet)
    
    result = EngineSystem.solve()
    #print(inlet)
    #print(pipe)
    #print(outlet)


    '''
    print(inlet["Source"].mass_flow)
    print(EngineSystem.get_iteration_vector())
    EngineSystem.set_iteration_vector([10])
    print(EngineSystem.get_iteration_vector())
    print(inlet["Source"].mass_flow)
 

    # Solve for mass flow
    print(pipe.residual())
    #mdot_solution = pipe.solve()
    #inlet["Source"].mass_flow = mdot_solution
    #print(f"Solved mass flow: {mdot_solution:.4f} kg/s")'''