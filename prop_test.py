from rocketprops.rocket_prop import get_prop
import rocketprops

rp1 = get_prop("RP-1")     # RP‑1 liquid hydrocarbon
lox = get_prop("LOX")     # Liquid Oxygen

rp1.set_std_state()

rp1.summ_print()