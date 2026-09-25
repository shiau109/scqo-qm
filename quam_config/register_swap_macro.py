"""Attach a bare-callable iSWAP macro to the q1_q2 pair (fix #1 for LCH_qc_swap_paramreset).

The LCH_qc_swap_paramreset node looks up `swap_pair.macros["iswap"].apply()` (bare call).
This script wires that macro onto `q1_q2` using `ISwapImplementation`, whose `apply()` plays
a single named `flux_pulse` on BOTH the control qubit's z line and the coupler, then aligns.

State of the pulses at time of writing (from the loaded QUAM):
  - coupler q1_q2.coupler.operations already has "swap_01_10_square" (SquarePulse).
  - control q1.z.operations does NOT have it yet -> this script adds it.

>>> FILL IN the calibrated amplitude/length below before running. <<<
The values here are PLACEHOLDERS. Run this once to persist into quam_state/state.json:

    python quam_config/register_swap_macro.py <state folder>
"""

from quam.components.pulses import SquarePulse
from quam_config._state_arg import state_folder
from scqo_qm.quam_io import load_state, save_state
from scqo_qm.components.macros.iswap_macro import ISwapImplementation

FLUX_PULSE = "flattop_cosine"  # op name played on both control.z and coupler

STATE = state_folder(__doc__.splitlines()[0])
machine = load_state(STATE)
pair = machine.qubit_pairs["q1_q2"]

# --- Control z-line swap pulse (PLACEHOLDER amplitude/length: calibrate these) ---------
# The coupler already carries `swap_01_10_square`; the control z-line needs its own copy
# because ISwapImplementation plays `flux_pulse` on both. Set the control amplitude to
# whatever your swap scheme requires (use a small/zero amplitude if the swap is
# coupler-driven and the control qubit only idles).
pair.qubit_control.z.operations[FLUX_PULSE] = SquarePulse(
    amplitude=0.1,  # PLACEHOLDER - calibrate
    length=64,     # PLACEHOLDER - calibrate (must match the swap duration)
)

# --- Attach the bare-callable macro under the key the node looks up ("iswap") -----------
pair.macros["partial_swap"] = ISwapImplementation(flux_pulse=FLUX_PULSE)

print("control.z ops:", list(pair.qubit_control.z.operations.keys()))
print("coupler ops:", list(pair.coupler.operations.keys()))
print("pair macros:", list(pair.macros.keys()))

save_state(machine, str(STATE))
print("Saved. q1_q2.macros['iswap'] is now bare-callable: pair.macros['iswap'].apply()")
