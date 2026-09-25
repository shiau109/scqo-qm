"""Attach a bare-callable parametric-reset macro to qubit q2 (fix #2 for LCH_qc_swap_paramreset).

The LCH_qc_swap_paramreset node looks up `reset_qubit.macros["reset"].apply()` (bare call).
This script wires that macro onto `q2` using `ParametricReset`, whose `apply()` sets the
z-line IF to a calibrated drive frequency, resets the IF phase, then plays a dedicated
`parametric_reset` z pulse (the same flux-line technique as LCH_qubit_parametric_drive_*).

State of the pulses at time of writing (from the loaded QUAM):
  - q2.z.operations does NOT have "parametric_reset" yet -> this script adds it.

>>> FILL IN the calibrated amplitude/length and drive frequency below before running. <<<
The values here are PLACEHOLDERS. Run this once to persist into quam_state/state.json:

    python quam_config/register_reset_macro.py <state folder>
"""

from quam.components.pulses import SquarePulse
from quam_config._state_arg import state_folder
from scqo_qm.quam_io import load_state, save_state
from scqo_qm.components.macros.parametric_reset_macro import ParametricReset

FLUX_PULSE = "parametric_reset"  # z op played by the macro
DRIVE_FREQUENCY = 375_000_000  # PLACEHOLDER - parametric drive frequency in Hz (z-line IF)

STATE = state_folder(__doc__.splitlines()[0])
machine = load_state(STATE)
q2 = machine.qubits["q2"]

# --- Dedicated parametric-reset z pulse (PLACEHOLDER amplitude/length: calibrate these) ---
# Amplitude/length live in the pulse op (not the macro), so the macro stays bare-callable.
q2.z.operations[FLUX_PULSE] = SquarePulse(
    amplitude=0.1,  # PLACEHOLDER - calibrate
    length=400,     # PLACEHOLDER - calibrate (parametric-drive duration in ns)
)

# --- Attach the bare-callable macro under the key the node looks up ("reset") -------------
q2.macros["reset"] = ParametricReset(drive_frequency=DRIVE_FREQUENCY, flux_pulse=FLUX_PULSE)

print("q2.z ops:", list(q2.z.operations.keys()))
print("q2 macros:", list(q2.macros.keys()))

save_state(machine, str(STATE))
print("Saved. q2.macros['reset'] is now bare-callable: q2.macros['reset'].apply()")
