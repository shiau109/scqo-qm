"""Register a `stark` XY (RF) operation on every qubit's xy line.

Adds one named operation (`OP`) carrying a `SquarePulse` (a constant flat tone) to
`q.xy.operations[OP]` for every qubit that has an xy line, so that
`qc_n_stark_amp`'s probe can play it as an AC-Stark drive:
`qubit.xy.play("stark", amplitude_scale=a)` with `a` the swept amplitude factor.

The xy line is an `MWChannel`; the template is the existing `saturation` tone (also
a plain `SquarePulse`). A `SquarePulse` carries NO per-pulse frequency/detuning, so
the OFF-RESONANCE that turns this tone into a genuine AC-Stark shift (rather than a
Rabi drive) is applied at QUA time in the probe via `update_frequency`
(`stark_detuning_hz`), NOT here — this script only defines the pulse's shape.

There is no separate "pulse registry": QUAM serializes each operation inline in
`quam_state/state.json` with a `"__class__"` import path and re-imports it on
`Quam.load()`. `SquarePulse` is a vendored class (importable), which is all QUAM
needs — plus the named operation on the channel, which is exactly what this writes.

>>> FILL IN the calibrated length / amplitude below before running. <<<
The values here are PLACEHOLDERS. Run this once to persist into quam_state/state.json:

    python quam_config/register_stark.py <state folder>

Constraints (else QM rejects the config):
  - length must be a multiple of 4 ns and >= 16
  - keep AMPLITUDE * (max amplitude_scale you will sweep) within the MW output
    headroom (a stored/played peak at or above full scale clips on hardware; the
    simulator hides it)

After running, per the repo CLAUDE.md: re-check the pulse `__class__` path survived
the save, and gate the config offline with `machine.generate_config()` under
`warnings.simplefilter("error")`.
"""

from quam.components.pulses import SquarePulse
from quam_config._state_arg import state_folder
from scqo_qm.quam_io import load_state, save_state

OP = "stark"  # operation name played via qc_n_stark_amp's stark_operation

# --- PLACEHOLDER shape params: calibrate these ------------------------------------------
LENGTH = 1000     # ns, multiple of 4 and >= 16
AMPLITUDE = 0.25  # MW-normalized amplitude. Keep AMPLITUDE * max_stark_amp within headroom;
                  # the swept amplitude_scale rides on top of this baked reference.
# ----------------------------------------------------------------------------------------

STATE = state_folder(__doc__.splitlines()[0])
machine = load_state(STATE)

n = 0
for q in machine.qubits.values():
    if q.xy is None:  # mirror populate_quam_lf_mw_fems.py's guard
        print(f"skip {q.name}: no xy line")
        continue
    # axis_angle=0.0 matches the other xy ops; a constant tone drives along +I.
    q.xy.operations[OP] = SquarePulse(length=LENGTH, amplitude=AMPLITUDE, axis_angle=0.0)
    n += 1
    print(f"{q.name}: xy ops -> {list(q.xy.operations.keys())}")

if n == 0:
    raise SystemExit("No qubits with an xy line found; nothing to register.")

save_state(machine, str(STATE))
print(f"Saved. '{OP}' (SquarePulse) registered on the xy line of {n} qubit(s).")
