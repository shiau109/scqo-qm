"""Register a `FlatTopCosinePulse` flux operation on couplers + qubit z lines.

Adds one named operation (`OP`) carrying a `FlatTopCosinePulse` (sine/raised-cosine edges
with a flat top) to, for every qubit pair that has a coupler:
  - `qp.coupler.operations[OP]`
  - `qp.qubit_control.z.operations[OP]`
  - `qp.qubit_target.z.operations[OP]`

so that `LCH_pair_qcq_fixed_time` (or any node that looks an op up by name) can play it as
the coupler flux op and/or the qubit-z flux op for either `flux_role`.

There is no separate "pulse registry": QUAM serializes each operation inline in
`quam_state/state.json` with a `"__class__"` import path and re-imports it on `Quam.load()`.
`FlatTopCosinePulse` only needs to be importable (it is, via `scqo_qm.components.pulses`),
and to exist as a named operation on the channel -- which is exactly what this script writes.

>>> FILL IN the calibrated length / amplitude / edge_width below before running. <<<
The values here are PLACEHOLDERS. Run this once to persist into quam_state/state.json:

    python quam_config/register_flattop_cosine.py <state folder>

Constraints (else FlatTopCosinePulse.waveform_function raises / QM rejects the config):
  - length must be a multiple of 4 ns and >= 16
  - 2 * edge_width <= length
"""

from quam_config._state_arg import state_folder
from scqo_qm.quam_io import load_state, save_state
from scqo_qm.components.pulses import FlatTopCosinePulse

OP = "flattop_cosine"  # operation name played via node.parameters.coupler_operation / qubit_operation

# --- PLACEHOLDER shape params: calibrate these -------------------------------------------
LENGTH = 64      # ns, multiple of 4 and >= 16
AMPLITUDE = 0.25  # V. Keep < 0.5 (OPX1000 LF-FEM "direct" output rail): a stored peak >= 0.5 V
                  # is clipped/corrupted on hardware (the simulator hides it). 0.25 leaves margin.
EDGE_WIDTH = 2   # samples per sine edge; flat top length = LENGTH - 2 * EDGE_WIDTH
# ----------------------------------------------------------------------------------------

STATE = state_folder(__doc__.splitlines()[0])
machine = load_state(STATE)

n_pairs = 0
for qp in machine.qubit_pairs.values():
    if qp.coupler is None:
        print(f"skip {qp.name}: no coupler")
        continue
    # Separate instances per channel so each amplitude can be calibrated independently later.
    qp.coupler.operations[OP] = FlatTopCosinePulse(length=LENGTH, amplitude=AMPLITUDE, edge_width=EDGE_WIDTH)
    qp.qubit_control.z.operations[OP] = FlatTopCosinePulse(length=LENGTH, amplitude=AMPLITUDE, edge_width=EDGE_WIDTH)
    qp.qubit_target.z.operations[OP] = FlatTopCosinePulse(length=LENGTH, amplitude=AMPLITUDE, edge_width=EDGE_WIDTH)
    n_pairs += 1
    print(f"{qp.name}:")
    print(f"  coupler ops:        {list(qp.coupler.operations.keys())}")
    print(f"  control.z ops:      {list(qp.qubit_control.z.operations.keys())}")
    print(f"  target.z ops:       {list(qp.qubit_target.z.operations.keys())}")

if n_pairs == 0:
    raise SystemExit("No qubit pairs with a coupler found; nothing to register.")

save_state(machine, str(STATE))
print(f"Saved. '{OP}' (FlatTopCosinePulse) registered on coupler + both z lines of {n_pairs} pair(s).")
