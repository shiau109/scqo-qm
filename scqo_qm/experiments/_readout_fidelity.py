"""Readout-fidelity acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Single-shot readout of each qubit prepared in |0> and |1>; the IQ blobs are fitted downstream to
characterise the readout fidelity at the current settings.

`prepared_states` selects WHICH states are prepared, and the shape of everything downstream follows
it: [0, 1] is the two-state default, and [0] alone is the ground-state-only cloud a
thermal-population measurement wants.
"""

from typing import Callable, Optional, Sequence

import numpy as np
import xarray as xr
from qm.qua import *

from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    qubits,
    *,
    operation: str,
    num_shots: int,
    reset_type: str,
    simulate: bool = False,
    prepared_states: Sequence[int] = (0, 1),
):
    """Build the readout-fidelity QUA program. Returns (program, sweep_axes).

    `num_shots` single shots are taken per prepared state (0 = ground, 1 = x180-excited);
    `qubits` is a BatchableList (see `_lib.select_qubits`).
    """
    num_qubits = len(qubits)
    prepared_states = list(prepared_states)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "shot_idx": xr.DataArray(np.arange(1, num_shots + 1), attrs={"long_name": "number of shots"}),
        "prepared_state": xr.DataArray(prepared_states, attrs={"long_name": "prepared qubit state", "units": ""}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        ps = declare(int)

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                # ground iq blobs for all qubits
                save(n, n_st)
                with for_each_(ps, prepared_states):
                    # Qubit initialization
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(reset_type, simulate)
                    align()

                    # Change qubit state
                    for i, qubit in multiplexed_qubits.items():
                        qubit.align()

                        with switch_(ps):
                            with case_(0):
                                pass
                            with case_(1):
                                qubit.xy.play("x180")

                        qubit.align()
                    # Qubit readout
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure(operation, qua_vars=(I[i], Q[i]))
                        qubit.align()
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(prepared_states)).buffer(num_shots).save(f"I{i + 1}")
                Q_st[i].buffer(len(prepared_states)).buffer(num_shots).save(f"Q{i + 1}")

    return prog, sweep_axes


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_shots: int,
    timeout: float,
    log: Optional[Callable] = None,
) -> xr.Dataset:
    """Connect to the QOP, execute the program and fetch the raw xr.Dataset."""
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout, log=log)
