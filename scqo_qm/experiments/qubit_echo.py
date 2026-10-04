"""T2 echo (Hahn) acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Echo sequence x90 - wait(t) - x180 - wait(t) - x90 on each qubit: the central pi
pulse refocuses quasi-static dephasing, so the envelope decays with T2_echo.
Sequence mirrors the vendored official ``calibrations/06b_echo.py`` core (which
ends in -x90; the final +x90 here refocuses to |1> instead of |0>, matching the
scqo physics half "X90 - tau/2 - X - tau/2 - X90" - the downstream exponential
fit is sign-aware either way).

QM qubit echo (Hahn) for scqo - supplies only ``probe()``.

Parameters, exponential-envelope fit and reporting are inherited from
``scqo.experiments.QubitEcho``. scqo sweeps ``wait_time_ns`` (the TOTAL echo idle
time tau); the QM builder sweeps the per-arm wait in clock cycles (two arms of
tau/2, 4 ns per cycle -> cycles = tau_ns / 8) and builds the same sweep on coord
``idle_time``, which the backend's ``_to_canonical`` renames back.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    qubits,
    *,
    idle_times_cycles,
    num_shots: int,
    reset_type: str,
    reset_max_attempts: int = 15,
    use_state_discrimination: bool = False,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the T2-echo QUA program. Returns (program, sweep_axes).

    `idle_times_cycles` is the PER-ARM idle-time sweep in clock cycles (4 ns); the
    echo has two idle arms, so the reported idle-time axis is 8 * idle_times_cycles
    ns (total idle time tau, as in the official 06b_echo node's 2 * 4 * t).
    `qubits` is a BatchableList (see `_lib.select_qubits`).
    """
    # Collapse sub-clock duplicates so the buffer length matches the axis (see ramsey).
    idle_times_cycles = np.unique(np.asarray(idle_times_cycles))

    num_qubits = len(qubits)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(
            8 * idle_times_cycles, attrs={"long_name": "total echo idle time", "units": "ns"}
        ),
    }
    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        t = declare(int)

        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)

                with for_each_(t, idle_times_cycles):
                    # Qubit initialization
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(reset_type, simulate, log_callable=log,
                                    max_attempts=reset_max_attempts)
                    align()
                    # Echo: x90 - tau/2 - x180 - tau/2 - x90 (official 06b_echo core)
                    for i, qubit in multiplexed_qubits.items():
                        qubit.align()
                        qubit.xy.play("x90")
                        qubit.xy.wait(t)
                        qubit.xy.play("x180")
                        qubit.xy.wait(t)
                        qubit.xy.play("x90")
                    align()
                    for i, qubit in multiplexed_qubits.items():
                        if use_state_discrimination:
                            qubit.readout_state(state[i])
                            save(state[i], state_st[i])
                        else:
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if use_state_discrimination:
                    state_st[i].buffer(len(idle_times_cycles)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(idle_times_cycles)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(idle_times_cycles)).average().save(f"Q{i + 1}")

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


from typing import Any, ClassVar

import numpy as np

from scqo import register
from scqo.experiments import QubitEcho
from scqo.requirements import Requirement

from ._requires import X90_AMPLITUDE


@register
class QMQubitEcho(QubitEcho):
    """Build a multiplexed Hahn-echo QUA program on the QM OPX."""

    #: what only this backend consumes, on top of the neutral requirements
    requires: ClassVar[tuple[Requirement, ...]] = (*QubitEcho.requires, X90_AMPLITUDE)
    #: true of THIS probe only; scqo shows them in `scqo run qubit_echo --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "the two pi/2 pulses are the x90 operation, whose amplitude is pi_amp_x90",
        "each arm is rounded to the 4 ns clock, so the total idle moves in 8 ns steps, and "
        "the dataset stores the realized times",
    )

    #: Readout is held at the calibrated point for the whole run and the reset is
    #: a genuine state reset, so reset_method='active' is valid here (_reset.py).
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        # Total idle time tau (ns) -> per-arm clock cycles: tau/2 per arm, 4 ns per
        # cycle. QUA wait() needs >= 1 cycle, hence the floor.
        wait_ns = self.sweep_axes["wait_time_ns"]
        arm_cycles = np.maximum(1, np.round(np.asarray(wait_ns) / 8)).astype(int)

        return build_program(
            machine,
            qubits,
            idle_times_cycles=arm_cycles,
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            reset_max_attempts=reset_max_attempts(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
        )
