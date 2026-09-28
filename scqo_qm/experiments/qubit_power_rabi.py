"""Power-Rabi acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Single-pulse power Rabi: sweep the qubit drive amplitude (as a pre-factor of the
current pulse amplitude); the operation is played exactly once per amplitude
point (no error-amplification loop). If `drive_qubit` is set, only that qubit is
driven while all qubits are still read out.

QM qubit power Rabi for scqo - supplies only ``probe()``.

Parameters, the cosine fit, opt_amp_prefactor extraction and the pi_amp writeback are
inherited from ``scqo.experiments.QubitPowerRabi``. scqo's ``amp_prefactor`` is already
a factor of the current pi pulse, which is exactly the QM builder's
``amplitude_scale``, so the sweep passes straight through — and since the probe emits
that same axis NAME, ``_to_canonical`` matches it by name instead of by position.

A COUPLER target (``drive_line`` + ``readout_member``; SCQO docs/coupler-transmon-plan.md)
runs the same program over a ``_mapped_target.MappedTarget``: its ``xy`` is the adopted
borrowed element, and its ``readout_state`` plays the map (the member's selective pi,
then its x180) before the member's own readout.
"""

from __future__ import annotations

from typing import Callable, Optional

import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array

from scqo_qm.experiments._amp_limits import check_amp_scale_window
from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    qubits,
    *,
    amps,
    operation: str,
    num_shots: int,
    reset_type: str,
    reset_max_attempts: int = 15,
    use_state_discrimination: bool,
    drive_qubit: Optional[str] = None,
    simulate: bool = False,
):
    """Build the power-Rabi QUA program. Returns (program, sweep_axes).

    `amps` is the amplitude pre-factor sweep (must be within [-2; 2), refused by
    name below); `qubits` is a BatchableList (see `_lib.select_qubits`).
    """
    check_amp_scale_window(amps, name=", ".join(qubits.get_names()))
    num_qubits = len(qubits)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "pulse amplitude prefactor"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(a, amps)):
                    # Qubit initialization
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(reset_type, simulate,
                                    max_attempts=reset_max_attempts)
                    align()

                    # Qubit manipulation: play the operation once (no error amplification).
                    # Drive only the selected qubit, or all qubits when drive_qubit is None.
                    for i, qubit in multiplexed_qubits.items():
                        if drive_qubit is None or qubit.name == drive_qubit:
                            qubit.xy.play(operation, amplitude_scale=a)
                    align()

                    # Qubit readout
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
                # Uniform save: always buffer over amps and average, regardless of the operation.
                if use_state_discrimination:
                    state_st[i].buffer(len(amps)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(amps)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(amps)).average().save(f"Q{i + 1}")

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

from scqo import register
from scqo.experiments import QubitPowerRabi


@register
class QMQubitPowerRabi(QubitPowerRabi):
    """Build a multiplexed power-Rabi QUA program on the QM OPX."""

    #: Readout is held at the calibrated point for the whole run and the reset is
    #: a genuine state reset, so reset_method='active' is valid here (_reset.py).
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.experiments._mapped_target import mapped_targets, routes_through

        machine = self.backend.machine  # type: ignore[attr-defined]
        if routes_through(self.params):
            # a target driven through a named line and/or read through a pair
            # member (a coupler: drive_line + readout_member) - one transmon-shaped
            # handle, so the program below is the qubits' own, unchanged
            qubits = mapped_targets(self, machine)
        else:
            qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        return build_program(
            machine,
            qubits,
            amps=self.sweep_axes["amp_prefactor"],
            operation="x180",
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            reset_max_attempts=reset_max_attempts(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            drive_qubit=None,
        )
