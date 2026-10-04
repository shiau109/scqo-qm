"""Phasor Ramsey acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Phase-tomography Ramsey: ``x90`` -> idle -> ``frame_rotation_2pi(frame)`` ->
``x90``, with the closing pulse's frame swept through a full turn at every idle
time. The fringe therefore lives in the FRAME axis, not the time axis.

NO VIRTUAL DETUNING. Plain ``qubit_ramsey`` ramps the frame proportionally to the
idle time to make the fringe oscillate in time, and that ramp has to be NEGATED
(a documented trap: get the sign wrong and every accepted update doubles the
residual detuning instead of cancelling it). Here the frame is the swept
tomography axis itself, so there is no ramp, no proportionality and no sign to
get wrong -- the accumulated phase is read out of the fringe's offset instead.

QM phasor Ramsey for scqo - supplies only ``probe()``. Parameters, the lock-in,
the stretched-exponential envelope fit and the writeback are all inherited from
``scqo.experiments.QubitRamseyPhasor``.
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
    frames,
    num_shots: int,
    reset_type: str,
    reset_max_attempts: int = 15,
    use_state_discrimination: bool,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the phasor-Ramsey QUA program. Returns (program, sweep_axes).

    ``idle_times_cycles`` is the LOG-spaced idle sweep in clock cycles (4 ns);
    ``frames`` is the closing-pulse phase axis in turns (endpoint-exclusive);
    ``qubits`` is a BatchableList (see ``_lib.select_qubits``).
    """
    # The QM clock resolves idle times to 4 ns and `wait` refuses fewer than 4
    # cycles. scqo's axis is already floored at 16 ns and snapped, but collapse
    # duplicates here too so the buffer length, the stream processing and the
    # dataset coordinate cannot disagree.
    idle_times_cycles = np.unique(np.asarray(idle_times_cycles))
    frames = np.asarray(frames, dtype=float)

    num_qubits = len(qubits)
    n_idle, n_frame = len(idle_times_cycles), len(frames)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time_ns": xr.DataArray(
            (4 * idle_times_cycles).astype(float),
            attrs={"long_name": "idle time", "units": "ns"}),
        "frame": xr.DataArray(
            frames, attrs={"long_name": "closing x90 frame rotation", "units": "turn"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        idle_time = declare(int)
        frame = declare(fixed)

        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)

                # idle OUTER, frame INNER -- the same nesting scqo's define_sweep
                # declares, so the stream buffers below match the axis order.
                with for_each_(idle_time, idle_times_cycles):
                    with for_each_(frame, frames):
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(reset_type, simulate, log_callable=log,
                                        max_attempts=reset_max_attempts)
                        align()
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("x90")
                            qubit.wait(idle_time)
                            # the tomography frame: a plain phase offset on the
                            # closing pulse, NOT a time-proportional ramp
                            qubit.xy.frame_rotation_2pi(frame)
                            qubit.xy.play("x90")
                            reset_frame(qubit.xy.name)
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
                # innermost axis buffers first
                if use_state_discrimination:
                    state_st[i].buffer(n_frame).buffer(n_idle).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(n_frame).buffer(n_idle).average().save(f"I{i + 1}")
                    Q_st[i].buffer(n_frame).buffer(n_idle).average().save(f"Q{i + 1}")

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
from scqo.experiments import QubitRamseyPhasor
from scqo.requirements import Requirement

from ._requires import X90_AMPLITUDE


@register
class QMQubitRamseyPhasor(QubitRamseyPhasor):
    """Build a multiplexed phasor-Ramsey QUA program on the QM OPX."""

    #: what only this backend consumes, on top of the neutral requirements
    requires: ClassVar[tuple[Requirement, ...]] = (
        *QubitRamseyPhasor.requires, X90_AMPLITUDE)
    #: true of THIS probe only; scqo shows them in `scqo run qubit_ramsey_phasor --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "both pi/2 pulses are the x90 operation, whose amplitude is pi_amp_x90",
        "the frame axis is a frame rotation applied before the second pulse and reset "
        "after it",
    )

    #: Readout is held at the calibrated point for the whole run and the reset is
    #: a genuine state reset, so reset_method='active' is valid here (_reset.py).
    #: Like plain Ramsey this is a sensitive test of the readout settle: residual
    #: photons Stark-shift the opening x90 and surface as a fitted-detuning error.
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        # scqo sweeps idle time in ns; the QUA program sweeps in clock cycles.
        # The floor is FOUR cycles, not one: `wait` under 4 cycles compiles here
        # and is refused only by the gateway compiler, which is how a latent
        # sub-16 ns wait once shipped undetected. scqo's axis is already floored
        # at 16 ns, so this only ever re-states that promise.
        idle_ns = self.sweep_axes["idle_time_ns"]
        idle_times_cycles = np.maximum(4, np.round(idle_ns / 4)).astype(int)

        return build_program(
            machine,
            qubits,
            idle_times_cycles=idle_times_cycles,
            frames=self.sweep_axes["frame"],
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            reset_max_attempts=reset_max_attempts(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
        )
