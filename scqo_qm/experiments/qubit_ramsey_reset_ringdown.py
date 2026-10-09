"""Ramsey-after-the-reset acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqat.

Every idle time plays its ARMS back to back, each over one full turn of the
closing pulse's frame::

    photon     x180 -> reset tone -> y90 -> idle -> frame_rotation_2pi -> x90 -> readout
    reference  wait -> reset tone -> y90 -> idle -> frame_rotation_2pi -> x90 -> readout
    baseline   wait ->    wait    -> y90 -> idle -> frame_rotation_2pi -> x90 -> readout

The reset tone is the target's stored macro (``reset_operation``, a
``ParametricReset``), played with its ring-down forced to 0: the wait is what
the run measures, so whatever is stored must not be in front of the first pi/2
pulse. The macro still returns its z line's oscillator to the stored IF.

THE ARMS ARE A PYTHON LOOP, not a QUA branch. A real-time ``if_`` on an arm
index would put FPGA arithmetic between the tone and the first pi/2 pulse, in
exactly the window being measured, and not the same amount in every arm. Unrolled
in Python each arm is its own straight-line block; the stream sees them in
order (idle, arm, frame), which is the order the buffers below fold.

An arm that does not play a pulse WAITS its length on the same line, so the
three arms share one timeline: the photon arm differs from the reference by the
x180 alone, and the reference from the baseline by the tone alone.

QM Ramsey after the reset for scqo - supplies only ``probe()``. Parameters, the
lock-in, the single-photon fit and the reported wait are inherited from
``scqo.experiments.QubitRamseyResetRingdown``.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import xarray as xr
from qm.qua import *

from scqo_qm.experiments._lib import acquire as _acquire

PHOTON_ARM, REFERENCE_ARM, BASELINE_ARM = "photon", "reference", "baseline"

#: QUA's floor for ``wait()``, in ns, and its clock. A shorter wait is a compile
#: error naming an internal variable, so the two fixed waits are checked by name.
MIN_WAIT_NS = 16
CLOCK_NS = 4

#: the xy operations the sequence plays
PREP, OPENING, CLOSING = "x180", "y90", "x90"


def _wait_cycles(length_ns, what: str) -> int:
    """A stored pulse length as a wait in clock cycles, or refuse BY NAME."""
    length_ns = int(length_ns)
    if length_ns < MIN_WAIT_NS or length_ns % CLOCK_NS:
        raise ValueError(
            f"{what} is {length_ns} ns long; an arm that leaves it out has to wait "
            f"exactly as long, and QUA waits in {CLOCK_NS} ns steps from "
            f"{MIN_WAIT_NS} ns up.")
    return length_ns // CLOCK_NS


def check_reset_tone(qubit, reset_operation: str):
    """The qubit's stored reset macro, or refuse BY NAME before any QUA is built."""
    macros = getattr(qubit, "macros", {}) or {}
    if reset_operation not in macros:
        raise ValueError(
            f"Qubit {qubit.name} has no macro {reset_operation!r}; available: "
            f"{sorted(macros)}. Store the reset tone first "
            f"(quam_config/register_reset_macro.py; SCQO "
            f"procedures/qubit-resonator-sideband-swap, Step 5).")
    macro = macros[reset_operation]
    if not hasattr(macro, "ringdown_ns") or not hasattr(macro, "flux_pulse"):
        raise ValueError(
            f"Macro {reset_operation!r} on {qubit.name} is a "
            f"{type(macro).__name__}, not a parametric reset: this experiment plays "
            f"the reset with its ring-down wait forced to 0, which only a "
            f"ParametricReset can do.")
    if macro.flux_pulse not in qubit.z.operations:
        raise ValueError(
            f"Macro {reset_operation!r} on {qubit.name} plays the z operation "
            f"{macro.flux_pulse!r}, which {qubit.z.name} does not carry.")
    return macro


def build_program(
    machine,
    qubits,
    *,
    idle_times_cycles,
    arms: Sequence[str],
    frames,
    num_shots: int,
    reset_type: str,
    reset_max_attempts: int = 15,
    reset_operation: str = "reset",
    use_state_discrimination: bool,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the Ramsey-after-the-reset QUA program. Returns (program, sweep_axes).

    ``idle_times_cycles`` is the idle sweep in clock cycles (4 ns); ``arms`` the
    arm labels in play order; ``frames`` the closing-pulse phase axis in turns
    (endpoint-exclusive); ``qubits`` a BatchableList (see ``_lib.select_qubits``).
    """
    idle_times_cycles = np.unique(np.asarray(idle_times_cycles))
    frames = np.asarray(frames, dtype=float)
    arms = [str(arm) for arm in arms]
    unknown = sorted(set(arms) - {PHOTON_ARM, REFERENCE_ARM, BASELINE_ARM})
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; expected photon, reference, baseline")

    # ---- everything that can refuse, before a single QUA statement ----------
    macros, prep_cycles, tone_cycles = {}, {}, {}
    for qubit in qubits:
        for operation in (PREP, OPENING, CLOSING):
            if operation not in qubit.xy.operations:
                raise ValueError(
                    f"Qubit {qubit.name} has no xy operation {operation!r}; "
                    f"available: {sorted(qubit.xy.operations)}.")
        macros[qubit.name] = check_reset_tone(qubit, reset_operation)
        prep_cycles[qubit.name] = _wait_cycles(
            qubit.xy.operations[PREP].length, f"{qubit.name}'s {PREP}")
        tone_cycles[qubit.name] = _wait_cycles(
            qubit.z.operations[macros[qubit.name].flux_pulse].length,
            f"{qubit.name}'s reset tone")

    num_qubits = len(qubits)
    n_idle, n_arm, n_frame = len(idle_times_cycles), len(arms), len(frames)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time_ns": xr.DataArray(
            (CLOCK_NS * idle_times_cycles).astype(float),
            attrs={"long_name": "idle time", "units": "ns"}),
        "arm": xr.DataArray(arms, attrs={"long_name": "arm"}),
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

                # idle OUTER, arm, frame INNER -- the nesting scqo's define_sweep
                # declares, so the stream buffers below match the axis order.
                with for_each_(idle_time, idle_times_cycles):
                    for arm in arms:  # unrolled: see the module docstring
                        with for_each_(frame, frames):
                            for i, qubit in multiplexed_qubits.items():
                                qubit.reset(reset_type, simulate, log_callable=log,
                                            max_attempts=reset_max_attempts)
                            align()
                            for i, qubit in multiplexed_qubits.items():
                                if arm == PHOTON_ARM:
                                    qubit.xy.play(PREP)
                                else:
                                    qubit.xy.wait(prep_cycles[qubit.name])
                                qubit.align()
                                if arm == BASELINE_ARM:
                                    qubit.z.wait(tone_cycles[qubit.name])
                                else:
                                    # the wait is what is being measured: none
                                    macros[qubit.name].apply(ringdown_ns=0)
                                qubit.align()
                                qubit.xy.play(OPENING)
                                qubit.wait(idle_time)
                                # the tomography frame: a plain phase offset on
                                # the closing pulse
                                qubit.xy.frame_rotation_2pi(frame)
                                qubit.xy.play(CLOSING)
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
                    (state_st[i].buffer(n_frame).buffer(n_arm).buffer(n_idle)
                     .average().save(f"state{i + 1}"))
                else:
                    (I_st[i].buffer(n_frame).buffer(n_arm).buffer(n_idle)
                     .average().save(f"I{i + 1}"))
                    (Q_st[i].buffer(n_frame).buffer(n_arm).buffer(n_idle)
                     .average().save(f"Q{i + 1}"))

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
from scqo.experiments import QubitRamseyResetRingdown
from scqo.requirements import Requirement

from ._requires import X90_AMPLITUDE


@register
class QMQubitRamseyResetRingdown(QubitRamseyResetRingdown):
    """Build the Ramsey-after-the-reset QUA program on the QM OPX."""

    #: what only this backend consumes, on top of the neutral requirements
    requires: ClassVar[tuple[Requirement, ...]] = (
        *QubitRamseyResetRingdown.requires, X90_AMPLITUDE)
    #: true of THIS probe only; scqo shows them in
    #: `scqo run qubit_ramsey_reset_ringdown --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "the reset is the QUAM macro reset_operation names on the target, a "
        "ParametricReset; it is played with its ring-down forced to 0, whatever is "
        "stored, and a missing or different macro is refused by name",
        "an arm that leaves a pulse out waits that pulse's stored length on the same "
        "line: the x180 on the drive, the reset tone on the flux line",
        "the opening pi/2 pulse is y90 and the closing one x90, both at pi_amp_x90; the "
        "frame axis is a frame rotation applied before the closing pulse and reset "
        "after it",
        "the arms of one idle time are separate straight-line blocks of the program, "
        "not a real-time branch, so nothing is computed between the tone and the "
        "opening pulse",
        "targets are measured one at a time",
        "not yet run on the instrument: where the opening pulse starts against the end "
        "of the tone has not been looked at, on the simulator or on a scope",
    )

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        # one at a time: a neighbour's reset tone during this target's Ramsey is
        # a second thing the ratio would have to be blind to
        qubits = select_qubits(machine, self.params.targets, multiplexed=False)

        # scqo sweeps idle time in ns on the 4 ns grid, floored at 16 ns; the QUA
        # program sweeps in clock cycles. The floor is restated here because a
        # `wait` under 4 cycles is refused only by the gateway compiler.
        idle_ns = self.sweep_axes["idle_time_ns"]
        idle_times_cycles = np.maximum(4, np.round(idle_ns / CLOCK_NS)).astype(int)

        return build_program(
            machine,
            qubits,
            idle_times_cycles=idle_times_cycles,
            arms=[str(arm) for arm in self.sweep_axes["arm"]],
            frames=self.sweep_axes["frame"],
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            reset_max_attempts=reset_max_attempts(self),
            reset_operation=self.params.reset_operation,
            use_state_discrimination=bool(self.params.use_state_discrimination),
        )
