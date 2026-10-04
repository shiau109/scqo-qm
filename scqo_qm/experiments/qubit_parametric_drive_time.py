"""Parametric-drive chevron (frequency x driving TIME) acquisition probe: vendor
code only (qm/quam) — no qualibrate, no scqo, no scqat.

Prepare the qubit (x180), then MODULATE its own flux (z) line with an RF tone —
the z element's oscillator is stepped to the swept drive frequency with
``update_frequency``, the IF phase is reset for shot-to-shot coherence, and the
z ``const`` operation is played at a FIXED, caller-given absolute amplitude for a
SWEPT duration — then read the qubit out. Where the modulation frequency matches
a sideband condition with a coupled component the excitation exchanges out of the
qubit and back, so each frequency row is an oscillating population trace and the
2D map over (frequency, time) is the chevron.

Sibling of ``qubit_parametric_drive_amp``, which fixes the driving time and
sweeps the AMPLITUDE. Everything the two share — the settle gap, the ns->cycles
guards, the z-oscillator config patch, the config-forwarding ``acquire`` and the
preview hook — lives in ``_parametric.py``; read its docstring for WHY these
probes carry their own config.

THE DURATION IS A REAL-TIME QUA VARIABLE, not a per-point compile: ``play()``
takes ``duration=`` in 4 ns clock cycles and accepts a swept QUA ``int``, so one
program covers the whole map (the same technique as ``pair_zz_coupler``'s coupler
pulse). scqo builds the neutral axis on that same 4 ns grid, so the conversion
here is an exact division and the stored axis IS what played — unlike the
neutral-grid time sweeps, this probe never re-declares the axis.

LOOP ORDER: frequency OUTER, time INNER. ``update_frequency`` then runs once per
frequency instead of once per point, and the inner (fast) axis is the one the
estimator reads as a trace.

Revived from the retired ``LCH_qubit_parametric_drive_freq_time`` probe (git
``cc04cec^``), modernized for scqo: EVERY selected qubit is driven (the node
drove only the first of the batch), the amplitude is absolute volts through the
shared rail + amplitude_scale + idle-sum guard ``check_flux_pulse_relative``, and
the axes carry the canonical scqo names.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array

from scqo_qm.experiments._flux_limits import (
    check_flux_pulse_relative,
    declared_idle_offset_v,
)
from scqo_qm.experiments._parametric import (
    PREP_SETTLE_NS,
    FluxOscillatorPreviewMixin,
    acquire,
    drive_time_cycles_array,
    ensure_flux_oscillators,
)


def build_program(
    machine,
    qubits,
    *,
    amp_v: float,
    freqs_hz,
    times_ns,
    num_shots: int,
    reset_type: str,
    use_state_discrimination: bool,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the parametric-drive chevron (fixed-amplitude, swept time) QUA
    program. Returns (program, sweep_axes).

    ``amp_v`` is the FIXED drive amplitude in ABSOLUTE volts — a z-pulse
    excursion riding on the standing bias ``initialize_qpu`` applied, validated
    per qubit by ``check_flux_pulse_relative`` (rail, amplitude_scale range and
    the idle + excursion sum) through a one-element window, so a fixed amplitude
    is gated exactly as the sibling's swept one. ``freqs_hz`` is the
    drive-frequency sweep in integer Hz (``update_frequency`` steps of 1 Hz);
    ``times_ns`` is the driving-time sweep in whole nanoseconds on the 4 ns grid.
    ``qubits`` is a BatchableList (see ``_lib.select_qubits``).
    """
    amp_v = float(amp_v)
    freqs_hz = np.asarray(freqs_hz).astype(int)
    times_ns = np.asarray(times_ns, dtype=float)
    cycles = drive_time_cycles_array(times_ns)
    num_qubits = len(qubits)

    # Per-qubit volts -> amplitude_scale denominators; the guard refuses a
    # missing/zero/over-rail `const`, an inexpressible scale, and an amplitude
    # that clips once it rides on the standing bias. One-element window: the
    # amplitude is fixed here, but the check is the sibling's.
    scales = {}
    for qubit in qubits:
        if getattr(qubit, "z", None) is None:
            raise ValueError(
                f"{qubit.name} has no z line; nothing to play the parametric "
                f"drive on.")
        ref = check_flux_pulse_relative(
            qubit.z, name=f"{qubit.name} parametric drive on {qubit.name}.z",
            idle_v=declared_idle_offset_v(qubit.z), amps_v=np.array([amp_v]),
            operation="const")
        # the amplitude is a python float, so the scale is too — no declare(fixed).
        scales[qubit.name] = amp_v / ref

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        # Outer loop -> y axis.
        "parametric_freq_hz": xr.DataArray(
            freqs_hz, attrs={"long_name": "parametric drive frequency", "units": "Hz"}),
        # Inner loop -> x axis.
        "drive_time_ns": xr.DataArray(
            times_ns, attrs={"long_name": "parametric driving time", "units": "ns"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        f_drive = declare(int)  # swept drive frequency (Hz, the z-line IF)
        t = declare(int)  # swept driving time (QUA clock cycles)
        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (the standing bias the
            # parametric excursion rides on).
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                # Frequency loop (outer -> y axis)
                with for_(*from_array(f_drive, freqs_hz)):
                    # Step the z oscillator ONCE per frequency, outside the
                    # duration loop -- the tone's frequency does not change
                    # along the inner axis.
                    for i, qubit in multiplexed_qubits.items():
                        qubit.z.update_frequency(f_drive)

                    # Duration loop (inner -> x axis)
                    with for_(*from_array(t, cycles)):
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(reset_type, simulate, log_callable=log)

                        # State prep: excite every driven qubit to |1>.
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("x180")
                        align()
                        wait(PREP_SETTLE_NS // 4)

                        # The parametric tone: the fixed volts on each qubit's
                        # own z line for the swept duration. reset_if_phase stays
                        # INSIDE the loop -- with the duration swept, the
                        # modulation phase at pulse start has to be reproducible
                        # shot to shot or the chevron washes out.
                        for i, qubit in multiplexed_qubits.items():
                            qubit.z.reset_if_phase()
                            qubit.z.play(
                                "const",
                                amplitude_scale=scales[qubit.name],
                                duration=t,
                            )
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
                # Inner buffer = driving time (x), next buffer = frequency (y).
                if use_state_discrimination:
                    state_st[i].buffer(len(cycles)).buffer(len(freqs_hz)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(cycles)).buffer(len(freqs_hz)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(cycles)).buffer(len(freqs_hz)).average().save(f"Q{i + 1}")

    return prog, sweep_axes


from functools import partial
from typing import Any, ClassVar

import numpy as np

from scqo import register
from scqo.experiments import QubitParametricDriveTime


@register
class QMQubitParametricDriveTime(FluxOscillatorPreviewMixin, QubitParametricDriveTime):
    """Build a multiplexed parametric-drive chevron QUA program on the QM OPX.

    Returns the 3-tuple probe shape ``(prog, sweep_axes, acquire)`` — not for
    heterogeneous streams (the fetch is the shared one) but to bind the
    oscillator-patched config into the acquire callable (``_parametric.py``).
    ``patch_preview_config`` is inherited from FluxOscillatorPreviewMixin."""

    #: true of THIS probe only; scqo shows them in
    #: `scqo run qubit_parametric_drive_time --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "the tone is the z line's const operation, played on an oscillator this probe "
        "adds to the generated configuration for the run; the stored state is unchanged",
        "a 200 ns wait separates the x180 from the tone",
        "the tone's length is played in whole 4 ns clock cycles, the grid the time axis is on",
        "an amplitude that, on top of the idle bias, takes the flux output beyond its "
        "range is refused",
    )

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        # update_frequency steps in whole Hz; re-declare the canonical axis as
        # what actually plays (the >= MHz-scale windows make rounding lossless).
        freqs = np.round(np.asarray(
            self.sweep_axes["parametric_freq_hz"], dtype=float)).astype(int)
        self.sweep_axes["parametric_freq_hz"] = freqs.astype(float)
        # The TIME axis is NOT re-declared: scqo builds it on the 4 ns grid, so
        # ns -> cycles is an exact division (module docstring).

        prog, sweep_axes = build_program(
            machine,
            qubits,
            amp_v=float(self.params.parametric_amp_v),
            freqs_hz=freqs,
            times_ns=np.asarray(self.sweep_axes["drive_time_ns"], dtype=float),
            num_shots=int(self.params.num_averages),
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
        )
        config = ensure_flux_oscillators(
            machine.generate_config(),
            [qubit.z.name for qubit in qubits],
            float(freqs[0]),
        )
        return prog, sweep_axes, partial(acquire, config=config)
