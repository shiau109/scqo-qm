"""Ramsey-vs-flux-pulse acquisition probe: vendor code only (qm/quam) - no scqo, no scqat.

Per shot: reset (+ frame reset) -> y90 at the idle point -> buffer -> a square
``const`` z pulse of relative amplitude ``a`` for the whole idle ``t``, with the
virtual phase ramp on the xy frame -> buffer -> x90 at the idle point -> readout at
the idle point. Loops: averages (outer) -> flux amplitude (in the order given) ->
idle (inner).

QM Ramsey vs flux PULSE for scqo - supplies only ``probe()``.

Parameters, the folding-safe sign of the virtual detuning, the fit and the park
writeback are inherited from ``scqo.experiments.QubitRamseyFluxPulse``.

PULSE CONTRACT: as in ``qubit_echo_flux_pulse`` - ``flux_amps_v`` are VOLTS measured
from the standing DC bias ``initialize_qpu`` applies, converted to an
``amplitude_scale`` of the element's ``const`` op, and rail-checked against
idle + excursion before any QUA is emitted (``_flux_limits``). The z pulse is the
square ``const``: a shaped op would zero-pad under ``duration=`` instead of
stretching. The ramp sign matches ``qubit_ramsey`` (the fringe sits at
``|D + (f_q - f_drive)|``), and D arrives SIGNED per target from the experiment.

Validated on 5Q4C q1 (2026-09-26) as a scratch prototype of this builder: against a
same-hour DC reference the pulse moves the qubit by g ~ 0.96 of the DC step with no
constant offset, and the phase stays linear in t over 4 us.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import xarray as xr
from qm.qua import *

from scqo_qm.experiments._flux_limits import check_flux_pulse_relative, idle_offset_v


def build_program(
    machine,
    qubits,
    *,
    flux_amps_v: Sequence[float],
    idle_cycles,
    ramp_detuning_hz: Mapping[str, float],
    buffer_cycles: int,
    num_shots: int,
    reset_type: str,
    reset_max_attempts: int = 15,
    use_state_discrimination: bool,
    flux_point: str = "joint",
    z_source: Optional[str] = None,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the Ramsey-vs-flux-pulse QUA program. Returns (program, sweep_axes).

    ``flux_amps_v``: the z-pulse amplitudes (V, relative to the standing bias), in
    the order to be swept. ``idle_cycles``: the idle = z-pulse lengths in clock
    cycles (4 ns, each >= 4). ``ramp_detuning_hz``: the SIGNED virtual detuning per
    qubit name. ``z_source``: a qubit name whose z line is pulsed INSTEAD of each
    target's own (crosstalk), or None.
    """
    flux_amps_v = [float(v) for v in flux_amps_v]
    idle_cycles = np.asarray(idle_cycles, dtype=int)
    if idle_cycles.min() < 4:
        raise ValueError(f"the shortest idle is {4 * idle_cycles.min()} ns; a z pulse "
                         f"plays for at least 16 ns (4 clock cycles)")
    if buffer_cycles and buffer_cycles < 4:
        raise ValueError(f"buffer of {4 * buffer_cycles} ns: a nonzero wait needs >= 16 ns")
    num_qubits = len(qubits)

    # every pulsed z element, rail-validated against idle + excursion
    if z_source is None:
        z_of = {}
        for qubit in qubits:
            z = getattr(qubit, "z", None)
            if z is None:
                raise ValueError(
                    f"{qubit.name}: no flux line, but this probe pulses every measured "
                    f"qubit's own z line - it cannot skip one and still report a flux axis")
            z_of[qubit.name] = z
    else:
        source = machine.qubits[z_source].z
        z_of = {qubit.name: source for qubit in qubits}
    amp_ref = {name: check_flux_pulse_relative(z, name=name if z_source is None else z_source,
                                               idle_v=idle_offset_v(z, flux_point),
                                               amps_v=flux_amps_v)
               for name, z in z_of.items()}
    ramp_coef = {name: -float(ramp_detuning_hz[name]) * 1e-9 for name in z_of}

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "flux_bias_v": xr.DataArray(
            np.asarray(flux_amps_v),
            attrs={"long_name": "flux pulse amplitude relative to the idle bias", "units": "V"}),
        "idle_time_ns": xr.DataArray(4 * idle_cycles, attrs={"long_name": "idle = z pulse length",
                                                             "units": "ns"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        t = declare(int)
        v = declare(fixed)
        phases = [declare(fixed) for _ in range(num_qubits)]
        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit, flux_point=flux_point)
            align()

            elements = sorted({e for q in multiplexed_qubits.values()
                               for e in (q.xy.name, q.z.name if getattr(q, "z", None) else None,
                                         q.resonator.name) if e}
                              | {z_of[q.name].name for q in multiplexed_qubits.values()})
            pulsed = {z_of[q.name].name: (z_of[q.name], amp_ref[q.name])
                      for q in multiplexed_qubits.values()}

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_each_(v, flux_amps_v):
                    with for_each_(t, idle_cycles):
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(reset_type, simulate, log_callable=log,
                                        max_attempts=reset_max_attempts)
                            reset_frame(qubit.xy.name)
                        align()
                        for i, qubit in multiplexed_qubits.items():
                            # same NEGATED ramp as qubit_ramsey: the fringe sits at
                            # |D + (f_q - f_drive)| with D signed per target
                            assign(phases[i], Cast.mul_fixed_by_int(ramp_coef[qubit.name], 4 * t))
                            qubit.xy.play("y90")
                        align(*elements)
                        if buffer_cycles:
                            wait(buffer_cycles, *elements)
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.frame_rotation_2pi(phases[i])
                        for z, ref in pulsed.values():
                            play("const" * amp(v / ref), z.name, duration=t)
                        align(*elements)
                        if buffer_cycles:
                            wait(buffer_cycles, *elements)
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("x90")
                        align(*elements)
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
                    state_st[i].buffer(len(idle_cycles)).buffer(len(flux_amps_v)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(idle_cycles)).buffer(len(flux_amps_v)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(idle_cycles)).buffer(len(flux_amps_v)).average().save(f"Q{i + 1}")

    return prog, sweep_axes


from typing import Any, ClassVar

from scqo import register
from scqo.experiments import QubitRamseyFluxPulse


@register
class QMQubitRamseyFluxPulse(QubitRamseyFluxPulse):
    """Build a multiplexed Ramsey-vs-flux-pulse QUA program on the QM OPX."""

    #: the same opt-in as qubit_ramsey: readout at the calibrated point, a genuine
    #: state reset, and the Ramsey is the sensitive test of the settle (_reset.py)
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from ._vendor import flux_source_name
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.quam_fields import GOVERNED_FLUX_POINT

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)
        # a foreign source is pulsed through machine.qubits[...].z; a coupler has no
        # such element here, so it is refused by name rather than failing in QUA
        z_source = (None if self.params.flux_component is None
                    else flux_source_name(self, self.params.flux_component, qubit_only=True))

        axes = self.sweep_axes or self.define_sweep()
        idle_cycles = np.round(np.asarray(axes["idle_time_ns"]) / 4).astype(int)
        return build_program(
            machine,
            qubits,
            flux_amps_v=list(axes["flux_bias_v"]),
            idle_cycles=idle_cycles,
            ramp_detuning_hz={q: self.ramp_detuning_hz(q) for q in self.params.targets},
            buffer_cycles=int(self.params.flux_buffer_ns) // 4,
            num_shots=int(self.params.num_averages),
            reset_type=check_reset_method(self),
            reset_max_attempts=reset_max_attempts(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            flux_point=GOVERNED_FLUX_POINT,
            z_source=z_source,
        )
