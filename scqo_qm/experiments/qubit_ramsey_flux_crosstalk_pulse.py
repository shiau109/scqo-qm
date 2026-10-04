"""Flux-crosstalk Ramsey acquisition probe: vendor code only (qm/quam) - no scqo, no scqat.

Per shot: reset (+ frame reset) -> y90 at the idle point -> buffer -> for the whole
idle ``t`` the target's own z line plays a square ``const`` of relative amplitude
``a`` AND the source flux line plays one of relative amplitude ``b``, with the
virtual phase ramp on the xy frame -> buffer -> x90 at the idle point -> readout at
the idle point. Loops: averages (outer) -> source amplitude -> own amplitude -> idle
(inner), each in the order given.

SOURCE LEAD: with ``source_lead_cycles`` the source pulse starts that long before the
own one and the two still end together. The source element then keeps its OWN clock
from the ``align()`` after the reset to the end of the pulses (it is left out of the
aligns in between), and the two sides are offset by plain waits: whichever side is
ahead waits the difference, and both carry a common ``_PAD_CYCLES`` so that neither
wait is shorter than QUA plays. Without a lead the program is the one above, unchanged.

QM flux-crosstalk Ramsey for scqo - supplies only ``probe()``.

Parameters, the folding-safe sign of the virtual detuning and the fit are inherited
from ``scqo.experiments.QubitRamseyFluxCrosstalkPulse``.

PULSE CONTRACT: as in ``qubit_ramsey_flux_pulse``, on BOTH lines - the swept values
are VOLTS measured from the standing DC bias ``initialize_qpu`` applies to that line,
converted to an ``amplitude_scale`` of the element's own ``const`` op, and
rail-checked against idle + excursion before any QUA is emitted (``_flux_limits``).
The source element is a qubit's ``z`` or a pair's ``TunableCoupler``; the two pulses
are square ``const`` (a shaped op would zero-pad under ``duration=`` instead of
stretching), start together and last the whole idle. The ramp sign matches
``qubit_ramsey`` (the fringe sits at ``|D + (f_q - f_drive)|``), and D arrives
SIGNED per target from the experiment.

AXIS-ORDER NOTE: the streams are buffered idle (inner), own flux, source (outer), so
the raw per-qubit array is (source, flux, idle) while scqo's contract lists the own
flux first. The sweep axes are returned under the canonical names IN RAW NESTING
ORDER; the backend then takes its name-based path and ``estimate()`` transposes by
name (the same arrangement as ``qubit_spectroscopy_flux_pulse``).
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import xarray as xr
from qm.qua import *

from scqo_qm.experiments._flux_limits import check_flux_pulse_relative, idle_offset_v

#: added to both sides' waits under a source lead: the shortest wait QUA plays
_PAD_CYCLES = 4


def build_program(
    machine,
    qubits,
    *,
    flux_amps_v: Sequence[float],
    source_amps_v: Sequence[float],
    source_z,
    source_name: str,
    source_idle_v: float,
    idle_cycles,
    ramp_detuning_hz: Mapping[str, float],
    buffer_cycles: int,
    source_lead_cycles: int = 0,
    num_shots: int,
    reset_type: str,
    reset_max_attempts: int = 15,
    use_state_discrimination: bool,
    flux_point: str = "joint",
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the flux-crosstalk Ramsey QUA program. Returns (program, sweep_axes).

    ``flux_amps_v``: the pulse amplitudes on each measured qubit's OWN z line (V,
    relative to its standing bias), in the order to be swept. ``source_amps_v``: the
    same for the source line. ``source_z``: the QUAM flux element pulsed alongside (a
    qubit's ``z`` or a ``TunableCoupler``), ``source_name`` its roster line (for the
    refusals) and ``source_idle_v`` the bias it stands at. ``idle_cycles``: the idle
    = pulse lengths in clock cycles (4 ns, each >= 4). ``ramp_detuning_hz``: the
    SIGNED virtual detuning per qubit name. ``source_lead_cycles``: how many clock
    cycles the source pulse is already on when the own pulse starts (0 = together).
    """
    flux_amps_v = [float(v) for v in flux_amps_v]
    source_amps_v = [float(v) for v in source_amps_v]
    idle_cycles = np.asarray(idle_cycles, dtype=int)
    if idle_cycles.min() < 4:
        raise ValueError(f"the shortest idle is {4 * idle_cycles.min()} ns; a z pulse "
                         f"plays for at least 16 ns (4 clock cycles)")
    if buffer_cycles and buffer_cycles < 4:
        raise ValueError(f"buffer of {4 * buffer_cycles} ns: a nonzero wait needs >= 16 ns")
    source_lead_cycles = int(source_lead_cycles)
    if source_lead_cycles < 0:
        raise ValueError(f"source lead of {4 * source_lead_cycles} ns: it cannot be negative")
    num_qubits = len(qubits)

    # both pulsed lines, each rail-validated against ITS idle + excursion
    own_ref = {}
    for qubit in qubits:
        z = getattr(qubit, "z", None)
        if z is None:
            raise ValueError(
                f"{qubit.name}: no flux line, but this probe locates the qubit's flux "
                f"apex on its own z line")
        if z is source_z:
            raise ValueError(
                f"{qubit.name}: the source line {source_name!r} is its own z line - "
                f"its response to that line is the arch itself, not a crosstalk")
        own_ref[qubit.name] = check_flux_pulse_relative(
            z, name=qubit.name, idle_v=idle_offset_v(z, flux_point), amps_v=flux_amps_v)
    source_ref = check_flux_pulse_relative(
        source_z, name=f"source line {source_name}", idle_v=float(source_idle_v),
        amps_v=source_amps_v)
    ramp_coef = {qubit.name: -float(ramp_detuning_hz[qubit.name]) * 1e-9 for qubit in qubits}
    source_wait = own_wait = 0
    if source_lead_cycles:
        # from the align() after the reset to the start of the own pulse, when nothing waits
        before_own = (max(int(q.xy.operations["y90"].length) for q in qubits) // 4
                      + buffer_cycles)
        source_wait = max(before_own - source_lead_cycles, 0) + _PAD_CYCLES
        own_wait = max(source_lead_cycles - before_own, 0) + _PAD_CYCLES

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "source_flux_v": xr.DataArray(
            np.asarray(source_amps_v),
            attrs={"long_name": "source-line pulse amplitude relative to its idle bias",
                   "units": "V"}),
        "flux_bias_v": xr.DataArray(
            np.asarray(flux_amps_v),
            attrs={"long_name": "own flux pulse amplitude relative to the idle bias",
                   "units": "V"}),
        "idle_time_ns": xr.DataArray(4 * idle_cycles, attrs={"long_name": "idle = pulse length",
                                                             "units": "ns"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        t = declare(int)
        a = declare(fixed)
        b = declare(fixed)
        phases = [declare(fixed) for _ in range(num_qubits)]
        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit, flux_point=flux_point)
            align()

            qubit_elements = sorted({e for q in multiplexed_qubits.values()
                                     for e in (q.xy.name, q.z.name, q.resonator.name)})
            # under a lead the source keeps its own clock between the outer aligns
            elements = (qubit_elements if source_lead_cycles
                        else sorted(set(qubit_elements) | {source_z.name}))

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_each_(b, source_amps_v):
                    with for_each_(a, flux_amps_v):
                        with for_each_(t, idle_cycles):
                            for i, qubit in multiplexed_qubits.items():
                                qubit.reset(reset_type, simulate, log_callable=log,
                                            max_attempts=reset_max_attempts)
                                reset_frame(qubit.xy.name)
                            align()
                            for i, qubit in multiplexed_qubits.items():
                                # same NEGATED ramp as qubit_ramsey: the fringe sits at
                                # |D + (f_q - f_drive)| with D signed per target
                                assign(phases[i],
                                       Cast.mul_fixed_by_int(ramp_coef[qubit.name], 4 * t))
                            if source_lead_cycles:
                                # the source pulse: on source_lead_cycles before the own
                                # one, off with it. align() again so the phase arithmetic
                                # above cannot sit between the two clocks
                                align()
                                wait(source_wait, source_z.name)
                                play("const" * amp(b / source_ref), source_z.name,
                                     duration=t + source_lead_cycles)
                                wait(own_wait, *qubit_elements)
                            for i, qubit in multiplexed_qubits.items():
                                qubit.xy.play("y90")
                            align(*elements)
                            if buffer_cycles:
                                wait(buffer_cycles, *elements)
                            for i, qubit in multiplexed_qubits.items():
                                qubit.xy.frame_rotation_2pi(phases[i])
                                play("const" * amp(a / own_ref[qubit.name]), qubit.z.name,
                                     duration=t)
                            if not source_lead_cycles:
                                # the source line, for the same idle: both start together
                                play("const" * amp(b / source_ref), source_z.name,
                                     duration=t)
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

        shape = (len(idle_cycles), len(flux_amps_v), len(source_amps_v))
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if use_state_discrimination:
                    (state_st[i].buffer(shape[0]).buffer(shape[1]).buffer(shape[2])
                     .average().save(f"state{i + 1}"))
                else:
                    (I_st[i].buffer(shape[0]).buffer(shape[1]).buffer(shape[2])
                     .average().save(f"I{i + 1}"))
                    (Q_st[i].buffer(shape[0]).buffer(shape[1]).buffer(shape[2])
                     .average().save(f"Q{i + 1}"))

    return prog, sweep_axes


from typing import Any

from scqo import register
from scqo.experiments import QubitRamseyFluxCrosstalkPulse


@register
class QMQubitRamseyFluxCrosstalkPulse(QubitRamseyFluxCrosstalkPulse):
    """Build the flux-crosstalk Ramsey QUA program on the QM OPX (one target).

    Thermal reset only: the active-reset opt-in is a hardware-gated decision per
    sequence (``_reset.py``), and this one has not been validated with it."""

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from ._vendor import flux_line_element
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.quam_fields import GOVERNED_FLUX_POINT

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)
        axes = self.sweep_axes or self.define_sweep()
        source_z, source_idle_v = flux_line_element(
            self, self.params.source_line, field="source_line",
            flux_point=GOVERNED_FLUX_POINT)
        idle_cycles = np.round(np.asarray(axes["idle_time_ns"]) / 4).astype(int)
        return build_program(
            machine,
            qubits,
            flux_amps_v=list(axes["flux_bias_v"]),
            source_amps_v=list(axes["source_flux_v"]),
            source_z=source_z,
            source_name=self.params.source_line,
            source_idle_v=source_idle_v,
            idle_cycles=idle_cycles,
            ramp_detuning_hz={q: self.ramp_detuning_hz(q) for q in self.params.targets},
            buffer_cycles=int(self.params.flux_buffer_ns) // 4,
            source_lead_cycles=int(self.params.source_lead_time_ns) // 4,
            num_shots=int(self.params.num_averages),
            reset_type=check_reset_method(self),
            reset_max_attempts=reset_max_attempts(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            flux_point=GOVERNED_FLUX_POINT,
        )
