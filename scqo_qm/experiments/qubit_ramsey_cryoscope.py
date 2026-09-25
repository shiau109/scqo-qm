"""Ramsey cryoscope acquisition probe: vendor code only (qm/quam) — no qualibrate, no scqo.

Flux-line step response via Ramsey phase tomography. Sequence per (duration,
frame): ``x90`` -> flux pulse of ``duration`` ns -> fixed total idle ->
``frame_rotation_2pi(frame)`` -> ``x90`` -> readout. The flux pulse's DURATION is
swept at 1 ns resolution (baked <=16 ns, a stretched ``const`` plus a baked
remainder above), and the closing pulse's FRAME is swept through a full turn so
the accumulated-phase fringe can be reconstructed per duration.

PULSE CONTRACT: ``flux_amp_v`` is the flux-pulse amplitude in VOLTS measured from
the standing DC bias that ``initialize_qpu`` applies — the pulse rides on that
offset (baked segments and the stretched ``const`` both emit ``flux_amp_v`` volts
ON TOP of the idle bias), so the amplitude is idle-relative. ``flux_point``
selects which named offset stands and is passed EXPLICITLY, because the DAC emits
``idle + excursion`` and the number validated against the rail must be the number
that plays.

PHASE HANDEDNESS: the frame sweep is phase TOMOGRAPHY (a full turn of the closing
pulse), not a virtual detuning — its sign only sets whether the reconstructed
phase runs +phi or -phi with increasing frame, and the estimator removes any
constant offset/sign in its ``unwrap`` + derivative + ``sqrt|.|`` chain. So,
unlike the Ramsey virtual-detuning ramp, no cross-repo sign convention is coupled
here; a plain ``frame_rotation_2pi(frame)`` is correct.

Single qubit only: the sequence is built per qubit (baked ops are filed on one z
element and the fixed idle is derived from one duration axis); run targets one at
a time.

QM ramsey cryoscope for scqo — supplies only ``probe()``.

Parameters, the step-response estimator and the paired-fact writeback are
inherited from ``scqo.experiments.QubitRamseyCryoscope``. scqo sweeps the flux-pulse
DURATION (``duration_ns``, every ns) x the closing-pulse FRAME (turns); the QM
probe realizes the 1 ns duration resolution with baking (<=16 ns baked, a
stretched ``const`` plus a baked remainder above) and plays the flux pulse
idle-relative at the governed flux point.

Like the swap chevron, ``probe()`` ACQUIRES and returns a ready ``xr.Dataset``:
the program only runs against the probe's own baked config (which carries the
``flux_pulse1..16`` operations), and the backend's shared fetch path would
regenerate a config without them. The probe's ``sweep_axes`` are already the
canonical scqo names in raw nesting order (``duration_ns`` outer, ``frame``
inner), so ``_to_canonical`` takes its name-based path.

Single target: the probe refuses more than one qubit (the sequence is built per
qubit); reset is resolved through ``_reset.check_reset_method`` so ``reset_method
="active"`` is refused by name rather than silently thermalized.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.bakery import baking

from scqo_qm.experiments._flux_limits import check_flux_pulse_relative, idle_offset_v
from scqo_qm.experiments._lib import acquire as _acquire


def baked_waveform(baked_config, waveform_amp: float, qubit, max_length: int = 16):
    """Bake the sub-16-ns flux segments at ABSOLUTE volts ``waveform_amp``.

    Element ``i-1`` is an ``i`` ns constant pulse at ``waveform_amp`` volts on
    ``qubit.z``, registered as op ``flux_pulse{i}``. Mutates ``baked_config`` —
    that config is the one that must be executed (pass it to
    ``acquire(..., config=baked_config)``). Reimplemented from
    ``calibration_utils.cryoscope.parameters.baked_waveform``, which this repo
    vendored until v3.13.0, so this probe stays free of qualibrate imports.
    """
    pulse_segments = []
    waveform = [waveform_amp] * max_length
    for i in range(1, max_length + 1):
        with baking(baked_config, padding_method="right") as b:
            b.add_op(f"flux_pulse{i}", qubit.z.name, waveform[:i])
            b.play(f"flux_pulse{i}", qubit.z.name)
        pulse_segments.append(b)
    return pulse_segments


def validate_inputs(qubits, durations_ns, flux_amp_v: float, flux_point: str) -> float:
    """Pure pre-flight checks (no QUA); return the flux ``const`` reference amplitude.

    Refuses, by name and before any hardware time:
    * more than one target — the sequence is built per qubit;
    * a duration axis that is not every nanosecond ``1..max`` — the QUA loop
      enumerates each ns and the Savitzky-Golay derivative assumes a uniform step;
    * a flux pulse whose ``idle + flux_amp_v`` clips the port, or that needs an
      ``amplitude_scale`` QUA cannot express (:func:`check_flux_pulse_relative`).
    """
    if len(qubits) != 1:
        names = list(qubits.get_names())
        raise ValueError(
            f"qubit_ramsey_cryoscope builds its phase-tomography sequence per qubit and "
            f"supports one target at a time, got {len(qubits)}: {names}. "
            f"Run them one at a time.")
    durations_ns = np.asarray(durations_ns)
    expected = np.arange(1, int(durations_ns[-1]) + 1)
    if durations_ns.size != expected.size or not np.array_equal(
            durations_ns.astype(int), expected):
        raise ValueError(
            "qubit_ramsey_cryoscope sweeps EVERY nanosecond 1..max (the QUA loop "
            "enumerates each ns and the derivative assumes a uniform 1 ns step); "
            f"got a {durations_ns.size}-point axis that is not arange(1, "
            f"{int(durations_ns[-1]) + 1}).")

    qubit = qubits[0]
    z = getattr(qubit, "z", None)
    if z is None:
        raise ValueError(
            f"{qubit.name}: no flux line, but the ramsey cryoscope plays a z pulse — it "
            f"cannot measure a flux step response on a qubit with no z channel.")
    return check_flux_pulse_relative(
        z, name=f"{qubit.name} ramsey cryoscope flux pulse",
        idle_v=idle_offset_v(z, flux_point), amps_v=[float(flux_amp_v)])


def build_program(
    machine,
    qubits,
    *,
    durations_ns,
    frames,
    flux_amp_v: float,
    num_shots: int,
    reset_type: str = "thermal",
    use_state_discrimination: bool = False,
    simulate: bool = False,
    flux_point: str = "joint",
    log: Optional[Callable] = None,
):
    """Build the cryoscope QUA program. Returns ``(program, sweep_axes, baked_config)``.

    The returned ``baked_config`` carries the baked ``flux_pulse1..16`` operations
    and MUST be the config used to execute (pass it to
    ``acquire(..., config=baked_config)``); a freshly generated config would lack
    them. ``durations_ns`` is every nanosecond ``1..max``; ``frames`` is the
    closing-pulse phase sweep in turns; ``flux_amp_v`` is the idle-relative pulse
    amplitude in volts (see the module docstring).
    """
    amp_ref = validate_inputs(qubits, durations_ns, flux_amp_v, flux_point)
    qubit = qubits[0]
    durations_ns = np.asarray(durations_ns).astype(int)
    frames = np.asarray(frames, dtype=float)
    max_len = int(durations_ns[-1])

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "duration_ns": xr.DataArray(
            durations_ns.astype(float),
            attrs={"long_name": "flux pulse duration", "units": "ns"}),
        "frame": xr.DataArray(
            frames, attrs={"long_name": "second x90 frame rotation", "units": "turn"}),
    }

    baked_config = machine.generate_config()
    # Bake the sub-16-ns segments at the absolute (idle-relative) excursion volts.
    baked_signals = baked_waveform(baked_config, float(flux_amp_v), qubit, max_length=16)
    # Volts -> amplitude_scale for the stretched (>16 ns) branch.
    const_scale = float(flux_amp_v) / amp_ref
    # cycles the z line idles while the first x90 (+16 ns buffer) plays, and the
    # fixed total idle the xy line holds so the 2nd x90 always lands after the
    # LONGEST flux pulse (constant free-evolution time -> constant T2 decay).
    x90_wait_cycles = (qubit.xy.operations["x90"].length + 16) // 4
    idle_wait_cycles = (max_len + 16) >> 2

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        idx = declare(int)
        frame = declare(fixed)
        t_left_ns = declare(int)
        t_cycles = declare(int)
        if use_state_discrimination:
            state = declare(int)
            state_st = declare_stream()

        machine.initialize_qpu(target=qubit, flux_point=flux_point)
        align()

        with for_(n, 0, n < num_shots, n + 1):
            save(n, n_st)
            with for_(idx, 1, idx <= max_len, idx + 1):
                with for_each_(frame, frames):
                    qubit.reset(reset_type, simulate, log_callable=log)
                    align()
                    # <=16 ns: baked segments exclusively (1 ns granularity).
                    with if_(idx <= 16):
                        with switch_(idx):
                            for j in range(1, 17):
                                with case_(j):
                                    align()
                                    qubit.xy.play("x90")
                                    qubit.z.wait(x90_wait_cycles)
                                    baked_signals[j - 1].run()
                                    qubit.xy.wait(idle_wait_cycles)
                                    qubit.xy.frame_rotation_2pi(frame)
                                    qubit.xy.play("x90")
                                    reset_frame(qubit.xy.name)
                    # >16 ns: a stretched `const` (multiple of 4 ns) plus a baked
                    # remainder for the leftover 1-3 ns.
                    with else_():
                        assign(t_cycles, idx >> 2)
                        assign(t_left_ns, idx - (t_cycles << 2))
                        with switch_(t_left_ns):
                            with case_(0):
                                align()
                                qubit.xy.play("x90")
                                qubit.z.wait(x90_wait_cycles)
                                qubit.z.play("const", duration=t_cycles,
                                             amplitude_scale=const_scale)
                                qubit.xy.wait(idle_wait_cycles)
                                qubit.xy.frame_rotation_2pi(frame)
                                qubit.xy.play("x90")
                                reset_frame(qubit.xy.name)
                            for j in range(1, 4):
                                with case_(j):
                                    align()
                                    qubit.xy.play("x90")
                                    qubit.z.wait(x90_wait_cycles)
                                    # keep the const and its baked tail gapless
                                    with strict_timing_():
                                        qubit.z.play("const", duration=t_cycles,
                                                     amplitude_scale=const_scale)
                                        baked_signals[j - 1].run()
                                    qubit.xy.wait(idle_wait_cycles)
                                    qubit.xy.frame_rotation_2pi(frame)
                                    qubit.xy.play("x90")
                                    reset_frame(qubit.xy.name)
                    align()
                    if use_state_discrimination:
                        qubit.readout_state(state)
                        save(state, state_st)
                    else:
                        qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                        save(I[0], I_st[0])
                        save(Q[0], Q_st[0])

        with stream_processing():
            n_st.save("n")
            if use_state_discrimination:
                state_st.buffer(len(frames)).buffer(max_len).average().save("state1")
            else:
                I_st[0].buffer(len(frames)).buffer(max_len).average().save("I1")
                Q_st[0].buffer(len(frames)).buffer(max_len).average().save("Q1")

    return prog, sweep_axes, baked_config


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_shots: int,
    timeout: float,
    log: Optional[Callable] = None,
    config: Optional[dict] = None,
) -> xr.Dataset:
    """Connect, execute and fetch. ``config`` MUST be the baked config from
    :func:`build_program` (it carries the baked flux-pulse ops)."""
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots,
                    timeout=timeout, log=log, config=config)


from typing import Any

import numpy as np
from scqo import register
from scqo.experiments import QubitRamseyCryoscope


@register
class QMQubitRamseyCryoscope(QubitRamseyCryoscope):
    """Build, run and fetch the ramsey cryoscope phase-tomography sequence on the QM OPX."""

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = ("it bakes a per-call config for the phase "
                           "tomography and fetches against it inside probe()")

    def probe(self) -> Any:
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.quam_fields import GOVERNED_FLUX_POINT

        from ._reset import check_reset_method

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)
        durations_ns = np.round(self.sweep_axes["duration_ns"]).astype(int)

        prog, sweep_axes, baked_config = build_program(
            machine,
            qubits,
            durations_ns=durations_ns,
            frames=self.sweep_axes["frame"],
            flux_amp_v=float(self.params.flux_pulse_amp_v),
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            flux_point=GOVERNED_FLUX_POINT,
        )
        # Acquire here: the baked config is per-call and cannot be reached through
        # the backend's (program, axes, module) shape.
        return acquire(
            machine, prog, sweep_axes,
            num_shots=int(self.params.num_averages),
            timeout=self.backend._timeout,
            config=baked_config,
        )
