"""XY-Z delay acquisition probe: vendor code only (qm/quam) — no qualibrate, no
scqo, no scqat.

Adapted from the official node ``16a_xyz_delay``, vendored in this repo until
v3.13.0 (its baker lived in ``calibration_utils/xyz_delay/parameters.py``, which a
probe may not import, so the ~30-line bake is reimplemented here). A fixed XY ``x180`` and a same-length Z
(flux) rectangle are baked TOGETHER into one segment per relative shift: the Z
waveform slides by ``i`` ns while the XY samples stay centred, so running segment
``i`` plays the two pulses offset by ``i - half_scan_ns`` ns. Sweeping the two
initial preparations (|e> via x180, |g> via idle) gives the ``|e> - |g>`` contrast
whose triangle peak marks alignment.

PULSE CONTRACT: ``z_pulse_amp_v`` is the flux-pulse amplitude in VOLTS, played as
baked raw samples on TOP of the standing idle bias ``flux_point`` applies (the AWG
adds the waveform to the DC offset), so the DAC emits ``idle + z_pulse_amp_v`` and
that sum is rail-checked below. The XY ``x180`` must be an arbitrary-sample
waveform (e.g. DragCosine) — the baker reads its I/Q samples out of the generated
config.

The baker MUTATES the passed config in place (it files ``<element>_baked_pulse_*``
ops), so the program must execute against the returned ``baked_config``; pass it
to ``acquire(..., config=baked_config)``. A freshly generated config would lack
the baked ops.

QM XY-Z delay for scqo — supplies ``probe()``.

Parameters, the triangle-peak fit and the ``flux_delay_s`` writeback are inherited
from ``scqo.experiments.QubitXyzDelay``; this adapter only builds and runs the
baked QM program.

Like ``pair_swap_chevron`` (and unlike the other QM adapters), ``probe()``
ACQUIRES and returns a ready ``xr.Dataset``: the program only runs against the
probe's own baked config (which carries the per-segment ``x180`` + ``flux_pulse``
ops), and the backend's shared fetch path would regenerate a config without them.
The probe already labels its axes with the scqo names (``prepared_state`` /
``relative_time_ns``), so only ``qubit`` -> ``target`` is left to the backend's
canonicalization.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.bakery import baking

from scqo_qm.experiments._lib import acquire as _acquire
from scqo_qm.experiments._flux_limits import dac_rail_v, idle_offset_v, rail_remedy


def _baked_xy_z_segments(config: dict, z_waveform, qubit, zeros_each_side: int):
    """Bake the XY x180 + Z flux pulse together for every relative shift.

    ``config`` is mutated in place (one ``flux_pulse`` + ``x180`` op per segment).
    Returns a list of ``2 * zeros_each_side`` baking objects; segment ``i`` plays
    the Z pulse shifted ``i`` samples right of the fixed-centre XY pulse.
    """
    total = 2 * zeros_each_side
    x180_name = qubit.xy.operations["x180"].name
    i_key = f"{x180_name}.wf.I"
    q_key = f"{x180_name}.wf.Q"
    try:
        i_samples = list(config["waveforms"][i_key]["samples"])
        q_samples = list(config["waveforms"][q_key]["samples"])
    except (KeyError, TypeError) as err:
        raise RuntimeError(
            f"{qubit.name}: the x180 has no arbitrary-sample I/Q waveform in the "
            f"config ({err}); the XY-Z delay bake needs a sampled x180 (e.g. "
            f"DragCosine), not a constant/analytic pulse."
        )

    segments = []
    zeros = [0.0] * zeros_each_side
    for i in range(total):
        with baking(config, padding_method="symmetric_l") as b:
            wf = [0.0] * i + list(z_waveform) + [0.0] * (total - i)
            i_wf = zeros + i_samples + zeros
            q_wf = zeros + q_samples + zeros
            assert len(wf) == len(i_wf) == len(q_wf), (
                "Flux and XY padded waveforms must have identical length "
                "(the x180 length must equal its sampled-waveform length)"
            )
            b.add_op("flux_pulse", qubit.z.name, wf)
            b.add_op("x180", qubit.xy.name, [i_wf, q_wf])
            b.play("flux_pulse", qubit.z.name)
            b.play("x180", qubit.xy.name)
        segments.append(b)
    return segments


def build_program(
    machine,
    qubits,
    *,
    half_scan_ns: int,
    z_pulse_amp_v: float,
    num_shots: int,
    reset_type: str = "thermal",
    use_state_discrimination: bool = False,
    flux_point: str = "joint",
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the XY-Z delay QUA program. Returns ``(program, sweep_axes, baked_config)``.

    The returned ``baked_config`` carries the per-segment baked ops and MUST be the
    config used to execute (``acquire(..., config=baked_config)``).

    ``prepared_state`` is ``[0, 1]`` (0 = |g> idle, 1 = |e> x180); ``relative_time_ns``
    runs over ``[-half_scan_ns, half_scan_ns)`` at 1 ns resolution.
    """
    num_qubits = len(qubits)
    half = int(half_scan_ns)
    number_of_segments = 2 * half
    relative_time = np.arange(-half, half, 1)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "prepared_state": xr.DataArray(
            np.array([0, 1]),
            attrs={"long_name": "prepared state (0=g, 1=e)", "units": "state"},
        ),
        "relative_time_ns": xr.DataArray(
            relative_time,
            attrs={"long_name": "relative delay between XY and Z pulses", "units": "ns"},
        ),
    }

    baked_config = machine.generate_config()

    delay_segments = {}
    for qubit in qubits:
        z = getattr(qubit, "z", None)
        if z is None:
            raise ValueError(
                f"{qubit.name}: no flux line, but the XY-Z delay probe plays a Z "
                f"pulse on every measured qubit and cannot skip one."
            )
        # The baked Z samples ride on the standing idle bias, so the DAC emits the
        # SUM; clipping there is invisible to the simulator. (The pulse is baked
        # raw samples, not const*amplitude_scale, so the const-based
        # check_flux_pulse_relative does not apply — check the rail sum directly.)
        idle = idle_offset_v(z, flux_point)
        rail = dac_rail_v(z)
        total = abs(idle + float(z_pulse_amp_v))
        if total > rail:
            raise ValueError(
                f"{qubit.name}: the XY-Z Z pulse rides on the {idle} V idle bias; "
                f"idle + {z_pulse_amp_v} V = {total} V exceeds the port's {rail} V "
                f"full scale. The DAC would clip and the SIMULATOR WOULD NOT SHOW "
                f"IT. " + rail_remedy(z, name=f"{qubit.name} XY-Z Z pulse",
                                      needed_v=total, rail=rail))
        z_waveform = [float(z_pulse_amp_v)] * qubit.xy.operations["x180"].length
        delay_segments[qubit.name] = _baked_xy_z_segments(
            baked_config, z_waveform, qubit, half)

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        segment = declare(int)

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit, flux_point=flux_point)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)

                # prepared_state OUTER (python), relative-time segment inner —
                # matches the buffer nesting below.
                for prepared_state in (0, 1):
                    with for_(segment, 0, segment < number_of_segments, segment + 1):
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(reset_type, simulate, log_callable=log)
                            qubit.align()

                            # State preparation: |e> via x180, |g> waits the same time.
                            if prepared_state == 1:
                                qubit.xy.play("x180")
                            else:
                                qubit.xy.wait(qubit.xy.operations["x180"].length)
                            qubit.align()

                            # Coarse pre-wait (clock cycles) covering the leading
                            # padding before the fine baked scan. FLOORED at 4:
                            # QUA's wait minimum is 4 cycles and half_scan_ns < 16
                            # would emit wait(<4) — legal to the qm client and to
                            # generate_qua_script, refused ONLY by the gateway
                            # compiler (5Q4C 2026-08-09). The floor is timing-safe:
                            # this wait only positions the segment start; the XY-Z
                            # RELATIVE axis rides inside the baked segments.
                            qubit.wait(max(4, half // 4))
                            with switch_(segment):
                                for j in range(number_of_segments):
                                    with case_(j):
                                        delay_segments[qubit.name][j].run()
                            qubit.align()

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
                    state_st[i].buffer(number_of_segments).buffer(2).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(number_of_segments).buffer(2).average().save(f"I{i + 1}")
                    Q_st[i].buffer(number_of_segments).buffer(2).average().save(f"Q{i + 1}")

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
    """Connect, execute and fetch. Pass the baked config from ``build_program`` as
    ``config`` — the shared helper would otherwise regenerate one without the
    baked ops."""
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout, log=log, config=config)


from typing import Any

from scqo import register
from scqo.experiments import QubitXyzDelay


@register
class QMQubitXyzDelay(QubitXyzDelay):
    """Build, run and fetch the multiplexed XY-Z delay scan on the QM OPX."""

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = ("it bakes a per-call config for the slid pulses "
                           "and fetches against it inside probe()")

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.quam_fields import GOVERNED_FLUX_POINT
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        prog, sweep_axes, baked_config = build_program(
            machine,
            qubits,
            half_scan_ns=int(self.params.half_scan_ns),
            z_pulse_amp_v=float(self.params.z_pulse_amp_v),
            num_shots=int(self.params.num_averages),
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            flux_point=GOVERNED_FLUX_POINT,
        )
        # Acquire here: the baked config is per-call and cannot be reached through
        # the backend's (program, axes, module) shape.
        return acquire(
            machine, prog, sweep_axes,
            num_shots=int(self.params.num_averages),
            timeout=self.backend._timeout,  # type: ignore[attr-defined]
            config=baked_config,
        )
