"""Coupler-crossing acquisition probe: vendor code only (qm/quam) - no scqo, no scqat.

Per shot: reset both members -> align -> the coupler plays its square ``const`` at
``amplitude_scale = b / ref`` for ``2 * buffer + x180`` -> each measured member
waits ``buffer`` and plays its ``x180`` (at its own drive frequency) -> align (the
coupler is back at idle) -> both members are read out 2-level -> the four joint
indicators are saved. Loops: averages (outer) -> ``b`` (inner, in the order given).

QM coupler-crossing scan for scqo - supplies ``probe()`` + the joint-population
reduction.

Parameters, the crossing / arch fit and the writeback (coupler ``flux_offset``,
``flux_per_phi0``, ``f_q_max_hz``; never the coupler's ``idle_flux``) are inherited
from ``scqo.experiments.PairCouplerCrossingPulse``.

PULSE CONTRACT: the swept values are VOLTS on the coupler line measured from the
standing bias ``initialize_qpu`` applies (for a coupler declared "off", its
``decouple_offset``). They become an ``amplitude_scale`` of the coupler's ``const``
op IN PYTHON, so the loop variable IS the scale and nothing is computed in front of
the play; the rail, the ``amplitude_scale`` bound and the idle + excursion sum are
checked before any QUA is built (``_flux_limits``). ``const`` is square, so
``duration=`` stretches it rather than zero-padding a shaped edge.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array

from scqo_qm.experiments._flux_limits import check_flux_pulse_relative, declared_idle_offset_v
from scqo_qm.experiments._lib import acquire as _acquire

#: QM's clock (ns) and the shortest play/wait it takes (4 cycles)
_CLOCK_NS = 4
_MIN_CYCLES = 4


def coupler_window_cycles(x180_lengths_ns: Sequence[float], buffer_ns: int) -> int:
    """The coupler pulse length in clock cycles: both buffers plus the longest x180,
    rounded UP to the clock (never shorter than the x180s it has to cover)."""
    total = 2 * int(buffer_ns) + max(float(v) for v in x180_lengths_ns)
    return max(_MIN_CYCLES, int(math.ceil(total / _CLOCK_NS)))


def build_program(
    machine,
    qubit_pair,
    *,
    coupler_amps_v: Sequence[float],
    measure_roles: Sequence[str],
    buffer_ns: int,
    num_shots: int,
    reset_type: str,
    simulate: bool = False,
):
    """Build the coupler-crossing QUA program for ONE pair. Returns (program, sweep_axes).

    ``coupler_amps_v``: the coupler pulse amplitudes (V, relative to its standing
    bias), in the order to be swept. ``measure_roles``: which vendor sides play the
    x180 - a subset of ``("control", "target")``; both are always read out.
    ``buffer_ns``: 0, or a multiple of 4 ns from 16 ns up.
    """
    qp = qubit_pair
    coupler = getattr(qp, "coupler", None)
    if coupler is None:
        raise ValueError(f"Qubit pair {qp.name} has no coupler; this probe pulses one.")
    if not measure_roles or set(measure_roles) - {"control", "target"}:
        raise ValueError(f"measure_roles must be a non-empty subset of control/target, "
                         f"got {list(measure_roles)}")
    if buffer_ns and (buffer_ns < _MIN_CYCLES * _CLOCK_NS or buffer_ns % _CLOCK_NS):
        raise ValueError(f"buffer of {buffer_ns} ns: use 0, or a multiple of 4 ns from 16 ns up")

    amps = np.asarray(coupler_amps_v, dtype=float)
    ref = check_flux_pulse_relative(
        coupler, name=f"{qp.name} coupler", idle_v=declared_idle_offset_v(coupler),
        amps_v=amps, operation="const")
    scales = amps / ref  # volts -> amplitude_scale here, never in front of the play

    members = {"control": qp.qubit_control, "target": qp.qubit_target}
    driven = [members[r] for r in ("control", "target") if r in measure_roles]
    lengths = [float(q.xy.operations["x180"].length) for q in driven]
    window = coupler_window_cycles(lengths, buffer_ns)
    buffer_cycles = int(buffer_ns) // _CLOCK_NS

    sweep_axes = {
        "qubit_pair": xr.DataArray([qp.name]),
        "coupler_amplitude": xr.DataArray(
            amps, attrs={"long_name": "coupler flux-pulse amplitude (relative to idle)",
                         "units": "V"}),
    }

    with program() as prog:
        scale = declare(fixed)
        n = declare(int)
        n_st = declare_stream()
        state_c = declare(int)
        state_t = declare(int)
        ind_gg, ind_ge, ind_eg, ind_ee = (declare(int) for _ in range(4))
        st_gg, st_ge, st_eg, st_ee = (declare_stream() for _ in range(4))

        machine.initialize_qpu(target=qp.qubit_control)
        machine.initialize_qpu(target=qp.qubit_target)
        align()
        with for_(n, 0, n < num_shots, n + 1):
            save(n, n_st)
            with for_(*from_array(scale, scales)):
                qp.qubit_control.reset(reset_type, simulate)
                qp.qubit_target.reset(reset_type, simulate)
                align()
                # the coupler sits at b for the whole window; the x180s land inside it
                coupler.play("const", amplitude_scale=scale, duration=window)
                for qubit in driven:
                    if buffer_cycles:
                        qubit.xy.wait(buffer_cycles)
                    qubit.xy.play("x180")
                align()
                # the coupler is back at idle: read both members, 2-level
                qp.qubit_control.readout_state(state_c)
                qp.qubit_target.readout_state(state_t)
                # joint indicators, first digit = control:
                #   ee(11)=c*t, eg(10)=c-ee, ge(01)=t-ee, gg(00)=1-c-t+ee
                assign(ind_ee, state_c * state_t)
                assign(ind_eg, state_c - ind_ee)
                assign(ind_ge, state_t - ind_ee)
                assign(ind_gg, 1 - state_c - state_t + ind_ee)
                save(ind_gg, st_gg)
                save(ind_ge, st_ge)
                save(ind_eg, st_eg)
                save(ind_ee, st_ee)

        with stream_processing():
            n_st.save("n")
            for stream, name in ((st_gg, "state_gg"), (st_ge, "state_ge"),
                                 (st_eg, "state_eg"), (st_ee, "state_ee")):
                stream.buffer(len(amps)).average().save(f"{name}1")

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


from typing import Any

from scqo import register
from scqo.experiments import PairCouplerCrossingPulse

from ._pair_roles import JointPopulationMixin


def _missing_thresholds(experiment: Any, members: Sequence[str]) -> list[str]:
    """Members whose governed ``readout_threshold`` is unset - ``readout_state``
    would compare against nothing."""
    missing = []
    for member in members:
        try:
            value = experiment.device.channel(member, "readout").readout_threshold
        except KeyError:
            value = None
        if value is None:
            missing.append(member)
    return missing


@register
class QMPairCouplerCrossingPulse(JointPopulationMixin, PairCouplerCrossingPulse):
    """Build the coupler-crossing scan on the QM OPX (one QCQ pair)."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from ._vendor import role_member, role_side, vendor_pair

        machine = self.backend.machine  # type: ignore[attr-defined]
        pair = self.params.targets[0]
        roster = self.device.roster
        members = [role_member(roster, pair, role) for role in ("high", "low")]
        missing = _missing_thresholds(self, members)
        if missing:
            raise ValueError(
                f"{type(self).name} reads the joint populations through state "
                f"discrimination, and {missing} have no readout_threshold. Run "
                f"single_shot_readout on them and accept its readout_rotation_rad / "
                f"readout_threshold suggestions first.")

        self._high_side = role_side(self, "high", field="targets")
        roles = ("high", "low") if self.params.measure == "both" else (self.params.measure,)
        measure_roles = [role_side(self, role, field="measure") for role in roles]

        prog, axes = build_program(
            machine,
            vendor_pair(self, pair),
            coupler_amps_v=self.sweep_axes["coupler_flux_v"],
            measure_roles=measure_roles,
            buffer_ns=self.params.flux_buffer_ns,
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
        )
        sweep_axes = {
            # the probe labels its target with the VENDOR pair key; scqo keys on
            # the ROSTER name the operator asked for
            "qubit_pair": xr.DataArray([pair]),
            "coupler_flux_v": axes["coupler_amplitude"],
        }
        return prog, sweep_axes
