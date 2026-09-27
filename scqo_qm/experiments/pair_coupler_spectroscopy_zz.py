"""Coupler ZZ-spectroscopy acquisition probe: vendor code only (qm/quam) - no scqo, no scqat.

Per shot and tone frequency, two arms back to back (pi, then reference): reset both
members -> align -> the tone member's xy plays ``saturation`` for the tone length at
IF = f - LO -> align -> pi arm: the pi member's xy plays the SELECTIVE pi - its
``saturation`` operation scaled down and stretched to ``selective_pi_len_ns`` - at its
own drive frequency; reference arm: that element waits as long -> align -> both
members read out 2-level -> the four joint indicators. Loops: averages (outer) ->
tone IF (in the order given) -> arm.

QM coupler ZZ spectroscopy for scqo - supplies ``probe()`` + the joint-population
reduction. Parameters, the fit and the writeback (the coupler's ``f_01_hz`` and, from
the multi-photon ladder, its ``anharmonicity_hz``) are inherited from
``scqo.experiments.PairCouplerSpectroscopyZZ``.

THE LINES: the tone rides on ``tone_on``'s drive element, the pi on the OTHER member's
- two elements on two ports, so no second upconverter. The tone member's port gets the
run's own LO (``_coupler_tone.moved_lo_config``: the MW-FEM band switched with the
port-pair partner when needed, the QUAM tree restored straight after the config is
built); when the pi member IS that partner (5Q4C q1_q2: q2's tone on ``6/3``, q1's pi
on ``6/2``) its port follows the band and keeps its LO, so its IF - and its pulses -
are unchanged. ``patch_preview_config`` gives ``--preview`` the same config.

THE SELECTIVE PI is a square pulse with the rotation AREA of the member's calibrated
``x180`` (:func:`selective_pi_scale`), so it needs no calibration of its own. The area
is the x180's sampled waveform summed with the pulse's own frame ``detuning`` undone:
that detuning compensates the Stark shift of a strong 16 ns pulse, which a weak pulse
microseconds long does not have (5Q4C q1: -8.7 MHz; undone, a 16 ns DragCosine of
amplitude A sums to exactly 7.5 A ns). Played as the member's ``saturation`` with
``amplitude_scale`` and ``duration`` - no new waveform in the config.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Sequence

import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array

from scqo_qm.experiments._coupler_tone import (
    MAX_IF_HZ,
    acquire,
    moved_lo_config,
    refuse_missing_thresholds,
)

#: the calibrated pulse whose rotation area the selective pi keeps
PI_REFERENCE = "x180"
#: the square operation the selective pi is played on
SQUARE_OPERATION = "saturation"
_CLOCK_NS = 4


def build_program(
    machine,
    qubit_pair,
    *,
    tone_ifs_hz: Sequence[int],
    tone_role: str,
    pi_ns: int,
    pi_scale: float,
    tone_ns: int,
    num_shots: int,
    reset_type: str,
    simulate: bool = False,
):
    """Build the ZZ-spectroscopy QUA program for ONE pair. Returns (program, sweep_axes).

    ``tone_ifs_hz``: the tone's IF per point (Hz, relative to the tone port's LO in
    the config the program runs against), in the order to be swept.
    ``tone_role``: ``"control"`` / ``"target"`` - the member whose drive carries the
    tone; the OTHER one plays :data:`SQUARE_OPERATION` scaled by ``pi_scale`` for
    ``pi_ns`` (the reference arm waits as long).
    """
    qp = qubit_pair
    if tone_role not in ("control", "target"):
        raise ValueError(f"tone_role must be control or target, got {tone_role!r}")
    if not 0 < pi_scale < 1:
        raise ValueError(f"pi_scale {pi_scale}: the selective pi is a scaled-DOWN "
                         f"{SQUARE_OPERATION!r}, so it must lie in (0, 1)")
    for what, ns in (("pi", pi_ns), ("tone", tone_ns)):
        if ns < 16 or ns % _CLOCK_NS:
            raise ValueError(f"{what} of {ns} ns: needs a multiple of 4 ns from 16 ns up")
    ifs = np.asarray(tone_ifs_hz, dtype=int)
    if np.max(np.abs(ifs)) > MAX_IF_HZ:
        raise ValueError(f"tone IF reaches {np.max(np.abs(ifs)) / 1e6:.0f} MHz, past "
                         f"+-{MAX_IF_HZ / 1e6:.0f} MHz around the LO")
    tone, pi = ((qp.qubit_control, qp.qubit_target) if tone_role == "control"
                else (qp.qubit_target, qp.qubit_control))
    pi_cycles = pi_ns // _CLOCK_NS
    tone_cycles = tone_ns // _CLOCK_NS

    sweep_axes = {
        "qubit_pair": xr.DataArray([qp.name]),
        "tone_if": xr.DataArray(ifs, attrs={"long_name": "tone IF", "units": "Hz"}),
        "pi_played": xr.DataArray([1, 0], attrs={"long_name": "pi played"}),
    }

    with program() as prog:
        f_if = declare(int)
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
            with for_(*from_array(f_if, ifs)):
                for pi_played in (True, False):  # back to back, same tone frequency
                    qp.qubit_control.reset(reset_type, simulate)
                    qp.qubit_target.reset(reset_type, simulate)
                    align()
                    tone.xy.update_frequency(f_if)
                    tone.xy.play("saturation", duration=tone_cycles)
                    align()
                    if pi_played:
                        pi.xy.play(SQUARE_OPERATION, amplitude_scale=float(pi_scale),
                                   duration=pi_cycles)
                    else:
                        wait(pi_cycles, pi.xy.name)
                    align()
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
                stream.buffer(2).buffer(len(ifs)).average().save(f"{name}1")

    return prog, sweep_axes


def pulse_area_ns(pulse) -> float:
    """A pulse's rotation area (amplitude x ns): its sampled waveform summed with the
    pulse's own frame ``detuning`` undone (samples are 1 ns apart)."""
    w = np.atleast_1d(np.asarray(pulse.calculate_waveform(), dtype=complex))
    if w.size == 1:                                  # a constant waveform
        w = np.full(int(pulse.length), w[0])
    t = np.arange(w.size) * 1e-9
    detuning = float(getattr(pulse, "detuning", 0.0) or 0.0)
    return float(abs(np.sum(w * np.exp(-1j * 2 * np.pi * detuning * t))))


def selective_pi_scale(qubit, length_ns: int) -> float:
    """The ``amplitude_scale`` on the member's :data:`SQUARE_OPERATION` that gives a
    square pulse ``length_ns`` long the rotation area of its :data:`PI_REFERENCE`.
    Refused by name when either operation is missing or the pulse would have to be
    louder than the square operation itself."""
    ops = getattr(qubit.xy, "operations", {}) or {}
    for op in (PI_REFERENCE, SQUARE_OPERATION):
        if op not in ops:
            raise ValueError(f"{qubit.name}: pair_coupler_spectroscopy_zz builds the "
                             f"selective pi from the pi member's {op!r}, and its drive "
                             f"has none")
    area = pulse_area_ns(ops[PI_REFERENCE])
    square = abs(float(ops[SQUARE_OPERATION].amplitude))
    if not (np.isfinite(area) and area > 0 and square > 0):
        raise ValueError(f"{qubit.name}: {PI_REFERENCE!r} has no usable area ({area}) or "
                         f"{SQUARE_OPERATION!r} no amplitude ({square})")
    scale = area / float(length_ns) / square
    if scale >= 1:
        raise ValueError(f"{qubit.name}: a {length_ns} ns selective pi needs "
                         f"{scale:.2f} x the {SQUARE_OPERATION!r} amplitude; make it "
                         f"longer")
    return scale


from scqo import register
from scqo.experiments import PairCouplerSpectroscopyZZ

from ._pair_roles import JointPopulationMixin


@register
class QMPairCouplerSpectroscopyZZ(JointPopulationMixin, PairCouplerSpectroscopyZZ):
    """Build the coupler ZZ spectroscopy on the QM OPX (one QCQ pair, MW-FEM)."""

    def _build(self) -> tuple[Any, dict, dict]:
        from ._reset import check_reset_method
        from ._vendor import role_member, role_side, vendor_pair

        machine = self.backend.machine  # type: ignore[attr-defined]
        p = self.params
        pair = p.targets[0]
        roster = self.device.roster
        refuse_missing_thresholds(
            self, [role_member(roster, pair, role) for role in ("high", "low")])
        reset_type = check_reset_method(self)

        qp = vendor_pair(self, pair)
        self._high_side = role_side(self, "high", field="targets")
        tone_role = role_side(self, p.tone_on, field="tone_on")
        tone_qubit, pi_qubit = ((qp.qubit_control, qp.qubit_target) if tone_role == "control"
                                else (qp.qubit_target, qp.qubit_control))
        pi_ns = int(p.selective_pi_len_ns)
        pi_scale = selective_pi_scale(pi_qubit, pi_ns)
        self._selective_pi = {"length_ns": pi_ns, "amplitude_scale": pi_scale,
                              "amplitude": pi_scale * abs(float(
                                  pi_qubit.xy.operations[SQUARE_OPERATION].amplitude))}

        lo = self.lo_hz()
        config, moved = moved_lo_config(machine, tone_qubit, lo_hz=lo, experiment=self.name)
        self._moved = moved

        freqs = np.asarray(self.sweep_axes["tone_freq_hz"], dtype=float)
        prog, axes = build_program(
            machine, qp,
            tone_ifs_hz=np.round(freqs - lo).astype(int),
            tone_role=tone_role, pi_ns=pi_ns, pi_scale=pi_scale,
            tone_ns=int(p.tone_len_ns),
            num_shots=p.num_averages, reset_type=reset_type)
        sweep_axes = {
            "qubit_pair": xr.DataArray([pair]),
            # the absolute tone frequencies the IFs realize (LO + IF, as played)
            "tone_freq_hz": xr.DataArray(lo + axes["tone_if"].values.astype(float),
                                         attrs={"units": "Hz"}),
            "pi_played": axes["pi_played"],
        }
        return prog, sweep_axes, config

    def probe(self) -> Any:
        prog, sweep_axes, config = self._build()
        self._config = config
        return prog, sweep_axes, partial(acquire, config=config)

    def patch_preview_config(self, config: dict) -> dict:
        """``--preview`` sees the config the run executes against: the moved LO and
        band."""
        return getattr(self, "_config", config)
