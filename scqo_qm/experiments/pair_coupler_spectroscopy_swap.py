"""Coupler swap-spectroscopy acquisition probe: vendor code only (qm/quam) - no scqo, no scqat.

Per shot and tone frequency, two arms back to back (ramp, then reference): reset
both members -> align -> the probe's xy plays ``saturation`` for the tone length at
IF = f - LO -> align (+ buffer) -> ramp arm: the ramped flux line plays one
arbitrary waveform (first sample at the ramp start, linear to the ramp end, then
the output drops back to idle); reference arm: that line waits the same length ->
align (+ buffer) -> both members read out 2-level -> the four joint indicators.
Loops: averages (outer) -> tone IF (in the order given) -> arm.

QM coupler swap spectroscopy for scqo - supplies ``probe()`` + the joint-population
reduction.

Parameters, the fit and the writeback (the coupler's ``f_01_hz``) are inherited from
``scqo.experiments.PairCouplerSpectroscopySwap``.

ONE LO PER RUN, AND ITS OWN CONFIG. The tone window sits far from the probe's drive
frequency (5Q4C q1: 5.14 GHz on band 1, LO 4.9 GHz; the coupler near 6.8 GHz), so
the run needs the probe's port at another LO and possibly another MW-FEM band. The
probe sets those on the QUAM tree only for ``generate_config()`` and restores them
in a ``finally`` straight after, then hands the backend that config through the
3-tuple acquire callable - nothing is left moved for the snapshot to call drift.
A band switch carries the port-pair partner ((2,3)(4,5)(6,7) share a band): its LO
stays where it is when the new band holds it, and is parked at the band's floor
otherwise. ``patch_preview_config`` gives ``--preview`` the same config. MW-FEM only;
an Octave tree is refused by name (its LO grid and shared synthesizers are
``broadband_qubit_spectroscopy``'s business).

THE RAMP is an arbitrary waveform on the ramped line's element, added to that
config (1 GS/s, linear from start to end), so it rides on the standing bias like
any flux pulse; the rail, the idle + excursion sum and the sample range are checked
before any QUA is built.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Sequence

import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array

from scqo_qm.experiments._flux_limits import (
    check_flux_pulse_relative,
    dac_rail_v,
    declared_idle_offset_v,
)
from scqo_qm.experiments._lib import acquire as _acquire

#: the config names the ramp is added under (waveform, pulse, operation)
RAMP_WAVEFORM = "coupler_swap_ramp_wf"
RAMP_PULSE = "coupler_swap_ramp_pulse"
RAMP_OPERATION = "coupler_swap_ramp"
#: the IF window one LO plays (the repo's MW-FEM convention)
MAX_IF_HZ = 250e6
_CLOCK_NS = 4


def ramp_samples(start_v: float, end_v: float, duration_ns: int) -> list[float]:
    """1 GS/s samples of the linear ramp: the first at ``start_v``, the last at
    ``end_v``; after the last one the element's output is idle again."""
    return [float(v) for v in np.linspace(float(start_v), float(end_v), int(duration_ns))]


def check_ramp(channel, *, name: str, start_v: float, end_v: float) -> None:
    """The ramp rides on the standing bias: the idle + excursion sum and QUA's
    amplitude range through the shared flux-pulse check, and every SAMPLE inside the
    port's full scale (an arbitrary waveform is not scaled by amplitude_scale)."""
    check_flux_pulse_relative(channel, name=name, idle_v=declared_idle_offset_v(channel),
                              amps_v=[start_v, end_v], operation="const")
    rail = dac_rail_v(channel)
    peak = max(abs(start_v), abs(end_v))
    if peak >= rail:
        raise ValueError(f"{name}: the ramp reaches {peak} V, at or past the port's "
                         f"{rail} V full scale")


def add_ramp_to_config(config: dict, element: str, samples: Sequence[float]) -> dict:
    """``config`` with the ramp operation on ``element`` (a single-output flux line)."""
    config.setdefault("waveforms", {})[RAMP_WAVEFORM] = {
        "type": "arbitrary", "samples": list(samples)}
    config.setdefault("pulses", {})[RAMP_PULSE] = {
        "operation": "control", "length": len(samples),
        "waveforms": {"single": RAMP_WAVEFORM}}
    config["elements"][element].setdefault("operations", {})[RAMP_OPERATION] = RAMP_PULSE
    return config


def build_program(
    machine,
    qubit_pair,
    *,
    tone_ifs_hz: Sequence[int],
    probe_role: str,
    ramp_element: str,
    ramp_ns: int,
    tone_ns: int,
    buffer_ns: int,
    num_shots: int,
    reset_type: str,
    simulate: bool = False,
):
    """Build the swap-spectroscopy QUA program for ONE pair. Returns (program, sweep_axes).

    ``tone_ifs_hz``: the tone's IF per point (Hz, relative to the probe port's LO
    in the config the program runs against), in the order to be swept.
    ``probe_role``: ``"control"`` / ``"target"`` - the member that gets the tone.
    ``ramp_element``: the element that plays :data:`RAMP_OPERATION` (it must be in
    that config); ``ramp_ns`` its length, the reference arm's wait.
    """
    qp = qubit_pair
    if probe_role not in ("control", "target"):
        raise ValueError(f"probe_role must be control or target, got {probe_role!r}")
    for what, ns in (("ramp", ramp_ns), ("tone", tone_ns)):
        if ns < 16 or ns % _CLOCK_NS:
            raise ValueError(f"{what} of {ns} ns: needs a multiple of 4 ns from 16 ns up")
    if buffer_ns and (buffer_ns < 16 or buffer_ns % _CLOCK_NS):
        raise ValueError(f"buffer of {buffer_ns} ns: use 0, or a multiple of 4 ns from 16 ns up")
    ifs = np.asarray(tone_ifs_hz, dtype=int)
    if np.max(np.abs(ifs)) > MAX_IF_HZ:
        raise ValueError(f"tone IF reaches {np.max(np.abs(ifs)) / 1e6:.0f} MHz, past "
                         f"+-{MAX_IF_HZ / 1e6:.0f} MHz around the LO")
    probe = qp.qubit_control if probe_role == "control" else qp.qubit_target
    ramp_cycles = ramp_ns // _CLOCK_NS
    tone_cycles = tone_ns // _CLOCK_NS
    buffer_cycles = buffer_ns // _CLOCK_NS

    sweep_axes = {
        "qubit_pair": xr.DataArray([qp.name]),
        "tone_if": xr.DataArray(ifs, attrs={"long_name": "tone IF", "units": "Hz"}),
        "ramp_played": xr.DataArray([1, 0], attrs={"long_name": "swap ramp played"}),
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
                for ramp_played in (True, False):  # back to back, same tone frequency
                    qp.qubit_control.reset(reset_type, simulate)
                    qp.qubit_target.reset(reset_type, simulate)
                    align()
                    probe.xy.update_frequency(f_if)
                    probe.xy.play("saturation", duration=tone_cycles)
                    align()
                    if buffer_cycles:
                        wait(buffer_cycles, ramp_element)
                    if ramp_played:
                        play(RAMP_OPERATION, ramp_element)
                    else:
                        wait(ramp_cycles, ramp_element)
                    if buffer_cycles:
                        wait(buffer_cycles, ramp_element)
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


# ---------------------------------------------------------------- the moved-LO config

def _mw_fem_bands():
    from scqo_qm.experiments.broadband_qubit_spectroscopy import _MW_FEM_BANDS

    return _MW_FEM_BANDS


def _choose_band(current: int, lo_hz: float, partner_lo_hz: Optional[float]) -> int:
    """Keep the current band when it holds the LO; otherwise the lowest band that
    holds it AND the partner's LO, else the lowest that holds it."""
    bands = _mw_fem_bands()
    lo_min, lo_max = bands[current]
    if lo_min <= lo_hz <= lo_max:
        return current
    holding = [b for b, (a, z) in sorted(bands.items()) if a <= lo_hz <= z]
    if not holding:
        raise ValueError(f"no MW-FEM band holds an LO of {lo_hz / 1e9:.3f} GHz")
    both = [b for b in holding if partner_lo_hz is not None
            and bands[b][0] <= partner_lo_hz <= bands[b][1]]
    return (both or holding)[0]


def moved_lo_config(machine, probe_qubit, *, lo_hz: float) -> tuple[dict, dict]:
    """``machine.generate_config()`` with the probe's port at ``lo_hz`` (band switched
    with its port-pair partner when needed) and the probe's drive RF at the LO
    (IF 0), then EVERY changed attribute restored. Returns (config, what_moved)."""
    from scqo_qm._family import RF_MW_FEM, rf_chain
    from scqo_qm.experiments.broadband_qubit_spectroscopy import (
        _get_port_info,
        _partner_port_id,
    )

    xy = probe_qubit.xy
    if rf_chain(xy) != RF_MW_FEM:
        raise ValueError(
            f"{probe_qubit.name}: pair_coupler_spectroscopy_swap moves the drive port's "
            f"LO, which this probe does on an MW-FEM only (the drive is on "
            f"{rf_chain(xy) or 'an unrecognized RF chain'}).")
    port = xy.opx_output
    info = _get_port_info(port)
    partner = None
    if info is not None:
        ctrl, fem, pid = info
        for q in machine.qubits.values():
            other = getattr(getattr(q, "xy", None), "opx_output", None)
            if other is not None and other is not port and \
                    _get_port_info(other) == (ctrl, fem, _partner_port_id(pid)):
                partner = q
                break
    partner_port = partner.xy.opx_output if partner is not None else None
    partner_lo = (float(partner_port.upconverter_frequency)
                  if partner_port is not None else None)
    band = _choose_band(int(port.band), float(lo_hz), partner_lo)

    saved: list[tuple[Any, str, Any]] = []

    def set_(obj, attr, value):
        saved.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, value)

    moved = {"lo_hz": float(lo_hz), "band": band, "band_before": int(port.band),
             "partner": partner.name if partner is not None else None, "partner_parked": 0}
    try:
        if band != int(port.band):
            set_(port, "band", band)
            if partner_port is not None:
                set_(partner_port, "band", band)
                lo_min, lo_max = _mw_fem_bands()[band]
                if not lo_min <= partner_lo <= lo_max:
                    set_(partner_port, "upconverter_frequency", lo_min)
                    set_(partner.xy, "RF_frequency", lo_min)
                    moved["partner_parked"] = 1
        set_(port, "upconverter_frequency", float(lo_hz))
        set_(xy, "RF_frequency", float(lo_hz))
        config = machine.generate_config()
    finally:
        for obj, attr, value in reversed(saved):
            setattr(obj, attr, value)
    return config, moved


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
    """Connect to the QOP, execute the program against ``config`` and fetch."""
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout,
                    log=log, config=config)


from scqo import register
from scqo.experiments import PairCouplerSpectroscopySwap

from ._pair_roles import JointPopulationMixin


def _missing_thresholds(experiment: Any, members: Sequence[str]) -> list[str]:
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
class QMPairCouplerSpectroscopySwap(JointPopulationMixin, PairCouplerSpectroscopySwap):
    """Build the coupler swap spectroscopy on the QM OPX (one QCQ pair, MW-FEM)."""

    def _build(self) -> tuple[Any, dict, dict]:
        from ._reset import check_reset_method
        from ._vendor import role_member, role_side, vendor_pair

        machine = self.backend.machine  # type: ignore[attr-defined]
        p = self.params
        pair = p.targets[0]
        roster = self.device.roster
        members = [role_member(roster, pair, role) for role in ("high", "low")]
        missing = _missing_thresholds(self, members)
        if missing:
            raise ValueError(
                f"{type(self).name} reads the joint populations through state "
                f"discrimination, and {missing} have no readout_threshold. Run "
                f"single_shot_readout on them and accept its readout_rotation_rad / "
                f"readout_threshold suggestions first.")
        reset_type = check_reset_method(self)

        qp = vendor_pair(self, pair)
        self._high_side = role_side(self, "high", field="targets")
        probe_role = role_side(self, p.probe, field="probe")
        probe_qubit = qp.qubit_control if probe_role == "control" else qp.qubit_target
        if p.ramp_on == "coupler":
            channel, what = qp.coupler, f"{pair} coupler"
        else:
            channel, what = probe_qubit.z, f"{pair} probe {probe_qubit.name}.z"
        check_ramp(channel, name=what, start_v=p.ramp_start_v, end_v=p.ramp_end_v)

        ramp_ns = self.ramp_duration_ns()
        self._ramp_duration_ns = float(ramp_ns)
        lo = self.lo_hz()
        config, moved = moved_lo_config(machine, probe_qubit, lo_hz=lo)
        add_ramp_to_config(config, channel.name,
                           ramp_samples(p.ramp_start_v, p.ramp_end_v, ramp_ns))
        self._moved = moved

        freqs = np.asarray(self.sweep_axes["tone_freq_hz"], dtype=float)
        prog, axes = build_program(
            machine, qp,
            tone_ifs_hz=np.round(freqs - lo).astype(int),
            probe_role=probe_role, ramp_element=channel.name, ramp_ns=ramp_ns,
            tone_ns=int(p.tone_len_ns), buffer_ns=p.flux_buffer_ns,
            num_shots=p.num_averages, reset_type=reset_type)
        sweep_axes = {
            "qubit_pair": xr.DataArray([pair]),
            # the absolute tone frequencies the IFs realize (LO + IF, as played)
            "tone_freq_hz": xr.DataArray(lo + axes["tone_if"].values.astype(float),
                                         attrs={"units": "Hz"}),
            "ramp_played": axes["ramp_played"],
        }
        return prog, sweep_axes, config

    def probe(self) -> Any:
        prog, sweep_axes, config = self._build()
        self._config = config
        return prog, sweep_axes, partial(acquire, config=config)

    def patch_preview_config(self, config: dict) -> dict:
        """``--preview`` sees the config the run executes against: the moved LO and
        band, and the ramp operation."""
        return getattr(self, "_config", config)
