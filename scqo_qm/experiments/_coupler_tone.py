"""The coupler tone's run config, shared by both coupler spectroscopies
(``pair_coupler_spectroscopy_swap`` and ``pair_coupler_spectroscopy_zz``).

ONE LO PER RUN, AND ITS OWN CONFIG. The tone window sits far from the tone member's
drive frequency (5Q4C q2: 4.84 GHz on band 1, LO 4.9 GHz; the couplers near 7.1 GHz),
so the run needs that port at another LO and possibly another MW-FEM band.
:func:`moved_lo_config` sets those on the QUAM tree only for ``generate_config()`` and
restores them in a ``finally`` straight after; the probe hands the backend that config
through the 3-tuple acquire callable (:func:`acquire`), so nothing is left moved for
the snapshot to call drift. A band switch carries the port-pair partner
((2,3)(4,5)(6,7) share a band): its LO stays where it is when the new band holds it,
and is parked at the band's floor otherwise. MW-FEM only; an Octave tree is refused
by name (its LO grid and shared synthesizers are ``broadband_qubit_spectroscopy``'s
business).

A port may carry a SECOND upconverter once a borrowed channel is adopted on it
(``scqo-qm adopt-channel``: 5Q4C q2's port holds 4.9 GHz for q2 and 7.1 GHz for the
couplers). Only the tone element's own upconverter moves - read and written through
``scqo_qm._mw_fem``, never the port's ``upconverter_frequency`` attribute, which a
two-upconverter port does not use - and a band switch that could not hold the other
upconverter's LO is refused by name rather than stranding the adopted channel.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import xarray as xr

from scqo_qm._mw_fem import (
    MW_FEM_BANDS,
    band_holds,
    partner_port_id,
    port_info,
    port_lo,
    port_los,
    set_port_lo,
)
from scqo_qm.experiments._lib import acquire as _acquire


def choose_band(current: int, lo_hz: float, partner_lo_hz: Optional[float]) -> int:
    """Keep the current band when it holds the LO; otherwise the lowest band that
    holds it AND the partner's LO, else the lowest that holds it."""
    bands = MW_FEM_BANDS
    lo_min, lo_max = bands[current]
    if lo_min <= lo_hz <= lo_max:
        return current
    holding = [b for b, (a, z) in sorted(bands.items()) if a <= lo_hz <= z]
    if not holding:
        raise ValueError(f"no MW-FEM band holds an LO of {lo_hz / 1e9:.3f} GHz")
    both = [b for b in holding if partner_lo_hz is not None
            and bands[b][0] <= partner_lo_hz <= bands[b][1]]
    return (both or holding)[0]


def _upconverter(channel) -> int:
    """The upconverter an MW channel plays on (QUAM's default is 1)."""
    return int(getattr(channel, "upconverter", 1) or 1)


def moved_lo_config(machine, tone_qubit, *, lo_hz: float, experiment: str) -> tuple[dict, dict]:
    """``machine.generate_config()`` with the tone qubit's upconverter at ``lo_hz``
    (band switched with its port-pair partner when needed) and its drive RF at the
    LO (IF 0), then EVERY changed value restored. Returns (config, what_moved).

    Only the tone element's own upconverter moves. A band switch is refused by name
    when the new band cannot hold the LO of ANOTHER upconverter on either port of
    the pair - an adopted borrowed channel rides it, and parking it would strand
    that channel for the run."""
    from scqo_qm._family import RF_MW_FEM, rf_chain

    xy = tone_qubit.xy
    if rf_chain(xy) != RF_MW_FEM:
        raise ValueError(
            f"{tone_qubit.name}: {experiment} moves the drive port's LO, which this "
            f"probe does on an MW-FEM only (the drive is on "
            f"{rf_chain(xy) or 'an unrecognized RF chain'}).")
    port = xy.opx_output
    up = _upconverter(xy)
    info = port_info(port)
    partner = None
    if info is not None:
        ctrl, fem, pid = info
        for q in machine.qubits.values():
            other = getattr(getattr(q, "xy", None), "opx_output", None)
            if other is not None and other is not port and \
                    port_info(other) == (ctrl, fem, partner_port_id(pid)):
                partner = q
                break
    partner_port = partner.xy.opx_output if partner is not None else None
    partner_up = _upconverter(partner.xy) if partner is not None else 1
    partner_lo = port_lo(partner_port, partner_up) if partner_port is not None else None
    band = choose_band(int(port.band), float(lo_hz), partner_lo)
    if band != int(port.band):
        stranded = [f"{label} upconverter {n} at {lo / 1e9:.4g} GHz"
                    for label, p, keep in (("the tone port's", port, up),
                                           ("its partner's", partner_port, partner_up))
                    if p is not None
                    for n, lo in port_los(p).items()
                    if n != keep and not band_holds(band, lo)]
        if stranded:
            raise ValueError(
                f"{tone_qubit.name}: {experiment} would switch the port pair to MW-FEM "
                f"band {band}, which cannot hold {', '.join(stranded)} - an adopted "
                f"borrowed channel rides it. Move the tone window into band "
                f"{int(port.band)}, or tone through the other member (tone_on).")

    undo: list[Callable[[], None]] = []

    def set_(obj, attr, value):
        old = getattr(obj, attr)
        undo.append(lambda o=obj, a=attr, v=old: setattr(o, a, v))
        setattr(obj, attr, value)

    def set_lo(p, n, hz):
        old = port_lo(p, n)
        undo.append(lambda p=p, n=n, v=old: set_port_lo(p, n, v))
        set_port_lo(p, n, hz)

    moved = {"lo_hz": float(lo_hz), "band": band, "band_before": int(port.band),
             "partner": partner.name if partner is not None else None, "partner_parked": 0}
    try:
        if band != int(port.band):
            set_(port, "band", band)
            if partner_port is not None:
                set_(partner_port, "band", band)
                if partner_lo is not None and not band_holds(band, partner_lo):
                    lo_min, _lo_max = MW_FEM_BANDS[band]
                    set_lo(partner_port, partner_up, lo_min)
                    set_(partner.xy, "RF_frequency", lo_min)
                    moved["partner_parked"] = 1
        set_lo(port, up, float(lo_hz))
        set_(xy, "RF_frequency", float(lo_hz))
        config = machine.generate_config()
    finally:
        for restore in reversed(undo):
            restore()
    return config, moved


def missing_thresholds(experiment: Any, members: Sequence[str]) -> list[str]:
    """The members without a ``readout_threshold`` (both are read out 2-level)."""
    missing = []
    for member in members:
        try:
            value = experiment.device.channel(member, "readout").readout_threshold
        except KeyError:
            value = None
        if value is None:
            missing.append(member)
    return missing


def refuse_missing_thresholds(experiment: Any, members: Sequence[str]) -> None:
    missing = missing_thresholds(experiment, members)
    if missing:
        raise ValueError(
            f"{type(experiment).name} reads the joint populations through state "
            f"discrimination, and {missing} have no readout_threshold. Run "
            f"single_shot_readout on them and accept its readout_rotation_rad / "
            f"readout_threshold suggestions first.")


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
