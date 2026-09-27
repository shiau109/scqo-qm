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
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import xarray as xr

from scqo_qm.experiments._lib import acquire as _acquire

#: the IF window one LO plays (the repo's MW-FEM convention)
MAX_IF_HZ = 250e6


def _mw_fem_bands():
    from scqo_qm.experiments.broadband_qubit_spectroscopy import _MW_FEM_BANDS

    return _MW_FEM_BANDS


def choose_band(current: int, lo_hz: float, partner_lo_hz: Optional[float]) -> int:
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


def moved_lo_config(machine, tone_qubit, *, lo_hz: float, experiment: str) -> tuple[dict, dict]:
    """``machine.generate_config()`` with the tone qubit's port at ``lo_hz`` (band
    switched with its port-pair partner when needed) and its drive RF at the LO (IF
    0), then EVERY changed attribute restored. Returns (config, what_moved)."""
    from scqo_qm._family import RF_MW_FEM, rf_chain
    from scqo_qm.experiments.broadband_qubit_spectroscopy import (
        _get_port_info,
        _partner_port_id,
    )

    xy = tone_qubit.xy
    if rf_chain(xy) != RF_MW_FEM:
        raise ValueError(
            f"{tone_qubit.name}: {experiment} moves the drive port's LO, which this "
            f"probe does on an MW-FEM only (the drive is on "
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
    band = choose_band(int(port.band), float(lo_hz), partner_lo)

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
