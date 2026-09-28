"""The SELECTIVE pi: a square pulse with a member's calibrated x180 rotation area.

Shared by ``pair_coupler_spectroscopy_zz`` (the pi a coupler excitation spoils
through the member-coupler ZZ) and the mapped readout (``_mapped_target``: the
same pi, then the member's x180, copy a coupler's state onto the member). It needs
no calibration of its own: the area is the member's x180 sampled waveform summed
with the pulse's own frame ``detuning`` undone. That detuning compensates the Stark
shift of a strong 16 ns pulse, which a weak pulse microseconds long does not have
(5Q4C q1: -8.7 MHz; undone, a 16 ns DragCosine of amplitude A sums to exactly
7.5 A ns). Played as the member's ``saturation`` operation with
``amplitude_scale`` and ``duration`` - no new waveform in the config.
"""

from __future__ import annotations

import numpy as np

#: the calibrated pulse whose rotation area the selective pi keeps
PI_REFERENCE = "x180"
#: the square operation the selective pi is played on
SQUARE_OPERATION = "saturation"


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
            raise ValueError(f"{qubit.name}: the selective pi is built from the "
                             f"member's {op!r}, and its drive has none")
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
