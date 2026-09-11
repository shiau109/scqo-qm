"""Absolute output power, per RF chain -- the arithmetic, with no vendor object in sight.

A neutral ``*_power_dbm`` is one number; realizing it takes TWO vendor knobs, and
which two depends on the RF chain:

* **MW-FEM** -- ``opx_output.full_scale_power_dbm`` (a -11..+16 dBm grid in 3 dB
  steps) scaling a normalized waveform amplitude. Solved by
  ``qm_backend._solve_full_scale``.
* **Octave** -- ``frequency_converter_up.gain`` (-20..+20 dB in 0.5 dB steps) plus
  an IF amplitude in VOLTS, capped at the OPX+'s 0.5 V DAC. Solved here.

**Why the two solves cannot share a policy.** The MW-FEM's coarse knob is free to
move: nothing downstream depends on its value. The Octave's is not. Its mixer
calibration is cached per ``(RF output, LO frequency, gain)`` -- gain is part of
the KEY -- so re-staging the gain silently invalidates the LO-leakage correction
for that output. And on a multiplexed readout line the gain belongs to the Octave
RF output, which every qubit on that feedline shares: moving it for one target
moves it for all of them, while the per-target difference can only ever live in
the amplitude.

So the policy here is **hold the gain, spend the amplitude**, and re-stage the
gain only when the amplitude genuinely cannot express the target -- reporting it
when that happens, because the caller owes the operator a mixer-calibration
warning. This is the same shape as the Qblox driver's ``output_att`` + pulse
amplitude solve: the coarse knob takes the decades, the amplitude carries the
exact residual, and the residual is where a per-target value can live at all.

This module is the POLICY. The hardware FACTS it is built on -- the gain grid,
the DAC ceiling, the mixer's optimum drive -- live in ``scqo_qm._octave``, which
is also where the LO grid and IF window the audits use live. One home per fact,
so a policy change never means re-deriving the instrument.

Pure: no quam import, no channel object, no warnings raised. The caller applies
the answer and decides how loudly to talk about it, which is what makes every
case below testable without an instrument or a tree.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

from scqo_qm._octave import (
    GAIN_MAX_DB,
    GAIN_MIN_DB,
    GAIN_STEP_DB,
    MAX_IF_AMP_V,
    OPTIMUM_IF_AMP_V,
)

#: Below this the amplitude is using under 1% of the DAC, so quantization noise
#: starts costing more than a gain change does. Deliberately far below the
#: optimum: the vendor's own multiplexed-readout recipe divides the target
#: amplitude by the qubit count (0.125/5 = 0.025 V on a five-qubit feedline), and
#: a floor that re-staged the gain for THAT would fight the standard config.
OCTAVE_MIN_AMP_V = 0.005


def volts_to_dbm(vp: float, z: float = 50.0) -> float:
    """Peak voltage -> dBm. Mirrors ``qualang_tools.units.unit.volts2dBm``
    exactly, so this module's solve inverts the vendor's reader rather than
    something merely similar (a half-dB disagreement would make every set/get
    round trip drift)."""
    return 10.0 * math.log10(((vp / math.sqrt(2.0)) ** 2 * 1000.0) / z)


def dbm_to_volts(p_dbm: float, z: float = 50.0) -> float:
    """dBm -> peak voltage. Mirrors ``qualang_tools.units.unit.dBm2volts``."""
    return math.sqrt(2.0 * z / 1000.0) * 10.0 ** (p_dbm / 20.0)


def snap_gain(gain_db: float) -> float:
    """The nearest legal Octave gain: on the 0.5 dB grid, inside [-20, +20].

    Applied to the CURRENT gain too, not just a new one. A tree can hold an
    off-grid value (a hand edit, or a float that round-tripped through JSON), and
    solving an amplitude against a gain the hardware will not actually take is a
    silent half-dB error in everything downstream.
    """
    snapped = round(float(gain_db) / GAIN_STEP_DB) * GAIN_STEP_DB
    return min(GAIN_MAX_DB, max(GAIN_MIN_DB, snapped))


#: The highest power the chain can reach: top gain at just under the DAC rail.
OCTAVE_MAX_POWER_DBM = GAIN_MAX_DB + volts_to_dbm(MAX_IF_AMP_V)
#: The lowest power reachable without dropping under the amplitude floor.
OCTAVE_MIN_POWER_DBM = GAIN_MIN_DB + volts_to_dbm(OCTAVE_MIN_AMP_V)


class OctaveChain(NamedTuple):
    """A solved Octave output chain.

    ``gain_moved`` is the caller's cue to warn: the mixer calibration for this RF
    output is keyed on the gain, so a moved gain means the cached LO-leakage
    correction no longer applies. ``reason`` says which way it had to move.
    """

    gain_db: float
    amplitude_v: float
    gain_moved: bool
    reason: Optional[str]


def solve_octave_chain(
    target_dbm: float, *, current_gain_db: float, name: str
) -> OctaveChain:
    """``target_dbm`` as an Octave ``(gain, amplitude)`` pair, preferring amplitude.

    Raises ``ValueError`` naming the reachable window when the target is outside
    what the chain can produce at all -- before anything is written, so a tree is
    never left half-staged.
    """
    held = snap_gain(current_gain_db)
    amplitude = dbm_to_volts(target_dbm - held)
    if OCTAVE_MIN_AMP_V <= amplitude < MAX_IF_AMP_V:
        return OctaveChain(held, amplitude, False, None)

    if not OCTAVE_MIN_POWER_DBM <= target_dbm <= OCTAVE_MAX_POWER_DBM:
        raise ValueError(
            f"{name}: {target_dbm} dBm is outside what an Octave output can "
            f"produce ([{OCTAVE_MIN_POWER_DBM:.1f}, {OCTAVE_MAX_POWER_DBM:.1f}] "
            f"dBm, gain [{GAIN_MIN_DB}, {GAIN_MAX_DB}] dB into a "
            f"{MAX_IF_AMP_V} V DAC). Change the fixed attenuation on that "
            f"line instead.")

    staged = snap_gain(target_dbm - volts_to_dbm(OPTIMUM_IF_AMP_V))
    amplitude = dbm_to_volts(target_dbm - staged)
    if not 0.0 < amplitude < MAX_IF_AMP_V:
        # Only reachable at a CLAMPED gain, where the optimum-seeking step above
        # cannot land the amplitude in range.
        raise ValueError(
            f"{name}: {target_dbm} dBm needs amplitude {amplitude:.4f} V at the "
            f"clamped gain of {staged} dB, past the {MAX_IF_AMP_V} V DAC "
            f"rail. Change the fixed attenuation on that line instead.")

    direction = "up" if staged > held else "down"
    reason = (
        f"amplitude alone could not reach {target_dbm} dBm at gain {held} dB "
        f"(it would need {dbm_to_volts(target_dbm - held):.4f} V, outside the "
        f"[{OCTAVE_MIN_AMP_V}, {MAX_IF_AMP_V}) V window), so the gain moved "
        f"{direction} to {staged} dB")
    return OctaveChain(staged, amplitude, True, reason)
