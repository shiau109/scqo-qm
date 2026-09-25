"""What an Octave IS: hardware facts, one home, no vendor import.

FACTS only -- the grids, windows and shared synthesizers the instrument imposes.
The POLICY built on them (how this driver chooses a gain against an amplitude)
lives in ``backend/_power.py``, which imports from here. Keeping the two apart is
what lets the policy change without anyone re-deriving the hardware, and stops a
second copy of a grid appearing next to a solve -- the failure
``experiments/_amp_limits`` already documents for the QUA amplitude bound.

Package root rather than under ``backend/`` because both halves need it and
``experiments/`` cannot import from ``backend/``: ``backend/__init__`` pulls in
``qm_backend``, which imports ``experiments``, so the reverse edge would cycle.
``quam_fields.py`` sits here for the same reason.

**Nothing in the vendor stack enforces any of this.** Checked across ``qm``,
``quam`` and ``quam_builder``: there is no LO-grid check and no IF-window check
anywhere. The only guards that ever existed are two ``assert`` lines in
``quam_config/populate_quam_opxp_octave.py``, which run when a tree is first
seeded and never again. So an out-of-range IF or an off-grid LO reaches the
instrument as a config the QOP may accept, quietly mistune, or reject with an
error naming neither the channel nor the number -- which is why the audits in
``quam_fields`` exist at all.
"""

from __future__ import annotations

from typing import Optional

#: Octave LO synthesizer range (Hz), inclusive.
LO_MIN_HZ = 2e9
LO_MAX_HZ = 18e9

#: The LO tunes in 250 MHz steps; anything between two grid points is not a
#: frequency the synthesizer can produce.
LO_STEP_HZ = 0.25e9

#: How far a stored LO may sit off the grid before it counts as off it. ulp noise
#: at 18 GHz is ~2e-6 Hz and a real mis-set is MHz, so 1 Hz separates float echo
#: from a genuine error with room to spare (the same reasoning, and the same
#: number, as quam_fields.DRIVE_FREQ_TOLERANCE_HZ).
LO_TOLERANCE_HZ = 1.0

#: |IF| = |RF - LO| may not exceed this. The OPX feeds the Octave a baseband IQ
#: pair, so the intermediate frequency is a real analog signal bounded by the
#: DAC and the mixer, not a digital offset inside a band.
IF_MAX_ABS_HZ = 400e6

#: The Octave gain grid (dB): quam_builder's power_tools refuses outside these
#: bounds, and the hardware quantizes to the step.
GAIN_MIN_DB = -20.0
GAIN_MAX_DB = 20.0
GAIN_STEP_DB = 0.5

#: Ceiling on the IF amplitude the OPX feeds the Octave (V) -- the OPX+ DAC rail.
#: ``power_tools.set_output_power_iq_channel`` refuses outside [-0.5, 0.5) for the
#: same reason. THE one home for this number: ``quam_config/instrument_limits.py``
#: used to state it too, for the qualibrate nodes' waveform capping, and left with
#: them. (An MW-FEM channel is bounded at 1.0 NORMALIZED instead -- the two families
#: do not even share the unit, so the ceilings were never comparable.)
MAX_IF_AMP_V = 0.5

#: The up-conversion mixer's optimum drive (V), per
#: ``quam_config/populate_quam_opxp_octave.py::get_octave_gain_and_amplitude``.
OPTIMUM_IF_AMP_V = 0.125

#: Which RF ports share an LO SYNTHESIZER. This is the constraint with no MW-FEM
#: analogue and the one most likely to surprise: the MW-FEM pairs ports only for
#: their ``band``, leaving each its own ``upconverter_frequency``, whereas two
#: Octave outputs on one synth are FORCED to the same LO. Park one and the other
#: moves with it.
#:
#: synth1 also serves RF INPUT 1, which is why a readout line using RF1 out and
#: RFin1 in cannot give its down-converter an independent LO -- and why the
#: up/down LO equality below is a hardware fact there, not merely a convention.
SYNTH_OUTPUTS: dict[int, tuple[int, ...]] = {1: (1,), 2: (2, 3), 3: (4, 5)}

#: RF INPUTS per synth, for the same reason.
SYNTH_INPUTS: dict[int, tuple[int, ...]] = {1: (1,), 2: (), 3: ()}


def synth_of_rf_output(rf_output: int) -> Optional[int]:
    """Which synthesizer drives an Octave RF OUTPUT, or None if there is no such
    output."""
    for synth, outputs in SYNTH_OUTPUTS.items():
        if rf_output in outputs:
            return synth
    return None


def rf_outputs_sharing_synth(rf_output: int) -> tuple[int, ...]:
    """The OTHER RF outputs forced to this one's LO. Empty when it has a synth to
    itself (RF1) or is not an RF output at all."""
    synth = synth_of_rf_output(rf_output)
    if synth is None:
        return ()
    return tuple(o for o in SYNTH_OUTPUTS[synth] if o != rf_output)


def snap_lo(lo_hz: float) -> float:
    """The nearest LO the synthesizer can actually produce: on the grid, in range.

    Anyone CHOOSING an LO must snap it, not merely check it. A broadband sweep
    computes a continuous LO per segment and then labels that segment's frequency
    axis with it; if the hardware quietly rounds the request to its own grid, every
    stitched point is mislabelled by up to half a step and the spectrum is wrong in
    a way no fit can see. Snapping first makes the requested and the emitted LO the
    same number, which is what the axis is derived from.
    """
    steps = round((float(lo_hz) - LO_MIN_HZ) / LO_STEP_HZ)
    snapped = LO_MIN_HZ + steps * LO_STEP_HZ
    return min(LO_MAX_HZ, max(LO_MIN_HZ, snapped))


def lo_grid_problem(lo_hz: float) -> Optional[str]:
    """Why ``lo_hz`` is not a frequency this synthesizer can produce, or None.

    Returns the REASON rather than a bool so every caller reports the same
    sentence -- an audit that said only "bad LO" would send the operator to
    measure which of the two rules they broke.
    """
    if lo_hz < LO_MIN_HZ - LO_TOLERANCE_HZ or lo_hz > LO_MAX_HZ + LO_TOLERANCE_HZ:
        return (f"{lo_hz / 1e9:g} GHz is outside the Octave synthesizer range "
                f"[{LO_MIN_HZ / 1e9:g}, {LO_MAX_HZ / 1e9:g}] GHz")
    off_grid = abs((lo_hz - LO_MIN_HZ) % LO_STEP_HZ)
    off_grid = min(off_grid, abs(LO_STEP_HZ - off_grid))
    if off_grid > LO_TOLERANCE_HZ:
        return (f"{lo_hz / 1e9:g} GHz is off the {LO_STEP_HZ / 1e9:g} GHz LO grid "
                f"by {off_grid / 1e6:g} MHz")
    return None


def if_window_problem(rf_hz: float, lo_hz: float) -> Optional[str]:
    """Why ``RF - LO`` is not an intermediate frequency this chain can carry, or
    None."""
    intermediate = rf_hz - lo_hz
    if abs(intermediate) > IF_MAX_ABS_HZ:
        return (f"IF = RF - LO = {intermediate / 1e6:g} MHz, outside the "
                f"+/-{IF_MAX_ABS_HZ / 1e6:g} MHz the Octave IF path carries")
    return None
