"""The broadband probes on an Octave: the LO is chosen, not merely checked.

Both probes STEP an LO and then label each stitched segment's frequency axis
with it. That makes two things load-bearing that no fit could ever catch:

* the LO must be one the synthesizer can actually produce. It tunes on a 250 MHz
  grid; a continuous LO handed to it is rounded by the hardware, and the segment
  is then mislabelled by the rounding -- every point of it, silently.
* whatever the sweep moved must come back. A restore that aborts partway leaves
  the remaining channels parked at a sub-band frequency, the session ends
  "cleanly", and the next experiment measures them there without complaint.

Before this, an Octave tree got neither: the LO limits degraded to
``[0, inf]`` (the synthesizer's real range is [2, 18] GHz) and the restore was a
straight line of writes with no guard.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from types import SimpleNamespace as NS

from scqo_qm._octave import LO_MAX_HZ, LO_MIN_HZ, LO_STEP_HZ, lo_grid_problem, snap_lo
from scqo_qm.experiments.broadband_qubit_spectroscopy import _restore_drive_channel

QUBIT_PROBE = pathlib.Path("scqo_qm/experiments/broadband_qubit_spectroscopy.py")
RESONATOR_PROBE = pathlib.Path("scqo_qm/experiments/broadband_resonator_spectroscopy.py")


# --- choosing an LO the hardware can produce ------------------------------


@pytest.mark.parametrize("requested", [5.1e9, 5.2e9, 6.3e9, 7.9e9, 4.13e9])
def test_a_snapped_lo_is_always_on_the_grid(requested):
    assert lo_grid_problem(snap_lo(requested)) is None


def test_snapping_moves_to_the_NEAREST_grid_point():
    assert snap_lo(5.1e9) == 5.0e9   # 100 MHz down beats 150 MHz up
    assert snap_lo(5.2e9) == 5.25e9  # 50 MHz up beats 200 MHz down
    assert snap_lo(5.125e9) in (5.0e9, 5.25e9)  # exactly between: either is nearest


def test_snapping_never_leaves_the_synthesizer_range():
    assert snap_lo(1.0e9) == LO_MIN_HZ
    assert snap_lo(25.0e9) == LO_MAX_HZ


def test_a_grid_point_snaps_to_itself():
    for lo in (LO_MIN_HZ, LO_MIN_HZ + LO_STEP_HZ, 6.25e9, LO_MAX_HZ):
        assert snap_lo(lo) == lo


def test_the_worst_case_mislabelling_snapping_prevents_is_half_a_step():
    """The size of the bug, stated: an unsnapped request can sit half a grid step
    from what the hardware plays, and the axis is derived from the request."""
    worst = max(abs(snap_lo(lo) - lo)
                for lo in (5.0e9 + k * 1e7 for k in range(26)))
    assert worst <= LO_STEP_HZ / 2 + 1.0


# --- putting the tree back ------------------------------------------------


class RecordingChannel:
    """A drive channel that records the ORDER of the writes it receives."""

    def __init__(self, *, octave=False, fails_on=None):
        self.writes: list[tuple[str, object]] = []
        self._fails_on = fails_on
        object.__setattr__(self, "_ready", True)
        if octave:
            self.frequency_converter_up = _Sub(self, "octave_lo")
        else:
            # A real MW port HAS a band; the restore guards on hasattr, so a stub
            # without one would silently exercise the wrong branch.
            self.opx_output = _Sub(self, "mw", band=1)

    def __setattr__(self, name, value):
        if getattr(self, "_ready", False) and name == "RF_frequency":
            if self._fails_on == "RF_frequency":
                raise ValueError("cannot overwrite a reference")
            self.writes.append(("RF_frequency", value))
        object.__setattr__(self, name, value)


class _Sub:
    """The port / converter half, reporting its writes to the parent."""

    def __init__(self, parent, kind, **preset):
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_kind", kind)
        for name, value in preset.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        self._parent.writes.append((f"{self._kind}.{name}", value))
        object.__setattr__(self, name, value)


def test_the_lo_is_restored_before_the_rf_that_depends_on_it():
    """An RF is only meaningful once its LO is correct, and the other order
    briefly asks the chain for an IF it cannot carry."""
    channel = RecordingChannel(octave=True)
    _restore_drive_channel(channel, band=None, mw_lo=None, octave_lo=5.5e9, rf=5.6e9)

    names = [name for name, _ in channel.writes]
    assert names == ["octave_lo.LO_frequency", "RF_frequency"]


def test_the_band_is_restored_before_the_lo_writes_that_must_land_inside_it():
    channel = RecordingChannel(octave=False)
    _restore_drive_channel(channel, band=2, mw_lo=5.5e9, octave_lo=None, rf=5.6e9)

    names = [name for name, _ in channel.writes]
    assert names == ["mw.band", "mw.upconverter_frequency", "RF_frequency"]


def test_none_means_the_sweep_recorded_nothing_not_restore_to_nothing():
    """A value the sweep never captured must be left alone, not zeroed."""
    channel = RecordingChannel(octave=True)
    _restore_drive_channel(channel, band=None, mw_lo=None, octave_lo=None, rf=None)
    assert channel.writes == []


def test_an_octave_channel_is_not_offered_the_mw_write_and_vice_versa():
    octave = RecordingChannel(octave=True)
    _restore_drive_channel(octave, band=1, mw_lo=5.5e9, octave_lo=5.5e9, rf=None)
    assert [n for n, _ in octave.writes] == ["octave_lo.LO_frequency"]

    mw = RecordingChannel(octave=False)
    _restore_drive_channel(mw, band=None, mw_lo=None, octave_lo=6.0e9, rf=None)
    assert mw.writes == []


# --- what the probes must not go back to ----------------------------------


@pytest.mark.parametrize("probe", [QUBIT_PROBE, RESONATOR_PROBE])
def test_neither_probe_still_sweeps_with_open_lo_limits(probe):
    """``0.0, float("inf")`` was the whole of the Octave bug: an undeclared chain
    swept the synthesizer below 2 GHz and labelled the result as though it had
    worked. Structural, because reaching the branch needs a cluster."""
    source = probe.read_text(encoding="utf-8")
    body = source[source.index("def probe("):]
    assert '0.0, float("inf")' not in body, "open LO limits are back"
    assert 'float("inf")' not in body, "an unbounded LO is back in some form"


@pytest.mark.parametrize("probe", [QUBIT_PROBE, RESONATOR_PROBE])
def test_both_probes_refuse_a_chain_whose_lo_range_they_cannot_bound(probe):
    body = probe.read_text(encoding="utf-8")
    assert "steps the drive LO" in body or "steps the readout LO" in body
    # A source grep, so the phrase must be one the message keeps on ONE line.
    assert "Refusing rather than sweeping with open limits" in body


@pytest.mark.parametrize("probe", [QUBIT_PROBE, RESONATOR_PROBE])
def test_both_probes_snap_the_lo_before_deriving_the_axis(probe):
    """``dfs`` is computed from the clamped LO, so the snap has to happen first
    or the segment carries an axis the hardware never played."""
    source = probe.read_text(encoding="utf-8")
    assert "snap_lo(" in source
    assert source.index("snap_lo(") < source.index("dfs =")


@pytest.mark.parametrize("probe", [QUBIT_PROBE, RESONATOR_PROBE])
def test_both_probes_warn_rather_than_strand_a_tree_they_could_not_restore(probe):
    source = probe.read_text(encoding="utf-8")
    assert "restore_failures" in source
    assert "still parked at a sub-band" in source


def test_the_qubit_probe_moves_the_synth_partner_with_its_target():
    """Two Octave outputs on one synthesizer cannot be tuned apart, so leaving
    the partner's stored LO behind would make the config claim one synthesizer
    is producing two frequencies."""
    source = QUBIT_PROBE.read_text(encoding="utf-8")
    assert "rf_outputs_sharing_synth" in source
    tree = ast.parse(source)
    assert any(isinstance(n, ast.Name) and n.id == "rf_outputs_sharing_synth"
               for n in ast.walk(tree))


def test_the_resonator_probe_says_it_does_not_track_a_shared_synth():
    """It steps only the readout line, so a partner output would be left where
    the last segment put it. An honest limitation beats a silent one."""
    source = RESONATOR_PROBE.read_text(encoding="utf-8")
    assert "shares a synthesizer" in source
    assert "restores only its own" in source
