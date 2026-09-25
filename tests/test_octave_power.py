"""The Octave output chain solve: hold the gain, spend the amplitude.

Absolute power needs two vendor knobs, and on an Octave the coarse one is not
free to move. The mixer calibration is cached per ``(RF output, LO, gain)`` --
gain is part of the KEY -- so re-staging it invalidates the cached LO-leakage
correction for that output. On a multiplexed feedline it is worse: the gain
belongs to the Octave RF output that every qubit on the line shares, so moving it
for one target moves it for all of them, and the per-target difference can only
ever live in the amplitude.

That makes IDEMPOTENCE the load-bearing property here, not accuracy: setting a
channel to the power it is already producing must not touch the gain. A solve
that re-derived both knobs from scratch each time would be arithmetically perfect
and would still invalidate the mixer calibration on every ``scqo set``.
"""

from __future__ import annotations

import math

import pytest

from scqo_qm._octave import (
    GAIN_MAX_DB,
    GAIN_MIN_DB,
    MAX_IF_AMP_V,
    OPTIMUM_IF_AMP_V,
)
from scqo_qm.backend._power import (
    OCTAVE_MAX_POWER_DBM,
    OCTAVE_MIN_AMP_V,
    OCTAVE_MIN_POWER_DBM,
    dbm_to_volts,
    snap_gain,
    solve_octave_chain,
    volts_to_dbm,
)


# --- the conversions must BE the vendor's, not merely resemble them --------


@pytest.mark.parametrize("volts", [0.5, 0.125, 0.05, 0.025, 0.005])
def test_volts_to_dbm_matches_qualang_tools_exactly(volts):
    """The solve inverts the reader in ``get_output_power_iq_channel``. A half-dB
    disagreement would make every set/get round trip drift by that much."""
    from qualang_tools.units import unit

    assert volts_to_dbm(volts) == pytest.approx(unit().volts2dBm(volts), abs=1e-12)


@pytest.mark.parametrize("dbm", [3.98, -8.06, -20.0, -40.0])
def test_dbm_to_volts_matches_qualang_tools_exactly(dbm):
    from qualang_tools.units import unit

    assert dbm_to_volts(dbm) == pytest.approx(unit().dBm2volts(dbm), abs=1e-12)


def test_the_two_conversions_invert_each_other():
    for volts in (0.5, 0.125, 0.005):
        assert dbm_to_volts(volts_to_dbm(volts)) == pytest.approx(volts)


# --- the gain grid --------------------------------------------------------


def test_snap_gain_lands_on_the_half_db_grid_and_inside_the_bounds():
    assert snap_gain(3.3) == 3.5
    assert snap_gain(3.2) == 3.0
    assert snap_gain(-100.0) == GAIN_MIN_DB
    assert snap_gain(100.0) == GAIN_MAX_DB


def test_the_CURRENT_gain_is_snapped_too_not_just_a_new_one():
    """A tree can hold an off-grid gain (a hand edit, or a float that round-tripped
    through JSON). Solving an amplitude against a gain the hardware will not take
    is a silent half-dB error in everything downstream."""
    solution = solve_octave_chain(-20.0, current_gain_db=-10.3, name="q1_ro")
    assert solution.gain_db == -10.5
    assert not solution.gain_moved  # snapping is not "moving"


# --- the policy -----------------------------------------------------------


def test_the_gain_is_held_whenever_the_amplitude_can_express_the_target():
    solution = solve_octave_chain(0.0, current_gain_db=0.0, name="q1_xy")
    assert solution.gain_db == 0.0
    assert solution.gain_moved is False
    assert solution.reason is None
    assert OCTAVE_MIN_AMP_V <= solution.amplitude_v < MAX_IF_AMP_V


def test_setting_the_power_a_chain_already_produces_moves_nothing():
    """THE property. Every ``scqo set`` of an unchanged value, and every writeback
    that lands on the same number, must leave the mixer calibration valid."""
    gain, amplitude = -20.0, 0.0316227766
    current = gain + volts_to_dbm(amplitude)

    solution = solve_octave_chain(current, current_gain_db=gain, name="q1_ro")

    assert solution.gain_db == gain
    assert solution.gain_moved is False
    assert solution.amplitude_v == pytest.approx(amplitude, rel=1e-9)


def test_solving_twice_is_stable_even_when_the_first_solve_moved_the_gain():
    """The second call must be a no-op, or a sweep would walk the gain (and the
    mixer calibration) one step per point."""
    first = solve_octave_chain(-40.0, current_gain_db=0.0, name="q1_ro")
    assert first.gain_moved

    second = solve_octave_chain(-40.0, current_gain_db=first.gain_db, name="q1_ro")
    assert second.gain_moved is False
    assert second.gain_db == first.gain_db
    assert second.amplitude_v == pytest.approx(first.amplitude_v)


def test_the_gain_moves_only_when_the_amplitude_cannot_reach_and_says_why():
    """+20 dBm at gain 0 would need 5 V, ten times the DAC rail."""
    solution = solve_octave_chain(20.0, current_gain_db=0.0, name="q1_xy")
    assert solution.gain_moved is True
    assert solution.gain_db > 0.0
    assert solution.amplitude_v < MAX_IF_AMP_V
    assert "gain moved up" in solution.reason
    assert "20.0 dBm" in solution.reason


def test_a_restaged_gain_aims_the_amplitude_at_the_mixer_optimum():
    """0.125 V is the Octave up-conversion mixer's optimum drive, so a gain that
    has to move should land there rather than anywhere merely legal."""
    solution = solve_octave_chain(10.0, current_gain_db=-20.0, name="q1_xy")
    assert solution.amplitude_v == pytest.approx(OPTIMUM_IF_AMP_V, rel=0.15)


def test_an_amplitude_far_under_the_floor_restages_the_gain_downward():
    """Not a hard limit -- a DAC-resolution one: below the floor the amplitude is
    using under 1% of full scale."""
    solution = solve_octave_chain(-50.0, current_gain_db=10.0, name="q1_ro")
    assert solution.gain_moved is True
    assert solution.gain_db < 10.0
    assert "gain moved down" in solution.reason


def test_the_vendors_own_multiplexed_readout_recipe_does_not_trip_the_floor():
    """``populate_quam_opxp_octave.py`` divides the target amplitude by the qubit
    count -- 0.125/5 = 0.025 V on a five-qubit feedline. A floor that re-staged
    the gain for THAT would fight the standard config on every readout set."""
    amplitude = OPTIMUM_IF_AMP_V / 5
    assert amplitude > OCTAVE_MIN_AMP_V
    target = -20.0 + volts_to_dbm(amplitude)
    solution = solve_octave_chain(target, current_gain_db=-20.0, name="q1_ro")
    assert solution.gain_moved is False


# --- the edges ------------------------------------------------------------


def test_a_target_above_the_chains_reach_is_refused_naming_the_window():
    with pytest.raises(ValueError) as err:
        solve_octave_chain(40.0, current_gain_db=0.0, name="q1_xy")
    message = str(err.value)
    assert "q1_xy" in message
    assert f"{OCTAVE_MAX_POWER_DBM:.1f}" in message
    assert "attenuation" in message  # what to actually change


def test_a_target_below_the_chains_reach_is_refused_the_same_way():
    with pytest.raises(ValueError, match="outside what an Octave output"):
        solve_octave_chain(OCTAVE_MIN_POWER_DBM - 10.0, current_gain_db=0.0,
                           name="q1_ro")


def test_the_reachable_window_is_the_gain_range_around_the_dac_rail():
    assert OCTAVE_MAX_POWER_DBM == pytest.approx(
        GAIN_MAX_DB + volts_to_dbm(MAX_IF_AMP_V))
    assert OCTAVE_MIN_POWER_DBM == pytest.approx(
        GAIN_MIN_DB + volts_to_dbm(OCTAVE_MIN_AMP_V))


def test_every_solution_inside_the_window_round_trips_to_its_target():
    """The solve and the vendor's reader must agree, at every reachable power."""
    for target in (-55.0, -40.0, -20.0, -8.0, 0.0, 10.0, 23.0):
        solution = solve_octave_chain(target, current_gain_db=0.0, name="q")
        read_back = solution.gain_db + volts_to_dbm(solution.amplitude_v)
        assert read_back == pytest.approx(target, abs=1e-9)
        assert math.isfinite(solution.amplitude_v)
        assert 0.0 < solution.amplitude_v < MAX_IF_AMP_V
        assert GAIN_MIN_DB <= solution.gain_db <= GAIN_MAX_DB
