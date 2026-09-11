"""What the Octave can actually produce -- and why nothing else was checking.

An MW-FEM tree needs none of these guards: its LO is a port field inside a
declared ``band``, and the instrument rejects a band that does not cover it. An
Octave has neither. Its LO is a synthesizer on a 250 MHz grid, its IF is a real
analog baseband signal bounded at +/-400 MHz rather than a digital offset, and
some of its RF outputs SHARE a synthesizer and cannot be tuned apart.

None of that is checked anywhere in ``qm``, ``quam`` or ``quam_builder`` -- the
only guards that ever existed are two ``assert`` lines in
``quam_config/populate_quam_opxp_octave.py``, which run when a tree is first
seeded and never again.

Two of the failures here are completely silent on hardware:

* an unset down-converter LO makes quam omit the whole ``RF_inputs`` entry, so
  the config has no receive path and nothing says so;
* ``output_mode`` defaults to ``always_off``, so a tree that never set it emits
  nothing and the run returns a clean flat line.
"""

from __future__ import annotations

import pytest

from types import SimpleNamespace as NS

from scqo_qm._octave import (
    IF_MAX_ABS_HZ,
    LO_MAX_HZ,
    LO_MIN_HZ,
    LO_STEP_HZ,
    if_window_problem,
    lo_grid_problem,
    rf_outputs_sharing_synth,
    synth_of_rf_output,
)
from scqo_qm.quam_fields import (
    octave_frequency_problems,
    octave_output_problems,
)


def up(lo: float, *, rf_output: int = 1, output_mode: str = "always_on",
       octave: str = "oct1"):
    return NS(LO_source="internal", LO_frequency=lo, gain=0.0,
              output_mode=output_mode, input_attenuators="off",
              id=rf_output, octave=NS(name=octave))


def down(lo):
    return NS(LO_frequency=lo, IF_mode_I="direct", IF_mode_Q="direct", id=1)


def octave_qubit(name: str, *, xy_lo=5.0e9, xy_rf=5.1e9, rr_lo=6.0e9,
                 rr_rf=6.1e9, xy_out=2, rr_out=1, down_lo="same",
                 xy_mode="always_on"):
    """A qubit whose drive and readout both run through an Octave."""
    resonator = NS(RF_frequency=rr_rf,
                   frequency_converter_up=up(rr_lo, rf_output=rr_out),
                   frequency_converter_down=down(rr_lo if down_lo == "same"
                                                 else down_lo))
    return NS(name=name, f_01=xy_rf,
              xy=NS(RF_frequency=xy_rf,
                    frequency_converter_up=up(xy_lo, rf_output=xy_out,
                                              output_mode=xy_mode)),
              resonator=resonator)


def machine_of(*qubits, active="all"):
    by_name = {q.name: q for q in qubits}
    return NS(qubits=by_name,
              active_qubits=list(by_name.values()) if active == "all" else active)


# --- the pure hardware facts ----------------------------------------------


def test_the_lo_grid_accepts_its_own_points():
    for lo in (LO_MIN_HZ, LO_MIN_HZ + LO_STEP_HZ, 5.0e9, 6.25e9, LO_MAX_HZ):
        assert lo_grid_problem(lo) is None, lo


def test_an_off_grid_lo_is_named_as_off_grid_not_as_out_of_range():
    """Two different mistakes with two different fixes; one message for both
    would send the operator to work out which they made."""
    reason = lo_grid_problem(5.1e9)
    assert "off the 0.25 GHz LO grid" in reason
    assert "range" not in reason


def test_an_out_of_range_lo_names_the_synthesizer_range():
    assert "outside the Octave synthesizer range" in lo_grid_problem(1.0e9)
    assert "outside the Octave synthesizer range" in lo_grid_problem(20.0e9)


def test_the_if_window_is_plus_minus_400_mhz_either_way():
    assert if_window_problem(5.0e9 + IF_MAX_ABS_HZ, 5.0e9) is None
    assert if_window_problem(5.0e9 - IF_MAX_ABS_HZ, 5.0e9) is None
    assert if_window_problem(5.0e9 + IF_MAX_ABS_HZ + 1e6, 5.0e9) is not None
    assert if_window_problem(5.0e9 - IF_MAX_ABS_HZ - 1e6, 5.0e9) is not None


def test_the_synth_map_is_the_octaves_own_grouping():
    """synth1: RF1 (+ RFin1), synth2: RF2 + RF3, synth3: RF4 + RF5."""
    assert synth_of_rf_output(1) == 1
    assert synth_of_rf_output(2) == synth_of_rf_output(3) == 2
    assert synth_of_rf_output(4) == synth_of_rf_output(5) == 3
    assert synth_of_rf_output(9) is None

    assert rf_outputs_sharing_synth(1) == ()  # a synth to itself
    assert rf_outputs_sharing_synth(2) == (3,)
    assert rf_outputs_sharing_synth(5) == (4,)


# --- the frequency audit --------------------------------------------------


def test_a_compliant_octave_tree_is_silent():
    assert octave_frequency_problems(machine_of(octave_qubit("q1"))) == []


def test_the_audit_is_chain_scoped_so_an_mw_tree_says_nothing():
    """It has no MW-FEM counterpart by construction -- the instrument enforces
    band coverage there -- so it must not invent findings on one."""
    mw = NS(name="q1", f_01=5e9,
            xy=NS(RF_frequency=5e9, opx_output=NS(band=1, full_scale_power_dbm=-11)),
            resonator=NS(RF_frequency=6e9,
                         opx_output=NS(band=2, full_scale_power_dbm=-11)))
    assert octave_frequency_problems(machine_of(mw)) == []


def test_an_up_converter_with_no_lo_is_caught():
    qubit = octave_qubit("q1")
    qubit.xy.frequency_converter_up.LO_frequency = None
    problems = octave_frequency_problems(machine_of(qubit))
    assert any("qubits.q1.xy" in p and "no LO frequency" in p for p in problems)


def test_an_off_grid_lo_is_caught_and_told_to_re_check_the_if():
    qubit = octave_qubit("q1", xy_lo=5.1e9)
    problem = next(p for p in octave_frequency_problems(machine_of(qubit))
                   if "qubits.q1.xy" in p)
    assert "off the 0.25 GHz LO grid" in problem
    assert "re-check IF = RF - LO" in problem


def test_an_if_outside_the_window_is_caught_and_points_at_the_LO():
    """The RF is the physics; the LO is the setting. Telling someone to move the
    RF would be telling them to measure a different qubit."""
    qubit = octave_qubit("q1", xy_lo=5.0e9, xy_rf=5.6e9)
    problem = next(p for p in octave_frequency_problems(machine_of(qubit))
                   if "qubits.q1.xy" in p)
    assert "600 MHz" in problem and "+/-400 MHz" in problem
    assert "Move the LO" in problem


def test_an_unset_down_converter_lo_is_caught_as_the_quietest_failure():
    qubit = octave_qubit("q1", down_lo=None)
    problem = next(p for p in octave_frequency_problems(machine_of(qubit))
                   if "DOWN-converter" in p)
    assert "no receive path" in problem
    assert "quietest failure" in problem


def test_a_down_converter_lo_that_disagrees_with_the_up_converter_is_caught():
    qubit = octave_qubit("q1", rr_lo=6.0e9, down_lo=6.25e9)
    problem = next(p for p in octave_frequency_problems(machine_of(qubit))
                   if "down-converter LO" in p)
    assert "6.25 GHz" in problem and "6 GHz" in problem
    assert "quadratures" in problem  # what actually goes wrong


# --- the constraint with no MW-FEM analogue -------------------------------


def test_two_outputs_on_one_synth_cannot_be_tuned_apart():
    """RF2 and RF3 share synth2. An MW-FEM pairs its ports only for their band
    and each keeps its own upconverter_frequency, so this is exactly the habit
    that makes the Octave surprising."""
    q1 = octave_qubit("q1", xy_out=2, xy_lo=5.0e9)
    q2 = octave_qubit("q2", xy_out=3, xy_lo=5.5e9, rr_out=1, rr_lo=6.0e9)
    problem = next(p for p in octave_frequency_problems(machine_of(q1, q2))
                   if "synth2" in p)
    assert "qubits.q1.xy at 5 GHz" in problem
    assert "qubits.q2.xy at 5.5 GHz" in problem
    assert "different synthesizers" in problem  # the re-wire
    assert "split them in the IF" in problem    # the alternative


def test_two_outputs_on_one_synth_at_the_SAME_lo_are_fine():
    q1 = octave_qubit("q1", xy_out=2, xy_lo=5.0e9)
    q2 = octave_qubit("q2", xy_out=3, xy_lo=5.0e9, xy_rf=5.2e9, rr_out=1)
    assert not [p for p in octave_frequency_problems(machine_of(q1, q2))
                if "synth" in p]


def test_outputs_on_DIFFERENT_synths_may_differ_freely():
    q1 = octave_qubit("q1", xy_out=2, xy_lo=5.0e9)
    q2 = octave_qubit("q2", xy_out=4, xy_lo=7.0e9, xy_rf=7.1e9, rr_out=1)
    assert not [p for p in octave_frequency_problems(machine_of(q1, q2))
                if "synth" in p]


def test_a_converter_with_no_id_is_skipped_rather_than_guessed_at():
    """An audit that guesses which output a channel is on would accuse the wrong
    pair of sharing a synthesizer."""
    q1 = octave_qubit("q1", xy_out=2, xy_lo=5.0e9)
    q2 = octave_qubit("q2", xy_out=3, xy_lo=5.5e9, rr_out=1)
    q2.xy.frequency_converter_up.id = None
    assert not [p for p in octave_frequency_problems(machine_of(q1, q2))
                if "synth" in p]


# --- the RF switch --------------------------------------------------------


def test_an_always_off_output_on_an_active_qubit_is_caught():
    qubit = octave_qubit("q1", xy_mode="always_off")
    problem = next(iter(octave_output_problems(machine_of(qubit))))
    assert "qubits.q1.xy" in problem
    assert "flat line" in problem
    assert "DEFAULT" in problem  # why a hand-assembled tree lands here


def test_parking_an_INACTIVE_line_by_switching_it_off_is_legitimate():
    """Refusing it everywhere would block a config someone deliberately chose."""
    qubit = octave_qubit("q1", xy_mode="always_off")
    assert octave_output_problems(machine_of(qubit, active=[])) == []


@pytest.mark.parametrize("mode", ["always_on", "triggered", "triggered_reversed"])
def test_the_real_modes_are_not_flagged(mode):
    """``triggered`` is driven by digital markers, which QUAM wires up."""
    assert octave_output_problems(machine_of(octave_qubit("q1", xy_mode=mode))) == []


def test_the_output_audit_is_chain_scoped_too():
    mw = NS(name="q1", f_01=5e9,
            xy=NS(RF_frequency=5e9, opx_output=NS(band=1, full_scale_power_dbm=-11)),
            resonator=NS(RF_frequency=6e9,
                         opx_output=NS(band=2, full_scale_power_dbm=-11)))
    assert octave_output_problems(machine_of(mw)) == []


# --- the console contract -------------------------------------------------


def test_every_finding_stays_ascii():
    """These land in a SystemExit at session start, on a lab console."""
    qubit = octave_qubit("q1", xy_lo=5.1e9, xy_rf=5.9e9, down_lo=None,
                         xy_mode="always_off")
    findings = (octave_frequency_problems(machine_of(qubit))
                + octave_output_problems(machine_of(qubit)))
    assert findings
    assert all(f.isascii() for f in findings)
