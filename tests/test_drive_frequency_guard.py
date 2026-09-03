"""The drive-frequency invariant: the knob scqo reads must BE the frequency that plays.

scqo's ``drive_freq_hz`` READS ``q.f_01`` (``quam_fields.get_drive_freq``), while
the drive line PLAYS ``q.xy.RF_frequency``: QUAM derives
``intermediate_frequency = RF - LO`` from it, and every probe's
``update_frequency`` starts from that IF. The official nodes write the pair
together (03a_qubit_spectroscopy, 06a_ramsey), ``set_drive_freq`` writes both
to one absolute value, and the live 5Q4C state carries them bit-identical. A
hand edit of one - or a node that moves only the RF, like the vendored
17_pi_vs_flux_long_distortions - splits them SILENTLY: the write succeeds, the
knob reports a number, and the hardware emits a different one. Hence an audit
at session construction rather than a convention.
"""

import math
from types import SimpleNamespace

from scqo_qm.quam_fields import (
    DRIVE_FREQ_TOLERANCE_HZ,
    drive_frequency_problems,
    set_drive_freq,
)

from conftest import make_stub_machine


def test_the_stub_tree_is_compliant():
    """Guard for every other case here: the shared fixture must start clean, or a
    later assertion could pass on the wrong problem."""
    assert drive_frequency_problems(make_stub_machine()) == []


def test_a_mismatch_is_named_with_both_numbers_and_the_fix():
    """The message must carry BOTH numbers and say which one the hardware plays:
    the operator's question is 'which value is live', and the audit cannot know
    which of the two was intended -- so it names the edit, not a winner."""
    machine = make_stub_machine()
    machine.qubits["q1"].xy.RF_frequency = 4.9e9  # f_01 stays 4.8e9

    problems = drive_frequency_problems(machine)
    assert len(problems) == 1
    msg = problems[0]
    assert "qubits.q1.f_01" in msg and "qubits.q1.xy.RF_frequency" in msg
    assert "4800000000.0" in msg      # what drive_freq_hz reads
    assert "4900000000.0" in msg      # what the drive line plays
    assert "100000000.0" in msg       # the disagreement
    assert "update_frequency" in msg  # why the RF is the one that plays
    assert "state.json" in msg        # the fix
    assert msg.isascii()


def test_the_setter_repairs_a_mismatch():
    """The invariant the audit enforces is the one set_drive_freq maintains: one
    absolute write lands on both stores, so an accepted drive_freq_hz on a split
    tree heals it instead of carrying the offset along (the old delta semantics)."""
    machine = make_stub_machine()
    q1 = machine.qubits["q1"]
    q1.xy.RF_frequency = 4.9e9
    assert drive_frequency_problems(machine)

    set_drive_freq(q1, 4.85e9)
    assert q1.f_01 == 4.85e9 and q1.xy.RF_frequency == 4.85e9
    assert drive_frequency_problems(machine) == []


def test_an_uncalibrated_qubit_is_not_flagged():
    """f_01=None is legitimate on a real state file (state_lib/10Q and
    quam_state_6q carry it; qm_backend.snapshot tolerates it). Nothing to
    compare against is not a disagreement."""
    machine = make_stub_machine()
    machine.qubits["q2"].f_01 = None
    assert drive_frequency_problems(machine) == []


def test_an_unresolved_rf_is_not_an_accusation():
    """A ``#``-reference that did not resolve (QUAM hands back the string), an
    unset RF, or a non-finite one: the tree cannot answer, so the audit skips
    rather than inventing a verdict."""
    for rf in ("#./inferred_RF_frequency", None, float("nan"), float("inf")):
        machine = make_stub_machine()
        machine.qubits["q2"].xy.RF_frequency = rf
        assert drive_frequency_problems(machine) == [], rf


def test_a_qubit_without_a_drive_line_is_not_flagged():
    """No xy subtree, nothing plays, nothing to govern."""
    machine = make_stub_machine()
    del machine.qubits["q3"].xy
    assert drive_frequency_problems(machine) == []


def test_float_echo_is_below_the_tolerance_and_a_hand_edit_is_above():
    """ulp noise at 5 GHz is ~1e-6 Hz; a hand edit is kHz or more. The 1 Hz
    tolerance separates the two with room to spare: one ulp passes, anything
    past the constant is flagged."""
    machine = make_stub_machine()
    q1 = machine.qubits["q1"]
    q1.xy.RF_frequency = math.nextafter(q1.f_01, math.inf)
    assert drive_frequency_problems(machine) == []

    q1.xy.RF_frequency = q1.f_01 + 2 * DRIVE_FREQ_TOLERANCE_HZ
    assert len(drive_frequency_problems(machine)) == 1


def test_every_bad_qubit_is_reported_in_one_pass():
    """The reason this is an audit and not a setter check: the operator fixing a
    state file wants the whole list, not the first qubit a probe touched."""
    machine = make_stub_machine()
    machine.qubits["q1"].xy.RF_frequency = 4.9e9
    machine.qubits["q3"].xy.RF_frequency = 5.5e9
    problems = drive_frequency_problems(machine)
    assert len(problems) == 2
    assert any("qubits.q1." in p for p in problems)
    assert any("qubits.q3." in p for p in problems)
    assert all("state.json" in p for p in problems)


def test_a_tree_without_qubits_is_empty_not_an_error():
    """A partial tree (no qubits attribute at all) is 'cannot determine'."""
    assert drive_frequency_problems(SimpleNamespace()) == []
