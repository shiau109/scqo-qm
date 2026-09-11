"""The Octave mixer calibration: the step that had no caller at all.

An Octave up-converts a baseband IQ pair with an ANALOG mixer, so every output
carries LO leakage and an image sideband until it is calibrated at the exact
``(RF output, LO, gain)`` and ``(LO, IF)`` it will run at. An MW-FEM synthesizes
microwave directly and has no equivalent step -- which is why nothing in this
driver ever called for one, and why an Octave setup would have run every
measurement uncalibrated with nothing saying so.

Two things beyond "it runs" are pinned here:

* the refusals are BY NAME. A calibration command that reports zero calibrations
  on an MW-FEM tree looks exactly like success.
* the results live OUTSIDE state.json, so ``vendor_config_snapshot`` cannot
  capture them and two runs with byte-identical QUAM state can carry different
  mixer corrections. ``power_context`` records which one was in force.
"""

from __future__ import annotations

import os

import pytest

from types import SimpleNamespace as NS

from scqo_qm.backend.calibrate_octave import (
    calibrate,
    calibration_db,
    octave_lines,
)
from scqo_qm.quam_fields import octave_calibration_warnings


def up(*, lo=6e9, rf_output=1, octave="oct1"):
    return NS(LO_source="internal", LO_frequency=lo, gain=0.0,
              output_mode="always_on", input_attenuators="off",
              id=rf_output, octave=NS(name=octave))


def mw_port(band=1):
    return NS(band=band, full_scale_power_dbm=-11)


class Qubit:
    """A qubit whose ``calibrate_octave`` records how it was asked."""

    def __init__(self, name, *, readout="octave", drive="octave", raises=None):
        self.name = name
        self.calls = []
        self._raises = raises
        self.resonator = (NS(frequency_converter_up=up(rf_output=1))
                          if readout == "octave" else NS(opx_output=mw_port(2)))
        self.xy = (NS(frequency_converter_up=up(rf_output=2))
                   if drive == "octave" else NS(opx_output=mw_port(1)))

    def calibrate_octave(self, qm, calibrate_drive=True, calibrate_resonator=True):
        self.calls.append({"drive": calibrate_drive, "readout": calibrate_resonator})
        if self._raises is not None:
            raise self._raises


def machine_of(*qubits, octaves=("oct1",), db_path=None, active=None):
    by_name = {q.name: q for q in qubits}
    return NS(
        qubits=by_name,
        active_qubit_names=list(by_name) if active is None else active,
        octaves={n: NS(calibration_db_path=db_path) for n in octaves},
        connect=lambda: NS(),
        generate_config=lambda: {},
    )


def session_of(machine):
    return NS(backend=NS(machine=machine), backend_label="qm")


# --- which lines are on an Octave -----------------------------------------


def test_an_all_mw_tree_has_no_octave_lines():
    machine = machine_of(Qubit("q1", readout="mw", drive="mw"))
    assert octave_lines(machine) == {}


def test_a_mixed_tree_is_resolved_PER_LINE():
    """A tree may legitimately run an Octave readout beside an MW-FEM drive, and
    asking the vendor to calibrate an MW channel raises naming the element but
    not the reason."""
    machine = machine_of(Qubit("q1", readout="octave", drive="mw"))
    assert octave_lines(machine) == {"q1": {"readout": True, "drive": False}}


def test_only_active_qubits_are_considered():
    machine = machine_of(Qubit("q1"), Qubit("q2"), active=["q1"])
    assert set(octave_lines(machine)) == {"q1"}


def test_an_unknown_target_is_refused_naming_what_is_active():
    machine = machine_of(Qubit("q1"))
    with pytest.raises(SystemExit) as err:
        octave_lines(machine, ["q9"])
    assert "q9" in str(err.value) and "q1" in str(err.value)


# --- where the results go -------------------------------------------------


def test_a_declared_path_is_reported_absolute_and_not_a_fallback(tmp_path):
    machine = machine_of(Qubit("q1"), db_path=str(tmp_path))
    where = calibration_db(machine)["oct1"]
    assert where["path"] == os.path.abspath(str(tmp_path))
    assert where["fallback"] is False


def test_an_unset_path_is_reported_as_the_cwd_fallback():
    """quam resolves it to os.getcwd(), so the calibration follows the process
    rather than the setup -- which is a finding, not a default."""
    where = calibration_db(machine_of(Qubit("q1")))["oct1"]
    assert where["path"] == os.path.abspath(os.getcwd())
    assert where["fallback"] is True


# --- the refusals ---------------------------------------------------------


def test_a_backend_with_no_quam_tree_is_refused_by_name():
    session = NS(backend=NS(), backend_label="qblox")
    with pytest.raises(SystemExit) as err:
        calibrate(session=session)
    assert "qblox" in str(err.value)
    assert "scqo user" in str(err.value)  # how to find the selected setup


def test_an_mw_only_tree_is_refused_rather_than_reported_as_zero_calibrations():
    """THE refusal. 'calibrated 0/0' on an MW-FEM tree looks like success."""
    session = session_of(machine_of(Qubit("q1", readout="mw", drive="mw")))
    with pytest.raises(SystemExit) as err:
        calibrate(session=session)
    message = str(err.value)
    assert "MW-FEM" in message
    assert "no mixer" in message or "needs no calibration" in message


def test_selecting_only_the_line_that_is_on_an_mw_fem_is_refused_by_name():
    session = session_of(machine_of(Qubit("q1", readout="octave", drive="mw")))
    with pytest.raises(SystemExit, match="all on an MW-FEM"):
        calibrate(session=session, readout=False, drive=True)


def test_asking_for_neither_line_is_refused():
    session = session_of(machine_of(Qubit("q1")))
    with pytest.raises(SystemExit, match="mutually exclusive"):
        calibrate(session=session, drive=False, readout=False)


# --- the dry run ----------------------------------------------------------


def test_a_dry_run_reports_the_plan_and_the_destination_and_calibrates_nothing():
    qubit = Qubit("q1")
    session = session_of(machine_of(qubit))
    report = calibrate(session=session, dry_run=True)

    assert report["planned"] == {"q1": ["drive", "readout"]}
    assert report["calibration_db"]["oct1"]["fallback"] is True
    assert report["calibrated"] == {}
    assert qubit.calls == []


def test_a_dry_run_narrowed_to_one_line_plans_only_that_line():
    session = session_of(machine_of(Qubit("q1")))
    report = calibrate(session=session, dry_run=True, drive=False)
    assert report["planned"] == {"q1": ["readout"]}


# --- the real run ---------------------------------------------------------


@pytest.fixture()
def no_cluster(monkeypatch):
    """``qm_session`` without a cluster: the calibration itself is the vendor's."""
    import contextlib

    import qualang_tools.multi_user as multi_user

    @contextlib.contextmanager
    def fake_session(qmm, config, timeout=None):
        yield NS(name="fake-qm")

    monkeypatch.setattr(multi_user, "qm_session", fake_session)


def test_each_planned_qubit_is_asked_for_exactly_its_own_lines(no_cluster):
    q1, q2 = Qubit("q1"), Qubit("q2", drive="mw")
    session = session_of(machine_of(q1, q2))

    report = calibrate(session=session)

    assert q1.calls == [{"drive": True, "readout": True}]
    assert q2.calls == [{"drive": False, "readout": True}]
    assert report["calibrated"] == {"q1": ["drive", "readout"], "q2": ["readout"]}
    assert report["success"] is True


def test_a_line_with_nothing_to_calibrate_is_RECORDED_not_printed(no_cluster):
    """The vendor's own calibrate_octave_ports swallows NoCalibrationElements
    with a print. 'this element had nothing to calibrate' is a finding an
    operator needs in the report, not console noise."""
    from qm.octave.octave_mixer_calibration import NoCalibrationElements

    qubit = Qubit("q1", raises=NoCalibrationElements("nothing to calibrate"))
    report = calibrate(session=session_of(machine_of(qubit)))

    assert report["skipped"] == {"q1": "no calibration elements"}
    assert report["calibrated"] == {}
    assert report["success"] is True  # not an error


def test_one_failing_element_does_not_abort_the_rest(no_cluster):
    """Half a calibration is the worst outcome: the operator would have to work
    out which half."""
    bad = Qubit("q1", raises=RuntimeError("mixer did not settle"))
    good = Qubit("q2")
    report = calibrate(session=session_of(machine_of(bad, good)))

    assert "q2" in report["calibrated"]
    assert any("q1" in e and "mixer did not settle" in e for e in report["errors"])
    assert report["success"] is False


def test_a_target_filter_calibrates_only_that_qubit(no_cluster):
    q1, q2 = Qubit("q1"), Qubit("q2")
    calibrate(session=session_of(machine_of(q1, q2)), targets=["q2"])
    assert q1.calls == []
    assert q2.calls == [{"drive": True, "readout": True}]


# --- the advisory ---------------------------------------------------------


def test_an_unset_calibration_db_path_warns_and_names_the_fix():
    advisories = octave_calibration_warnings(machine_of(Qubit("q1")))
    assert len(advisories) == 1
    assert "working directory" in advisories[0]
    assert "backend_config/" in advisories[0]
    assert advisories[0].isascii()  # it lands on a lab console


def test_a_declared_calibration_db_path_is_silent(tmp_path):
    machine = machine_of(Qubit("q1"), db_path=str(tmp_path))
    assert octave_calibration_warnings(machine) == []


def test_a_tree_with_no_octaves_is_silent():
    assert octave_calibration_warnings(machine_of(Qubit("q1"), octaves=())) == []


# --- what the run record says ---------------------------------------------


def test_power_context_records_which_mixer_calibration_was_in_force(
        backend, stub_machine, tmp_path):
    """The provenance gap this closes: the corrections live OUTSIDE state.json,
    so a setup snapshot cannot capture them and two runs with identical QUAM
    state can carry different ones."""
    db = tmp_path / "calibration_db.json"
    db.write_text('{"lo_cal": {}}', encoding="utf-8")

    resonator = stub_machine.qubits["q1"].resonator
    del resonator.opx_output
    resonator.frequency_converter_up = NS(
        LO_source="internal", LO_frequency=5.9e9, gain=-10.0, id=1,
        octave=NS(name="oct1", calibration_db_path=str(tmp_path)))

    recorded = backend.power_context(["q1"])["q1"]["octave_calibration_db"]
    assert recorded["path"] == str(db)
    assert recorded["present"] is True
    assert recorded["declared"] is True
    assert len(recorded["sha16"]) == 16
    assert "lo_cal" not in str(recorded)  # digest, never the contents


def test_an_absent_calibration_db_is_recorded_as_absent(
        backend, stub_machine, tmp_path):
    """Its ABSENCE is exactly what a later reader wants recorded against a run
    whose spectra look mirrored."""
    resonator = stub_machine.qubits["q1"].resonator
    del resonator.opx_output
    resonator.frequency_converter_up = NS(
        LO_source="internal", LO_frequency=5.9e9, gain=0.0, id=1,
        octave=NS(name="oct1", calibration_db_path=str(tmp_path)))

    recorded = backend.power_context(["q1"])["q1"]["octave_calibration_db"]
    assert recorded["present"] is False
    assert "sha16" not in recorded


def test_an_mw_chain_records_no_calibration_block_at_all(backend):
    """It has no mixer, so an empty or absent fingerprint would be a claim about
    a file that has nothing to do with it."""
    assert "octave_calibration_db" not in backend.power_context(["q1"])["q1"]
