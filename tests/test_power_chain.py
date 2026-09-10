"""Absolute power is the one knob family whose vendor home differs per RF chain.

``readout_power_dbm`` / ``drive_power_dbm`` are one neutral number realized by TWO
vendor knobs, and which two depends on the chain: MW-FEM stages
``opx_output.full_scale_power_dbm`` against a normalized amplitude, an Octave
stages ``frequency_converter_up.gain`` against an amplitude in volts. Before this
dispatch existed the whole family failed on an Octave in the quietest possible
way:

* the accessor raised a bare ``AttributeError`` naming neither target nor reason;
* ``_read_or_none`` caught it and put ``None`` in the device snapshot -- for every
  qubit, every session;
* ``power_context``'s ``except Exception`` caught it too and wrote ``{}`` into the
  run record, which is exactly what it writes for a target that has no readout
  chain at all.

So the operator's evidence for "this driver cannot do absolute power here" was two
indistinguishable blanks. These tests pin the dispatch, the loud refusal for the
chains that remain unsolvable, and what the run record ends up saying.
"""

from __future__ import annotations

import pytest

from types import SimpleNamespace as NS

from scqo_qm import quam_fields
from scqo_qm.backend._power import volts_to_dbm
from scqo_qm.backend.qm_backend import _power_chain


def octave_channel(gain: float = 0.0, amplitude: float = 0.05):
    """A drive/readout channel on an Octave: an IQ pair plus an upconverter, and
    conspicuously no ``opx_output``.

    It carries real operations because the point under test is the CHAIN, and a
    channel with no pulses would fail earlier for an unrelated reason -- which is
    how the first draft of these tests passed while asserting nothing.
    """
    return NS(
        RF_frequency=6e9,
        LO_frequency=5.9e9,
        opx_output_I=NS(feedforward_filter=None),
        opx_output_Q=NS(feedforward_filter=None),
        frequency_converter_up=NS(LO_source="internal", LO_frequency=5.9e9,
                                  gain=gain, output_mode="always_on",
                                  input_attenuators="off"),
        operations={
            quam_fields.READOUT_OPERATION: NS(amplitude=amplitude, length=2000),
            quam_fields.SATURATION_OPERATION: NS(amplitude=amplitude, length=1000),
        },
    )


def mixer_channel(amplitude: float = 0.02):
    """The external-analog-mixer chain: same IQ shape, a converter with no gain
    knob this driver can reach.

    Carries operations for the same reason ``octave_channel`` does: a channel with
    no pulses fails earlier, for an unrelated reason, and a test that accepts THAT
    failure is asserting nothing about the chain.
    """
    return NS(frequency_converter_up=NS(local_oscillator=NS(frequency=6e9),
                                        mixer=NS(), gain=0.0),
              operations={
                  quam_fields.READOUT_OPERATION: NS(amplitude=amplitude, length=2000),
                  quam_fields.SATURATION_OPERATION: NS(amplitude=amplitude, length=1000),
              })


# --- the dispatch ---------------------------------------------------------


def test_both_solved_chains_are_named_rather_than_refused():
    assert _power_chain(NS(opx_output=NS(band=2, full_scale_power_dbm=-11)),
                        name="q1_ro", field="readout_power_dbm") == "mw_fem"
    assert _power_chain(octave_channel(), name="q1_ro",
                        field="readout_power_dbm") == "octave"


def test_an_external_mixer_is_refused_naming_the_target_field_and_the_remedy():
    """Its power lives on the LO's own source, so pointing the operator at an
    Octave gain would send them to a knob that does not exist."""
    with pytest.raises(ValueError) as err:
        _power_chain(mixer_channel(), name="q1_xy", field="drive_power_dbm")
    message = str(err.value)
    assert "q1_xy.drive_power_dbm" in message
    assert "external analog mixer" in message
    assert "drive_amp" in message  # what to reach for instead


def test_a_channel_declaring_no_chain_says_so_rather_than_guessing():
    with pytest.raises(ValueError) as err:
        _power_chain(NS(), name="q1_ro", field="readout_power_dbm")
    message = str(err.value)
    assert "no RF chain" in message
    assert "state.json" in message  # where to go look


# --- through the channel views --------------------------------------------


def test_the_readout_view_reads_and_writes_an_octave_chain(backend, stub_machine):
    resonator = octave_channel(gain=-20.0, amplitude=0.0316227766)
    stub_machine.qubits["q1"].resonator = resonator
    view = backend.device.component("q1_ro")

    assert view.readout_power_dbm == pytest.approx(-40.0, abs=1e-6)

    view.readout_power_dbm = -35.0
    assert view.readout_power_dbm == pytest.approx(-35.0, abs=1e-6)
    assert resonator.frequency_converter_up.gain == -20.0  # amplitude carried it


def test_the_drive_view_reads_and_writes_an_octave_chain(backend, stub_machine):
    xy = octave_channel(gain=0.0, amplitude=0.125)
    stub_machine.qubits["q1"].xy = xy
    view = backend.device.component("q1_xy")

    assert view.drive_power_dbm == pytest.approx(volts_to_dbm(0.125), abs=1e-9)

    view.drive_power_dbm = -12.0
    assert view.drive_power_dbm == pytest.approx(-12.0, abs=1e-6)


def test_a_write_that_must_move_the_gain_warns_about_the_mixer_calibration(
        backend, stub_machine):
    """The coupling that makes this driver's policy different from the MW one: the
    Octave mixer calibration is cached per (RF output, LO, gain), so a moved gain
    silently invalidates it -- and on a multiplexed feedline it moved every other
    channel on that output too."""
    stub_machine.qubits["q1"].resonator = octave_channel(gain=0.0, amplitude=0.05)
    view = backend.device.component("q1_ro")

    with pytest.warns(RuntimeWarning, match="mixer calibration"):
        view.readout_power_dbm = 20.0

    assert view.readout_power_dbm == pytest.approx(20.0, abs=1e-6)


def test_a_write_the_amplitude_can_absorb_warns_about_nothing(
        backend, stub_machine, recwarn):
    stub_machine.qubits["q1"].resonator = octave_channel(gain=-20.0, amplitude=0.03)
    view = backend.device.component("q1_ro")

    view.readout_power_dbm = -38.0

    assert not [w for w in recwarn if "mixer calibration" in str(w.message)]


def test_an_external_mixer_view_still_refuses_by_name(backend, stub_machine):
    stub_machine.qubits["q1"].resonator = mixer_channel()
    view = backend.device.component("q1_ro")
    with pytest.raises(ValueError, match="external analog mixer"):
        view.readout_power_dbm


# --- what the run record ends up saying -----------------------------------


def test_power_context_records_the_octave_gain_not_a_full_scale(
        backend, stub_machine):
    """Recording a gain under the MW key would be worse than omitting it: a reader
    comparing two runs would silently compare a full scale against a gain."""
    stub_machine.qubits["q1"].resonator = octave_channel(gain=-12.5, amplitude=0.05)
    block = backend.power_context(["q1"])["q1"]

    assert block["octave_gain_db"] == -12.5
    assert "full_scale_power_dbm" not in block
    assert block["readout_power_dbm"] == pytest.approx(
        -12.5 + volts_to_dbm(0.05), abs=1e-9)


def test_power_context_records_why_an_unsolvable_chain_could_not_answer(
        backend, stub_machine):
    """The provenance fix: never fail a run, but never write a silent blank either."""
    stub_machine.qubits["q1"].resonator = mixer_channel()
    block = backend.power_context(["q1"])["q1"]
    assert "readout_power_dbm" not in block
    assert "external analog mixer" in block["readout_unavailable"]


def test_power_context_still_reports_an_unknown_target_as_simply_empty(backend):
    """The two empties stay apart: a name the roster does not serve has no chain,
    so there is nothing to explain -- and an internal AttributeError string would
    be worse than nothing in the run record."""
    assert backend.power_context(["nonexistent"])["nonexistent"] == {}


def test_the_two_chain_blocks_stay_independent(backend, stub_machine):
    """A qubit whose drive is on an unsolvable chain still reports its MW readout."""
    stub_machine.qubits["q1"].xy = mixer_channel()
    block = backend.power_context(["q1"])["q1"]
    assert block["readout_power_dbm"] == pytest.approx(
        backend.device.component("q1_ro").readout_power_dbm)
    assert "external analog mixer" in block["drive_unavailable"]


def test_the_chain_independent_provenance_survives_an_unpriceable_chain(
        backend, stub_machine):
    """Amplitude and LO do not depend on the RF chain, so losing them along with
    the dBm would throw away provenance this driver can perfectly well record --
    and the LO is the value a hand edit makes invisible everywhere else."""
    resonator = mixer_channel(amplitude=0.02)
    resonator.LO_frequency = 5.9e9
    stub_machine.qubits["q1"].resonator = resonator

    block = backend.power_context(["q1"])["q1"]
    assert block["readout_lo_freq_hz"] == pytest.approx(5.9e9)
    assert block["readout_amplitude"] == pytest.approx(0.02)
    assert "external analog mixer" in block["readout_unavailable"]
    assert "readout_power_dbm" not in block


def test_each_power_block_names_its_rf_chain(backend, stub_machine):
    """The key that makes the rest of the block legible.

    "no full_scale_power_dbm here" is a fault on an MW-FEM run and simply the
    truth on an Octave one, and a run record that does not say which chain it was
    cannot tell a later reader apart.
    """
    stub_machine.qubits["q1"].xy = octave_channel()
    block = backend.power_context(["q1"])["q1"]
    assert block["readout_rf_chain"] == "mw_fem"
    assert block["drive_rf_chain"] == "octave"


def test_an_undeclared_chain_is_recorded_as_unknown_not_omitted(backend, stub_machine):
    channel = octave_channel()
    del channel.frequency_converter_up
    stub_machine.qubits["q1"].resonator = channel
    assert backend.power_context(["q1"])["q1"]["readout_rf_chain"] == "unknown"
