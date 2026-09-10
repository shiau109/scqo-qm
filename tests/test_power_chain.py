"""Absolute power is the one knob family whose vendor home differs per RF chain.

``readout_power_dbm`` / ``drive_power_dbm`` reach
``channel.opx_output.full_scale_power_dbm`` through quam_builder's MW-only
``power_tools``. An Octave channel has no ``opx_output`` at all, so before this
door existed the whole family failed in the quietest possible way:

* the accessor raised a bare ``AttributeError`` naming neither target nor reason;
* ``_read_or_none`` caught it and put ``None`` in the device snapshot — for every
  qubit, every session;
* ``power_context``'s ``except Exception`` caught it too and wrote ``{}`` into the
  run record, which is exactly what it writes for a target that has no readout
  chain at all.

So the operator's evidence for "this driver cannot do absolute power on an Octave"
was two indistinguishable blanks. These tests pin the loud version.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from scqo_qm import quam_fields
from scqo_qm.backend.qm_backend import _mw_power_channel


def octave_channel():
    """A drive/readout channel on an Octave: an IQ pair plus an upconverter, and
    conspicuously no ``opx_output``.

    It carries real operations because the point under test is the CHAIN, and a
    channel with no pulses would fail earlier for an unrelated reason — which is
    how the first draft of these tests passed while asserting nothing.
    """
    return NS(
        RF_frequency=6e9,
        LO_frequency=5.9e9,
        opx_output_I=NS(feedforward_filter=None),
        opx_output_Q=NS(feedforward_filter=None),
        frequency_converter_up=NS(LO_source="internal", LO_frequency=5.9e9,
                                  gain=0.0, output_mode="always_on",
                                  input_attenuators="off"),
        operations={
            quam_fields.READOUT_OPERATION: NS(amplitude=0.01, length=2000),
            quam_fields.SATURATION_OPERATION: NS(amplitude=0.05, length=1000),
        },
    )


# --- the door -------------------------------------------------------------


def test_an_mw_fem_channel_passes_through_unchanged():
    channel = NS(opx_output=NS(band=2, full_scale_power_dbm=-11))
    assert _mw_power_channel(channel, name="q1_ro", field="readout_power_dbm") is channel


def test_an_octave_channel_is_refused_naming_the_target_field_and_the_remedy():
    with pytest.raises(ValueError) as err:
        _mw_power_channel(octave_channel(), name="q1_ro", field="readout_power_dbm")
    message = str(err.value)
    assert "q1_ro.readout_power_dbm" in message
    assert "Octave" in message
    assert "gain" in message and "amplitude" in message
    assert "readout_amp" in message  # what to reach for instead


def test_an_external_mixer_is_refused_separately_from_the_octave():
    """Different remedy: the power lives on the LO's own source, so pointing the
    operator at an Octave gain would send them to a knob that does not exist."""
    channel = NS(frequency_converter_up=NS(local_oscillator=NS(frequency=6e9),
                                           mixer=NS(), gain=0.0))
    with pytest.raises(ValueError) as err:
        _mw_power_channel(channel, name="q1_xy", field="drive_power_dbm")
    assert "external analog mixer" in str(err.value)
    assert "Octave" not in str(err.value)


def test_a_channel_declaring_no_chain_says_so_rather_than_guessing():
    with pytest.raises(ValueError) as err:
        _mw_power_channel(NS(), name="q1_ro", field="readout_power_dbm")
    message = str(err.value)
    assert "no RF chain" in message
    assert "state.json" in message  # where to go look


# --- through the channel views --------------------------------------------


def test_the_readout_view_refuses_by_name_on_an_octave_tree(backend, stub_machine):
    stub_machine.qubits["q1"].resonator = octave_channel()
    view = backend.device.component("q1_ro")
    with pytest.raises(ValueError, match="Octave"):
        view.readout_power_dbm
    with pytest.raises(ValueError, match="Octave"):
        view.readout_power_dbm = -40.0


def test_the_drive_view_refuses_by_name_on_an_octave_tree(backend, stub_machine):
    stub_machine.qubits["q1"].xy = octave_channel()
    view = backend.device.component("q1_xy")
    with pytest.raises(ValueError, match="Octave"):
        view.drive_power_dbm
    with pytest.raises(ValueError, match="Octave"):
        view.drive_power_dbm = 0.0


# --- what the run record ends up saying -----------------------------------


def test_power_context_records_why_an_octave_chain_could_not_answer(
        backend, stub_machine):
    """The provenance fix: never fail a run, but never write a silent blank either."""
    stub_machine.qubits["q1"].resonator = octave_channel()
    block = backend.power_context(["q1"])["q1"]
    assert "readout_power_dbm" not in block
    assert "Octave" in block["readout_unavailable"]


def test_power_context_still_reports_an_unknown_target_as_simply_empty(backend):
    """The two empties stay apart: a name the roster does not serve has no chain,
    so there is nothing to explain — and an internal AttributeError string would
    be worse than nothing in the run record."""
    assert backend.power_context(["nonexistent"])["nonexistent"] == {}


def test_the_two_chain_blocks_stay_independent(backend, stub_machine):
    """A qubit whose drive is on an Octave still reports its MW readout chain."""
    stub_machine.qubits["q1"].xy = octave_channel()
    block = backend.power_context(["q1"])["q1"]
    assert block["readout_power_dbm"] == pytest.approx(
        backend.device.component("q1_ro").readout_power_dbm)
    assert "Octave" in block["drive_unavailable"]


def test_the_chain_independent_provenance_survives_an_unpriceable_chain(
        backend, stub_machine):
    """Amplitude and LO do not depend on the RF chain, so losing them along with
    the dBm would throw away provenance this driver can perfectly well record —
    and the LO is the value a hand edit makes invisible everywhere else."""
    resonator = octave_channel()
    stub_machine.qubits["q1"].resonator = resonator

    block = backend.power_context(["q1"])["q1"]
    assert block["readout_lo_freq_hz"] == pytest.approx(5.9e9)
    assert block["readout_amplitude"] == pytest.approx(
        float(resonator.operations["readout"].amplitude))
    assert "Octave" in block["readout_unavailable"]
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
