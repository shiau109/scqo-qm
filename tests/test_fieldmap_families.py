"""``scqo state --fields`` must describe the instrument in front of the operator.

The vendor-only inventory is the ONLY place an operator discovers these knobs --
they are not scqo subcommands and not neutral fields, so nothing else lists them.
That makes a wrong entry worse than a missing one: an MW-FEM ``band`` shown to
someone running an Octave names a ``state.json`` key their hardware does not have,
and they will go looking for it. The reverse is just as bad -- hiding the Octave
gain hides the knob that realizes their readout power.

So the inventory is split by RF chain and filtered against the loaded tree. The
UNION stays whole for the structural self-checks, because a ``coupled`` name may
cross families and the union is where every such name resolves.
"""

from __future__ import annotations

import pytest

from types import SimpleNamespace as NS

from scqo_qm.backend import fieldmap
from scqo_qm.backend.fieldmap import (
    VENDOR_ONLY,
    VENDOR_ONLY_COMMON,
    VENDOR_ONLY_MW_FEM,
    VENDOR_ONLY_OCTAVE,
    vendor_only_for,
)


# --- the split ------------------------------------------------------------


def test_the_three_blocks_partition_the_union():
    assert VENDOR_ONLY == {**VENDOR_ONLY_COMMON, **VENDOR_ONLY_MW_FEM,
                           **VENDOR_ONLY_OCTAVE}
    assert not set(VENDOR_ONLY_COMMON) & set(VENDOR_ONLY_MW_FEM)
    assert not set(VENDOR_ONLY_COMMON) & set(VENDOR_ONLY_OCTAVE)
    assert not set(VENDOR_ONLY_MW_FEM) & set(VENDOR_ONLY_OCTAVE)


def test_every_mw_entry_reaches_a_port_and_every_octave_entry_does_not():
    """The structural reason the split exists: MW-FEM knobs live on
    ``opx_output`` / ``opx_input`` PORTS, which an Octave channel does not have at
    all, while the Octave's live on a ``frequency_converter`` COMPONENT or on the
    Octave itself. An entry filed under the wrong block is an AttributeError
    waiting for whoever follows its path."""
    for name, entry in VENDOR_ONLY_MW_FEM.items():
        assert "opx_output" in entry.path or "opx_input" in entry.path, name
    for name, entry in VENDOR_ONLY_OCTAVE.items():
        assert ("frequency_converter" in entry.path
                or "octaves[" in entry.path), name
        assert "opx_output." not in entry.path, name


def test_the_common_block_names_neither_families_vendor_objects():
    for name, entry in VENDOR_ONLY_COMMON.items():
        assert "frequency_converter" not in entry.path, name
        assert ".band" not in entry.path, name
        assert "full_scale_power_dbm" not in entry.path, name


# --- the filter -----------------------------------------------------------


def test_each_chain_brings_in_exactly_its_own_block():
    assert set(vendor_only_for(["mw_fem"])) == set(VENDOR_ONLY_COMMON) | set(
        VENDOR_ONLY_MW_FEM)
    assert set(vendor_only_for(["octave"])) == set(VENDOR_ONLY_COMMON) | set(
        VENDOR_ONLY_OCTAVE)


def test_a_mixed_tree_gets_both_blocks_because_it_genuinely_has_both():
    """An LF-FEM flux line beside an Octave drive is a real wiring, and so is one
    OPX1000 carrying both an MW-FEM and an Octave."""
    assert vendor_only_for(["mw_fem", "octave"]) == VENDOR_ONLY


def test_an_unrecognized_chain_contributes_nothing_and_is_not_an_error():
    """The common block still describes everything that does not depend on the
    chain, which is more useful than refusing to answer."""
    assert vendor_only_for(["unknown"]) == VENDOR_ONLY_COMMON
    assert vendor_only_for([]) == VENDOR_ONLY_COMMON


def test_the_filter_returns_a_fresh_mapping_each_time():
    listed = vendor_only_for(["octave"])
    listed.clear()
    assert vendor_only_for(["octave"])


# --- the Octave entries carry the operational half ------------------------


def test_the_octave_realizers_point_at_the_governed_write():
    """Same contract the MW-FEM realizers keep: a ``realizer`` entry must say
    which governed write to use INSTEAD of a direct edit."""
    for name in ("readout_octave_gain", "drive_octave_gain"):
        entry = VENDOR_ONLY_OCTAVE[name]
        assert entry.kind == "realizer", name
        assert "scqo set" in entry.edit, name
        assert "_power_dbm" in entry.edit, name


def test_the_gain_entries_warn_that_moving_them_invalidates_the_mixer_calibration():
    """The one fact that makes the Octave power policy different from the MW one.
    It has to reach the operator somewhere, and the inventory is where they look."""
    for name in ("readout_octave_gain", "drive_octave_gain"):
        assert "mixer calibration" in VENDOR_ONLY_OCTAVE[name].edit, name


def test_the_output_mode_entries_name_the_always_off_default():
    """A tree assembled without setting it emits nothing at all, with no error
    anywhere -- the first thing to check when a new Octave setup reads flat."""
    for name in ("readout_octave_output_mode", "drive_octave_output_mode"):
        assert "always_off" in VENDOR_ONLY_OCTAVE[name].doc, name


def test_the_downconverter_lo_entry_names_the_silent_unset_case():
    """quam omits the whole RF_inputs entry when it is not a number, so the
    generated config has no receive path and reports no error."""
    entry = VENDOR_ONLY_OCTAVE["octave_downconverter_lo_frequency"]
    assert "readout_octave_lo_frequency" in entry.coupled
    assert "no receive path" in entry.doc


def test_the_calibration_db_entry_names_the_cwd_fallback():
    """Unset, quam falls back to os.getcwd(), so the calibration lands wherever
    the process started and a later run silently finds none."""
    assert "getcwd" in VENDOR_ONLY_OCTAVE["octave_calibration_db_path"].edit


# --- through the backend --------------------------------------------------


def test_the_backend_lists_the_chains_its_own_tree_declares(backend):
    """The live fixture tree is MW-FEM."""
    listed = backend.vendor_only()
    assert set(VENDOR_ONLY_MW_FEM) <= set(listed)
    assert not set(VENDOR_ONLY_OCTAVE) & set(listed)


def test_the_backend_lists_the_octave_block_on_an_octave_tree(backend, stub_machine):
    for qubit in stub_machine.qubits.values():
        for line in ("xy", "resonator"):
            channel = getattr(qubit, line, None)
            if channel is None:
                continue
            del channel.opx_output
            channel.frequency_converter_up = NS(LO_source="internal", gain=0.0,
                                                LO_frequency=5.9e9)

    listed = backend.vendor_only()
    assert set(VENDOR_ONLY_OCTAVE) <= set(listed)
    assert not set(VENDOR_ONLY_MW_FEM) & set(listed)


def test_with_no_tree_to_inspect_the_complete_inventory_is_served():
    """"What does this driver know about" still has an answer, and it is a better
    one than an empty dict."""
    from scqo_qm.backend.qm_backend import QMBackend

    assert QMBackend.vendor_only(None) == VENDOR_ONLY


# --- the operator commands ------------------------------------------------


def test_apply_distortion_is_hidden_when_every_flux_line_is_an_opx_plus(
        backend, stub_machine):
    """An inventory whose whole purpose is DISCOVERY must not advertise a door
    that is walled up: there is no ``exponential_filter`` on an OPX+ analog
    output, so the command there can only refuse."""
    for qubit in stub_machine.qubits.values():
        z = getattr(qubit, "z", None)
        if z is not None:
            z.opx_output = NS(feedforward_filter=None, feedback_filter=None)
    for pair in getattr(stub_machine, "qubit_pairs", {}).values():
        coupler = getattr(pair, "coupler", None)
        if coupler is not None:
            coupler.opx_output = NS(feedforward_filter=None, feedback_filter=None)

    names = {c.name for c in backend.operator_commands()}
    assert "apply_distortion" not in names
    assert "close_qm" in names  # unrelated to the flux port, must survive


def test_apply_distortion_survives_on_an_lf_fem_tree(backend):
    assert "apply_distortion" in {c.name for c in backend.operator_commands()}


def test_its_doc_says_which_flux_lines_it_serves():
    entry = next(c for c in fieldmap.OPERATOR_COMMANDS
                 if c.name == "apply_distortion")
    assert "LF-FEM" in entry.doc


@pytest.mark.parametrize("block", [VENDOR_ONLY_MW_FEM, VENDOR_ONLY_OCTAVE])
def test_every_family_entry_stays_ascii_for_a_lab_console(block):
    for name, entry in block.items():
        assert all(s.isascii() for s in (entry.path, entry.unit, entry.doc,
                                         entry.edit, entry.counterpart)), name
