"""The whole Octave half of the driver, against a REAL tree rather than stubs.

Every other state on disk is MW-FEM -- ``quam_state/``, ``quam_state_6q/``,
``state_lib/*`` and both live device configs -- so until this file existed the
Octave paths were only ever exercised against hand-built ``SimpleNamespace``
channels. A stub proves the branch is reachable; it cannot prove the branch
matches what ``quam_builder`` actually produces, and the shapes that matter here
are exactly the ones the builder decides.

The tree is BUILT, not stored, for the same reason the sibling state folders are
gitignored: a committed fixture drifts from the builder, and a fresh clone should
need no local state at all. ``scripts/make_opxp_fixture.py`` is the tracked
artifact.

Wiring: 1 OPX+ + 1 Octave, two flux-tunable qubits on one multiplexed readout.
That is the real ceiling for the pairing (2 + 3N <= 10 analog outputs), so the
fixture doubles as a standing statement of what fits -- and it puts an Octave RF
chain beside an OPX+ baseband flux port, which is the combination that exercises
BOTH scoping rules at once.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pytest

from scqo_qm._family import FLUX_OPX_PLUS, RF_OCTAVE, flux_port_family, rf_chain, tree_families
from scqo_qm.backend import fieldmap
from scqo_qm.backend._distortion import apply_exponential_filter
from scqo_qm.backend.calibrate_octave import calibration_db, octave_lines
from scqo_qm.experiments._flux_limits import dac_rail_v, rail_remedy
from scqo_qm.quam_fields import (
    drive_frequency_problems,
    flux_headroom_problems,
    flux_point_problems,
    octave_calibration_warnings,
    octave_frequency_problems,
    octave_output_problems,
    rf_frequency_reference_problems,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "make_opxp_fixture", Path("scripts/make_opxp_fixture.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def opxp_state(tmp_path_factory) -> str:
    """A freshly built OPX+/Octave tree. Module-scoped: the build is the cost."""
    out = tmp_path_factory.mktemp("quam_state_opxp") / "tree"
    _load_generator().build(out)
    return str(out)


@pytest.fixture(scope="module")
def opxp_machine(opxp_state):
    from quam_config import Quam

    return Quam.load(opxp_state)


@pytest.fixture(scope="module")
def opxp_backend(opxp_state, opxp_machine):
    from scqo.roster import parse_components

    from scqo_qm.backend.qm_backend import QMBackend
    from scqo_qm.backend.roster_gen import roster_toml_for

    roster = parse_components(roster_toml_for(opxp_machine))
    return QMBackend.load(roster=roster, state_path=opxp_state)


# --- the tree is what we think it is --------------------------------------


def test_the_builder_produces_an_octave_rf_chain_on_an_opx_plus_baseband(opxp_machine):
    """The combination that exercises both scoping rules at once, and the one
    that proves 'OPX+ vs OPX1000' was the wrong axis: the RF chain and the
    baseband port answer separately."""
    assert tree_families(opxp_machine) == {
        "rf_chain": [RF_OCTAVE], "flux_port": [FLUX_OPX_PLUS], "octaves": ["oct1"]}


def test_detection_agrees_with_the_real_vendor_classes(opxp_machine):
    q1 = opxp_machine.qubits["q1"]
    assert rf_chain(q1.xy) == RF_OCTAVE
    assert rf_chain(q1.resonator) == RF_OCTAVE
    assert flux_port_family(q1.z) == FLUX_OPX_PLUS
    # The structural fact the whole power dispatch turns on.
    assert not hasattr(q1.xy, "opx_output")


def test_the_builder_stores_rf_frequency_as_a_literal_not_a_reference(opxp_machine):
    """The trap the startup audit exists for: a reference READS fine and refuses
    every write. quam_builder gets this right, which is why the audit targets
    hand-assembled trees."""
    assert rf_frequency_reference_problems(opxp_machine) == []


# --- every startup audit, on a tree a session could actually open ---------


@pytest.mark.parametrize("audit", [
    flux_point_problems,
    flux_headroom_problems,
    drive_frequency_problems,
    rf_frequency_reference_problems,
    octave_frequency_problems,
    octave_output_problems,
])
def test_a_freshly_built_tree_opens_clean(audit, opxp_machine):
    assert audit(opxp_machine) == []


def test_the_calibration_db_advisory_is_silent_when_the_path_is_pinned(opxp_machine):
    """The fixture pins it to the tree's own folder, which is what a real setup
    should do -- quam's fallback is os.getcwd()."""
    assert octave_calibration_warnings(opxp_machine) == []
    assert calibration_db(opxp_machine)["oct1"]["fallback"] is False


# --- the inventory describes THIS instrument ------------------------------


def test_the_octave_block_is_listed_and_the_mw_one_is_not(opxp_backend):
    listed = set(opxp_backend.vendor_only())
    assert set(fieldmap.VENDOR_ONLY_OCTAVE) <= listed
    assert not set(fieldmap.VENDOR_ONLY_MW_FEM) & listed


def test_both_command_scoping_rules_fire_on_this_one_tree(opxp_backend):
    """An Octave RF chain brings calibrate_octave in; an OPX+ baseband flux port
    takes apply_distortion out, because there is no exponential_filter there."""
    names = {c.name for c in opxp_backend.operator_commands()}
    assert "calibrate_octave" in names
    assert "apply_distortion" not in names
    assert "close_qm" in names


# --- absolute power, on the real chain ------------------------------------


def test_the_power_the_builder_staged_reads_back_as_the_power_it_asked_for(
        opxp_backend):
    generator = _load_generator()
    assert opxp_backend.device.component("q1_ro").readout_power_dbm == pytest.approx(
        generator.READOUT_POWER_DBM, abs=1e-6)


def test_re_setting_the_current_power_leaves_the_gain_and_the_calibration_alone(
        opxp_backend, opxp_machine):
    """THE property, now on a real tree: the gain keys the mixer calibration, so
    an idempotent write must not touch it."""
    view = opxp_backend.device.component("q1_ro")
    gain_before = opxp_machine.qubits["q1"].resonator.frequency_converter_up.gain
    power = view.readout_power_dbm

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        view.readout_power_dbm = power

    assert opxp_machine.qubits["q1"].resonator.frequency_converter_up.gain == gain_before
    assert view.readout_power_dbm == pytest.approx(power, abs=1e-6)
    assert not [w for w in caught if "mixer calibration" in str(w.message)]


def test_a_small_change_is_absorbed_by_the_amplitude(opxp_backend, opxp_machine):
    view = opxp_backend.device.component("q1_ro")
    resonator = opxp_machine.qubits["q1"].resonator
    gain_before = resonator.frequency_converter_up.gain
    original = view.readout_power_dbm
    try:
        view.readout_power_dbm = original + 3.0
        assert resonator.frequency_converter_up.gain == gain_before
        assert view.readout_power_dbm == pytest.approx(original + 3.0, abs=1e-6)
    finally:
        view.readout_power_dbm = original


# --- what a run of this tree would record ---------------------------------


def test_power_context_speaks_the_octave_chains_own_vocabulary(opxp_backend):
    block = opxp_backend.power_context(["q1"])["q1"]

    assert block["readout_rf_chain"] == "octave"
    assert "octave_gain_db" in block
    assert "full_scale_power_dbm" not in block
    assert "drive_octave_gain_db" in block
    assert "drive_full_scale_power_dbm" not in block


def test_power_context_records_the_mixer_calibration_even_before_one_exists(
        opxp_backend):
    """Its ABSENCE is exactly what a later reader wants recorded against a run
    whose spectra look mirrored."""
    recorded = opxp_backend.power_context(["q1"])["q1"]["octave_calibration_db"]
    assert recorded["declared"] is True
    assert recorded["present"] is False  # nothing has calibrated this fixture


# --- the flux side --------------------------------------------------------


def test_an_opx_plus_flux_port_gets_its_own_rail_and_its_own_remedy(opxp_machine):
    z = opxp_machine.qubits["q1"].z
    assert dac_rail_v(z) == 0.5
    message = rail_remedy(z, name="q1.z", needed_v=1.0, rail=0.5)
    assert "OPX+" in message
    assert "no 'amplified' mode" in message


def test_writing_a_distortion_filter_here_is_refused_rather_than_dropped(
        opxp_machine):
    """quam would ACCEPT exponential_filter on this port and drop it on save.
    Proven against the real port class, not a stub."""
    with pytest.raises(ValueError, match="OPX\\+"):
        apply_exponential_filter(opxp_machine, "q1", [0.1], [1e-6])
    assert not hasattr(opxp_machine.qubits["q1"].z.opx_output, "exponential_filter")


# --- the calibration command ----------------------------------------------


def test_calibrate_octave_finds_both_lines_of_both_qubits(opxp_machine):
    assert octave_lines(opxp_machine) == {
        "q1": {"readout": True, "drive": True},
        "q2": {"readout": True, "drive": True},
    }


# --- the config this tree compiles to -------------------------------------


def test_the_tree_generates_a_qua_config_with_an_octave_block(opxp_machine):
    config = opxp_machine.generate_config()
    assert list(config["octaves"]) == ["oct1"]
    assert config["octaves"]["oct1"]["RF_outputs"]
    assert config["octaves"]["oct1"]["RF_inputs"], (
        "an unset down-converter LO makes quam omit RF_inputs entirely - no "
        "receive path, and no error")


def test_config_generation_warns_only_where_the_vendor_itself_is_deprecated(
        opxp_machine):
    """The repo convention says to gate config edits under warnings-as-errors.
    On quam 0.5.0 that gate cannot pass on ANY tree: the vendor's own builders
    construct through deprecated accessors -- ``InIQChannel.opx_input_offset_*``
    on the IQ path here, ``SingleChannel.filter_*_taps`` on the live MW tree. So
    the honest gate is "nothing but known vendor deprecations", and this test is
    what will notice when a version bump makes the stricter one possible.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opxp_machine.generate_config()

    non_deprecation = [w for w in caught
                       if not issubclass(w.category, DeprecationWarning)]
    assert non_deprecation == [], [str(w.message) for w in non_deprecation]
    assert all("opx_input_offset" in str(w.message) for w in caught), \
        [str(w.message)[:80] for w in caught]
