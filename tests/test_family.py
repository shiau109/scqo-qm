"""Which hardware family a QUAM channel declares — and what it must refuse to guess.

This driver serves both Quantum Machines RF chains (MW-FEM and Octave) and both
baseband DACs (LF-FEM and OPX+) behind one backend name, so a wrong answer here
does not raise: it picks the wrong power model, the wrong flux rail, or the wrong
predistortion field, and the run looks fine. Hence two properties are pinned
here rather than left to the call sites:

1. Detection is DUCK-TYPED. Half this suite builds channels as ``SimpleNamespace``
   and the flux guards are pinned against a port that is ``None``; an
   ``isinstance`` check against the quam classes would answer wrong for all of
   them, and every one of those tests would still pass.
2. Ignorance is reported as ignorance. ``None`` means "the tree does not say", and
   is never silently promoted to a family — the caller takes the conservative
   branch and names it.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace as NS

from scqo_qm._family import (
    FLUX_LF_FEM,
    FLUX_OPX_PLUS,
    RF_EXTERNAL_MIXER,
    RF_MW_FEM,
    RF_OCTAVE,
    UNKNOWN,
    flux_port_family,
    rf_chain,
    tree_families,
)


def octave_converter(**over):
    """An ``OctaveUpConverter``'s discriminating fields."""
    return NS(**{"LO_source": "internal", "LO_frequency": 5e9, "gain": 0.0,
                 "output_mode": "always_on", "input_attenuators": "off", **over})


def mixer_converter(**over):
    """The generic ``FrequencyConverter``'s (external analog mixer) fields."""
    return NS(**{"local_oscillator": NS(frequency=5e9), "mixer": NS(),
                 "gain": 0.0, **over})


# --- RF chain -------------------------------------------------------------


def test_an_octave_upconverter_is_the_octave_chain():
    assert rf_chain(NS(frequency_converter_up=octave_converter())) == RF_OCTAVE


def test_an_mw_fem_port_is_the_mw_chain():
    assert rf_chain(NS(opx_output=NS(band=2, full_scale_power_dbm=-11))) == RF_MW_FEM


def test_gain_alone_cannot_separate_an_octave_from_an_external_mixer():
    """Both converters declare ``gain``, so it is not a discriminator.

    The Octave is identified by ``LO_source``/``input_attenuators`` and the
    external mixer by ``local_oscillator``/``mixer``. Getting this backwards
    would send an external-mixer tree down the Octave power solve, which writes a
    ``gain`` the mixer's LO does not read.
    """
    assert rf_chain(NS(frequency_converter_up=octave_converter())) == RF_OCTAVE
    assert rf_chain(NS(frequency_converter_up=mixer_converter())) == RF_EXTERNAL_MIXER


def test_a_channel_that_declares_nothing_is_unknown_not_a_family():
    assert rf_chain(NS()) is None
    assert rf_chain(NS(opx_output=None)) is None
    assert rf_chain(NS(frequency_converter_up=NS())) is None


def test_a_bare_lf_port_is_not_an_rf_chain():
    """An LF-FEM/OPX+ analog output carries no band, so it names no RF chain."""
    assert rf_chain(NS(opx_output=NS(output_mode="direct"))) is None


# --- baseband flux port ---------------------------------------------------


def test_output_mode_marks_an_lf_fem_flux_port():
    port = NS(output_mode="amplified", exponential_filter=None,
              feedforward_filter=None)
    assert flux_port_family(NS(opx_output=port)) == FLUX_LF_FEM


def test_an_opx_plus_flux_port_is_the_absence_of_the_lf_fem_extras():
    """``OPXPlusAnalogOutputPort`` has the LF base fields and none of the FEM ones.

    It is recognised POSITIVELY by ``feedforward_filter`` rather than by "not
    LF-FEM", so a bare stub cannot be mistaken for a real OPX+ port — the
    distinction that decides whether ``apply_distortion`` refuses.
    """
    port = NS(feedforward_filter=None, feedback_filter=None, delay=0, offset=None)
    assert flux_port_family(NS(opx_output=port)) == FLUX_OPX_PLUS


def test_a_channel_with_no_port_stays_unknown():
    """The shape the flux guards are already pinned against."""
    assert flux_port_family(NS(opx_output=None)) is None
    assert flux_port_family(NS()) is None


def test_a_microwave_port_is_not_a_baseband_dac():
    """Naming an MW-FEM port a flux family would be worse than admitting ignorance."""
    assert flux_port_family(NS(opx_output=NS(band=1, full_scale_power_dbm=-11))) is None


def test_an_undeclared_port_is_unknown_rather_than_opx_plus():
    assert flux_port_family(NS(opx_output=NS())) is None


# --- census ---------------------------------------------------------------


def test_the_census_lists_families_so_a_mixed_tree_stays_visible():
    """An LF-FEM flux line beside an Octave drive is a REAL wiring
    (``wiring_lffem_octave.py``). Collapsing the census to one label would hide
    exactly the tree a reader needs to see."""
    machine = NS(
        qubits={
            "q1": NS(xy=NS(frequency_converter_up=octave_converter()),
                     resonator=NS(frequency_converter_up=octave_converter()),
                     z=NS(opx_output=NS(output_mode="direct"))),
            "q2": NS(xy=NS(opx_output=NS(band=1)),
                     resonator=NS(opx_output=NS(band=2)),
                     z=NS(opx_output=NS(feedforward_filter=None))),
        },
        qubit_pairs={},
        octaves={"oct1": NS()},
    )
    census = tree_families(machine)
    assert census["rf_chain"] == [RF_MW_FEM, RF_OCTAVE]
    assert census["flux_port"] == [FLUX_LF_FEM, FLUX_OPX_PLUS]
    assert census["octaves"] == ["oct1"]


def test_the_census_reports_an_undeclared_channel_as_unknown():
    machine = NS(qubits={"q1": NS(xy=NS(), resonator=NS(), z=NS(opx_output=None))},
                 qubit_pairs={}, octaves={})
    census = tree_families(machine)
    assert census["rf_chain"] == [UNKNOWN]
    assert census["flux_port"] == [UNKNOWN]
    assert "octaves" not in census


def test_the_census_counts_coupler_flux_ports_too():
    machine = NS(
        qubits={},
        qubit_pairs={"q1_q2": NS(coupler=NS(opx_output=NS(output_mode="direct")))},
        octaves={},
    )
    assert tree_families(machine)["flux_port"] == [FLUX_LF_FEM]


def test_the_census_never_raises_on_a_tree_it_cannot_walk():
    """A census that aborts a run would be worse than a partial one."""
    assert tree_families(NS()) == {"rf_chain": [], "flux_port": []}
    assert tree_families(object()) == {"rf_chain": [], "flux_port": []}


# --- the isinstance ban ---------------------------------------------------


def test_family_detection_never_imports_or_isinstance_checks_a_vendor_class():
    """Static, not behavioural: an ``isinstance`` added later would answer right
    for the live tree and wrong for every stub in this suite, so no behavioural
    test in this file would catch it."""
    source = pathlib.Path("scqo_qm/_family.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "isinstance", "_family.py must stay duck-typed"
        assert not isinstance(node, (ast.Import, ast.ImportFrom)) or (
            getattr(node, "module", "") in ("__future__", "typing")
        ), "_family.py must import no vendor package"


def test_either_mw_only_field_identifies_the_mw_chain():
    """``band`` and ``full_scale_power_dbm`` are both MWFEMAnalogOutputPort-only.

    Requiring both would make the answer depend on how completely a caller built
    the port — and this suite's own MW stub carries only ``full_scale_power_dbm``.
    A partial port reading as "unknown" would send absolute power down the refusal
    path, which is the failure this module exists to prevent.
    """
    assert rf_chain(NS(opx_output=NS(band=2))) == RF_MW_FEM
    assert rf_chain(NS(opx_output=NS(full_scale_power_dbm=-11))) == RF_MW_FEM
    assert flux_port_family(NS(opx_output=NS(full_scale_power_dbm=-11))) is None
