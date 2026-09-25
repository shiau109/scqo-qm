"""The probes' door out of the neutral surface (``scqo_qm/experiments/_vendor.py``).

QM probes take the raw QUAM tree and address it by VENDOR name, while scqo hands
the experiment ROSTER names — a foreign flux source (``flux_component``) and a
pair target are the two places where the two namespaces meet. These tests pin
that join against the stub QUAM tree from ``conftest.py``, whose names disagree
on purpose (roster ``q1_q2`` / ``q1_q2_c`` vs QUAM ``coupler_q1_q2``), so a
resolution that fell back to string arithmetic would fail here.

No QUA program is built (that needs an instrument-grade config); what is under
test is exactly the name resolution the probe call sites do first.
"""

from __future__ import annotations

import pytest

from conftest import make_experiment


def _flux_experiment(backend, roster, **kw):
    from scqo_qm.experiments.resonator_spectroscopy_flux import (
        QMResonatorSpectroscopyFlux,
    )

    params = QMResonatorSpectroscopyFlux.Parameters(targets=["q1"], **kw)
    return make_experiment(QMResonatorSpectroscopyFlux, backend, roster, params)


def test_flux_source_resolves_a_qubits_own_z_line(backend, roster):
    from scqo_qm.experiments._vendor import flux_source_name

    exp = _flux_experiment(backend, roster, flux_component="q2")
    assert flux_source_name(exp, "q2") == "q2"  # a key of machine.qubits


def test_flux_source_resolves_a_coupler_to_its_quam_pair(backend, roster):
    """The roster names the coupler MODE ``q1_q2_c``; the probe reaches it as
    ``machine.qubit_pairs['coupler_q1_q2'].coupler``. The join is the vendor
    OBJECT the backend resolved, never the name."""
    from scqo_qm.experiments._vendor import flux_source_name

    exp = _flux_experiment(backend, roster, flux_component="q1_q2_c")
    assert flux_source_name(exp, "q1_q2_c") == "coupler_q1_q2"


def test_flux_source_refuses_a_coupler_for_a_z_pulse_probe(backend, roster):
    """qubit_spectroscopy_flux plays the flux as a z PULSE on a QUAM qubit, and a
    tunable coupler has none — refused here with the reason, not deep inside the
    QUA build."""
    from scqo_qm.experiments._vendor import flux_source_name

    exp = _flux_experiment(backend, roster, flux_component="q1_q2_c")
    with pytest.raises(ValueError, match="coupler"):
        flux_source_name(exp, "q1_q2_c", qubit_only=True)


def test_flux_source_refuses_an_entity_with_no_flux_channel(backend, roster):
    """Going through the channel KIND is the capability guard: q3 is a fixed
    transmon with no flux rider, so the roster refuses before any vendor hop."""
    from scqo.roster import RosterError

    from scqo_qm.experiments._vendor import flux_source_name

    exp = _flux_experiment(backend, roster)
    with pytest.raises((RosterError, KeyError), match="flux"):
        flux_source_name(exp, "q3")


def test_pair_target_resolves_to_the_quam_pair_key(backend, roster):
    """``targets`` are ROSTER composite names; ``_lib.select_qubit_pairs``
    selects by QUAM key. QM names its pairs after the coupler, so the two
    genuinely differ."""
    from scqo_qm.experiments._vendor import vendor_pair, vendor_pair_name

    exp = _pair_experiment(backend, roster)
    assert vendor_pair_name(exp, "q1_q2") == "coupler_q1_q2"
    assert vendor_pair(exp, "q1_q2") is backend.machine.qubit_pairs["coupler_q1_q2"]


def _pair_experiment(backend, roster, measure="low"):
    from scqo_qm.experiments.pair_zz_coupler import QMPairZZCoupler

    params = QMPairZZCoupler.Parameters(targets=["q1_q2"], measure=measure)
    return make_experiment(QMPairZZCoupler, backend, roster, params)


@pytest.mark.parametrize("measure,side", [("low", "control"), ("high", "target")])
def test_pair_measure_role_maps_onto_the_vendor_side(backend, roster, measure, side):
    """The neutral high/low ROLE is roster topology; control/target is vendor gate
    plumbing. The stub pair is control=q1 / target=q2 while the roster says
    high=q2 / low=q1, so a positional guess would invert this."""
    exp = _pair_experiment(backend, roster, measure=measure)
    assert exp._measure_side(backend.machine) == side


def _swap_experiment(cls_name, backend, roster, **kw):
    import importlib

    module = importlib.import_module(f"scqo_qm.experiments.{cls_name}")
    cls = next(v for k, v in vars(module).items() if k.startswith("QMPair"))
    return make_experiment(cls, backend, roster,
                           cls.Parameters(targets=["q1_q2"], **kw))


SWAP_MAPS = ["pair_swap_chevron", "pair_swap_flux_map"]


@pytest.mark.parametrize("module", SWAP_MAPS)
@pytest.mark.parametrize("side,vendor", [("low", "control"), ("high", "target")])
def test_swap_map_role_selectors_map_onto_the_vendor_side(backend, roster, module,
                                                          side, vendor):
    """``drive_side`` (who gets the pi pulse) and ``flux_side`` (whose z line
    carries the pulse) are independent roster ROLES; the probes take vendor
    control/target. The stub pair is control=q1 / target=q2 while the roster
    says high=q2 / low=q1, so a positional guess would invert both."""
    from scqo_qm.experiments._vendor import role_side

    exp = _swap_experiment(module, backend, roster, drive_side=side, flux_side=side)
    assert role_side(exp, exp.params.drive_side, field="drive_side") == vendor
    assert role_side(exp, exp.params.flux_side, field="flux_side",
                     needs_flux=True) == vendor


@pytest.mark.parametrize("module", SWAP_MAPS)
def test_swap_map_high_side_orientation_drives_the_reduction(backend, roster, module):
    """``reduce_raw`` labels the joint populations by ROLE, and the probes label
    them by vendor side (first digit = control). The orientation is resolved
    once, here."""
    from scqo_qm.experiments._vendor import role_side

    exp = _swap_experiment(module, backend, roster)
    assert role_side(exp, "high", field="targets") == "target"  # roster high=q2


def test_qc_n_swap_amp_refuses_non_control_roles(backend, roster):
    """The qc_N_swap_amp vendor probe excites AND flux-drives the vendor
    CONTROL member only (the swap macro's ctrl_amp is the one swept knob), so a
    role selection resolving elsewhere is refused BY NAME before any QUA is
    built. The stub pair is control=q1 while the roster's high=q2, so
    drive_side='high' resolves to the vendor target — the refusing case."""
    from scqo_qm.experiments.qc_n_swap_amp import QMQcNSwapAmp

    exp = make_experiment(QMQcNSwapAmp, backend, roster,
                          QMQcNSwapAmp.Parameters(targets=["q1_q2"],
                                                  drive_side="high",
                                                  flux_side="high"))
    exp.sweep_axes = exp.define_sweep()
    with pytest.raises(ValueError, match="CONTROL member"):
        exp.probe()


def test_role_side_refuses_the_selected_member_without_a_flux_channel(backend):
    """The params-AWARE half of the gate: ``validate_targets`` is a classmethod
    over (roster, targets) and cannot see ``flux_side``, so it can only ask
    whether SOME member has a flux channel. Choosing the one that does not is
    refused here — still pre-probe, before any QUA is built."""
    from scqo.roster import parse_components

    from conftest import ROSTER_TOML
    from scqo_qm.backend.qm_backend import QMBackend
    from scqo_qm.experiments._vendor import role_side

    # same chip, but q2's flux wire is gone -> roster high (=q2) has no flux
    no_z2 = parse_components(ROSTER_TOML.replace('[lines.z2]\nflux = ["q2"]\n\n', ""))
    unfluxed = QMBackend(backend.machine, roster=no_z2)
    exp = _swap_experiment("pair_swap_chevron", unfluxed, no_z2, flux_side="high")

    assert role_side(exp, "low", field="flux_side", needs_flux=True) == "control"
    with pytest.raises(ValueError, match="no flux channel"):
        role_side(exp, "high", field="flux_side", needs_flux=True)


def test_thermal_population_probe_prepares_the_ground_state_only(backend, roster,
                                                                 monkeypatch):
    """One prepared state, no drive pulse: the cloud is whatever the passive wait
    leaves behind. Anything else would measure the thing it is trying to count."""
    from scqo_qm.experiments import _readout_fidelity as readout_fidelity
    from scqo_qm.experiments.qubit_thermal_population import QMQubitThermalPopulation

    seen = {}
    monkeypatch.setattr(readout_fidelity, "build_program",
                        lambda machine, qubits, **kw: seen.update(kw) or ("prog", {}))
    monkeypatch.setattr("scqo_qm.experiments._lib.select_qubits",
                        lambda machine, targets, **kw: list(targets))

    exp = make_experiment(QMQubitThermalPopulation, backend, roster,
                          QMQubitThermalPopulation.Parameters(targets=["q1"]))
    exp.probe()
    assert seen["prepared_states"] == [0]
    assert seen["reset_type"] == "thermal"


def test_single_shot_readout_addresses_the_readout_channel(backend, roster):
    """The discriminator proposal writes through the READOUT CHANNEL entity — the
    qubit MODE name carries no knobs since the greenfield split."""
    from scqo_qm.experiments.single_shot_readout import QMSingleShotReadout

    exp = make_experiment(QMSingleShotReadout, backend, roster,
                          QMSingleShotReadout.Parameters(targets=["q1"]))
    view = exp.device.channel("q1", "readout")
    view.readout_rotation_rad = 1.25
    view.readout_threshold = -3e-4
    view.readout_rus_threshold = -1e-4

    pulse = backend.machine.qubits["q1"].resonator.operations["readout"]
    assert pulse.integration_weights_angle == pytest.approx(1.25)
    assert pulse.threshold == pytest.approx(-3e-4)
    assert pulse.rus_exit_threshold == pytest.approx(-1e-4)
    with pytest.raises(KeyError):
        exp.device.component("q1")  # the mode itself carries no knobs
