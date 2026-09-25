"""The QM T1-tracking shells' pure decisions (no QUA, no instrument).

Covers both trackers — `qubit_t1_ade` and `qubit_t1_bayesian`. Everything here
is either arithmetic/policy the shells own, a vendor-prerequisite refusal on a
stub machine, or an AST property of the probe modules (the fixed-point rules a
QUA build cannot check for itself):

* the ADE delay geometry (t0 + {0,1,3} * dt) is pinned across BOTH repos —
  the probe's DELAY_MULTS must equal the neutral experiment's, or the closed
  form fits a different quadratic than the sequence played;
* the Bayesian ms -> cycles conversion must go through Cast.mul_int_by_fixed —
  the fixed-arithmetic product `tau_ms * 250000` exceeds 8 and WRAPS (modulo
  16) to a garbage wait, a silent failure the simulator does not show;
* both probes reset through the one door (`qubit.reset(reset_type, ...)`) and
  never call `reset_qubit_active` directly;
* the shells refuse BY NAME: a missing readout threshold (both — the
  sequences always discriminate) and unmeasured readout fidelities (Bayesian —
  the SPAM-aware likelihood reads alpha/beta from `fidelity_e`/`fidelity_g`).
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from scqo_qm.experiments.qubit_t1_ade import (
    QMQubitT1Ade,
    discriminator_problems,
)
from scqo_qm.experiments.qubit_t1_bayesian import (
    QMQubitT1Bayesian,
    spam_terms,
)

_PROBES_DIR = Path(__file__).resolve().parents[1] / "scqo_qm" / "experiments"
PROBE_ADE = _PROBES_DIR / "qubit_t1_ade.py"
PROBE_BAYES = _PROBES_DIR / "qubit_t1_bayesian.py"

BOTH_PROBES = pytest.mark.parametrize(
    "probe", [PROBE_ADE, PROBE_BAYES], ids=["ade", "bayesian"])


def _called_names(probe: Path) -> set[str]:
    """Every callee spelling in the probe, as source text (AST walk, not a
    substring scan — the docstrings DESCRIBE forbidden calls)."""
    tree = ast.parse(probe.read_text(encoding="utf-8"))
    return {ast.unparse(node.func) for node in ast.walk(tree)
            if isinstance(node, ast.Call)}


def _stub_qubit(*, threshold=0.001):
    return SimpleNamespace(
        xy=SimpleNamespace(operations={"x180": SimpleNamespace(length=40)}),
        resonator=SimpleNamespace(
            operations={"readout": SimpleNamespace(threshold=threshold)},
        ),
    )


def _stub_machine(**qubit_kwargs):
    return SimpleNamespace(qubits={"q1": _stub_qubit(**qubit_kwargs)})


def _stub_device(*, fidelity_g=0.91, fidelity_e=0.95):
    """The scqo device surface, reduced to the readout monitors the Bayesian
    shell reads (``single_shot_readout`` stores both on a successful run)."""
    view = SimpleNamespace(fidelity_g=fidelity_g, fidelity_e=fidelity_e)
    return SimpleNamespace(channel=lambda target, kind: view)


def _shell(cls, machine, device=None, **params):
    exp = cls.__new__(cls)
    exp.params = cls.Parameters(targets=["q1"], **params)
    exp.backend = SimpleNamespace(machine=machine)
    exp.device = device if device is not None else _stub_device()
    return exp


class TestDelayGeometry:

    def test_delay_mults_pinned_across_repos(self):
        """The probe and the neutral experiment must agree on the 1:3 spacing —
        the closed form is derived from exactly these multipliers."""
        from scqo_qm.experiments.qubit_t1_ade import DELAY_MULTS as probe_mults
        from scqo.experiments.qubit_t1_ade import DELAY_MULTS as neutral_mults

        assert tuple(probe_mults) == tuple(neutral_mults) == (0, 1, 3)

    def test_one_state_stream_per_delay(self):
        from scqo_qm.experiments.qubit_t1_ade import DELAY_MULTS, _STATE_STREAMS

        assert len(_STATE_STREAMS) == len(DELAY_MULTS) == 3


class TestProbeAstProperties:

    @BOTH_PROBES
    def test_reset_goes_through_the_one_door(self, probe):
        called = _called_names(probe)
        assert "qubit.reset" in called
        assert not any("reset_qubit_active" in name for name in called)

    def test_bayesian_tau_conversion_uses_int_by_fixed(self):
        """`Cast.mul_int_by_fixed(250_000, tau_ms)` — the fixed product wraps."""
        called = _called_names(PROBE_BAYES)
        assert "Cast.mul_int_by_fixed" in called

    def test_ade_math_is_floored_not_raw(self):
        """Every sqrt/ln argument goes through a relu clamp; a probe edit that
        drops Math.relu reverts to raw fixed math that dies at the range edge."""
        called = _called_names(PROBE_ADE)
        assert {"Math.relu", "Math.sqrt", "Math.ln", "Math.div"} <= called


class TestVendorPrerequisites:

    def test_discriminator_problems_names_the_qubit(self):
        machine = _stub_machine(threshold=None)
        assert discriminator_problems(machine, ["q1"]) == [
            "q1 has no readout threshold"]
        assert discriminator_problems(_stub_machine(), ["q1"]) == []

    def test_spam_terms_are_the_off_diagonals_of_the_measured_fidelities(self):
        spam, problems = spam_terms(
            _stub_device(fidelity_g=0.91, fidelity_e=0.95), ["q1"])
        assert problems == []
        # alpha = P(read 0 | prep 1) = 1 - fidelity_e; beta = 1 - fidelity_g
        assert spam["q1"] == pytest.approx((0.05, 0.09))

    def test_spam_terms_name_the_missing_or_uninformative_fidelity(self):
        _, problems = spam_terms(_stub_device(fidelity_e=None), ["q1"])
        assert problems == ["q1 has no measured fidelity_e"]
        # at 0.5 the readout says nothing about that state: the SPAM span is zero
        _, problems = spam_terms(_stub_device(fidelity_g=0.5), ["q1"])
        assert problems == ["q1 has no measured fidelity_g"]
        _, problems = spam_terms(_stub_device(fidelity_g=None, fidelity_e=None), ["q1"])
        assert problems == ["q1 has no measured fidelity_e and fidelity_g"]

    def test_ade_probe_refuses_without_a_threshold(self):
        exp = _shell(QMQubitT1Ade, _stub_machine(threshold=None))
        with pytest.raises(ValueError, match="single_shot_readout"):
            exp.probe()

    def test_bayesian_probe_refuses_without_measured_readout_fidelities(self):
        exp = _shell(QMQubitT1Bayesian, _stub_machine(),
                     device=_stub_device(fidelity_e=None))
        with pytest.raises(ValueError, match="single_shot_readout"):
            exp.probe()


class TestShellToProbeMapping:
    """Pin the shell -> probe parameter mapping with no QUAM (the readout-probe
    pattern: monkeypatch build_program + select_qubits, capture kwargs)."""

    def _capture(self, monkeypatch, probe_module_name):
        captured = {}

        def fake_build_program(machine, qubits, **kwargs):
            captured.update(kwargs)
            return "PROG", {"qubit": xr.DataArray(["q1"]),
                            "block_idx": xr.DataArray(np.arange(kwargs["num_blocks"]))}

        monkeypatch.setattr(
            f"scqo_qm.experiments.{probe_module_name}.build_program",
            fake_build_program,
        )
        monkeypatch.setattr(
            "scqo_qm.experiments._lib.select_qubits",
            lambda machine, names, multiplexed: SimpleNamespace(
                get_names=lambda: list(names)),
        )
        return captured

    def test_ade_mapping(self, monkeypatch):
        captured = self._capture(monkeypatch, "qubit_t1_ade")
        exp = _shell(QMQubitT1Ade, _stub_machine(),
                     num_blocks=7, num_averages=50, t0_ns=16,
                     t1_guess_s=40e-6, dt_factor=1.0)
        prog, sweep_axes, acquire_fn = exp.probe()
        assert prog == "PROG"
        from scqo_qm.experiments import qubit_t1_ade as ade_mod
        assert acquire_fn is ade_mod.acquire
        assert captured["num_blocks"] == 7
        assert captured["n_avg"] == 50
        assert captured["t0_cycles"] == 4
        # dt = dt_factor * t1_guess = 40 us -> 10_000 cycles
        assert captured["dt_cycles"] == 10_000
        assert captured["adaptive_dt"] is False
        assert captured["reset_type"] == "thermal"

    def test_bayesian_mapping_and_side_channel(self, monkeypatch):
        captured = self._capture(monkeypatch, "qubit_t1_bayesian")
        exp = _shell(QMQubitT1Bayesian, _stub_machine(),
                     num_blocks=5, num_probes=20, t1_prior_s=35e-6)
        exp._t1_prior_s = {"q1": 35e-6}  # define_sweep's job, done by run()
        prog, sweep_axes, acquire_fn = exp.probe()
        from scqo_qm.experiments import qubit_t1_bayesian as bayes_mod
        assert acquire_fn is bayes_mod.acquire
        assert captured["num_blocks"] == 5
        assert captured["num_probes"] == 20
        assert captured["c_adaptive"] == pytest.approx(0.51)
        assert captured["t1_prior_s"] == {"q1": 35e-6}
        # the SPAM pair the builder plants as QUA literals, from the monitors
        assert captured["spam_pairs"]["q1"] == pytest.approx((0.05, 0.09))
        assert captured["interleaved"] is True
        assert len(captured["lin_wait_cycles"]) == 20
        # the validation grid rides sweep_axes so acquire() can label lin_wait_s
        assert "lin_wait_cycles" in sweep_axes
        assert sweep_axes["lin_wait_cycles"].values.tolist() == \
            list(captured["lin_wait_cycles"])
