"""Tests for the scqo QM backend (scqo_qm).

Three tiers:

* ``_to_canonical`` and catalog registration are pure (no instrument, no QUAM).
* The greenfield ENTITY surface (component resolution, the per-kind channel
  views, the composite pair knobs, snapshot/power_context) runs against the stub
  QUAM tree from ``conftest.py`` — always, on every machine.
* Probe equivalence and the absolute-power chain solve load the LIVE
  ``quam_state/``. These no longer skip: the state file and ``quam_config.Quam``
  both name ``MixedTransmonQuam``, which validates fixed, tunable and mixed trees
  alike, so there is no root class left for them to disagree about.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from scqo_qm.backend.qm_backend import QMBackend, _progress_shot_total
from scqo_qm.backend.roster_gen import roster_toml_for


# --------------------------------------------------------------------------- pure

def _raw(sweep_dim: str, n_qubits: int = 2, n_sweep: int = 5) -> xr.Dataset:
    data = np.zeros((n_qubits, n_sweep))
    return xr.Dataset(
        {"I": (("qubit", sweep_dim), data), "Q": (("qubit", sweep_dim), data)},
        coords={"qubit": [f"q{i}" for i in range(n_qubits)], sweep_dim: np.arange(n_sweep)},
    )


class _FakeExp:
    def __init__(self, sweep_axes):
        self.sweep_axes = sweep_axes


# ---------------------------------------------- progress-counter denominator
# The progress counter divides an int by this total, so a None here is a hard
# TypeError AFTER the whole QUA program has run (qubit_parity_switch on QM,
# 2026-08-04): its num_shots is None because the count is derived from
# record_time_s, and getattr(params, "num_shots", 1) returns the present None,
# not the default. The total must fall back to resolved_num_shots().

def test_progress_total_falls_back_to_resolved_for_derived_shots():
    """qubit_parity_switch: params.num_shots is None (record_time_s-derived),
    the count lives in resolved_num_shots(). The total must be that, not None."""
    exp = SimpleNamespace(params=SimpleNamespace(num_shots=None),
                          resolved_num_shots=lambda: 2_951_594)
    total = _progress_shot_total(exp)
    assert total == 2_951_594
    assert total is not None            # the exact regression: never None


def test_progress_total_prefers_num_averages():
    """A sweep experiment declares num_averages; the resolver is never consulted."""
    exp = SimpleNamespace(
        params=SimpleNamespace(num_averages=200),
        resolved_num_shots=lambda: pytest.fail("resolver must not be called"),
    )
    assert _progress_shot_total(exp) == 200


def test_progress_total_uses_explicit_num_shots():
    """single_shot_readout et al. pass a concrete num_shots straight through."""
    exp = SimpleNamespace(params=SimpleNamespace(num_shots=4000))
    assert _progress_shot_total(exp) == 4000


def test_progress_total_defaults_to_one_without_a_resolver():
    """Neither averaging nor shots nor a resolver -> the original 1 default,
    so nothing that used to work now divides by None or zero."""
    exp = SimpleNamespace(params=SimpleNamespace())
    assert _progress_shot_total(exp) == 1


def test_to_canonical_renames_ramsey_axis():
    raw = _raw("idle_time")
    out = QMBackend._to_canonical(raw, _FakeExp({"idle_time_ns": np.arange(5)}))
    assert "idle_time_ns" in out.dims and "idle_time" not in out.dims
    assert set(out.data_vars) == {"I", "Q"}


def test_power_rabi_axis_needs_no_rename_at_all():
    """The probe and scqo now agree on ``amp_prefactor``, so this axis takes the
    NAME-based path and never the positional fallback.

    It used to differ (probe ``amp_prefactor`` vs scqo ``amp_factor``) and was
    rescued only by position — the less-safe branch, which cannot tell two
    same-length axes apart. The positional fallback itself stays covered by the
    ramsey and resonator cases above/below, which are still genuinely positional.
    """
    raw = _raw("amp_prefactor")
    out = QMBackend._to_canonical(raw, _FakeExp({"amp_prefactor": np.arange(5)}))
    assert "amp_prefactor" in out.dims
    assert "amp_factor" not in out.dims


def test_to_canonical_renames_resonator_spec_axis():
    raw = _raw("detuning")
    out = QMBackend._to_canonical(raw, _FakeExp({"detuning_hz": np.arange(5)}))
    assert "detuning_hz" in out.dims and "detuning" not in out.dims


def test_to_canonical_noop_when_names_match():
    raw = _raw("idle_time_ns")
    out = QMBackend._to_canonical(raw, _FakeExp({"idle_time_ns": np.arange(5)}))
    assert "idle_time_ns" in out.dims


def test_to_canonical_renames_two_axes():
    """2D sweeps (punchout): both axes rename positionally with size checks."""
    data = np.zeros((2, 5, 3))
    raw = xr.Dataset(
        {"I": (("qubit", "detuning", "power"), data), "Q": (("qubit", "detuning", "power"), data)},
        coords={"qubit": ["q0", "q1"], "detuning": np.arange(5), "power": np.arange(3)},
    )
    out = QMBackend._to_canonical(
        raw, _FakeExp({"detuning_hz": np.arange(5), "power_dbm": np.arange(3)})
    )
    assert {"detuning_hz", "power_dbm"} <= set(out.dims)
    assert out["I"].dims == ("target", "detuning_hz", "power_dbm")


def test_to_canonical_name_based_ignores_order_2d():
    """Flux spectroscopy: raw nesting (detuning_hz, flux_bias_v) vs scqo declaration
    (flux_bias_v, detuning_hz), with EQUAL sizes — positional renaming would swap the
    axes silently; the name-based path must leave the data untouched."""
    n = 4  # equal-length axes: the dangerous case
    data = np.arange(2 * n * n, dtype=float).reshape(2, n, n)
    raw = xr.Dataset(
        {"I": (("qubit", "detuning_hz", "flux_bias_v"), data), "Q": (("qubit", "detuning_hz", "flux_bias_v"), data)},
        coords={"qubit": ["q0", "q1"], "detuning_hz": np.linspace(-1e6, 1e6, n), "flux_bias_v": np.linspace(-0.1, 0.1, n)},
    )
    out = QMBackend._to_canonical(
        raw, _FakeExp({"flux_bias_v": np.zeros(n), "detuning_hz": np.zeros(n)})
    )
    assert out["I"].dims == ("target", "detuning_hz", "flux_bias_v")  # raw order kept
    np.testing.assert_array_equal(out["I"].values, data)
    np.testing.assert_array_equal(out["detuning_hz"].values, raw["detuning_hz"].values)


def test_to_canonical_name_based_single_shot():
    """Per-shot readout: raw nesting (shot_idx, prepared_state) vs scqo declaration
    (prepared_state, shot_idx) — resolved by name, sizes checked per name."""
    n_shots = 7
    data = np.zeros((2, n_shots, 2))
    raw = xr.Dataset(
        {"I": (("qubit", "shot_idx", "prepared_state"), data), "Q": (("qubit", "shot_idx", "prepared_state"), data)},
        coords={"qubit": ["q0", "q1"], "shot_idx": np.arange(1, n_shots + 1), "prepared_state": [0, 1]},
    )
    out = QMBackend._to_canonical(
        raw, _FakeExp({"prepared_state": np.array([0, 1]), "shot_idx": np.arange(n_shots)})
    )
    assert out["I"].dims == ("target", "shot_idx", "prepared_state")
    # size check is per NAME even though the declaration order differs
    bad = _FakeExp({"prepared_state": np.array([0, 1]), "shot_idx": np.arange(n_shots + 1)})
    with pytest.raises(ValueError):
        QMBackend._to_canonical(raw, bad)


def test_to_canonical_rejects_axis_count_mismatch():
    raw = _raw("idle_time")  # one sweep axis
    with pytest.raises(NotImplementedError):
        QMBackend._to_canonical(raw, _FakeExp({"a": np.arange(5), "b": np.arange(2)}))


def test_to_canonical_rejects_axis_size_mismatch():
    raw = _raw("idle_time", n_sweep=5)
    with pytest.raises(ValueError):
        QMBackend._to_canonical(raw, _FakeExp({"idle_time_ns": np.arange(7)}))


def test_catalog_registers_qm_experiments():
    import scqo_qm  # noqa: F401  (side effect: register)
    from scqo import catalog

    names = {e["name"] for e in catalog()}
    assert {"qubit_ramsey", "qubit_power_rabi", "resonator_spectroscopy"} <= names
    # the pair family: registered here, so `scqo run` on a QM setup gets a
    # probe() instead of the core class's NotImplementedError
    assert {"pair_zz_coupler", "pair_swap_chevron", "pair_swap_flux_map"} <= names


# ------------------------------------------------ entity surface (stub QUAM tree)

def test_component_resolves_channel_entities_per_kind(backend, stub_machine):
    """One view class per CHANNEL KIND over the SAME QUAM qubit: the three names
    a qubit's channels carry land on q.xy / q.resonator / q.z, and each view's
    ``.name`` is the ENTITY name while the vendor object is the subtree."""
    q1 = stub_machine.qubits["q1"]

    xy = backend.device.component("q1_xy")
    ro = backend.device.component("q1_ro")
    z = backend.device.component("q1_z")

    assert (xy.kind, ro.kind, z.kind) == ("drive", "readout", "flux")
    assert (xy.name, ro.name, z.name) == ("q1_xy", "q1_ro", "q1_z")
    assert xy.vendor is q1.xy and ro.vendor is q1.resonator and z.vendor is q1.z
    assert xy.qubit is q1 and ro.qubit is q1 and z.qubit is q1


def test_component_refuses_everything_that_carries_no_knobs(backend):
    """The contract scqo degrades gracefully against: a KeyError, naming what to
    address instead, for an unknown name, a MODE, a LINE, and a resonator mode
    (knobs live on channels since the greenfield split)."""
    with pytest.raises(KeyError, match="not in this device's roster"):
        backend.device.component("nope")
    with pytest.raises(KeyError, match="q1_ro"):
        backend.device.component("q1")       # a mode: address its channels
    with pytest.raises(KeyError, match="q1_z"):
        backend.device.component("q1")
    with pytest.raises(KeyError):
        backend.device.component("fl")       # a line
    with pytest.raises(KeyError):
        backend.device.component("q1_res")   # the minted resonator mode


def test_component_names_the_missing_subtree_on_a_fixed_frequency_qubit(backend,
                                                                        roster):
    """q3 is a fixed ``transmon``: the roster declares no flux rider for it, so
    no q3_z exists at all — and if one were declared the vendor hop would fail
    naming the absent subtree rather than returning a half-wired view."""
    assert ("q3", "flux") not in roster.defaults
    assert backend.device.component("q3_ro").kind == "readout"


def test_flux_channel_serves_both_vendor_shapes(backend, stub_machine):
    """``idle_flux`` over a qubit's FluxLine AND over the pair's TunableCoupler —
    the coupler's STANDING bias is an ordinary knob on the COUPLER MODE's own
    flux channel (the pair-level coupler_decouple_v field is gone)."""
    q1_z = backend.device.component("q1_z")
    q1_z.idle_flux = -0.042
    assert stub_machine.qubits["q1"].z.joint_offset == pytest.approx(-0.042)

    coupler_z = backend.device.component("q1_q2_c_z")
    assert coupler_z.qubit is None                      # not a QUAM qubit at all
    assert coupler_z.vendor is stub_machine.qubit_pairs["coupler_q1_q2"].coupler
    coupler_z.idle_flux = 0.031                         # flux_point 'off'
    assert coupler_z.vendor.decouple_offset == pytest.approx(0.031)


def test_channel_views_round_trip_the_neutral_knobs(backend, stub_machine):
    """Neutral get/set maps onto QUAM through scqo_qm.quam_fields; a
    drive_freq_hz write shifts both f_01 and xy.RF_frequency."""
    q2 = stub_machine.qubits["q2"]
    xy = backend.device.component("q2_xy")
    ro = backend.device.component("q2_ro")

    rf0 = float(q2.xy.RF_frequency)
    xy.drive_freq_hz = 5.102e9
    assert float(q2.f_01) == pytest.approx(5.102e9)
    assert float(q2.xy.RF_frequency) == pytest.approx(rf0 + 2e6)

    xy.pi_amp = 0.123
    assert xy.pi_amp == pytest.approx(0.123)
    xy.pi_duration_s = 4.0e-8
    assert q2.xy.operations["x180"].length == 40
    with pytest.raises(ValueError, match="multiple of 4 ns"):
        xy.pi_duration_s = 4.2e-8  # the QM pulse grid REFUSES, never rounds

    ro.readout_freq_hz = 6.25e9
    assert float(q2.resonator.RF_frequency) == pytest.approx(6.25e9)
    assert float(q2.resonator.f_01) == pytest.approx(6.25e9)
    ro.readout_amp = 0.111
    assert float(q2.resonator.operations["readout"].amplitude) == pytest.approx(0.111)
    ro.readout_threshold = -1.5e-4
    assert q2.resonator.operations["readout"].threshold == pytest.approx(-1.5e-4)


def test_thermalization_time_round_trips_on_the_qubit(backend, stub_machine):
    """The reset wait is a DRIVE-channel knob neutrally, but it lives on the
    QUAM qubit (not q.xy) — and it is rounded to the 4 ns QUA wait grid rather
    than refused like pi_duration_s, because it is a policy wait, not a
    calibrated pulse."""
    q2 = stub_machine.qubits["q2"]
    xy = backend.device.component("q2_xy")

    assert xy.thermalization_time_s is None  # never calibrated == unset
    xy.thermalization_time_s = 3.715492e-4
    assert q2.thermalization_time_ns == 371548  # floor to the 4 ns grid
    assert xy.thermalization_time_s == pytest.approx(3.71548e-4)

    with pytest.raises(ValueError, match="must be positive"):
        xy.thermalization_time_s = 0.0


def test_thermalization_time_refuses_a_stock_quam_class(backend, stub_machine):
    """Stock QUAM derives the wait as factor x T1 through a READ-ONLY property,
    so there is nowhere to store an absolute one. The refusal must name the fix
    (the qubit's state.json __class__) instead of silently doing nothing."""
    del stub_machine.qubits["q2"].thermalization_time_ns  # a stock transmon
    xy = backend.device.component("q2_xy")

    assert xy.thermalization_time_s is None
    with pytest.raises(NotImplementedError, match="Thermalizing"):
        xy.thermalization_time_s = 2e-4


def test_per_run_override_sets_and_reverts_exactly(backend, stub_machine, roster):
    """The per-run override is baked into the compiled program, so bracketing
    the BUILD is enough — and the revert must be exact, including restoring
    "never calibrated" rather than fabricating a value."""
    from scqo.experiments import get

    from conftest import make_experiment

    cls = get("qubit_relaxation")
    q1, q2 = stub_machine.qubits["q1"], stub_machine.qubits["q2"]
    q1.thermalization_time_ns = 100_000  # q1 calibrated, q2 never touched

    seen = {}
    exp = make_experiment(cls, backend, roster,
                          cls.Parameters(targets=["q1", "q2"],
                                         thermalization_time_ns=8_000.0))
    with backend._thermalization_override(exp):
        seen = {"q1": q1.thermalization_time_ns, "q2": q2.thermalization_time_ns}
    assert seen == {"q1": 8_000, "q2": 8_000}
    assert q1.thermalization_time_ns == 100_000  # exact revert
    assert q2.thermalization_time_ns is None     # restored to unset, not 0

    # no override -> the standing QUAM values are left completely alone
    plain = make_experiment(cls, backend, roster, cls.Parameters(targets=["q1"]))
    with backend._thermalization_override(plain):
        assert q1.thermalization_time_ns == 100_000


def test_per_run_override_expands_a_pair_to_its_member_qubits(backend, stub_machine,
                                                              roster):
    """A composite target carries no drive channel of its own; the reset happens
    on its MEMBER modes, resolved through the roster (the QUAM pair is named
    after its coupler, so a name-based shortcut would miss)."""
    from scqo.experiments import get

    from conftest import make_experiment

    cls = get("pair_zz_coupler")
    exp = make_experiment(cls, backend, roster,
                          cls.Parameters(targets=["q1_q2"],
                                         thermalization_time_ns=12_000.0))
    with backend._thermalization_override(exp):
        assert stub_machine.qubits["q1"].thermalization_time_ns == 12_000
        assert stub_machine.qubits["q2"].thermalization_time_ns == 12_000


def test_snapshot_reports_the_bound_knobs_per_entity(backend):
    """The pull-mode seed source: every realized channel reports exactly the
    knobs the fieldmap BINDS for its kind, and the composite reports the
    per-operation knobs the ROSTER compiled for it."""
    from scqo_qm.backend.fieldmap import FIELD_BINDINGS

    snap = backend.device.snapshot()
    assert set(snap["q1_xy"]) == set(FIELD_BINDINGS["drive"])
    assert set(snap["q1_ro"]) == set(FIELD_BINDINGS["readout"])
    assert set(snap["q1_z"]) == set(FIELD_BINDINGS["flux"])
    # the composite's names are per-OPERATION, instantiated from the roster
    assert "cz_coupler_flux" in snap["q1_q2"]
    assert snap["q1_q2"]["cz_coupler_flux"] == pytest.approx(-0.125)
    # an Unrealized composite knob degrades to None instead of crashing the seed
    assert snap["q1_q2"]["cz_duration_s"] is None


def test_composite_view_reads_and_writes_the_gate_knobs(backend, stub_machine):
    """The QM pair surface Qblox has no counterpart for: per-operation knobs by
    full field name, resolved against the roster's DECLARED operations and the
    QUAM gate macro (matched case-insensitively — QUAM spells it "CZ")."""
    macro = stub_machine.qubit_pairs["coupler_q1_q2"].macros["CZ"]
    pair = backend.device.component("q1_q2")

    assert pair.read_knob("cz_coupler_flux") == pytest.approx(-0.125)
    pair.write_knob("cz_coupler_flux", -0.2)
    assert macro.coupler_flux_pulse.amplitude == pytest.approx(-0.2)

    # virtual Z: rad <-> turns, and the roster's high role (q2) is the QUAM
    # pair's TARGET here — resolved by name, never guessed
    pair.write_knob("cz_vz_high_rad", np.pi)
    assert macro.phase_shift_target == pytest.approx(0.5)
    assert macro.phase_shift_control == pytest.approx(0.0)
    pair.write_knob("cz_vz_low_rad", -np.pi / 2)
    assert macro.phase_shift_control == pytest.approx(-0.25)
    assert pair.read_knob("cz_vz_low_rad") == pytest.approx(-np.pi / 2)


def test_composite_view_refuses_undeclared_and_unrealized_knobs(backend):
    """Exact-cause errors: an undeclared operation names the declared set, a
    non-knob name names the legal suffixes, and a suffix QM cannot realize
    raises NotImplementedError with its reason (never a silent no-op)."""
    pair = backend.device.component("q1_q2")

    with pytest.raises(KeyError, match="not declared on this composite"):
        pair.read_knob("iswap_coupler_flux")
    with pytest.raises(KeyError, match="not a per-operation knob"):
        pair.read_knob("coupler_flux")
    with pytest.raises(NotImplementedError, match="FLUX-activated"):
        pair.read_knob("cz_drive_freq_hz")
    with pytest.raises(NotImplementedError):
        pair.write_knob("cz_duration_s", 40e-9)


def test_power_context_matches_the_views(backend, stub_machine):
    """Run-record provenance, addressed by MODE name: each target's readout and
    drive chains resolved through the roster's DEFAULT channels, never failing."""
    ctx = backend.power_context(["q1", "nonexistent"])
    q1 = stub_machine.qubits["q1"]

    assert ctx["q1"]["full_scale_power_dbm"] == q1.resonator.opx_output.full_scale_power_dbm
    assert ctx["q1"]["readout_amplitude"] == pytest.approx(
        float(q1.resonator.operations["readout"].amplitude))
    assert ctx["q1"]["readout_power_dbm"] == pytest.approx(
        backend.device.component("q1_ro").readout_power_dbm)
    assert ctx["q1"]["drive_power_dbm"] == pytest.approx(
        backend.device.component("q1_xy").drive_power_dbm)
    assert ctx["q1"]["readout_lo_freq_hz"] == pytest.approx(
        float(q1.resonator.LO_frequency))
    assert ctx["nonexistent"] == {}  # unknown target degrades, never raises


def test_distortion_apply_command_is_the_hint_hook_scqo_asks_for(backend):
    """The cryoscope hint hook: scqo's two cryoscopes record the taps as FACTS and
    print THIS command as the manual vendor step. Run-addressed when the run is
    known (accept order then cannot change what lands on the OPX), facts-addressed
    when it is not — and the hook NAME is the cross-repo contract, so the hint
    module's own constant is what the test resolves it through."""
    from scqo.experiments._distortion_hint import HOOK, apply_hint_lines

    assert callable(getattr(backend, HOOK, None))  # the name scqo actually asks
    assert backend.distortion_apply_command("q1", "RUN-1") == (
        "python -m scqo_qm.backend.apply_distortion --target q1 --run RUN-1")
    assert backend.distortion_apply_command("q2") == (
        "python -m scqo_qm.backend.apply_distortion --target q2")

    lines = apply_hint_lines("qubit_ramsey_cryoscope", backend, ["q1"], "RUN-1")
    assert any("apply_distortion --target q1 --run RUN-1" in line for line in lines)


def test_readout_power_dbm_solves_the_chain_bidirectionally(backend, stub_machine):
    """Absolute power: the setter re-solves (full_scale_power_dbm, amplitude) with
    the SMALLEST grid full-scale keeping amp <= 0.5 — bidirectional (a lower target
    lowers full scale again, unlike the bare power_tools helper)."""
    view = backend.device.component("q1_ro")
    res = stub_machine.qubits["q1"].resonator

    view.readout_power_dbm = -2.0
    assert view.readout_power_dbm == pytest.approx(-2.0, abs=1e-6)
    assert res.opx_output.full_scale_power_dbm == 7  # smallest grid value >= -2+6.02
    assert 0.354 < float(res.operations["readout"].amplitude) <= 0.5

    view.readout_power_dbm = -24.3
    assert res.opx_output.full_scale_power_dbm == -11  # back DOWN to the grid floor
    assert float(res.operations["readout"].amplitude) == pytest.approx(
        10 ** ((-24.3 + 11) / 20.0))

    with pytest.warns(UserWarning, match="canonical operating point"):
        view.readout_power_dbm = 10.0
    assert res.opx_output.full_scale_power_dbm == 16

    # zero amplitude -> the absolute power is UNDEFINED, and snapshot degrades it
    res.operations["readout"].amplitude = 0.0
    with pytest.raises(ValueError, match="absolute power undefined"):
        _ = view.readout_power_dbm
    assert backend.device.snapshot()["q1_ro"]["readout_power_dbm"] is None


def test_drive_power_dbm_solves_the_same_chain_on_xy(backend, stub_machine):
    """The drive twin: same grid solve on the xy channel + the saturation op;
    drive_amp is the coupled residual."""
    view = backend.device.component("q1_xy")
    xy = stub_machine.qubits["q1"].xy

    view.drive_power_dbm = -21.0
    assert xy.opx_output.full_scale_power_dbm == -11  # grid floor at weak drive
    assert view.drive_amp == pytest.approx(10 ** ((-21.0 + 11) / 20.0))

    view.drive_power_dbm = -2.0
    assert xy.opx_output.full_scale_power_dbm == 7  # back UP: bidirectional
    assert 0.354 < view.drive_amp <= 0.5

    xy.operations["saturation"].amplitude = 0.0
    with pytest.raises(ValueError, match="absolute power undefined"):
        _ = view.drive_power_dbm
    assert backend.device.snapshot()["q1_xy"]["drive_power_dbm"] is None


def test_recording_device_seeds_and_pushes_through_the_channel_entities(backend,
                                                                        roster):
    """End to end the way a Session drives it: RecordingDevice seeds its runtime
    config from the vendor (pull) and a neutral write lands on QUAM."""
    from conftest import recording_device

    device = recording_device(backend, roster)
    assert device.channel("q1", "readout").readout_freq_hz == pytest.approx(6.10e9)
    assert device.channel("q1_q2_c", "flux").idle_flux == pytest.approx(0.0)

    device.channel("q1", "drive").pi_amp = 0.31
    assert backend.device.component("q1_xy").pi_amp == pytest.approx(0.31)


def test_roster_toml_for_a_quam_tree_parses(stub_machine):
    """Pin the live-state roster generator against the stub tree: fixed-frequency
    qubits get no flux rider, the coupler becomes an ordinary mode with its own
    flux wire, and the pair composite carries the coupler role.

    Both names are derived from the MEMBERS (``q1_q2`` / ``q1_q2_c``), never from
    the QUAM pair key — the stub's key is ``coupler_q1_q2`` and the live tree's
    is ``q1_q2``, and reusing the latter collided with the composite. The join
    from either vendor spelling to these roster names therefore has to go through
    the roster, which is the property these fixtures exist to exercise."""
    from scqo.roster import parse_components

    generated = parse_components(roster_toml_for(stub_machine))
    assert generated.entities["q3"].kind == "transmon"
    assert ("q3", "flux") not in generated.defaults      # no flux on a fixed one
    assert ("q1_q2_c", "flux") in generated.defaults
    # the vendor's own name for the pair is NOT a roster name
    assert "coupler_q1_q2" not in generated.entities
    pair = generated.entities["q1_q2"]
    assert pair.roles["high"] == ("q2",) and pair.roles["low"] == ("q1",)
    assert pair.roles["coupler"] == ("q1_q2_c",)

    # ...and every name it declares resolves through the backend against the
    # same tree — which is what the skipped live-machine tests below rely on
    generated_backend = QMBackend(stub_machine, roster=generated)
    assert set(generated_backend.device.components()) == (
        set(generated.channels()) | set(generated.composites()))
    assert generated_backend.device.component(
        generated.default_channel("q1", "readout")).kind == "readout"


# ------------------------------------------------------------------ requires QUAM

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")


@pytest.fixture(scope="module")
def machine():
    # No root-class skip any more. This used to tolerate my_quam.py being toggled
    # between FluxTunableQuam and FixedFrequencyQuam, under which a flux-tunable
    # quam_state could not validate. Both the state file and the default now name
    # MixedTransmonQuam, which accepts fixed, tunable and mixed trees alike, so a
    # TypeError here is a real failure rather than a working-tree situation.
    return quam_config.Quam.load(str(Path(__file__).resolve().parents[1] / "quam_state"))


@pytest.fixture(scope="module")
def live_roster(machine):
    """A schema-3 roster mirroring whatever the loaded quam_state holds."""
    from scqo.roster import parse_components

    return parse_components(roster_toml_for(machine))


def test_probe_matches_direct_build(machine, live_roster):
    """QMQubitRamsey/QMQubitPowerRabi.probe() must produce the same QUA program as calling the
    module-level build_program directly with the mapped kwargs (proves the param mapping)."""
    from qm import generate_qua_script

    def script(prog):  # drop the volatile "generated at <timestamp>" header line
        return "\n".join(ln for ln in generate_qua_script(prog, config).splitlines() if "generated at" not in ln)

    from scqo_qm.experiments._lib import select_qubits
    from scqo_qm.experiments import qubit_ramsey as ramsey_probe
    from scqo_qm.experiments import qubit_power_rabi as power_rabi_probe
    from scqo_qm.experiments import _resonator_spectroscopy as resonator_spec_probe
    from scqo_qm.experiments.qubit_ramsey import QMQubitRamsey
    from scqo_qm.experiments.qubit_power_rabi import QMQubitPowerRabi
    from scqo_qm.experiments.resonator_spectroscopy import QMResonatorSpectroscopy

    backend = QMBackend(machine, roster=live_roster)
    config = machine.generate_config()
    qubits_names = ["q4", "q5"]
    qubits = select_qubits(machine, qubits_names, multiplexed=True)

    # Ramsey
    r = QMQubitRamsey(backend, QMQubitRamsey.Parameters(targets=qubits_names, num_averages=200))
    r.sweep_axes = r.define_sweep()
    r_prog, _ = r.probe()
    idle_cycles = np.maximum(1, np.round(r.sweep_axes["idle_time_ns"] / 4)).astype(int)
    r_direct, _ = ramsey_probe.build_program(
        machine, qubits, idle_times_cycles=idle_cycles,
        detuning_hz=int(r.params.frequency_detuning_hz), num_shots=200,
        reset_type="thermal", use_state_discrimination=False,
    )
    assert script(r_prog) == script(r_direct)

    # Power Rabi
    p = QMQubitPowerRabi(backend, QMQubitPowerRabi.Parameters(targets=qubits_names, num_averages=200))
    p.sweep_axes = p.define_sweep()
    p_prog, _ = p.probe()
    p_direct, _ = power_rabi_probe.build_program(
        machine, qubits, amps=p.sweep_axes["amp_prefactor"], operation="x180",
        num_shots=200, reset_type="thermal", use_state_discrimination=False, drive_qubit=None,
    )
    assert script(p_prog) == script(p_direct)

    # Resonator spectroscopy
    rs = QMResonatorSpectroscopy(
        backend, QMResonatorSpectroscopy.Parameters(targets=qubits_names, num_averages=200)
    )
    rs.sweep_axes = rs.define_sweep()
    rs_prog, _ = rs.probe()
    rs_direct, _ = resonator_spec_probe.build_program(
        machine, qubits, dfs=rs.sweep_axes["detuning_hz"], num_shots=200,
    )
    assert script(rs_prog) == script(rs_direct)


def _preview_ramsey(machine, live_roster):
    from scqo_qm.experiments.qubit_ramsey import QMQubitRamsey

    backend = QMBackend(machine, roster=live_roster)
    exp = QMQubitRamsey(backend, QMQubitRamsey.Parameters(
        targets=["q4", "q5"], num_averages=200))
    exp.sweep_axes = exp.define_sweep()  # the Session's job before the hook
    return backend, exp


def test_preview_writes_qua_script(machine, live_roster, tmp_path,
                                   monkeypatch):
    """QMBackend.preview with no_simulate: the QUA-script dump renders from
    the live state with no QOP connection at all (generate_config walks the
    tree in memory; the socket probe is armed to fail the test if touched)."""
    import socket

    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: pytest.fail(
                            "no_simulate must never touch the network"))
    backend, exp = _preview_ramsey(machine, live_roster)
    out_dir = tmp_path / "prev"
    files = backend.preview(exp, out_dir, no_simulate=True)
    assert files == [out_dir / "qua_script.py"]
    text = files[0].read_text(encoding="utf-8")
    assert text.startswith("# scqo preview: qubit_ramsey\n# backend: qm\n")
    assert "# params:" in text
    assert len(text.splitlines()) > 20  # a real program body, not just header


def test_tomography_preview_renders_through_the_hook(machine, live_roster,
                                                     tmp_path, monkeypatch):
    """A NORMAL (non-self-acquiring) shell's ``preview_program()`` is what the
    preview renders — the dispatch is 'has a hook', not 'is self-acquiring'.

    ``qubit_tomography``'s hook omits the training shots, so the proof is in the
    dumped script: the real ``probe()`` build saves I_train/Q_train streams and
    the previewed one does not. Multi-target on purpose — the single-target gate
    constrains self-acquiring shells only.
    """
    import socket

    from conftest import make_experiment
    from scqo_qm.experiments.qubit_tomography import QMQubitTomography

    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: pytest.fail(
                            "no_simulate must never touch the network"))
    qubits_names = list(machine.qubits.keys())[:2]
    backend = QMBackend(machine, roster=live_roster)
    exp = make_experiment(
        QMQubitTomography, backend, live_roster,
        QMQubitTomography.Parameters(targets=qubits_names, num_averages=10,
                                     num_training_shots=100))
    exp.sweep_axes = exp.define_sweep()

    files = backend.preview(exp, tmp_path / "prev", no_simulate=True)
    text = files[0].read_text(encoding="utf-8")
    assert "I_tomo1" in text          # the measurement itself is still rendered
    assert "I_train1" not in text     # ... without the training shots
    assert "Q_train1" not in text

    # and the ordinary probe() build DOES carry them, so the absence above is
    # the hook's doing rather than a property of the experiment
    from qm import generate_qua_script

    prog, _axes, _acquire = exp.probe()
    assert "I_train1" in generate_qua_script(prog, machine.generate_config())


def test_stark_amp_single_target_preview_builds(machine, live_roster, tmp_path,
                                                monkeypatch):
    """A self-acquiring pair shell (qc_n_stark_amp) is previewable with EXACTLY
    ONE --target: the single program it would build is rendered, no acquire, no
    network. Skips when the live state has no pair with a 'stark' xy op + an
    iswap-style swap macro (control-z flux pulse)."""
    import socket

    from conftest import make_experiment
    from scqo_qm.experiments.qc_n_stark_amp import QMQcNStarkAmp

    choice = None
    for _key, qp in machine.qubit_pairs.items():
        ctrl = qp.qubit_control
        if "stark" not in getattr(ctrl.xy, "operations", {}):
            continue
        target = f"{ctrl.name}_{qp.qubit_target.name}"
        ent = live_roster.entities.get(target)
        if ent is None:
            continue
        side = next((r for r in ("high", "low") if ctrl.name in ent.roles.get(r, ())),
                    None)
        if side is None or (ctrl.name, "flux") not in live_roster.defaults:
            continue
        swap_op = next((n for n, m in (qp.macros or {}).items()
                        if isinstance(getattr(m, "flux_pulse", None), str)
                        and getattr(m, "flux_pulse") in ctrl.z.operations), None)
        if swap_op is not None:
            choice = (target, side, swap_op)
            break
    if choice is None:
        pytest.skip("no live pair with a 'stark' xy op + an iswap-style swap macro")
    target, side, swap_op = choice

    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: pytest.fail(
                            "no_simulate must never touch the network"))
    backend = QMBackend(machine, roster=live_roster)
    exp = make_experiment(
        QMQcNStarkAmp, backend, live_roster,
        QMQcNStarkAmp.Parameters(targets=[target], swap_operation=swap_op,
                                 drive_side=side, flux_side=side,
                                 num_amp_points=5, swap_counts=[0, 1, 2],
                                 num_averages=10))
    exp.sweep_axes = exp.define_sweep()
    out_dir = tmp_path / "prev"
    files = backend.preview(exp, out_dir, no_simulate=True)
    assert files == [out_dir / "qua_script.py"]
    text = files[0].read_text(encoding="utf-8")
    assert text.startswith("# scqo preview: qc_n_stark_amp\n# backend: qm\n")
    assert len(text.splitlines()) > 20  # a real program body, not just header


def test_swap_amp_single_target_preview_builds(machine, live_roster, tmp_path,
                                               monkeypatch):
    """qc_n_swap_amp (self-acquiring) is previewable with EXACTLY ONE --target:
    the single program is built (no acquire, no network). Skips when the live
    state has no pair with an iswap-style swap macro (control-z flux pulse)."""
    import socket

    from conftest import make_experiment
    from scqo_qm.experiments.qc_n_swap_amp import QMQcNSwapAmp

    choice = None
    for _key, qp in machine.qubit_pairs.items():
        ctrl = qp.qubit_control
        target = f"{ctrl.name}_{qp.qubit_target.name}"
        ent = live_roster.entities.get(target)
        if ent is None:
            continue
        side = next((r for r in ("high", "low") if ctrl.name in ent.roles.get(r, ())),
                    None)
        if side is None or (ctrl.name, "flux") not in live_roster.defaults:
            continue
        swap_op = next((n for n, m in (qp.macros or {}).items()
                        if isinstance(getattr(m, "flux_pulse", None), str)
                        and getattr(m, "flux_pulse") in ctrl.z.operations), None)
        if swap_op is not None:
            choice = (target, side, swap_op)
            break
    if choice is None:
        pytest.skip("no live pair with an iswap-style swap macro")
    target, side, swap_op = choice

    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: pytest.fail(
                            "no_simulate must never touch the network"))
    backend = QMBackend(machine, roster=live_roster)
    exp = make_experiment(
        QMQcNSwapAmp, backend, live_roster,
        QMQcNSwapAmp.Parameters(targets=[target], swap_operation=swap_op,
                                drive_side=side, flux_side=side,
                                min_flux_amp_v=0.0, max_flux_amp_v=0.05,
                                num_amp_points=5, swap_counts=[0, 1, 2],
                                num_averages=10))
    exp.sweep_axes = exp.define_sweep()
    out_dir = tmp_path / "prev"
    files = backend.preview(exp, out_dir, no_simulate=True)
    assert files == [out_dir / "qua_script.py"]
    text = files[0].read_text(encoding="utf-8")
    assert text.startswith("# scqo preview: qc_n_swap_amp\n# backend: qm\n")
    assert len(text.splitlines()) > 20  # a real program body, not just header


class _FakeQMM:
    """Stands in for machine.connect(): serves canned simulated samples."""

    def __init__(self, samples):
        self._samples = samples
        self.closed = False
        self.simulated_with = None

    def simulate(self, config, prog, sim_config):
        self.simulated_with = sim_config
        outer = self

        class _Job:
            def get_simulated_samples(self):
                return outer._samples

        return _Job()

    def close(self):
        self.closed = True


def _fake_samples(signal: bool):
    # the REAL vendor sample classes, so the plot path exercises the true API
    import numpy as np
    from qm.simulate._simulator_samples import (
        SimulatorControllerSamples,
        SimulatorSamples,
    )

    analog = np.zeros(64)
    if signal:
        analog[8:24] = 0.25
    cs = SimulatorControllerSamples(
        analog={"1-1": analog}, digital={"1-2": np.zeros(64, dtype=bool)},
        analog_sampling_rate={"1-1": 2e9})
    return SimulatorSamples({"con1": cs})


def test_preview_simulate_writes_waveforms(machine, live_roster, tmp_path,
                                           monkeypatch):
    """The auto-simulate path: probe passes, the (stubbed) gateway serves
    samples, the interactive waveform plot lands beside the script, and the
    QMM connection is closed."""
    import socket

    class _Probe:
        def close(self):
            pass

    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: _Probe())
    backend, exp = _preview_ramsey(machine, live_roster)
    fake = _FakeQMM(_fake_samples(signal=True))
    monkeypatch.setattr(machine, "connect", lambda: fake, raising=False)
    out_dir = tmp_path / "prev"
    files = backend.preview(exp, out_dir)
    assert [f.name for f in files] == ["qua_script.py",
                                       "simulated_waveforms.html"]
    assert files[1].stat().st_size > 0
    assert fake.closed  # the gRPC channel never leaks
    assert fake.simulated_with.duration == 20_000 // 4  # default window, cycles


def test_preview_simulate_empty_window_warns_no_file(machine, live_roster,
                                                     tmp_path, monkeypatch):
    """An all-zero simulated window (thermal shot = long leading wait) warns
    with guidance instead of shipping a blank plot."""
    import socket

    from scqo.backend import PreviewWarning

    class _Probe:
        def close(self):
            pass

    monkeypatch.setattr(socket, "create_connection",
                        lambda *a, **k: _Probe())
    backend, exp = _preview_ramsey(machine, live_roster)
    monkeypatch.setattr(machine, "connect",
                        lambda: _FakeQMM(_fake_samples(signal=False)),
                        raising=False)
    with pytest.warns(PreviewWarning, match="contains no pulses"):
        files = backend.preview(exp, tmp_path / "prev")
    assert [f.name for f in files] == ["qua_script.py"]


def test_preview_simulate_degrades_when_gateway_is_dead(machine, live_roster,
                                                        tmp_path,
                                                        monkeypatch):
    """A dead host fails the 2 s TCP probe and the preview degrades to the
    script with a PreviewWarning — never an error."""
    import socket

    from scqo.backend import PreviewWarning

    def refuse(*args, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr(socket, "create_connection", refuse)
    backend, exp = _preview_ramsey(machine, live_roster)
    with pytest.warns(PreviewWarning, match="simulated waveforms skipped"):
        files = backend.preview(exp, tmp_path / "prev")
    assert [f.name for f in files] == ["qua_script.py"]


def test_preview_simulate_ns_cap_refuses(machine, live_roster, tmp_path):
    backend, exp = _preview_ramsey(machine, live_roster)
    with pytest.raises(ValueError, match="simulate_ns"):
        backend.preview(exp, tmp_path / "prev", simulate_ns=10_000_000)
    with pytest.raises(ValueError, match="contradict"):
        backend.preview(exp, tmp_path / "prev", simulate_ns=1000,
                        no_simulate=True)
    assert not (tmp_path / "prev").exists()


@pytest.mark.parametrize("gate", ["x180", "x90"])
def test_gate_target_probes_build_for_both_gates(machine, gate):
    """Issue #24: the drag-equator probe died with a TypeError (stale ``lock_x90``
    kwarg into quam_fields.set_drag_beta) before any QUA was emitted, for BOTH
    target gates — and nothing imported the probe, so no test caught it. Build all
    three target_gate probes end-to-end on the live state for each gate, and prove
    the equator's install/restore leaves the stored QUAM alphas unchanged."""
    from scqo_qm import quam_fields
    from scqo_qm.experiments import qubit_deterministic_benchmarking as db_probe
    from scqo_qm.experiments import qubit_drag_alternating as drag_alternating_probe
    from scqo_qm.experiments import qubit_drag_equator as drag_equator_probe
    from scqo_qm.experiments._lib import select_qubits

    qubits_names = ["q4", "q5"]
    qubits = select_qubits(machine, qubits_names, multiplexed=True)

    before = {q: quam_fields.get_drag_beta(machine.qubits[q], operation=gate)
              for q in qubits_names}
    prog, axes, config = drag_equator_probe.build_program(
        machine, qubits, num_shots=10, beta_array=[-0.2, 0.0, 0.2],
        pulse_repetitions=3, reset_type="thermal",
        use_state_discrimination=False, target_gate=gate)
    assert prog is not None and config is not None
    after = {q: quam_fields.get_drag_beta(machine.qubits[q], operation=gate)
             for q in qubits_names}
    assert after == pytest.approx(before)

    prog, _ = drag_alternating_probe.build_program(
        machine, qubits, num_shots=10, beta_array=[-0.2, 0.0, 0.2],
        nb_pulses_array=[2, 4], reset_type="thermal",
        use_state_discrimination=False, target_gate=gate)
    assert prog is not None

    prog, _ = db_probe.build_program(
        machine, qubits, num_shots=10, repetitions=[0, 4, 8],
        target_gate=gate, reset_type="thermal", use_state_discrimination=False)
    assert prog is not None


def test_live_readout_window_round_trip(machine, live_roster):
    """The window accessors against a REAL QUAM ReadoutPulse (its default-weights
    reference semantics are what the stub only mimics) — restored afterwards."""
    backend = QMBackend(machine, roster=live_roster)
    view = backend.device.component(live_roster.default_channel("q4", "readout"))
    pulse = machine.qubits["q4"].resonator.operations["readout"]
    length_ns = int(pulse.length)
    if pulse.integration_weights != [(1, length_ns)]:
        pytest.skip("q4's readout weights are not in the default-reference form")
    half = (length_ns // 2 // 4 * 4) * 1e-9
    try:
        view.readout_integration_s = half
        assert pulse.integration_weights[0][0] == 1.0
        assert view.readout_integration_s == pytest.approx(half)
    finally:
        view.readout_integration_s = length_ns * 1e-9  # restore the reference form


def test_absolute_punchout_probe_matches_direct_build(machine, live_roster):
    """Chain-stepped contract: QMResonatorSpectroscopyPowerChain.probe() builds
    the plain 1D resonator-spectroscopy program at the current device state — the
    core run() loop solves the chain per point and swaps in the 1D detuning axis."""
    from qm import generate_qua_script

    from scqo_qm.experiments._lib import select_qubits
    from scqo_qm.experiments import _resonator_spectroscopy as res_spec_probe
    from scqo_qm.experiments.resonator_spectroscopy_power_chain import (
        QMResonatorSpectroscopyPowerChain,
    )

    backend = QMBackend(machine, roster=live_roster)
    config = machine.generate_config()

    def script(prog):
        return "\n".join(
            ln for ln in generate_qua_script(prog, config).splitlines() if "generated at" not in ln
        )

    qubits_names = ["q4", "q5"]
    qubits = select_qubits(machine, qubits_names, multiplexed=True)

    exp = QMResonatorSpectroscopyPowerChain(
        backend,
        QMResonatorSpectroscopyPowerChain.Parameters(
            targets=qubits_names, max_power_dbm=-15.0, min_power_dbm=-45.0, num_averages=100
        ),
    )
    axes = exp.define_sweep()
    # uniform grid straight from the core
    power_dbm = np.asarray(axes["power_dbm"])
    steps = np.diff(power_dbm)
    assert np.allclose(steps, steps[0])
    # mimic one per-point call (the run loop swaps in the 1D axis)
    exp.sweep_axes = {"detuning_hz": axes["detuning_hz"]}
    prog, _ = exp.probe()

    direct, _ = res_spec_probe.build_program(
        machine, qubits, dfs=axes["detuning_hz"], num_shots=100,
    )
    assert script(prog) == script(direct)


def test_power_amp_probe_builds_with_new_loop_order(machine, live_roster):
    """The fast absolute punchout (amp -> averages -> freq loop order, middle-axis
    stream averaging) compiles to a QUA program: prefactors 10**((P - max)/20)
    relative to the window top the core run() solved the chain for (top exactly
    1.0, all <= 1 — inside QUA's amplitude_scale range), and
    readout_depletion_ns reaches the program (the generated script changes
    when it is set)."""
    from conftest import make_experiment
    from qm import generate_qua_script

    from scqo_qm.experiments.resonator_spectroscopy_power_amp import (
        QMResonatorSpectroscopyPowerAmp,
    )

    backend = QMBackend(machine, roster=live_roster)
    config = machine.generate_config()

    def script(params):
        # make_experiment, not a bare constructor: the ring-down wait is resolved
        # through the neutral device surface now (per-run override -> the
        # readout_depletion_s knob), so the probe needs the channel views a
        # Session would have attached.
        exp = make_experiment(
            QMResonatorSpectroscopyPowerAmp, backend, live_roster,
            QMResonatorSpectroscopyPowerAmp.Parameters(**params))
        exp.sweep_axes = exp.define_sweep()
        # the axis is the absolute window straight from the core
        power_dbm = np.asarray(exp.sweep_axes["power_dbm"])
        assert power_dbm[0] == -50.0 and power_dbm[-1] == -20.0  # the defaults
        prog, axes = exp.probe()
        assert set(axes) == {"qubit", "detuning", "power"}
        return "\n".join(
            ln for ln in generate_qua_script(prog, config).splitlines() if "generated at" not in ln
        )

    base = dict(targets=["q4"], num_power_points=5, num_readout_freq_points=3, num_averages=10)
    default = script(base)
    overridden = script({**base, "readout_depletion_ns": 25000.0})
    assert default != overridden  # the relaxation override reaches the QUA program


def test_readout_fidelity_probe_state_selection_compiles(machine, live_roster):
    """The one probe now serves three experiments by prepared_states alone. All
    three must COMPILE against the live config — the |f> branch plays inside a
    switch_ and retunes the drive element mid-case, which is where a QUA build
    would break — and the two-state default must stay byte-identical to what it
    generated before the parameter existed."""
    from qm import generate_qua_script

    from scqo_qm.experiments._lib import select_qubits
    from scqo_qm.experiments import _readout_fidelity as fidelity_probe

    config = machine.generate_config()

    def script(prog):
        return "\n".join(
            ln for ln in generate_qua_script(prog, config).splitlines() if "generated at" not in ln
        )

    names = ["q1"]
    if not machine.qubits[names[0]].xy.operations.get("EF_x180"):
        pytest.skip("q1 has no calibrated EF_x180 in the live state")
    qubits = select_qubits(machine, names, multiplexed=True)
    common = dict(operation="readout", num_shots=10, reset_type="thermal")

    default, axes = fidelity_probe.build_program(machine, qubits, **common)
    explicit, _ = fidelity_probe.build_program(machine, qubits, prepared_states=(0, 1), **common)
    assert script(default) == script(explicit)
    assert list(axes["prepared_state"].values) == [0, 1]

    gef, gef_axes = fidelity_probe.build_program(
        machine, qubits, prepared_states=(0, 1, 2), readout_freq_shift_hz=-600e3, **common)
    assert script(gef)  # the EF case + the resonator shift assemble
    assert list(gef_axes["prepared_state"].values) == [0, 1, 2]

    ground, ground_axes = fidelity_probe.build_program(machine, qubits, prepared_states=(0,), **common)
    assert script(ground)
    assert list(ground_axes["prepared_state"].values) == [0]


def test_readout_sweeps_build_in_both_readout_modes(machine, live_roster):
    """readout_power / readout_frequency realize BOTH readout modes off one
    program: the shot loop is identical and only the stream terminal differs
    (`.buffer(num_shots)` vs `.average()`), so average mode must compile against
    the live config and must drop `shot_idx` from the sweep axes — scqo's
    contract accepts the averaged form only without that axis."""
    from qm import generate_qua_script

    from scqo_qm.experiments._lib import select_qubits
    from scqo_qm.experiments import readout_power as power_probe
    from scqo_qm.experiments import readout_frequency as freq_probe

    config = machine.generate_config()
    qubits = select_qubits(machine, ["q1"], multiplexed=True)
    common = dict(num_shots=10, reset_type="thermal")
    cases = [
        (power_probe, {"amps": np.linspace(0.5, 1.0, 3)}),
        (freq_probe, {"dfs": np.linspace(-1e6, 1e6, 3)}),
    ]
    for module, sweep in cases:
        shot_prog, shot_axes = module.build_program(machine, qubits, **sweep, **common)
        avg_prog, avg_axes = module.build_program(
            machine, qubits, **sweep, average_shots=True, **common)

        assert "shot_idx" in shot_axes and "shot_idx" not in avg_axes
        shot_script = generate_qua_script(shot_prog, config)
        avg_script = generate_qua_script(avg_prog, config)
        assert "average()" in avg_script and "average()" not in shot_script
        # same measurement sequence either way — only the terminal moved
        assert avg_script.count("measure(") == shot_script.count("measure(")


# ------------------------------------------- the pair swap maps, live QUAM tree

def _live_pair(machine) -> tuple[str, object]:
    """The first live QUAM pair and the ROSTER composite name for it.

    ``roster_toml_for`` names composites ``<control>_<target>`` — QM's own pairs
    are named after the coupler, which is exactly the join under test."""
    key, qp = next(iter(machine.qubit_pairs.items()))
    return f"{qp.qubit_control.name}_{qp.qubit_target.name}", qp


def _swap_experiment(cls, machine, live_roster, **params):
    from conftest import make_experiment

    backend = QMBackend(machine, roster=live_roster)
    target, _ = _live_pair(machine)
    return make_experiment(cls, backend, live_roster,
                           cls.Parameters(targets=[target], **params))


def test_swap_chevron_probe_builds_against_the_baked_config(machine, live_roster,
                                                            monkeypatch):
    """The chevron acquires itself because its program only runs against the
    probe's own BAKED config: the 1..16 ns segments are registered there, and
    the shared fetch path would hand the QOP a freshly generated config without
    them. Pin that the config actually travels, and that the program compiles."""
    from qm import generate_qua_script

    from scqo_qm.experiments import pair_swap_chevron as chevron_probe
    from scqo_qm.experiments.pair_swap_chevron import QMPairSwapChevron

    captured = {}

    def fake_acquire(m, prog, sweep_axes, *, num_shots, timeout, log=None, config=None):
        captured.update(prog=prog, sweep_axes=sweep_axes, config=config,
                        num_shots=num_shots)
        return xr.Dataset()

    monkeypatch.setattr(chevron_probe, "acquire", fake_acquire)

    exp = _swap_experiment(QMPairSwapChevron, machine, live_roster,
                           min_flux_amp_v=0.0, max_flux_amp_v=0.05, num_amp_points=5,
                           min_swap_time_ns=1.0, max_swap_time_ns=40.0,
                           num_time_points=20, num_averages=10)
    exp.sweep_axes = exp.define_sweep()
    exp.probe()

    # the 16 baked segments exist in the config that was passed, on the FLUX
    # member's z element, and a freshly generated config has none of them —
    # which is the whole reason this probe cannot use the shared fetch path
    fresh = machine.generate_config()
    baked = set(captured["config"]["pulses"]) - set(fresh["pulses"])
    assert len(baked) == 16, sorted(baked)
    _, qp = _live_pair(machine)
    assert all(name.startswith(f"{qp.qubit_control.z.name}_baked") for name in baked)
    assert captured["num_shots"] == 10

    # canonical axis names, roster target names, and the REAL quantized time grid
    assert set(captured["sweep_axes"]) == {"qubit_pair", "flux_amp_v", "swap_time_ns"}
    target, _ = _live_pair(machine)
    assert list(captured["sweep_axes"]["qubit_pair"].values) == [target]
    np.testing.assert_allclose(exp.sweep_axes["swap_time_ns"],
                               captured["sweep_axes"]["swap_time_ns"].values)
    # absolute mode: the axis carries VOLTS, not the QUA scale factor
    assert captured["sweep_axes"]["flux_amp_v"].attrs["units"] == "V"
    assert float(captured["sweep_axes"]["flux_amp_v"].values.max()) == pytest.approx(0.05)

    assert generate_qua_script(captured["prog"], captured["config"])


def test_xyz_delay_probe_builds_against_the_baked_config(machine, live_roster,
                                                         monkeypatch):
    """XY-Z delay acquires itself against its OWN baked config: every relative
    shift bakes an x180 + flux_pulse segment together, which a freshly generated
    config lacks. Pin that the baked ops travel and the program compiles."""
    from qm import generate_qua_script

    from scqo_qm.experiments import qubit_xyz_delay as xyz_probe
    from scqo_qm.experiments.qubit_xyz_delay import QMQubitXyzDelay

    captured = {}

    def fake_acquire(m, prog, sweep_axes, *, num_shots, timeout, log=None, config=None):
        captured.update(prog=prog, sweep_axes=sweep_axes, config=config,
                        num_shots=num_shots)
        return xr.Dataset()

    monkeypatch.setattr(xyz_probe, "acquire", fake_acquire)

    backend = QMBackend(machine, roster=live_roster)
    exp = QMQubitXyzDelay(
        backend,
        QMQubitXyzDelay.Parameters(
            targets=["q4", "q5"], half_scan_ns=5, z_pulse_amp_v=0.0, num_averages=10
        ),
    )
    exp.sweep_axes = exp.define_sweep()
    exp.probe()

    # both the flux and the XY line carry baked segments in the passed config, and
    # a freshly generated one has none of them — the reason this probe self-acquires
    fresh = machine.generate_config()
    baked = set(captured["config"]["pulses"]) - set(fresh["pulses"])
    assert baked, "no baked segments travelled in the config"
    q4 = machine.qubits["q4"]
    assert any(name.startswith(f"{q4.z.name}_baked") for name in baked)
    assert any(name.startswith(f"{q4.xy.name}_baked") for name in baked)
    assert captured["num_shots"] == 10

    # canonical scqo axis names, [0, 1] prep, the full 2*half relative-time grid
    assert set(captured["sweep_axes"]) == {"qubit", "prepared_state", "relative_time_ns"}
    assert list(captured["sweep_axes"]["prepared_state"].values) == [0, 1]
    assert len(captured["sweep_axes"]["relative_time_ns"].values) == 10  # 2 * half_scan_ns
    assert generate_qua_script(captured["prog"], captured["config"])


def test_xyz_delay_small_half_scan_emits_no_illegal_wait(machine, live_roster,
                                                         monkeypatch):
    """half_scan_ns < 16 used to bake the coarse pre-wait as wait(<4) — legal to
    the qm client AND to generate_qua_script, refused ONLY by the gateway
    compiler ("must be a minimum 4", 5Q4C 2026-08-09). The fix floors the
    pre-wait at 4 cycles, so the offline pin is the gateway's own rule applied
    to the generated script: every wait duration >= 4."""
    import re

    from qm import generate_qua_script

    from scqo_qm.experiments import qubit_xyz_delay as xyz_probe
    from scqo_qm.experiments.qubit_xyz_delay import QMQubitXyzDelay

    captured = {}

    def fake_acquire(m, prog, sweep_axes, *, num_shots, timeout, log=None, config=None):
        captured.update(prog=prog, config=config)
        return xr.Dataset()

    monkeypatch.setattr(xyz_probe, "acquire", fake_acquire)

    backend = QMBackend(machine, roster=live_roster)
    exp = QMQubitXyzDelay(
        backend,
        QMQubitXyzDelay.Parameters(
            targets=["q4"], half_scan_ns=5, z_pulse_amp_v=0.0, num_averages=2
        ),
    )
    exp.sweep_axes = exp.define_sweep()
    exp.probe()

    script = generate_qua_script(captured["prog"], captured["config"])
    waits = [int(m) for m in re.findall(r"\bwait\((\d+)[,)]", script)]
    assert waits, "expected wait statements in the generated script"
    assert min(waits) >= 4, f"illegal sub-4-cycle wait reached the script: {sorted(set(waits))[:5]}"


def test_swap_flux_map_probe_builds(machine, live_roster):
    """The 2D map needs no baking, so it returns the ordinary (program, axes)
    pair and the backend's shared fetch runs it."""
    from qm import generate_qua_script

    from scqo_qm.experiments.pair_swap_flux_map import QMPairSwapFluxMap

    exp = _swap_experiment(QMPairSwapFluxMap, machine, live_roster,
                           min_coupler_flux_v=-0.02, max_coupler_flux_v=0.02,
                           num_coupler_points=5, min_qubit_flux_v=0.0,
                           max_qubit_flux_v=0.02, num_qubit_points=5,
                           swap_time_ns=45.0, num_averages=10)
    exp.sweep_axes = exp.define_sweep()
    prog, axes = exp.probe()

    assert set(axes) == {"qubit_pair", "qubit_flux_v", "coupler_flux_v"}
    assert axes["coupler_flux_v"].attrs["units"] == "V"
    # 45 ns is not on QM's 4 ns clock: the driver quantizes to 44 and RECORDS
    # that, so result.fit reports what actually played, not what was asked for
    assert exp._flux_time_ns == 44.0
    assert generate_qua_script(prog, machine.generate_config())


def _live_flux_qubit(machine):
    """The first live QUAM qubit carrying both a flux `const` op and an `x90` —
    everything the ramsey cryoscope sequence plays. Roster mode name == QUAM name."""
    for name, q in machine.qubits.items():
        z = getattr(q, "z", None)
        if (z is not None and "const" in getattr(z, "operations", {})
                and "x90" in q.xy.operations):
            return name
    return None


def test_ramsey_cryoscope_probe_builds_against_the_baked_config(machine, live_roster,
                                                         monkeypatch):
    """Like the swap chevron, the ramsey cryoscope acquires itself: its 1..16 ns baked
    segments live only in the probe's own config. Pin that the baked config
    travels, the canonical axes come back, and the phase-tomography program
    COMPILES against the live QUAM (the pure validate_inputs test cannot)."""
    from qm import generate_qua_script

    from scqo_qm.experiments import qubit_ramsey_cryoscope as ramsey_cryoscope_probe
    from scqo_qm.experiments.qubit_ramsey_cryoscope import QMQubitRamseyCryoscope

    name = _live_flux_qubit(machine)
    if name is None:
        pytest.skip("no flux-tunable qubit with a const op in the live state")

    captured = {}

    def fake_acquire(m, prog, sweep_axes, *, num_shots, timeout, log=None, config=None):
        captured.update(prog=prog, sweep_axes=sweep_axes, config=config,
                        num_shots=num_shots)
        return xr.Dataset()

    monkeypatch.setattr(ramsey_cryoscope_probe, "acquire", fake_acquire)

    backend = QMBackend(machine, roster=live_roster)
    exp = QMQubitRamseyCryoscope(
        backend,
        QMQubitRamseyCryoscope.Parameters(targets=[name], max_duration_ns=32,
                                    num_frames=8, num_averages=10,
                                    flux_pulse_amp_v=0.02),
    )
    exp.sweep_axes = exp.define_sweep()
    exp.probe()

    # the 16 baked segments exist in the passed config, absent from a fresh one
    fresh = machine.generate_config()
    baked = set(captured["config"]["pulses"]) - set(fresh["pulses"])
    assert len(baked) == 16, sorted(baked)
    assert captured["num_shots"] == 10

    # canonical axes in raw nesting order, with their units
    assert set(captured["sweep_axes"]) == {"qubit", "duration_ns", "frame"}
    assert list(captured["sweep_axes"]["qubit"].values) == [name]
    assert captured["sweep_axes"]["duration_ns"].attrs["units"] == "ns"
    assert captured["sweep_axes"]["frame"].attrs["units"] == "turn"
    assert len(captured["sweep_axes"]["duration_ns"]) == 32
    assert len(captured["sweep_axes"]["frame"]) == 8

    assert generate_qua_script(captured["prog"], captured["config"])


def test_spectroscopy_cryoscope_probe_builds_against_the_live_quam(machine, live_roster):
    """The long-time (spectroscopy) cryoscope needs no baking, so it returns the
    ordinary (program, sweep_axes) pair and the backend's shared fetch runs it.
    Pin the canonical axes, the RUN-SCOPED drive op's lifecycle (probe() files it,
    so machine.generate_config() — which the shared acquire calls AFTER probe() —
    carries it at exactly drive_len_ns, and the teardown removes it again so it can
    never reach machine.save()), and that the parked spectroscopy program COMPILES
    against the live QUAM (the pure validate_inputs test cannot)."""
    from qm import generate_qua_script

    from scqo_qm.experiments.qubit_spectroscopy_cryoscope import (
        DRIVE_OP,
        QMQubitSpectroscopyCryoscope,
    )

    name = _live_flux_qubit(machine)
    if name is None:
        pytest.skip("no flux-tunable qubit with a const op in the live state")

    backend = QMBackend(machine, roster=live_roster)
    exp = QMQubitSpectroscopyCryoscope(
        backend,
        QMQubitSpectroscopyCryoscope.Parameters(
            targets=[name], start_drive_detuning_hz=-150e6, end_drive_detuning_hz=0.0,
            num_drive_freq_points=11, drive_len_ns=400, min_wait_ns=16, max_wait_ns=2000,
            num_wait_points=8, num_averages=10, flux_pulse_amp_v=0.02,
        ),
    )
    xy = machine.qubits[name].xy
    assert DRIVE_OP not in xy.operations  # nothing persisted from an earlier run
    exp.sweep_axes = exp.define_sweep()
    try:
        prog, sweep_axes = exp.probe()

        # the run-scoped tone: on the line at exactly drive_len_ns and weaker than
        # the calibrated x180 (it spreads the same rotation area over 400 ns, not
        # 16), and rendered into the config generated after probe() returns.
        assert xy.operations[DRIVE_OP].length == 400
        assert 0 < xy.operations[DRIVE_OP].amplitude < xy.operations["x180"].amplitude
        config = machine.generate_config()
        pulse_id = config["elements"][xy.name]["operations"][DRIVE_OP]
        assert config["pulses"][pulse_id]["length"] == 400
    finally:
        exp._drop_drive_op()  # what run()'s finally does
    assert DRIVE_OP not in xy.operations
    assert DRIVE_OP not in machine.generate_config()["elements"][xy.name]["operations"]

    # canonical axes in raw nesting order, with their units — no baking, so the
    # adapter returns (program, sweep_axes) rather than self-acquiring
    assert set(sweep_axes) == {"qubit", "detuning_hz", "wait_time_ns"}
    assert list(sweep_axes["qubit"].values) == [name]
    assert sweep_axes["detuning_hz"].attrs["units"] == "Hz"
    assert sweep_axes["wait_time_ns"].attrs["units"] == "ns"
    assert len(sweep_axes["detuning_hz"]) == 11
    # the asymmetric window flows through define_sweep -> probe, ascending edges
    assert sweep_axes["detuning_hz"].values[0] == pytest.approx(-150e6)
    assert sweep_axes["detuning_hz"].values[-1] == pytest.approx(0.0)
    assert 2 <= len(sweep_axes["wait_time_ns"]) <= 8  # log axis dedups on the 4 ns grid

    # against the config captured while the op was installed — the one the real
    # acquire path uses.
    assert generate_qua_script(prog, config)


def test_apply_exponential_filter_extends_a_live_quam_port(machine):
    """apply_exponential_filter's EXTEND path must mutate the live QuamList IN PLACE.
    Reassigning old+new re-parents the existing QuamList children and QUAM refuses
    ("Cannot overwrite parent attribute") — a plain-list stub cannot see this, so
    pin it against the real QUAM (snapshot + restore as plain lists, never saved)."""
    from scqo_qm.backend._distortion import apply_exponential_filter

    name = _live_flux_qubit(machine)
    if name is None:
        pytest.skip("no flux-tunable qubit with a const op in the live state")
    port = machine.qubits[name].z.opx_output
    saved = [list(pair) for pair in port.exponential_filter]  # plain-list snapshot
    try:
        out = apply_exponential_filter(machine, name, [0.05, -0.03], [100e-9, 3000e-9])
        assert [list(p) for p in port.exponential_filter] == [
            [0.05, 100.0], [-0.03, 3000.0]]  # replace + tau s->ns
        assert out["scale"] == 1.0
        # the gotcha: extend must not raise on the live QuamList
        apply_exponential_filter(machine, name, [0.02], [50e-9], replace=False)
        assert [list(p) for p in port.exponential_filter] == [
            [0.05, 100.0], [-0.03, 3000.0], [0.02, 50.0]]  # appended, not re-parented
    finally:
        port.exponential_filter = saved  # restore (plain lists -> no re-parent)


def test_active_reset_program_builds_on_live_state(machine, live_roster):
    """Building the ramsey program with reset_type='active' EXECUTES QUAM's
    reset_qubit_active, so this proves offline everything offline can: the kwarg
    threads through to max_attempts, the discriminator thresholds are consumed,
    and the whole QUA program serialises against the live config. It deliberately
    BYPASSES check_reset_method — a build test must not be gated on whether the
    live state happens to be calibrated; the feedback loop itself is hardware.
    Thresholds are written in memory and restored (the module-scoped fixture is
    never saved)."""
    from qm import generate_qua_script

    from scqo_qm.experiments._lib import select_qubits
    from scqo_qm.experiments import qubit_ramsey as ramsey_probe

    # a live qubit whose xy carries both x90 (ramsey) and x180 (the reset pi)
    name = next((n for n, q in machine.qubits.items()
                 if "x90" in q.xy.operations and "x180" in q.xy.operations
                 and "readout" in q.resonator.operations), None)
    if name is None:
        pytest.skip("no live qubit with x90/x180 and a readout op")

    pulse = machine.qubits[name].resonator.operations["readout"]
    saved = (pulse.threshold, pulse.rus_exit_threshold)
    try:
        pulse.threshold = -1.0e-4
        pulse.rus_exit_threshold = -2.0e-4
        qubits = select_qubits(machine, [name], multiplexed=True)
        # reset_max_attempts=2 -> reset_qubit_active(max_attempts=2): a wrong
        # kwarg name is a TypeError right here, before serialisation.
        prog, _ = ramsey_probe.build_program(
            machine, qubits, idle_times_cycles=np.array([4, 8, 12]),
            detuning_hz=1_000_000, num_shots=10,
            reset_type="active", reset_max_attempts=2,
            use_state_discrimination=False,
        )
        assert generate_qua_script(prog, machine.generate_config())
    finally:
        pulse.threshold, pulse.rus_exit_threshold = saved


def test_ade_tracking_program_builds_on_live_state(machine, live_roster):
    """The repo's first on-FPGA-arithmetic probe: Math.div/sqrt/ln and the relu
    clamps only serialise against a real config, so a successful
    generate_qua_script IS the maximum offline proof. Built with
    reset_type='active' and adaptive_dt=True so the whole surface (the reset
    door, the Cast.mul_int_by_fixed dt retune) threads. Thresholds are written
    in memory and restored (the module-scoped fixture is never saved)."""
    from qm import generate_qua_script

    from scqo_qm.experiments._lib import select_qubits
    from scqo_qm.experiments import qubit_t1_ade as ade_probe

    name = next((n for n, q in machine.qubits.items()
                 if "x180" in q.xy.operations and "readout" in q.resonator.operations),
                None)
    if name is None:
        pytest.skip("no live qubit with x180 and a readout op")

    pulse = machine.qubits[name].resonator.operations["readout"]
    saved = (pulse.threshold, pulse.rus_exit_threshold)
    try:
        pulse.threshold = -1.0e-4
        pulse.rus_exit_threshold = -2.0e-4
        qubits = select_qubits(machine, [name], multiplexed=False)
        prog, _ = ade_probe.build_program(
            machine, qubits, num_blocks=3, n_avg=5,
            t0_cycles=4, dt_cycles=2500, adaptive_dt=True, dt_factor=1.0,
            min_dt_cycles=4, max_dt_cycles=50_000,
            reset_type="active", reset_max_attempts=2,
        )
        assert generate_qua_script(prog, machine.generate_config())
    finally:
        pulse.threshold, pulse.rus_exit_threshold = saved


def test_bayesian_tracking_program_builds_on_live_state(machine, live_roster):
    """The u = 1/k posterior update (Math.inv/ln/exp, both phi branches) and
    the QUAM confusion-matrix reads serialise against the live config.
    Thresholds — and the confusion matrix when the live state lacks one — are
    written in memory and restored as plain lists (no QuamList re-parenting)."""
    from qm import generate_qua_script

    from scqo_qm.experiments._lib import select_qubits
    from scqo_qm.experiments import qubit_t1_bayesian as bayes_probe

    name = next((n for n, q in machine.qubits.items()
                 if "x180" in q.xy.operations and "readout" in q.resonator.operations),
                None)
    if name is None:
        pytest.skip("no live qubit with x180 and a readout op")

    qubit = machine.qubits[name]
    pulse = qubit.resonator.operations["readout"]
    cm = qubit.resonator.confusion_matrix
    saved_cm = None if cm is None else [list(row) for row in cm]
    saved = (pulse.threshold, pulse.rus_exit_threshold)
    try:
        pulse.threshold = -1.0e-4
        pulse.rus_exit_threshold = -2.0e-4
        if qubit.resonator.confusion_matrix is None:
            qubit.resonator.confusion_matrix = [[0.95, 0.05], [0.09, 0.91]]
        qubits = select_qubits(machine, [name], multiplexed=False)
        prog, _ = bayes_probe.build_program(
            machine, qubits, num_blocks=2, num_probes=5,
            c_adaptive=0.51, k0=1.0, t1_prior_s={name: 35e-6},
            t1_min_s=1e-6, t1_max_s=100e-6, k_min=0.2, k_max=100.0,
            interleaved=True,
            lin_wait_cycles=np.array([4, 50, 500, 5000, 50000]),
            active_reset_per_probe=False,
            reset_type="active", reset_max_attempts=2,
        )
        assert generate_qua_script(prog, machine.generate_config())
    finally:
        pulse.threshold, pulse.rus_exit_threshold = saved
        qubit.resonator.confusion_matrix = saved_cm
