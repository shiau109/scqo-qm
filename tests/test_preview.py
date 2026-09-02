"""``QMBackend.preview``: the self-acquiring census + the refusal dispatch.

QUAM-free (conftest stub): preview's refusal for a ``probe_self_acquires``
shell must fire BEFORE ``probe()`` — the stub machine cannot build a QUA
program, so getting past the refusal would blow up, which is exactly the
ordering proof. The census keeps the declarations in lockstep with the shells
that really acquire inside ``probe()``: a new self-acquiring shell MUST
declare the attribute, or preview would reach the instrument before the
backend's defensive ready-Dataset check could fire. The live-state script
dump is ``test_qm_backend.py::test_preview_writes_qua_script`` (needs the
``machine`` fixture, so it does not live here).
"""

from __future__ import annotations

import pytest

from conftest import make_experiment

import scqo_qm.experiments  # noqa: F401  (import side effect: @register)
from scqo_qm.backend.qm_backend import SELF_ACQUIRING_ATTR
from scqo.experiments import catalog, get

#: the shells whose probe() executes on the instrument and returns a ready
#: Dataset — the ONLY legitimate carriers of the opt-out attribute
SELF_ACQUIRING = {
    "broadband_qubit_spectroscopy",
    "broadband_resonator_spectroscopy",
    "pair_swap_angle",
    "pair_swap_chevron",
    "qc_n_stark_amp",
    "qc_n_swap_amp",
    "qubit_drag_equator",
    "qubit_drag_alternating",
    "qubit_ramsey_cryoscope",
    "qubit_xyz_delay",
}


def _qm_shells() -> dict[str, type]:
    """Every registered experiment whose class comes from THIS driver."""
    return {entry["name"]: get(entry["name"]) for entry in catalog()
            if get(entry["name"]).__module__.startswith("scqo_qm.")}


def test_declarations_match_the_self_acquiring_set_exactly():
    shells = _qm_shells()
    assert set(SELF_ACQUIRING) <= set(shells), "census names a non-QM shell"
    declared = {name for name, cls in shells.items()
                if getattr(cls, SELF_ACQUIRING_ATTR, None)}
    assert declared == SELF_ACQUIRING  # both directions: no stray, no missing


def test_every_reason_is_a_nonempty_string():
    for name in sorted(SELF_ACQUIRING):
        reason = getattr(get(name), SELF_ACQUIRING_ATTR)
        assert isinstance(reason, str) and reason.strip(), name


def test_refusal_fires_before_probe_and_creates_nothing(backend, roster,
                                                        tmp_path):
    from scqo_qm.experiments.qubit_drag_alternating import (
        QMQubitDragAlternating,
    )

    exp = make_experiment(QMQubitDragAlternating, backend, roster,
                          QMQubitDragAlternating.Parameters(targets=["q1"]))
    out_dir = tmp_path / "prev"
    with pytest.raises(ValueError,
                       match="qubit_drag_alternating cannot be previewed"):
        backend.preview(exp, out_dir)
    assert not out_dir.exists()


# ---------------------------------------------------------------- single-target
# preview gate: a self-acquiring shell that exposes a preview_program() hook may
# be previewed with EXACTLY ONE target (the single program it would build), and
# is refused otherwise. The build itself needs the live machine (see
# test_qm_backend.py); the gate logic is exercised here QUAM-free.

from types import SimpleNamespace  # noqa: E402


def _fake_experiment(targets, *, self_acquiring: bool, hook: bool):
    attrs: dict = {}
    if self_acquiring:
        attrs[SELF_ACQUIRING_ATTR] = "it acquires in a Python loop"
    if hook:
        attrs["preview_program"] = lambda self: "PROG"
    cls = type("Fake", (), attrs)
    obj = cls()
    obj.params = SimpleNamespace(targets=list(targets))
    return obj


def test_preview_refusal_gate():
    from scqo_qm.backend.qm_backend import _preview_refusal

    # a normal build-and-return shell is never refused
    assert _preview_refusal(
        _fake_experiment(["p1"], self_acquiring=False, hook=False)) is None
    # self-acquiring + hook + exactly one target -> previewable (the new case)
    assert _preview_refusal(
        _fake_experiment(["p1"], self_acquiring=True, hook=True)) is None
    # self-acquiring + hook + more than one target -> refused, single-target msg
    multi = _preview_refusal(
        _fake_experiment(["p1", "p2"], self_acquiring=True, hook=True))
    assert multi is not None and "exactly one --target" in multi
    # self-acquiring, no hook -> refused with the original reason
    nohook = _preview_refusal(
        _fake_experiment(["p1"], self_acquiring=True, hook=False))
    assert nohook is not None and "inside a real run" in nohook


def test_hook_without_self_acquiring_is_legal_at_any_target_count():
    """A NORMAL shell may expose ``preview_program()`` too, and is never gated by
    the single-target rule — that rule constrains SELF-ACQUIRING shells, whose
    ``probe()`` is not a legal fallback.

    ``qubit_tomography`` is the live case: it builds and returns like any normal
    shell (so it is correctly absent from :data:`SELF_ACQUIRING`) and exposes the
    hook only to render a cheaper program — training shots omitted. Pinned so the
    census is not misread as 'only a self-acquiring shell may carry a hook'.
    """
    from scqo_qm.backend.qm_backend import _preview_refusal

    cls = get("qubit_tomography")
    assert cls.__module__.startswith("scqo_qm.")
    assert "qubit_tomography" not in SELF_ACQUIRING
    assert not getattr(cls, SELF_ACQUIRING_ATTR, None)
    assert hasattr(cls, "preview_program")

    for targets in (["p1"], ["p1", "p2", "p3"]):
        assert _preview_refusal(
            _fake_experiment(targets, self_acquiring=False, hook=True)) is None


def test_multi_target_self_acquiring_preview_refuses(backend, roster, tmp_path):
    """qc_n_stark_amp HAS a preview_program hook, so with >1 target it is refused
    by the single-target gate (before any program is built — QUAM-free)."""
    from scqo_qm.experiments.qc_n_stark_amp import QMQcNStarkAmp

    exp = make_experiment(QMQcNStarkAmp, backend, roster,
                          QMQcNStarkAmp.Parameters(targets=["q1_q2", "q2_q3"]))
    out_dir = tmp_path / "prev"
    with pytest.raises(ValueError, match="select exactly one --target"):
        backend.preview(exp, out_dir)
    assert not out_dir.exists()
