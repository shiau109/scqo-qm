"""The QM active-reset door: one helper, opt-in carriers, refusals, backstop.

Active reset landed on the Qblox backend first; scqo widened the neutral
``reset_method`` Literal repo-wide, so the flag validates on every backend. The
QM policy lives in one place, ``scqo_qm/experiments/_reset.py``:
:func:`check_reset_method` refuses everything QM cannot honour BY NAME rather
than thermalizing quietly, and :func:`reset_max_attempts` maps scqo's
``active_reset_rounds`` onto QUAM's upper-bound loop count.

Most tests are QUAM-free: the policy is a pure function of Parameters, the class
and the governed device views. Two builders:

* :func:`_shell` — ``cls.__new__`` with ``.params`` set, no backend. Proves the
  checks that must fire BEFORE any device read (thermal pass-through, the opt-in
  and override refusals) never touch ``.device``.
* stub-backed experiments via conftest's ``make_experiment`` over the hand-built
  QUAM stand-in, so the discriminator/depletion refusals — which DO read the
  device — run on every checkout regardless of the ``my_quam.py`` toggle. An
  uncalibrated value is made by mutating the stub tree BEFORE the recording
  device seeds from it (a QUAM ``None`` threshold, a resonator with no
  ``depletion_time``), exactly the shapes a cold chip carries.

The live-instrument half — that the QUA program actually BUILDS with the kwarg
threaded and the thresholds consumed — is
``test_qm_backend.py::test_active_reset_program_builds_on_live_state``; it needs
the ``machine`` fixture, so it does not live here.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from scqo_qm.experiments._reset import (
    ACTIVE_RESET_ATTR,
    check_reset_method,
    reset_max_attempts,
)

REPO = Path(__file__).resolve().parents[1]
SHELLS = REPO / "scqo_qm" / "experiments"
CARRIERS = {"qubit_relaxation", "qubit_ramsey", "qubit_ramsey_phasor",
            "qubit_echo", "qubit_power_rabi",
            "qubit_t1_ade", "qubit_t1_bayesian",
            "qubit_spectroscopy"}


def _shell(name, **params):
    """A registered shell with no backend: ``check_reset_method`` reads only
    ``.params`` and the class until it reaches the device, so skipping __init__
    keeps the pure-policy tests off the vendor stack entirely."""
    import scqo_qm.experiments  # noqa: F401  (registers the QM shells)
    from scqo.experiments import get

    cls = get(name)
    exp = cls.__new__(cls)
    exp.params = cls.Parameters(targets=["q1"], **params)
    return exp


def _registered_names():
    import scqo_qm.experiments  # noqa: F401
    from scqo.experiments import catalog

    return sorted(e["name"] for e in catalog())


def _qm_reset_shells():
    """Registered names whose class is a QM SHELL (source under this repo) AND
    whose Parameters carry ``reset_method`` — the set that must resolve through
    the helper. A base scqo class with no QM probe (``qc_n_swap_amp`` today) is
    covered by the ``acquire()`` backstop, not the shell, so it is excluded."""
    from scqo.experiments import get

    out = []
    for name in _registered_names():
        cls = get(name)
        fields = getattr(getattr(cls, "Parameters", None), "model_fields", {})
        if "reset_method" not in fields:
            continue
        src = Path(inspect.getsourcefile(cls)).resolve()
        if SHELLS in src.parents:
            out.append(name)
    return out


def _module_source(name):
    from scqo.experiments import get

    return Path(inspect.getsourcefile(get(name))).read_text(encoding="utf-8")


# ------------------------------------------------------------------ pure policy

def test_thermal_passes_through_unchanged():
    """The neutral vocabulary matches QUAM's own ``reset_type``, so the supported
    case is a pass-through. The device-less shell proves it never reads state."""
    exp = _shell("qubit_relaxation")
    assert not hasattr(exp, "device")
    assert check_reset_method(exp) == "thermal"
    assert check_reset_method(
        _shell("qubit_relaxation", reset_method="thermal")) == "thermal"


def test_only_the_named_shells_opt_in():
    """The census: the coherent-drive carriers plus qubit_spectroscopy, whose
    saturation drive is now a finite pulse — so its reset is a real state reset,
    and the only thing it sweeps is the DRIVE frequency. Widening this set is a
    hardware-gated decision (a new sequence must be validated on the instrument),
    so it is pinned here and an addition must edit this line."""
    from scqo.experiments import get

    opted = {n for n in _registered_names()
             if getattr(get(n), ACTIVE_RESET_ATTR, False)}
    assert opted == CARRIERS


def test_every_qm_reset_shell_resolves_through_the_helper():
    """The invariant that would have caught the original leak (shells carrying
    ``reset_method`` but hardcoding thermal in their probe): every QM shell with
    the field calls ``check_reset_method``, discovered dynamically so a new shell
    cannot slip past by never being listed here."""
    shells = _qm_reset_shells()
    assert CARRIERS <= set(shells)  # sanity: the carriers are in the discovered set
    missing = [n for n in shells if "check_reset_method(" not in _module_source(n)]
    assert not missing, (
        "QM shells carry reset_method but never call check_reset_method "
        f"(active would thermalize silently): {missing}")


def test_no_probe_hardcodes_a_reset_literal():
    """The probe-side half: no probe or fused shell passes a string literal to
    ``qubit.reset`` — every reset threads the shell-resolved ``reset_type``.
    Catches the next copy-paste of the old ``qubit.reset("thermal", ...)``."""
    pattern = re.compile(r"\.reset\(\s*['\"]")
    offenders = []
    for folder in (SHELLS,):
        for path in sorted(folder.glob("*.py")):
            for lineno, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line.split("#", 1)[0]):
                    offenders.append(f"{folder.name}/{path.name}:{lineno}: {line.strip()}")
    assert not offenders, "\n  ".join(["thread reset_type instead:"] + offenders)


@pytest.mark.parametrize("name", [
    "readout_frequency",        # sweeps the readout condition
    "readout_power",            # sweeps the readout amplitude
    "single_shot_readout",      # IS the discriminator calibration
    "single_shot_readout_gef",
    "qubit_sqrb",               # ex-leaker, now refused by name
])
def test_denied_shells_refuse_active_by_name(name):
    """A shell that does not opt in refuses active BEFORE any device read (the
    opt-in check precedes them), so a device-less shell suffices. The message
    names the experiment, points at thermal, and hints the carriers."""
    with pytest.raises(ValueError) as err:
        check_reset_method(_shell(name, reset_method="active"))
    msg = str(err.value)
    assert name in msg
    assert "thermal" in msg
    assert "qubit_ramsey" in msg  # the _CARRIERS hint


def test_thermal_population_refuses_active_at_validation():
    """qubit_thermal_population refuses active UPSTREAM of the driver, at pydantic
    validation, so the flag never reaches check_reset_method — the neutral layer
    owns that one because an active reset removes the population it counts."""
    from pydantic import ValidationError
    from scqo.experiments import get

    cls = get("qubit_thermal_population")
    cls.Parameters(targets=["q1"])  # thermal default is fine
    with pytest.raises(ValidationError):
        cls.Parameters(targets=["q1"], reset_method="active")


def test_active_refuses_the_thermalization_override():
    """thermalization_time_ns + active is refused, not ignored — and BEFORE any
    device read, so the override context manager never sees the combination."""
    exp = _shell("qubit_ramsey", reset_method="active",
                 thermalization_time_ns=5000.0)
    assert not hasattr(exp, "device")
    with pytest.raises(ValueError, match="thermalization_time_ns"):
        check_reset_method(exp)


def test_rounds_map_to_max_attempts_as_an_upper_bound():
    """active_reset_rounds -> QUAM max_attempts. QM reads it as an upper bound
    (the loop exits early on success) where Qblox plays exactly N; the mapping
    lives in one place."""
    assert reset_max_attempts(_shell("qubit_ramsey", active_reset_rounds=3)) == 3
    assert reset_max_attempts(_shell("qubit_ramsey")) == 1  # neutral default


def test_carrier_shells_thread_the_rounds():
    """Each carrier shell forwards reset_max_attempts into its probe (the source
    half of the live-build proof in test_qm_backend.py)."""
    for name in CARRIERS:
        assert "reset_max_attempts=reset_max_attempts(self)" in _module_source(name)


# ---------------------------------------------------------- device-backed policy

def _calibrate(stub_machine, *, target="q1", threshold=-3.0e-4, rus=-2.0e-4,
               depletion_ns=800):
    """Set the readout discriminator + depletion on the stub tree so a recording
    device seeded AFTER this reads calibrated values. ``None`` for a threshold is
    QUAM's uncalibrated shape; ``depletion_ns=None`` leaves the attribute absent
    (the resonator a cold chip carries)."""
    res = stub_machine.qubits[target].resonator
    res.operations["readout"].threshold = threshold
    res.operations["readout"].rus_exit_threshold = rus
    if depletion_ns is not None:
        res.depletion_time = int(depletion_ns)


def _active_carrier(name, backend, roster, **params):
    from conftest import make_experiment
    from scqo.experiments import get

    cls = get(name)
    return make_experiment(cls, backend, roster,
                           cls.Parameters(targets=["q1"], reset_method="active",
                                          **params))


def test_active_returns_active_on_a_calibrated_stub(backend, stub_machine, roster):
    _calibrate(stub_machine)
    exp = _active_carrier("qubit_ramsey", backend, roster)
    assert check_reset_method(exp) == "active"


def test_active_refuses_an_uncalibrated_discriminator(backend, stub_machine, roster):
    """Both thresholds None (QUAM's uncalibrated shape). The recording view reads
    strict, so the None surfaces as KeyError, which the guard turns into a named
    refusal — where an unguarded run would die as ``I > None`` deep in the QUA
    DSL naming nothing. The guard does NOT ride on use_state_discrimination."""
    _calibrate(stub_machine, threshold=None, rus=None)
    exp = _active_carrier("qubit_ramsey", backend, roster,
                          use_state_discrimination=False)
    with pytest.raises(ValueError, match="single_shot_readout"):
        check_reset_method(exp)


def test_active_refuses_when_only_the_rus_threshold_is_missing(
        backend, stub_machine, roster):
    """rus_exit_threshold alone is uncalibrated: required even at rounds=1,
    because the repeat-until-success exit condition is BUILT regardless."""
    _calibrate(stub_machine, threshold=-3.0e-4, rus=None)
    exp = _active_carrier("qubit_ramsey", backend, roster, active_reset_rounds=1)
    with pytest.raises(ValueError, match="readout_rus_threshold"):
        check_reset_method(exp)


def test_active_refuses_a_depletion_that_was_never_seeded(
        backend, stub_machine, roster):
    """The stub resonator has no depletion_time at all -> depletion_wait_ns
    raises KeyError, which the guard treats as ungoverned and refuses (naming
    resonator_spectroscopy). Discriminator is calibrated, so step 6 is reached."""
    _calibrate(stub_machine, depletion_ns=None)  # leave the attribute absent
    assert not hasattr(stub_machine.qubits["q1"].resonator, "depletion_time")
    exp = _active_carrier("qubit_ramsey", backend, roster)
    with pytest.raises(ValueError, match="resonator_spectroscopy"):
        check_reset_method(exp)


def test_active_refuses_the_factory_default_depletion(
        backend, stub_machine, roster):
    """QUAM's depletion_time defaults to 16 ns and is never None, so a None-only
    guard would be dead code on QM: the honest refusal is the factory default,
    which a real depletion_factor/(2 pi x kappa) proposal never equals by luck."""
    _calibrate(stub_machine, depletion_ns=16)
    exp = _active_carrier("qubit_ramsey", backend, roster)
    with pytest.raises(ValueError, match="resonator_spectroscopy"):
        check_reset_method(exp)


def test_the_backend_refuses_before_it_probes(backend, stub_machine, roster):
    """The acquire() backstop fires check_reset_method before probe(), so a shell
    that forgot the helper still refuses. It works QM-free ONLY because the
    backstop precedes the _lib import — do not reorder."""
    _calibrate(stub_machine)  # calibrated, so only the opt-in refusal can fire
    exp = _active_carrier("single_shot_readout", backend, roster)

    def explode():
        raise AssertionError("probe() ran — the backstop fired too late")

    exp.probe = explode
    with pytest.raises(ValueError, match="single_shot_readout"):
        backend.acquire(exp)
