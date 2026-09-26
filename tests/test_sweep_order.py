"""A descending window reaches the QOP as a descending sweep.

scqo's flux, detuning and amplitude windows are a TRAVERSAL ORDER (decided
2026-09-26): ``start`` -> ``end`` in either direction, never re-sorted. The QM probes
hand the axis to QUA verbatim — ``for_(*from_array(v, axis))``, which branches on
the step sign (``v >= stop``, ``v + -step``), or ``for_each_(v, axis)``, which
plays a declared array in order — so the order survives unless something on
the way sorts it. Checked on every QM carrier of those windows (derived from the
Parameters mixins, so a new carrier is covered automatically), in the generated
QUA itself: some sweep counts DOWN — a ``for_`` whose increment is negative, or a
``for_each_`` over a declared array that decreases. An ascending build of the same
probe must not (the contrast, so the pattern cannot pass by accident).

scqo proves the stored axis keeps the order and that no estimator can tell; this
file proves the program really walks it. Live-QUAM, like test_sequential_probe:
what from_array renders is a property of the real tree and config.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")

from conftest import recording_device  # noqa: E402
from test_qm_backend import roster_toml_for  # noqa: E402

import scqo_qm.experiments  # noqa: E402,F401  (registers the QM probes)
from scqo.experiments import catalog, get  # noqa: E402
from scqo.experiments._capabilities import (  # noqa: E402
    AMP_AXIS,
    DETUNING_AXIS,
    FLUX_AXIS,
    AmplitudeSweepParameters,
    DriveDetuningSweepParameters,
    FluxSweepParameters,
    ReadoutDetuningSweepParameters,
)
from scqo.roster import parse_components  # noqa: E402
from scqo_qm.backend.qm_backend import QMBackend  # noqa: E402

TARGET = "q4"

#: window mixin -> (start field, end field, the axis it sweeps)
WINDOWS = {
    FluxSweepParameters: ("start_flux_v", "end_flux_v", FLUX_AXIS),
    DriveDetuningSweepParameters: (
        "start_drive_detuning_hz", "end_drive_detuning_hz", DETUNING_AXIS),
    ReadoutDetuningSweepParameters: (
        "start_readout_detuning_hz", "end_readout_detuning_hz", DETUNING_AXIS),
    AmplitudeSweepParameters: ("start_amp_factor", "end_amp_factor", AMP_AXIS),
}

#: keep every program small; each carrier takes only the fields it declares
SMALL = {"num_averages": 10, "num_shots": 100, "num_amp_points": 5,
         "num_flux_points": 5, "num_drive_freq_points": 5,
         "num_readout_freq_points": 5, "num_power_points": 3,
         "num_wait_points": 5, "max_wait_ns": 400, "num_idle_points": 5,
         "max_repetitions": 4}

QM_PROBES = sorted(e["name"] for e in catalog()
                   if get(e["name"]).__module__.startswith("scqo_qm."))

CASES = [
    pytest.param(name, *fields, id=f"{name}-{fields[0]}")
    for name in QM_PROBES
    for mixin, fields in WINDOWS.items()
    if issubclass(get(name).Parameters, mixin)
]


@pytest.fixture(scope="module")
def machine():
    return quam_config.Quam.load(str(Path(__file__).resolve().parents[1] / "quam_state"))


@pytest.fixture(scope="module")
def live_roster(machine):
    return parse_components(roster_toml_for(machine))


@pytest.fixture(scope="module")
def config(machine):
    return machine.generate_config()


def test_every_window_has_a_qm_carrier():
    """Guard against the derivation silently finding nothing."""
    assert {c.values[1] for c in CASES} == {f[0] for f in WINDOWS.values()}


def _params(name, start, end, *, descending):
    cls = get(name)
    fields = set(cls.Parameters.model_fields)
    base = cls.Parameters(targets=[TARGET],
                          **{k: v for k, v in SMALL.items() if k in fields})
    low, high = sorted((getattr(base, start), getattr(base, end)))
    first, last = (high, low) if descending else (low, high)
    return base.model_copy(update={start: first, end: last})


def _script(machine, live_roster, config, name, params):
    from qm import generate_qua_script

    backend = QMBackend(machine, roster=live_roster)
    exp = get(name)(backend, params)
    exp.device = recording_device(backend, live_roster)
    exp.sweep_axes = exp.define_sweep()
    built = exp.probe()
    prog, axes = built[0], built[1]
    try:
        return generate_qua_script(prog, config), axes, exp
    finally:
        drop = getattr(exp, "_drop_drive_op", None)  # the cryoscope's run-scoped op
        if drop is not None:
            drop()


_NUM = r"-?[\d.]+(?:e[-+]?\d+)?"


def _counts_down(script: str) -> bool:
    """A ``for_`` with a NEGATIVE increment, or a ``for_each_`` over a declared
    array that decreases — the two ways QUA walks an axis high -> low."""
    if re.search(rf"for_\((v\d+),{_NUM},\(\1>=?{_NUM}\),\(\1\+-{_NUM}\)\)", script):
        return True
    for values in re.findall(r"declare\(fixed,\s*value=\[([^\]]*)\]\)", script):
        array = np.array([float(v) for v in values.split(",") if v.strip()])
        if array.size > 1 and np.all(np.diff(array) < 0):
            return True
    return False


@pytest.mark.parametrize("name,start,end,axis", CASES)
def test_a_descending_window_is_swept_descending(machine, live_roster, config,
                                                 name, start, end, axis):
    down = _params(name, start, end, descending=True)
    script, axes, _exp = _script(machine, live_roster, config, name, down)
    # a few probes label the detuning axis by its pre-canonical name; the
    # backend's _to_canonical renames it, by position, after acquisition
    key = axis if axis in axes else axis.removesuffix("_hz")
    labelled = np.asarray(axes[key].values, dtype=float)
    assert labelled[0] == pytest.approx(getattr(down, start))
    assert labelled[0] > labelled[-1], f"{name}: the axis was re-sorted"
    assert _counts_down(script), f"{name}: no sweep counts down in the QUA"

    up = _params(name, start, end, descending=False)
    script_up, _, _ = _script(machine, live_roster, config, name, up)
    assert not _counts_down(script_up), (
        f"{name}: an ASCENDING build already counts down somewhere - the check "
        f"cannot tell the two apart for this probe")
