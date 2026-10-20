"""``qubit_power_rabi_ef``: the 1->2 Rabi from |1>, asserted on generated QUA.

The x180 plays first at the 0->1 frequency, the element is retuned to the 1->2
IF (``f_12 - LO``) for the EF_x180 pulse scaled by the swept factor, and the
0->1 IF is restored before the barrier and the readout. An unmeasured or
unplayable 1->2 frequency is refused by name before anything is built.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")

from conftest import recording_device  # noqa: E402
from test_qm_backend import roster_toml_for  # noqa: E402

from scqo_qm.backend.qm_backend import QMBackend  # noqa: E402
from scqo.roster import parse_components  # noqa: E402

TARGET = "q4"


@pytest.fixture()
def machine():
    # function scope: the tests write q.f_12 on the tree
    return quam_config.Quam.load(str(Path(__file__).resolve().parents[1] / "quam_state"))


def _experiment(machine, **params):
    from scqo.experiments import get

    import scqo_qm.experiments  # noqa: F401  (registers the QM probes)

    roster = parse_components(roster_toml_for(machine))
    backend = QMBackend(machine, roster=roster)
    cls = get("qubit_power_rabi_ef")
    exp = cls(backend, cls.Parameters(targets=[TARGET], num_amp_points=5,
                                      num_averages=10, **params))
    exp.device = recording_device(backend, roster)  # pulls q.f_12 as it is now
    exp.sweep_axes = exp.define_sweep()
    return exp


def _body(prog, machine) -> list[str]:
    from qm import generate_qua_script

    lines = [ln.strip().replace(chr(34), chr(39))
             for ln in generate_qua_script(prog, machine.generate_config()).splitlines()]
    start = next(i for i, ln in enumerate(lines) if ln.startswith("play('x180'"))
    stop = next(i for i, ln in enumerate(lines[start:]) if ln.startswith("save(")) + start
    return lines[start:stop]


def _index(body, prefix) -> int:
    return next(i for i, ln in enumerate(body) if ln.startswith(prefix))


def test_x180_then_the_retuned_ef_pulse_then_the_0_1_if_then_the_readout(machine):
    q = machine.qubits[TARGET]
    q.f_12 = float(q.f_01) - 210e6
    if_ef = round(float(q.f_12) - float(q.xy.LO_frequency))
    exp = _experiment(machine)
    body = _body(exp.probe()[0], machine)

    x180 = _index(body, "play('x180'")
    retunes = [i for i, ln in enumerate(body) if ln.startswith("update_frequency(")]
    ef = _index(body, "play('EF_x180'")
    barrier = next(i for i in range(ef, len(body)) if body[i].startswith("align("))
    measure = _index(body, "measure('readout'")
    assert x180 < retunes[0] < ef < retunes[1] < barrier < measure
    first = float(re.search(r",\s*(-?[\d.e+-]+),\s*'Hz'", body[retunes[0]]).group(1))
    restored = float(re.search(r",\s*(-?[\d.e+-]+),\s*'Hz'", body[retunes[1]]).group(1))
    assert first == pytest.approx(if_ef)
    assert restored == pytest.approx(float(q.xy.intermediate_frequency))
    assert "amp(" in body[ef], f"the EF pulse must carry the swept factor: {body[ef]}"


def test_an_unmeasured_1_2_frequency_is_refused_by_name(machine):
    machine.qubits[TARGET].f_12 = None
    exp = _experiment(machine)
    with pytest.raises(KeyError, match="drive_freq_ef_hz"):
        exp.probe()


def test_a_1_2_frequency_past_the_upconverter_band_is_refused(machine):
    q = machine.qubits[TARGET]
    q.f_12 = float(q.xy.LO_frequency) - 450e6  # a hand-edited tree; the setter refuses it
    exp = _experiment(machine)
    with pytest.raises(ValueError, match="400 MHz"):
        exp.probe()
