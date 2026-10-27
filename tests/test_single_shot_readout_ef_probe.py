"""``single_shot_readout_ef``: g, e and f interleaved, asserted on generated QUA.

Prepared state 2 plays the x180 at the 0->1 frequency, retunes the element to
the 1->2 IF (``f_12 - LO``) for the unscaled EF_x180 pulse and restores the
0->1 IF before the readout; states 0 and 1 are the two-state readout's. Every
shot is saved, three per shot index. An unmeasured 1->2 frequency is refused by
name before anything is built.
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
    cls = get("single_shot_readout_ef")
    exp = cls(backend, cls.Parameters(targets=[TARGET], num_shots=100, **params))
    exp.device = recording_device(backend, roster)  # pulls q.f_12 as it is now
    exp.sweep_axes = exp.define_sweep()
    return exp


def _script(prog, machine) -> list[str]:
    from qm import generate_qua_script

    return [ln.strip().replace(chr(34), chr(39))
            for ln in generate_qua_script(prog, machine.generate_config()).splitlines()]


def _case(lines, k) -> list[str]:
    """The body of ``case_(k)`` in the prepared-state switch (the generated
    script spells a switch as an if_/elif_ chain)."""
    start = next(i for i, ln in enumerate(lines)
                 if re.match(rf"with (el)?if_\(\(\w+=={k}\)\)", ln))
    stop = next((i for i in range(start + 1, len(lines))
                 if lines[i].startswith(("with elif_(", "with else_(", "align("))),
                len(lines))
    return lines[start + 1:stop]


def test_state_2_is_x180_then_the_retuned_ef_pulse_then_the_0_1_if(machine):
    from scqo_qm.experiments.single_shot_readout_ef import QMSingleShotReadoutEf

    q = machine.qubits[TARGET]
    q.f_12 = float(q.f_01) - 210e6
    if_ef = round(float(q.f_12) - float(q.xy.LO_frequency))
    exp = _experiment(machine)
    assert type(exp) is QMSingleShotReadoutEf
    prog, axes = exp.probe()
    lines = _script(prog, machine)

    body = _case(lines, 2)
    plays = [ln for ln in body if ln.startswith("play(")]
    retunes = [ln for ln in body if ln.startswith("update_frequency(")]
    assert [p.split(",")[0] for p in plays] == ["play('x180'", "play('EF_x180'"]
    assert "amp(" not in plays[1], f"the 1->2 pulse plays at its stored amplitude: {plays[1]}"
    order = [i for i, ln in enumerate(body) if ln.startswith(("play(", "update_frequency("))]
    assert [body[i].split("(")[0] for i in order] == [
        "play", "update_frequency", "play", "update_frequency"]
    first = float(re.search(r",\s*(-?[\d.e+-]+),\s*'Hz'", retunes[0]).group(1))
    restored = float(re.search(r",\s*(-?[\d.e+-]+),\s*'Hz'", retunes[1]).group(1))
    assert first == pytest.approx(if_ef)
    assert restored == pytest.approx(float(q.xy.intermediate_frequency))

    assert [ln.split(",")[0] for ln in _case(lines, 1) if ln.startswith("play(")] == ["play('x180'"]
    assert list(axes["prepared_state"].values) == [0, 1, 2]
    assert any("buffer(3)" in ln and "buffer(100)" in ln for ln in lines)
    assert not any("average()" in ln for ln in lines)


def test_an_unmeasured_1_2_frequency_is_refused_by_name(machine):
    machine.qubits[TARGET].f_12 = None
    exp = _experiment(machine)
    with pytest.raises(KeyError, match="drive_freq_ef_hz"):
        exp.probe()


def test_the_two_state_program_carries_no_state_2(machine):
    """The shared builder adds case 2 only when asked: single_shot_readout's
    program is unchanged."""
    from scqo_qm.experiments import _readout_fidelity as fidelity_probe
    from scqo_qm.experiments._lib import select_qubits

    qubits = select_qubits(machine, [TARGET], multiplexed=True)
    prog, _ = fidelity_probe.build_program(machine, qubits, operation="readout",
                                           num_shots=100, reset_type="thermal")
    lines = _script(prog, machine)
    assert not any(re.match(r"with (el)?if_\(\(\w+==2\)\)", ln) for ln in lines)
    assert not any(ln.startswith("update_frequency(") for ln in lines)
    with pytest.raises(ValueError, match="ef_ifs"):
        fidelity_probe.build_program(machine, qubits, operation="readout", num_shots=100,
                                     reset_type="thermal", prepared_states=(0, 1, 2))
