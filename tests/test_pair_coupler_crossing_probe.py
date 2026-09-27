"""``pair_coupler_crossing_pulse``: the coupler window, the x180s and the readout.

Three halves:

* pure arithmetic - the coupler pulse covers both buffers and the longest x180,
  rounded UP to the clock;
* the class's params -> builder mapping on the stub tree, whose pair is
  control=q1 / target=q2 while the roster says high=q2 / low=q1 (so a positional
  guess about ``measure`` would invert it), plus the pre-probe refusals;
* the program BUILT on the live ``quam_state`` and read back as QUA script: the
  coupler plays ``const`` scaled by a plain loop variable for the whole window,
  the measured members wait one buffer and play ``x180`` inside it, and both
  members are read out after an ``align``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from conftest import make_experiment

from scqo_qm.experiments.pair_coupler_crossing_pulse import coupler_window_cycles

STATE = str(Path(__file__).resolve().parents[1] / "quam_state")


# ------------------------------------------------------------------ arithmetic

def test_the_window_covers_both_buffers_and_the_longest_x180():
    assert coupler_window_cycles([16, 40], 100) == 60          # (200 + 40) / 4
    assert coupler_window_cycles([42], 100) == 61              # 242 ns -> rounded UP
    assert coupler_window_cycles([16], 0) == 4                 # never below 4 cycles
    assert coupler_window_cycles([8], 0) == 4


# ------------------------------------------------------------------ the class, on stubs

def _experiment(backend, roster, **kw):
    from scqo_qm.experiments.pair_coupler_crossing_pulse import QMPairCouplerCrossingPulse

    exp = make_experiment(QMPairCouplerCrossingPulse, backend, roster,
                          QMPairCouplerCrossingPulse.Parameters(targets=["q1_q2"], **kw))
    exp.sweep_axes = exp.define_sweep()
    return exp


@pytest.fixture
def captured(monkeypatch):
    from scqo_qm.experiments import pair_coupler_crossing_pulse as module

    seen = {}

    def fake_build(machine, qubit_pair, **kw):
        seen.update(kw, qubit_pair=qubit_pair)
        return "prog", {"coupler_amplitude": kw["coupler_amps_v"]}

    monkeypatch.setattr(module, "build_program", fake_build)
    return seen


@pytest.mark.parametrize("measure,roles", [
    ("both", ["target", "control"]), ("high", ["target"]), ("low", ["control"])])
def test_measure_roles_map_onto_the_vendor_sides(backend, roster, captured, measure, roles):
    exp = _experiment(backend, roster, measure=measure)
    prog, axes = exp.probe()
    assert captured["measure_roles"] == roles
    assert captured["qubit_pair"] is backend.machine.qubit_pairs["coupler_q1_q2"]
    assert exp._high_side == "target"                  # roster high = q2 = vendor target
    # the scqo axis carries the swept volts in the order asked, keyed by the roster name
    np.testing.assert_allclose(axes["coupler_flux_v"], exp.sweep_axes["coupler_flux_v"])
    assert list(axes["qubit_pair"].values) == ["q1_q2"]


def test_the_buffer_and_averages_reach_the_builder(backend, roster, captured):
    _experiment(backend, roster, flux_buffer_ns=48, num_averages=77).probe()
    assert captured["buffer_ns"] == 48 and captured["num_shots"] == 77
    assert captured["reset_type"] == "thermal"


def test_a_member_without_a_threshold_is_refused_by_name(stub_machine, roster, captured):
    from scqo_qm.backend.qm_backend import QMBackend

    stub_machine.qubits["q1"].resonator.operations["readout"].threshold = None
    exp = _experiment(QMBackend(stub_machine, roster=roster), roster)
    with pytest.raises(ValueError, match=r"\['q1'\] have no readout_threshold"):
        exp.probe()
    assert captured == {}                              # refused before any build


# ------------------------------------------------------------------ built on the live tree

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")


@pytest.fixture(scope="module")
def machine():
    return quam_config.Quam.load(STATE)


@pytest.fixture(scope="module")
def pair(machine):
    if "q1_q2" not in machine.qubit_pairs:
        pytest.skip("live state has no pair q1_q2")
    return machine.qubit_pairs["q1_q2"]


def _script(machine, prog) -> str:
    from qm import generate_qua_script

    return generate_qua_script(prog, machine.generate_config()).replace(chr(34), chr(39))


def _build(machine, pair, amps, roles=("control", "target"), buffer_ns=100):
    from scqo_qm.experiments.pair_coupler_crossing_pulse import build_program

    return build_program(machine, pair, coupler_amps_v=amps, measure_roles=list(roles),
                         buffer_ns=buffer_ns, num_shots=10, reset_type="thermal")


def test_the_live_build_plays_the_sequence(machine, pair):
    amps = np.linspace(-0.2, 0.1, 7)
    prog, axes = _build(machine, pair, amps)
    script = _script(machine, prog)
    ref = float(pair.coupler.operations["const"].amplitude)
    lengths = [pair.qubit_control.xy.operations["x180"].length,
               pair.qubit_target.xy.operations["x180"].length]
    window = coupler_window_cycles(lengths, 100)

    coupler_play = [ln for ln in script.splitlines()
                    if "play('const'" in ln and f"'{pair.coupler.name}'" in ln]
    assert len(coupler_play) == 1, coupler_play
    # a plain loop variable as the scale, and the window as the duration
    assert re.search(r"amp\(v\d+\)", coupler_play[0]), coupler_play[0]
    assert f"duration={window}" in coupler_play[0].replace(" ", ""), coupler_play[0]
    for qubit in (pair.qubit_control, pair.qubit_target):
        assert f"play('x180', '{qubit.xy.name}')" in script
        assert re.search(rf"wait\(25, '{re.escape(qubit.xy.name)}'\)", script)
        assert f"'{qubit.resonator.name}'" in script
    # the loop walks the SCALES, the data axis keeps the volts
    np.testing.assert_allclose(axes["coupler_amplitude"].values, amps)
    assert script.index(f"'{pair.coupler.name}'") < script.index("measure(")
    del ref


def test_one_measured_member_plays_one_x180(machine, pair):
    prog, _ = _build(machine, pair, np.linspace(-0.1, 0.1, 5), roles=("target",))
    script = _script(machine, prog)
    assert f"play('x180', '{pair.qubit_target.xy.name}')" in script
    assert f"play('x180', '{pair.qubit_control.xy.name}')" not in script
    # both are still read out
    for qubit in (pair.qubit_control, pair.qubit_target):
        assert f"'{qubit.resonator.name}'" in script


def test_a_window_past_the_rail_is_refused(machine, pair):
    with pytest.raises(ValueError, match="full scale|amplitude_scale"):
        _build(machine, pair, np.linspace(-0.6, 0.1, 5))


def test_zero_buffer_plays_without_a_wait(machine, pair):
    prog, _ = _build(machine, pair, np.linspace(-0.1, 0.1, 5), buffer_ns=0)
    script = _script(machine, prog)
    assert not re.search(rf"wait\(\d+, '{re.escape(pair.qubit_control.xy.name)}'\)", script)
