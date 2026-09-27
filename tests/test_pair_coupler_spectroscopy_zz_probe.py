"""``pair_coupler_spectroscopy_zz``: the moved-LO config, the tone and the pi, the two arms.

Built on the live ``quam_state``: the config handed to the backend has the TONE
member's port at the window centre (band 2 for a ~6.8 GHz coupler) while the QUAM
tree is exactly as it was, and the pi member - that port's partner on q1_q2 - follows
the band with its LO unchanged. The generated QUA: per tone point the IF update on
the tone member, the stretched ``saturation`` in both arms, then the pi member's
``x180`` in one arm and a wait of the same length in the other, both members read
out. The pi goes to the member the tone does NOT ride on.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scqo_qm.experiments.pair_coupler_spectroscopy_zz import PI_OPERATION, pi_length_ns

STATE = str(Path(__file__).resolve().parents[1] / "quam_state")


def test_a_pi_member_without_an_x180_is_refused_by_name():
    bare = SimpleNamespace(name="qX", xy=SimpleNamespace(operations={}))
    with pytest.raises(ValueError, match="qX.*'x180'"):
        pi_length_ns(bare)
    ok = SimpleNamespace(name="qY", xy=SimpleNamespace(
        operations={PI_OPERATION: SimpleNamespace(length=16)}))
    assert pi_length_ns(ok) == 16


# ------------------------------------------------------------------ built on the live tree

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")

from conftest import recording_device  # noqa: E402
from test_qm_backend import roster_toml_for  # noqa: E402


@pytest.fixture(scope="module")
def machine():
    return quam_config.Quam.load(STATE)


@pytest.fixture(scope="module")
def live_roster(machine):
    from scqo.roster import parse_components

    return parse_components(roster_toml_for(machine))


def _experiment(machine, live_roster, **kw):
    from scqo_qm.backend.qm_backend import QMBackend
    from scqo_qm.experiments.pair_coupler_spectroscopy_zz import QMPairCouplerSpectroscopyZZ

    if "q1_q2" not in machine.qubit_pairs:
        pytest.skip("live state has no pair q1_q2")
    backend = QMBackend(machine, roster=live_roster)
    params = QMPairCouplerSpectroscopyZZ.Parameters(
        targets=["q1_q2"], **{"num_tone_freq_points": 11, "num_averages": 10, **kw})
    exp = QMPairCouplerSpectroscopyZZ(backend, params)
    exp.device = recording_device(backend, live_roster)
    exp.sweep_axes = exp.define_sweep()
    return exp


def _qubits(machine, exp):
    """(tone qubit, pi qubit) as QUAM objects."""
    return (machine.qubits[exp.tone_member("q1_q2")], machine.qubits[exp.pi_member("q1_q2")])


def _tree(machine):
    out = {}
    for name, q in machine.qubits.items():
        port = q.xy.opx_output
        out[name] = (getattr(port, "band", None), getattr(port, "upconverter_frequency", None),
                     q.xy.RF_frequency)
    return out


def _port(config, channel):
    port = channel.opx_output
    return config["controllers"][port.controller_id]["fems"][port.fem_id][
        "analog_outputs"][port.port_id]


def test_the_run_config_moves_the_tone_port_and_the_tree_does_not(machine, live_roster):
    before = _tree(machine)
    exp = _experiment(machine, live_roster)
    prog, axes, acquire = exp.probe()
    assert _tree(machine) == before, "the QUAM tree was left moved"
    config = exp._config
    tone_q, pi_q = _qubits(machine, exp)
    qp = machine.qubit_pairs["q1_q2"]
    assert {tone_q.name, pi_q.name} == {qp.qubit_control.name, qp.qubit_target.name}
    tone_port = _port(config, tone_q.xy)
    assert tone_port["band"] == 2
    assert tone_port["upconverter_frequency"] == pytest.approx(6.80e9)
    # q1 and q2 share a port pair: the pi member follows the band, keeps its LO
    pi_port = _port(config, pi_q.xy)
    assert pi_port["band"] == 2
    assert pi_port["upconverter_frequency"] == pytest.approx(before[pi_q.name][1])
    assert exp._moved["partner"] == pi_q.name and exp._moved["partner_parked"] == 0
    np.testing.assert_allclose(axes["tone_freq_hz"].values,
                               np.linspace(6.55e9, 7.05e9, 11), atol=1.0)
    assert list(axes["pi_played"].values) == [1, 0]
    assert exp.patch_preview_config({}) is config
    assert acquire.keywords["config"] is config


def test_the_program_plays_the_tone_then_the_pi_in_one_arm(machine, live_roster):
    from qm import generate_qua_script

    exp = _experiment(machine, live_roster)
    prog, _axes, _acq = exp.probe()
    script = generate_qua_script(prog, exp._config).replace(chr(34), chr(39))
    tone_q, pi_q = _qubits(machine, exp)
    tone_xy, pi_xy = tone_q.xy.name, pi_q.xy.name
    pi_cycles = pi_length_ns(pi_q) // 4
    assert re.search(rf"update_frequency\('{re.escape(tone_xy)}', v\d+", script)
    assert f"update_frequency('{pi_xy}'" not in script
    # the tone (10 us = 2500 cycles) in BOTH arms, the pi in one, a wait in the other
    assert script.count(f"play('saturation', '{tone_xy}', duration=2500)") == 2
    assert script.count(f"play('{PI_OPERATION}', '{pi_xy}')") == 1
    assert f"play('{PI_OPERATION}', '{tone_xy}')" not in script
    assert f"wait({pi_cycles}, '{pi_xy}')" in script
    for q in (tone_q, pi_q):
        assert script.count(f"'{q.resonator.name}'") >= 2   # read out in both arms


def test_tone_on_picks_the_line_and_the_pi_goes_to_the_other(machine, live_roster):
    low = _experiment(machine, live_roster, tone_on="low")
    high = _experiment(machine, live_roster, tone_on="high")
    assert _qubits(machine, low) == _qubits(machine, high)[::-1]
    high.probe()
    tone_q, _pi_q = _qubits(machine, high)
    assert _port(high._config, tone_q.xy)["upconverter_frequency"] == pytest.approx(6.80e9)
