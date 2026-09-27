"""``pair_coupler_spectroscopy_swap``: the moved-LO config, the ramp, the two arms.

Three halves:

* pure - the ramp samples, and which MW-FEM band an LO lands in (keep the current
  one when it holds the LO, else the lowest one holding the LO AND the port-pair
  partner's LO, so the partner is parked only when it has to be);
* the class built on the live ``quam_state``: the config it hands the backend has
  the probe's port at the window centre (band 2 for the ~6.8 GHz coupler) with the
  ramp operation on the coupler, while the QUAM tree is exactly as it was;
* the generated QUA: per tone point the IF update, the stretched ``saturation``,
  then the ramp in one arm and a wait of the same length in the other, both
  members read out.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from scqo_qm.experiments.pair_coupler_spectroscopy_swap import (
    RAMP_OPERATION,
    _choose_band,
    ramp_samples,
)

STATE = str(Path(__file__).resolve().parents[1] / "quam_state")


# ------------------------------------------------------------------ pure

def test_the_ramp_runs_from_start_to_end_one_sample_per_ns():
    s = ramp_samples(0.08, 0.14, 400)
    assert len(s) == 400
    assert s[0] == pytest.approx(0.08) and s[-1] == pytest.approx(0.14)
    assert np.all(np.diff(s) > 0)
    down = ramp_samples(0.0, -0.1, 16)
    assert down[0] == 0.0 and down[-1] == pytest.approx(-0.1)


@pytest.mark.parametrize("current,lo,partner,band", [
    (1, 5.2e9, 4.9e9, 1),     # the current band holds it: nothing moves
    (1, 6.8e9, 4.9e9, 2),     # 5Q4C q1_q2_c: band 2 holds both LOs
    (1, 7.17e9, 5.0e9, 2),    # 5Q4C q2_q3_c: band 2 still holds q4's 5.0 GHz
    (1, 9.0e9, 4.9e9, 3),     # only band 3 holds it: the partner gets parked
    (2, 6.8e9, 4.9e9, 2),
])
def test_the_band_is_the_least_disruptive_one(current, lo, partner, band):
    assert _choose_band(current, lo, partner) == band


def test_an_lo_no_band_holds_is_refused():
    with pytest.raises(ValueError, match="no MW-FEM band"):
        _choose_band(1, 11e9, None)


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
    from scqo_qm.experiments.pair_coupler_spectroscopy_swap import (
        QMPairCouplerSpectroscopySwap,
    )

    if "q1_q2" not in machine.qubit_pairs:
        pytest.skip("live state has no pair q1_q2")
    backend = QMBackend(machine, roster=live_roster)
    params = QMPairCouplerSpectroscopySwap.Parameters(
        targets=["q1_q2"], **{"ramp_end_v": 0.14, "num_tone_freq_points": 11,
                              "num_averages": 10, **kw})
    exp = QMPairCouplerSpectroscopySwap(backend, params)
    exp.device = recording_device(backend, live_roster)
    exp.sweep_axes = exp.define_sweep()
    return exp


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


def test_the_run_config_moves_the_lo_and_the_tree_does_not(machine, live_roster):
    before = _tree(machine)
    exp = _experiment(machine, live_roster)          # probe = high = q2 (vendor target)
    prog, axes, acquire = exp.probe()
    assert _tree(machine) == before, "the QUAM tree was left moved"
    config = exp._config
    qp = machine.qubit_pairs["q1_q2"]
    probe_q, partner_q = qp.qubit_target, qp.qubit_control
    probe_port = _port(config, probe_q.xy)
    assert probe_port["band"] == 2
    assert probe_port["upconverter_frequency"] == pytest.approx(6.80e9)
    # the partner follows the band but keeps its LO: band 2 holds 4.9 GHz
    partner_port = _port(config, partner_q.xy)
    assert partner_port["band"] == 2
    assert partner_port["upconverter_frequency"] == pytest.approx(
        before[partner_q.name][1])
    assert exp._moved["partner_parked"] == 0
    # the ramp: on the coupler, 936 samples (0.14 V at 0.15 V/us, up to 4 ns)
    ops = config["elements"][qp.coupler.name]["operations"]
    pulse = config["pulses"][ops[RAMP_OPERATION]]
    samples = config["waveforms"][pulse["waveforms"]["single"]]["samples"]
    assert pulse["length"] == len(samples) == 936
    assert samples[0] == 0.0 and samples[-1] == pytest.approx(0.14)
    # the data axis is the ABSOLUTE tone frequency the IFs realize
    np.testing.assert_allclose(axes["tone_freq_hz"].values,
                               np.linspace(6.55e9, 7.05e9, 11), atol=1.0)
    assert list(axes["ramp_played"].values) == [1, 0]
    assert exp.patch_preview_config({}) is config
    assert acquire.keywords["config"] is config


def test_the_program_plays_both_arms(machine, live_roster):
    from qm import generate_qua_script

    exp = _experiment(machine, live_roster, ramp_start_v=0.08, flux_buffer_ns=100)
    prog, _axes, _acq = exp.probe()
    script = generate_qua_script(prog, exp._config).replace(chr(34), chr(39))
    qp = machine.qubit_pairs["q1_q2"]
    probe_xy = qp.qubit_target.xy.name
    coupler = qp.coupler.name
    ramp_cycles = exp.ramp_duration_ns() // 4           # 400 ns -> 100 cycles
    assert exp.ramp_duration_ns() == 400
    assert re.search(rf"update_frequency\('{re.escape(probe_xy)}', v\d+", script)
    # the tone (10 us = 2500 cycles) in BOTH arms, the ramp in one
    assert script.count(f"play('saturation', '{probe_xy}', duration=2500)") == 2
    assert script.count(f"play('{RAMP_OPERATION}', '{coupler}')") == 1
    assert f"wait({ramp_cycles}, '{coupler}')" in script
    for q in (qp.qubit_control, qp.qubit_target):
        assert script.count(f"'{q.resonator.name}'") >= 2   # read out in both arms


def test_fast_then_slow_plays_the_ramp_backwards(machine, live_roster):
    """The jump goes to the far end at once and the slow segment runs back toward
    idle - the swap happens on the way back. Same length as the other shape."""
    exp = _experiment(machine, live_roster, ramp_shape="fast_then_slow", probe="low")
    exp.probe()
    coupler = machine.qubit_pairs["q1_q2"].coupler.name
    pulse = exp._config["pulses"][exp._config["elements"][coupler]["operations"][RAMP_OPERATION]]
    samples = exp._config["waveforms"][pulse["waveforms"]["single"]]["samples"]
    assert len(samples) == 936
    assert samples[0] == pytest.approx(0.14) and samples[-1] == pytest.approx(0.0)
    assert np.all(np.diff(samples) < 0)


def test_a_ramp_past_the_rail_is_refused(machine, live_roster):
    exp = _experiment(machine, live_roster, ramp_end_v=0.6)
    with pytest.raises(ValueError, match="full scale|amplitude_scale"):
        exp.probe()


def test_ramp_on_probe_plays_on_the_probes_z(machine, live_roster):
    exp = _experiment(machine, live_roster, ramp_on="probe", ramp_end_v=-0.05)
    exp.probe()
    z = machine.qubit_pairs["q1_q2"].qubit_target.z.name
    assert RAMP_OPERATION in exp._config["elements"][z]["operations"]
