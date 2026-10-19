"""``qubit_spectroscopy_ef``: the 1->2 search, asserted on generated QUA.

What the experiment rests on is only visible in the emitted program:

* the x180 plays BEFORE the element is retuned, so it prepares |1> at the 0->1
  frequency; the saturation pulse plays at the retuned IF, and the 0->1 IF is
  restored right after it, before the barrier and the readout;
* the saturation pulse is scaled from the standing saturation power, so the
  port's full scale - shared with the x180 - is never moved;
* a window the xy upconverter cannot play, or a power the held port cannot
  reach, is refused by name before anything is built.

Live-QUAM: the retune is ``update_frequency`` on the real element and the scale
reads the real drive chain.
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


@pytest.fixture(scope="module")
def machine():
    return quam_config.Quam.load(str(Path(__file__).resolve().parents[1] / "quam_state"))


@pytest.fixture(scope="module")
def live_roster(machine):
    return parse_components(roster_toml_for(machine))


@pytest.fixture(scope="module")
def config(machine):
    return machine.generate_config()


def _experiment(machine, live_roster, **params):
    from scqo.experiments import get

    import scqo_qm.experiments  # noqa: F401  (registers the QM probes)

    backend = QMBackend(machine, roster=live_roster)
    cls = get("qubit_spectroscopy_ef")
    kwargs = {"num_drive_freq_points": 5, "num_averages": 10,
              "start_drive_detuning_hz": -250e6, "end_drive_detuning_hz": -20e6,
              "drive_len_ns": 8000.0, **params}
    exp = cls(backend, cls.Parameters(targets=[TARGET], **kwargs))
    exp.device = recording_device(backend, live_roster)
    exp.sweep_axes = exp.define_sweep()
    return exp


def _body(prog, config) -> list[str]:
    """One sweep point's QUA, from the x180 to the first save, quotes normalised."""
    from qm import generate_qua_script

    lines = [ln.strip().replace(chr(34), chr(39))
             for ln in generate_qua_script(prog, config).splitlines()]
    start = next(i for i, ln in enumerate(lines) if ln.startswith("play('x180'"))
    stop = next(i for i, ln in enumerate(lines[start:]) if ln.startswith("save(")) + start
    return lines[start:stop]


def _index(body, prefix) -> int:
    return next(i for i, ln in enumerate(body) if ln.startswith(prefix))


def test_x180_then_the_retuned_tone_then_the_0_1_if_then_the_readout(
        machine, live_roster, config):
    exp = _experiment(machine, live_roster)
    prog, axes = exp.probe()
    body = _body(prog, config)
    if_01 = float(machine.qubits[TARGET].xy.intermediate_frequency)

    x180 = _index(body, "play('x180'")
    retunes = [i for i, ln in enumerate(body) if ln.startswith("update_frequency(")]
    drive = _index(body, "play('saturation'")
    barrier = next(i for i in range(drive, len(body)) if body[i].startswith("align("))
    measure = _index(body, "measure('readout'")
    assert x180 < retunes[0] < drive < retunes[1] < barrier < measure
    # the first retune adds the swept detuning to the 0->1 IF; the second is the
    # 0->1 IF alone, so the next shot's x180 plays at its own frequency
    assert "+" in body[retunes[0]]
    restored = float(re.search(r",\s*(-?[\d.e+-]+),\s*'Hz'", body[retunes[1]]).group(1))
    assert restored == pytest.approx(if_01)
    assert int(re.search(r"duration=(\d+)", body[drive]).group(1)) == 2000  # 8000 ns / 4
    assert list(axes["detuning"].values) == list(exp.sweep_axes["detuning_hz"])


def test_the_saturation_power_scales_the_pulse_and_holds_the_port(
        machine, live_roster, config):
    """10 dB under the standing saturation power is an amplitude_scale of
    10**(-10/20); the port's full scale is the same before and after the build."""
    exp = _experiment(machine, live_roster)
    standing = exp.device.channel(TARGET, "drive").drive_power_dbm
    full_scale = machine.qubits[TARGET].xy.opx_output.full_scale_power_dbm
    exp.params = exp.params.model_copy(update={"drive_power_dbm": standing - 10.0})
    prog, _axes = exp.probe()
    drive = _body(prog, config)[_index(_body(prog, config), "play('saturation'")]
    scale = float(re.search(r"amp\(([\d.e+-]+)\)", drive).group(1))
    assert scale == pytest.approx(10 ** (-10.0 / 20.0), rel=1e-9)
    assert machine.qubits[TARGET].xy.opx_output.full_scale_power_dbm == full_scale


def test_a_window_past_the_upconverter_band_is_refused_with_the_reachable_edges(
        machine, live_roster):
    exp = _experiment(machine, live_roster, start_drive_detuning_hz=-450e6)
    with pytest.raises(ValueError, match=r"start_drive_detuning_hz / end_drive_detuning_hz within"):
        exp.probe()


def test_a_power_the_held_port_cannot_reach_is_refused(machine, live_roster):
    exp = _experiment(machine, live_roster, drive_power_dbm=10.0)
    with pytest.raises(ValueError, match="port's level is not moved"):
        exp.probe()


def test_the_ef_knobs_are_served_from_quam_own_homes(machine, live_roster):
    """pi_amp_ef / pi_duration_ef_s are the EF_x180 operation's; drive_freq_ef_hz
    is q.f_12, unset until measured."""
    from scqo_qm import quam_fields

    exp = _experiment(machine, live_roster)
    view = exp.device.channel(TARGET, "drive")
    op = machine.qubits[TARGET].xy.operations[quam_fields.EF_OPERATION]
    assert view.pi_amp_ef == pytest.approx(float(op.amplitude))
    assert view.pi_duration_ef_s == pytest.approx(float(op.length) / 1e9)
