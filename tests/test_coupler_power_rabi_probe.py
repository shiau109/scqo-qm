"""``qubit_power_rabi`` on a COUPLER, on a copy of the live ``quam_state/`` with the
borrowed channel adopted (SCQO docs/coupler-transmon-plan.md section 4).

The program is the qubits' own, over a ``MappedTarget``: the x180 plays on the
ADOPTED element (``xy_q2.q1_q2_c``, q2's port, upconverter 2) with the swept
``amplitude_scale``; the read is the MAP - the member q1's selective pi (its
``saturation`` scaled to the x180's area, 500 clocks) then q1's own x180 - and q1's
readout. Nothing plays on q2's own element. Also here: the coupler tone and the
broadband search on the adopted port pair, and the run record's power context.
"""

from __future__ import annotations

import re

import pytest

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")

from conftest import recording_device  # noqa: E402
from test_adopt_channel import ADDRESS, LO, _adopt, _session, setup_dir  # noqa: E402,F401


def _experiment(session, cls=None, **kw):
    if cls is None:
        from scqo_qm.experiments.qubit_power_rabi import QMQubitPowerRabi as cls
    base = {"targets": ["q1_q2_c"], "drive_line": "xy_q2", "readout_member": "q1",
            "use_state_discrimination": True, "num_averages": 10, "num_amp_points": 11}
    exp = cls(session.backend, cls.Parameters(**{**base, **kw}))
    exp.device = recording_device(session.backend, session.roster)
    exp.sweep_axes = exp.define_sweep()
    return exp


def _script(session, prog) -> str:
    from qm import generate_qua_script

    config = session.backend.machine.generate_config()
    return generate_qua_script(prog, config).replace(chr(34), chr(39))


def test_the_coupler_is_driven_through_the_adopted_channel_and_read_through_q1(setup_dir):
    from scqo_qm.experiments._selective_pi import selective_pi_scale

    _adopt(setup_dir)
    session = _session(setup_dir)
    exp = _experiment(session)
    prog, axes = exp.probe()
    assert list(axes["qubit"].values) == ["q1_q2_c"]
    assert axes["amp_prefactor"].size == 11
    script = _script(session, prog)
    machine = session.backend.machine
    q1, q2 = machine.qubits["q1"], machine.qubits["q2"]

    drive = re.search(rf"play\('x180'\*amp\(v\d+\), '{re.escape(ADDRESS)}'\)", script)
    selective = re.search(rf"play\('saturation'\*amp\(([0-9.e-]+)\), "
                          rf"'{re.escape(q1.xy.name)}', duration=500\)", script)
    flip = script.find(f"play('x180', '{q1.xy.name}')")
    read = script.find(f"'{q1.resonator.name}'", flip)
    assert drive and selective and flip > 0 and read > 0
    assert drive.start() < selective.start() < flip < read      # drive, map, read
    assert float(selective.group(1)) == pytest.approx(selective_pi_scale(q1, 2000), rel=1e-6)
    assert not re.search(rf"play\([^)]*'{re.escape(q2.xy.name)}'", script)


def test_a_member_without_a_threshold_is_refused_before_any_qua(setup_dir):
    _adopt(setup_dir)
    session = _session(setup_dir)
    session.backend.machine.qubits["q1"].resonator.operations["readout"].threshold = None
    exp = _experiment(session)
    with pytest.raises(ValueError, match="q1 discriminated.*no readout_threshold"):
        exp.probe()


def test_a_drive_line_alone_plays_on_the_borrowed_channel_and_reads_the_target(setup_dir):
    """drive_line without readout_member: q1 driven through q2's line (xy_q2.q1,
    adopted on upconverter 2 near q1's frequency), read by q1's own resonator."""
    f01 = float(_session(setup_dir).backend.machine.qubits["q1"].xy.RF_frequency)
    _adopt(setup_dir, "xy_q2.q1", lo_hz=round(f01 / 1e8) * 1e8, freq_hz=f01)
    session = _session(setup_dir)
    exp = _experiment(session, targets=["q1"], readout_member=None,
                      use_state_discrimination=False)
    prog, _axes = exp.probe()
    script = _script(session, prog)
    q1 = session.backend.machine.qubits["q1"]
    assert re.search(r"play\('x180'\*amp\(v\d+\), 'xy_q2\.q1'\)", script)
    assert not re.search(rf"play\('x180'\*amp\(v\d+\), '{re.escape(q1.xy.name)}'\)", script)
    assert f"'{q1.resonator.name}'" in script


def test_the_power_context_names_the_borrowed_port_and_the_member(setup_dir):
    from scqo_qm.backend.qm_backend import _routes_of

    _adopt(setup_dir)
    session = _session(setup_dir)
    backend = session.backend
    exp = _experiment(session)
    backend._routes = _routes_of(exp)
    ctx = backend.power_context(["q1_q2_c"])["q1_q2_c"]
    port = backend.machine.qubits["q2"].xy.opx_output
    assert ctx["drive_channel"] == ADDRESS and ctx["drive_port"].endswith(" up2")
    assert ctx["drive_lo_freq_hz"] == pytest.approx(LO)
    assert ctx["drive_full_scale_power_dbm"] == port.full_scale_power_dbm
    assert ctx["readout_member"] == "q1" and "readout_amplitude" in ctx
    assert backend._routes == {}          # consumed: it can never describe a later run
    assert "drive_channel" not in backend.power_context(["q1_q2_c"])["q1_q2_c"]


def test_the_coupler_tone_moves_only_its_own_upconverter(setup_dir):
    from scqo_qm._mw_fem import port_info, port_los
    from scqo_qm.experiments._coupler_tone import moved_lo_config

    _adopt(setup_dir)
    machine = _session(setup_dir).backend.machine
    q2 = machine.qubits["q2"]
    port = q2.xy.opx_output
    before = port_los(port)
    config, moved = moved_lo_config(machine, q2, lo_hz=7.0e9, experiment="test")
    assert port_los(port) == before                     # the tree is restored
    ctrl, fem, pid = port_info(port)
    compiled = config["controllers"][ctrl]["fems"][fem]["analog_outputs"][pid]
    assert compiled.get("upconverter_frequency") is None
    assert {int(k): v["frequency"] for k, v in compiled["upconverters"].items()} == {
        1: 7.0e9, 2: LO}
    assert moved["band"] == 2 and moved["partner_parked"] == 0
    with pytest.raises(ValueError, match="upconverter 2 at 7.1 GHz"):
        moved_lo_config(machine, q2, lo_hz=4.0e9, experiment="test")   # band 1 strands it
    assert port_los(port) == before


def test_broadband_refuses_a_port_pair_with_an_adopted_channel(setup_dir):
    from scqo_qm.experiments.broadband_qubit_spectroscopy import QMBroadbandQubitSpectroscopy

    _adopt(setup_dir)
    session = _session(setup_dir)
    params = QMBroadbandQubitSpectroscopy.Parameters(targets=["q1"])
    exp = QMBroadbandQubitSpectroscopy(session.backend, params)
    exp.device = recording_device(session.backend, session.roster)
    exp.sweep_axes = exp.define_sweep()
    with pytest.raises(ValueError, match="second upconverter"):
        exp.probe()
