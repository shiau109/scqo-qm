"""``scqo-qm adopt-channel`` on a COPY of the live ``quam_state/``.

Adopting a borrowed drive channel adds ONE element (``borrowed_channels[<address>]``,
an MWChannel on the line's port, upconverter 2) and changes the port pair's band / LO
spelling - and nothing else: the element compiles onto the port's second upconverter,
the line's own qubit keeps its LO and IF, ``wiring.json`` is byte-identical, and every
refusal leaves both files untouched. The roster is the one the live tree implies
(``roster_gen``: drive lines ``xy_<q>``, coupler modes ``<low>_<high>_c``), so the
coupler q1_q2_c through q2's line is ``xy_q2.q1_q2_c``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")

from scqo_qm.backend.adopt_channel import _strip, adopt_channel, list_channels, main  # noqa: E402

LIVE = Path(__file__).resolve().parents[1] / "quam_state"
ADDRESS = "xy_q2.q1_q2_c"
FREQ = 7.0569545e9
LO = 7.1e9


@pytest.fixture()
def setup_dir(tmp_path):
    if not (LIVE / "state.json").exists():
        pytest.skip("no local quam_state/ to copy")
    folder = tmp_path / "backend_config"
    folder.mkdir()
    for name in ("state.json", "wiring.json"):
        shutil.copy(LIVE / name, folder / name)
    return folder


def _machine(folder):
    from scqo_qm.quam_io import load_state

    return load_state(str(folder))


def _session(folder, physical=None):
    from scqo.roster import parse_components

    from scqo_qm.backend.qm_backend import QMBackend
    from scqo_qm.backend.roster_gen import roster_toml_for

    machine = _machine(folder)
    roster = parse_components(roster_toml_for(machine))
    if ADDRESS not in roster.entities:
        pytest.skip(f"the live tree implies no {ADDRESS}")
    return SimpleNamespace(roster=roster, backend=QMBackend(machine, roster=roster),
                           physical=physical, cooldown_id="", setup_name="")


def _adopt(folder, address=ADDRESS, **kw):
    kw = {"lo_hz": LO, "freq_hz": FREQ, **kw}
    return adopt_channel(address, session=_session(folder), state_dir=folder, **kw)


def _files(folder):
    return {n: (folder / n).read_bytes() for n in ("state.json", "wiring.json")}


def _port_keys(port):
    from scqo_qm._mw_fem import port_info

    ctrl, fem, pid = port_info(port)
    return str(ctrl), str(fem), str(pid)


def test_adopting_adds_the_element_and_moves_only_the_port_pair(setup_dir):
    before = _files(setup_dir)
    live = json.loads(before["state.json"])
    machine = _machine(setup_dir)
    q2_port = _port_keys(machine.qubits["q2"].xy.opx_output)

    out = _adopt(setup_dir)
    assert out["saved"] and out["band"] == 2 and out["port"].endswith(" up2")
    assert out["if_hz"] == pytest.approx(FREQ - LO)

    after = _files(setup_dir)
    assert after["wiring.json"] == before["wiring.json"]
    new = json.loads(after["state.json"])
    element = new["borrowed_channels"][ADDRESS]
    assert element["id"] == ADDRESS and element["upconverter"] == 2
    assert element["operations"]["x180"]["amplitude"] == 0.25
    assert element["operations"]["x90"]["amplitude"] == 0.125
    assert element["operations"]["x180"]["length"] == 200
    ctrl, fem, pid = q2_port
    port = new["ports"]["mw_outputs"][ctrl][fem][pid]
    assert port["upconverter_frequency"] is None and port["band"] == 2
    lo1 = live["ports"]["mw_outputs"][ctrl][fem][pid]["upconverter_frequency"]
    assert {int(k): v["frequency"] for k, v in port["upconverters"].items()} == {1: lo1, 2: LO}
    touched = [q2_port, (ctrl, fem, out["partner"].split("/")[2])]
    assert _strip(new, ADDRESS, touched) == _strip(live, ADDRESS, touched)


def test_the_adopted_element_compiles_and_the_line_qubit_keeps_its_if(setup_dir):
    machine = _machine(setup_dir)
    before = machine.generate_config()["elements"]
    _adopt(setup_dir)
    config = _machine(setup_dir).generate_config()
    element = config["elements"][ADDRESS]
    assert element["MWInput"]["upconverter"] == 2
    assert element["intermediate_frequency"] == pytest.approx(FREQ - LO)
    for q in ("q1", "q2"):
        name = machine.qubits[q].xy.name
        assert config["elements"][name]["intermediate_frequency"] == pytest.approx(
            before[name]["intermediate_frequency"])
        assert config["elements"][name]["MWInput"] == before[name]["MWInput"]


def test_the_backend_serves_the_adopted_channel(setup_dir):
    from scqo_qm.backend.qm_backend import QMBorrowedDriveChannel

    _adopt(setup_dir)
    session = _session(setup_dir)
    device = session.backend.device
    view = device.component(ADDRESS)
    assert isinstance(view, QMBorrowedDriveChannel)
    assert view.pi_amp == pytest.approx(0.25) and view.pi_amp_x90 == pytest.approx(0.125)
    assert view.drive_freq_hz == pytest.approx(FREQ)
    assert view.pi_duration_s == pytest.approx(2e-7)
    view.pi_amp = 0.3
    assert view.vendor.operations["x180"].amplitude == pytest.approx(0.3)
    assert view.pi_amp_x90 == pytest.approx(0.125)          # its own knob
    view.pi_duration_s = 240e-9                              # x180 and x90 together
    assert [view.vendor.operations[op].length for op in ("x180", "x90")] == [240, 240]
    with pytest.raises(NotImplementedError, match="I28"):
        view.drive_power_dbm = -10.0
    with pytest.raises(ValueError, match="IF"):
        view.drive_freq_hz = LO + 300e6
    snap = device.snapshot()[ADDRESS]
    assert snap["pi_amp"] == pytest.approx(0.3) and snap["drive_amp"] is None
    assert device.components()[ADDRESS].kind == "drive"


def test_a_second_channel_on_the_port_shares_upconverter_2(setup_dir):
    _adopt(setup_dir)
    session = _session(setup_dir)
    others = sorted(n for n, e in session.roster.borrowed_channels().items()
                    if e.line == "xy_q2" and n.endswith("_c") and n != ADDRESS)
    if not others:
        pytest.skip("the live tree has no second coupler to adopt through xy_q2")
    with pytest.raises(SystemExit, match="already runs upconverter 2"):
        _adopt(setup_dir, others[0], lo_hz=7.2e9, freq_hz=7.15e9)
    out = _adopt(setup_dir, others[0], freq_hz=7.1555e9)
    assert out["if_hz"] == pytest.approx(7.1555e9 - LO)


def test_the_measured_f01_is_the_default_frequency(setup_dir):
    physical = SimpleNamespace(
        get=lambda entity, field: FREQ if (entity, field) == ("q1_q2_c", "f_01_hz") else None)
    out = adopt_channel(ADDRESS, lo_hz=LO, session=_session(setup_dir, physical),
                        state_dir=setup_dir)
    assert out["rf_hz"] == pytest.approx(FREQ)


@pytest.mark.parametrize("address, kw, words", [
    ("xy_q2.q2", {}, "DESIGNED channel"),
    ("xy_q2.nope", {}, "unknown entity"),
    (ADDRESS, {"freq_hz": LO + 400e6}, "IF window"),
    (ADDRESS, {"freq_hz": None}, "no --freq-hz"),
    (ADDRESS, {"lo_hz": 11.0e9, "freq_hz": 11.0e9}, "no MW-FEM band holds both"),
    (ADDRESS, {"pi_amp": 1.5}, r"in \(0, 1\)"),
    (ADDRESS, {"length_ns": 10}, "multiple of 4"),
])
def test_every_refusal_leaves_the_folder_untouched(setup_dir, address, kw, words):
    before = _files(setup_dir)
    with pytest.raises(SystemExit, match=words):
        _adopt(setup_dir, address, **kw)
    assert _files(setup_dir) == before


def test_adopting_twice_is_refused(setup_dir):
    _adopt(setup_dir)
    before = _files(setup_dir)
    with pytest.raises(SystemExit, match="already adopted"):
        _adopt(setup_dir)
    assert _files(setup_dir) == before


def test_a_file_changed_on_disk_after_loading_is_never_overwritten(setup_dir):
    session = _session(setup_dir)
    state = json.loads((setup_dir / "state.json").read_text(encoding="utf-8"))
    state.setdefault("extras", {})["edited_meanwhile"] = 1.0
    (setup_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    edited = _files(setup_dir)
    with pytest.raises(SystemExit, match="changed on disk"):
        adopt_channel(ADDRESS, lo_hz=LO, freq_hz=FREQ, session=session, state_dir=setup_dir)
    assert _files(setup_dir) == edited


def test_dry_run_and_list_write_nothing(setup_dir):
    before = _files(setup_dir)
    out = _adopt(setup_dir, dry_run=True)
    assert out["saved"] is False and _files(setup_dir) == before
    rows = list_channels(_session(setup_dir))
    assert any(r["line"] == "xy_q2" and r["los"] for r in rows["lines"])
    assert rows["adopted"] == []
    assert _files(setup_dir) == before


def test_the_command_is_listed_on_an_mw_fem_tree(setup_dir):
    names = {c.name for c in _session(setup_dir).backend.operator_commands()}
    assert "adopt_channel" in names


@pytest.mark.parametrize("argv", [["--list", ADDRESS], [ADDRESS]])
def test_bad_flag_combinations_exit_2(argv):
    with pytest.raises(SystemExit) as err:
        main(argv)
    assert err.value.code == 2
