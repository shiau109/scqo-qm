"""register_partial_swap on a COPY of the local live ``quam_state``: the three entries
land, nothing else moves, and every refusal leaves the live folder untouched.

The session is injected (no scqo config, no QM cluster), so the ``build_session`` door is
the only part not exercised here; it is the same door ``apply_distortion`` uses.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")

from scqo_qm.backend.register_partial_swap import (  # noqa: E402
    NAME_PATTERN,
    list_partial_swaps,
    main,
    pulse_name,
    register_partial_swap,
)

LIVE = Path(__file__).resolve().parents[1] / "quam_state"
PAIR = "q1_q2"
NAME = "partial_swap_077"  # an angle no live tree carries
PULSE = "partial_swap_square_077"


@pytest.fixture
def setup_dir(tmp_path):
    """A setup's backend_config folder: a copy of the local live tree."""
    if not (LIVE / "state.json").is_file():
        pytest.skip("no local quam_state (it is gitignored)")
    folder = tmp_path / "backend_config"
    folder.mkdir()
    for name in ("state.json", "wiring.json"):
        shutil.copy2(LIVE / name, folder / name)
    return folder


def _session(folder):
    return SimpleNamespace(backend=SimpleNamespace(machine=quam_config.Quam.load(str(folder))))


def _register(folder, name=NAME, **kw):
    return register_partial_swap(PAIR, name, session=_session(folder), state_dir=folder, **kw)


def _state(folder):
    return json.loads((folder / "state.json").read_text(encoding="utf-8"))


def _files(folder):
    return {n: (folder / n).read_bytes() for n in ("state.json", "wiring.json")}


def _without_op(state, control):
    state = json.loads(json.dumps(state))
    state["qubit_pairs"][PAIR]["macros"].pop(NAME, None)
    state["qubit_pairs"][PAIR]["coupler"]["operations"].pop(PULSE, None)
    state["qubits"][control]["z"]["operations"].pop(PULSE, None)
    return state


def test_the_name_contract():
    assert pulse_name("partial_swap_060") == "partial_swap_square_060"
    for bad in ("partial_swap", "partial_swap_60", "partial_swap_0600", "iswap", "partial_swap_x60"):
        with pytest.raises(SystemExit, match="three digits"):
            pulse_name(bad)


def test_adds_the_three_entries_and_nothing_else(setup_dir):
    before = _state(setup_dir)
    wiring = (setup_dir / "wiring.json").read_bytes()

    out = _register(setup_dir, z_amp=-0.15, coupler_amp=0.0875)

    assert (out["action"], out["saved"], out["length_ns"]) == ("added", True, 40)
    after = _state(setup_dir)
    control = out["control"]
    macro = after["qubit_pairs"][PAIR]["macros"][NAME]
    assert macro["flux_pulse"] == PULSE
    assert macro["__class__"] == "scqo_qm.components.macros.iswap_macro.ISwapImplementation"
    z = after["qubits"][control]["z"]["operations"][PULSE]
    coupler = after["qubit_pairs"][PAIR]["coupler"]["operations"][PULSE]
    assert (z["amplitude"], z["length"]) == (-0.15, 40)
    assert (coupler["amplitude"], coupler["length"]) == (0.0875, 40)
    assert _without_op(after, control) == _without_op(before, control)
    assert (setup_dir / "wiring.json").read_bytes() == wiring  # never rewritten
    assert not list(setup_dir.glob(".state.json*"))  # no partial file left behind

    # the saved macro loads back as its real class (QUAM would fall back silently)
    from scqo_qm.components.macros.iswap_macro import ISwapImplementation

    reloaded = quam_config.Quam.load(str(setup_dir))
    assert isinstance(reloaded.qubit_pairs[PAIR].macros[NAME], ISwapImplementation)


def test_update_retunes_one_amplitude_and_keeps_the_other(setup_dir):
    _register(setup_dir, z_amp=-0.15, coupler_amp=0.0875)

    out = _register(setup_dir, update=True, z_amp=-0.14987)

    assert out["action"] == "retuned"
    assert out["before"] == {"z_amp": -0.15, "coupler_amp": 0.0875}
    assert out["after"] == {"z_amp": -0.14987, "coupler_amp": 0.0875}
    after = _state(setup_dir)
    assert after["qubits"][out["control"]]["z"]["operations"][PULSE]["amplitude"] == -0.14987
    assert after["qubit_pairs"][PAIR]["coupler"]["operations"][PULSE]["amplitude"] == 0.0875


@pytest.mark.parametrize("pair, kwargs, fragment", [
    ("q9_q10", dict(z_amp=-0.15, coupler_amp=0.08), "no qubit pair"),
    (PAIR, dict(z_amp=-0.15), "needs both"),
    (PAIR, dict(z_amp=-0.15, coupler_amp=0.08, length_ns=42), "multiple of 4"),
    (PAIR, dict(z_amp=-0.15, coupler_amp=0.08, length_ns=12), "multiple of 4"),
    (PAIR, dict(update=True, z_amp=-0.15), "drop --update"),
    (PAIR, dict(z_amp=float("nan"), coupler_amp=0.08), "finite"),
    (PAIR, dict(z_amp=-3.0, coupler_amp=0.08), "full scale"),  # the port would clip it
])
def test_refusals_leave_the_live_folder_untouched(setup_dir, pair, kwargs, fragment):
    files = _files(setup_dir)
    with pytest.raises(SystemExit, match=fragment):
        register_partial_swap(pair, NAME, session=_session(setup_dir), state_dir=setup_dir, **kwargs)
    assert _files(setup_dir) == files


def test_an_existing_operation_is_retuned_only_with_update(setup_dir):
    _register(setup_dir, z_amp=-0.15, coupler_amp=0.0875)
    files = _files(setup_dir)

    with pytest.raises(SystemExit, match="use --update"):
        _register(setup_dir, z_amp=-0.15, coupler_amp=0.0875)
    with pytest.raises(SystemExit, match="needs --z-amp and/or --coupler-amp"):
        _register(setup_dir, update=True)
    with pytest.raises(SystemExit, match="amplitudes only"):
        _register(setup_dir, update=True, z_amp=-0.15, length_ns=48)
    assert _files(setup_dir) == files


def test_an_edit_that_landed_on_disk_after_loading_is_never_overwritten(setup_dir):
    session = _session(setup_dir)  # loaded BEFORE the other edit
    edited = _state(setup_dir)
    control = session.backend.machine.qubit_pairs[PAIR].qubit_control.name
    edited["qubits"][control]["z"]["operations"]["const"]["amplitude"] = 0.123
    (setup_dir / "state.json").write_text(json.dumps(edited), encoding="utf-8")

    with pytest.raises(SystemExit, match="changed on disk"):
        register_partial_swap(PAIR, NAME, session=session, state_dir=setup_dir,
                              z_amp=-0.15, coupler_amp=0.0875)
    assert _state(setup_dir) == edited


def test_dry_run_verifies_and_writes_nothing(setup_dir):
    files = _files(setup_dir)
    out = _register(setup_dir, z_amp=-0.15, coupler_amp=0.0875, dry_run=True)
    assert (out["action"], out["saved"]) == ("added", False)
    assert _files(setup_dir) == files


def test_list_reports_managed_and_legacy_operations(setup_dir):
    _register(setup_dir, z_amp=-0.15, coupler_amp=0.0875)
    machine = _session(setup_dir).backend.machine

    rows = list_partial_swaps(machine, PAIR)

    mine = next(r for r in rows if r["name"] == NAME)
    assert mine["managed"] and mine["flux_pulse"] == PULSE
    assert (mine["z_amp"], mine["coupler_amp"], mine["length_ns"]) == (-0.15, 0.0875, 40)
    for row in rows:
        if NAME_PATTERN.fullmatch(row["name"]) is None:
            assert not row["managed"]  # e.g. the live tree's bare "partial_swap": shown, never managed
    assert {r["pair"] for r in list_partial_swaps(machine)} <= set(machine.qubit_pairs)


@pytest.mark.parametrize("argv", [
    ["--list", "--name", NAME],
    ["--list", "--z-amp", "0"],
    ["--name", NAME, "--z-amp", "-0.15", "--coupler-amp", "0.08"],  # no --pair
])
def test_cli_refuses_contradictory_or_incomplete_flags(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2  # argparse usage error, before any session is built
