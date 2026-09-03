"""Driver-side scqo glue: the `scqo` CLI works in THIS venv + the qm factory.

The real CLI coverage lives in SCQO/tests (test_cli_*.py) against the built-in
simulated backend; this smoke test only proves the driver-side glue: the `scqo`
command runs end-to-end in the qm venv, the per-CHANNEL-KIND fieldmap cannot
drift from scqo's catalog, and the `scqo.backends` entry point resolves to a
working factory (``build_backend(cfg, setup, roster)`` — the setup is a NAMED
record, backend + note plus the DERIVED "instrument_config" vendor folder
injected by scqo, and the roster is the device's authority on which entities
exist).

Greenfield: the temp lab writes a schema-3 components.toml (modes + lines; the
readout rider mints q0_res/q0_ro, the drive rider q0_xy) plus the design.toml
the simulated vendor seeds its knobs from — without a datasheet no knob has a
standing value and every run fails pre-probe.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: the whole served CHANNEL surface: one view class + one binding table per kind
SERVED_KINDS = {"drive", "readout", "flux"}


def _env(tmp_path: Path) -> dict:
    data_root = tmp_path / "data"
    (data_root / "simdev").mkdir(parents=True)
    (data_root / "simdev" / "cooldowns.toml").write_text(
        '[cd1]\nstart = 2026-07-01\n[cd1.setup.practice]\nbackend = "simulated"\n',
        encoding="utf-8",
    )
    # post-cutover a CONFIGURED device REQUIRES a component roster
    (data_root / "simdev" / "components.toml").write_text(
        "schema = 3\n"
        "[modes.q0]\n"
        'kind = "transmon"\n'
        "[lines.fl]\n"
        'readout = ["q0"]\n'      # mints q0_res (mode) + q0_ro (channel)
        "[lines.xy0]\n"
        'drive = ["q0"]\n',       # mints q0_xy
        encoding="utf-8",
    )
    # ...and a datasheet: the simulated vendor seeds readout_freq_hz from the
    # resonator's f_dress0_hz and drive_freq_hz from the qubit's f_01_hz
    (data_root / "simdev" / "design.toml").write_text(
        "schema = 1\n[q0]\nf_01_hz = 3.8e9\n[q0_res]\nf_dress0_hz = 5.95e9\n",
        encoding="utf-8",
    )
    # Always pin parameters_file (empty): without it the CLI falls back to the
    # runner's real ~/.scqo/parameters.toml, whose standing defaults can flip
    # the sim fit to failed (same guard as SCQO's test_cli_run).
    params = tmp_path / "parameters.toml"
    params.write_text("", encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text(
        f"[lab]\ndevice = \"simdev\"\ndata_root = '{data_root.as_posix()}'\n"
        f"parameters_file = '{params.as_posix()}'\n",
        encoding="utf-8",
    )
    return {**os.environ, "SCQO_CONFIG": str(config), "SCQO_USER_CONFIG": "none"}


def test_scqo_run_end_to_end(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "scqo.cli", "run", "resonator_spectroscopy",
         "--targets", "q0"],
        capture_output=True, text=True, env=_env(tmp_path), cwd=REPO,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.split("\nsaved:")[0])
    assert result["outcomes"] == {"q0": "successful"}


def test_field_catalog_matches_implementation():
    """The declared field catalog cannot drift: per CHANNEL KIND, bindings plus the
    declared Unrealized entries cover EXACTLY scqo's KNOB fields of that kind (a new
    core knob fails here until this driver binds or declines it — the combo-release
    alarm; monitors and facts are never pushed and must appear in neither), coupled
    names are real sibling knobs, the vendor-only inventory collides with no neutral
    field name, and the module is pure data (importable without qm/quam — enforced
    on its import statements)."""
    import ast

    from scqo.catalog import ALL_STATIC_FIELDS, CHANNELS

    from scqo_qm.backend import fieldmap

    assert set(fieldmap.FIELD_BINDINGS) == SERVED_KINDS
    assert set(fieldmap.UNREALIZED) <= SERVED_KINDS
    for kind in SERVED_KINDS:
        knobs = {f for f, spec in CHANNELS[kind].fields.items()
                 if spec.role == "knob"}
        bindings = fieldmap.FIELD_BINDINGS[kind]
        unrealized = fieldmap.UNREALIZED.get(kind, {})
        assert set(bindings) | set(unrealized) == knobs, kind
        assert not set(bindings) & set(unrealized)  # realized XOR unrealized
        for name, binding in bindings.items():
            assert binding.path, f"{kind}.{name}: empty vendor path"
            assert set(binding.coupled) <= knobs - {name}, name
        for name, entry in unrealized.items():
            # the scqo dataclass attribute is still spelled 'category'; since
            # the greenfield model it carries the channel KIND
            assert entry.category == kind and entry.field == name, name
            assert entry.reason, name

    assert not set(fieldmap.VENDOR_ONLY) & ALL_STATIC_FIELDS
    assert all(v.path and v.doc for v in fieldmap.VENDOR_ONLY.values())

    # every entry carries a valid placement-rule kind; unique entries must state
    # the lock-in fact (no counterpart on the other backend)
    from scqo.fieldmap import VENDOR_ONLY_KINDS

    for name, v in fieldmap.VENDOR_ONLY.items():
        assert v.kind in VENDOR_ONLY_KINDS, name
        if v.kind == "unique":
            assert "no qblox counterpart" in v.doc.lower(), name

    tree = ast.parse(Path(fieldmap.__file__).read_text(encoding="utf-8"))
    imported = {
        name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for name in ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module])
    }
    assert imported <= {"__future__", "scqo.fieldmap"}, imported

    # the backend class serves exactly the declared catalog (methods are pure),
    # and it serves a view class for exactly the kinds the catalog declares
    from scqo_qm.backend.qm_backend import _CHANNEL_VIEWS, QMBackend

    assert QMBackend.field_bindings(None) == fieldmap.FIELD_BINDINGS
    assert QMBackend.unrealized(None) == fieldmap.UNREALIZED
    assert QMBackend.vendor_only(None) == fieldmap.VENDOR_ONLY
    assert set(_CHANNEL_VIEWS) == SERVED_KINDS


def test_composite_knob_catalog_covers_every_op_knob():
    """The COMPOSITE half of the catalog (QM has a pair surface Qblox does not):
    a pair's knob names are PER-OPERATION full names, so they are tabulated by
    OP_KNOBS SUFFIX instead of by static field name. Bindings plus declared
    Unrealized suffixes must cover scqo's OP_KNOBS exactly — the same drift
    alarm, one level down."""
    from scqo.catalog import OP_KNOBS

    from scqo_qm.backend import fieldmap
    from scqo_qm.backend.qm_backend import QMBackend

    bound = set(fieldmap.OP_KNOB_BINDINGS)
    declined = set(fieldmap.OP_KNOB_UNREALIZED)
    assert bound | declined == set(OP_KNOBS)
    assert not bound & declined
    for name, binding in fieldmap.OP_KNOB_BINDINGS.items():
        assert binding.path, f"{name}: empty vendor path"
    for name, entry in fieldmap.OP_KNOB_UNREALIZED.items():
        assert entry.category == "qubit_pair" and entry.field == name, name
        assert entry.reason, name
    assert QMBackend.op_knob_bindings(None) == (fieldmap.OP_KNOB_BINDINGS,
                                                fieldmap.OP_KNOB_UNREALIZED)


def test_backend_entry_point_resolves_and_guards_fire(tmp_path, roster):
    """The qm factory loads (vendor-free import); the pull-guard fires BEFORE any
    QUAM state is touched; missing canonical files are named — no hardware needed."""
    from importlib.metadata import entry_points

    import pytest

    from scqo.labconfig import LabConfig

    eps = {ep.name: ep for ep in entry_points(group="scqo.backends")}
    assert "qm" in eps, ("reinstall the editable (uv pip install -e . --no-deps) "
                         "to register entry points")
    factory = eps["qm"].load()

    empty = tmp_path / "empty"
    empty.mkdir()
    setup = {"backend": "qm", "instrument_config": str(empty)}

    push_cfg = LabConfig(state_sync="push")
    with pytest.raises(SystemExit, match="pull"):
        factory(push_cfg, setup, roster)  # state-authority guard, before any file

    pull_cfg = LabConfig(state_sync="pull")
    with pytest.raises(SystemExit, match="state.json"):
        factory(pull_cfg, setup, roster)  # canonical QUAM files required

    with pytest.raises(SystemExit, match="qm"):
        factory(pull_cfg, {"backend": "qblox"}, roster)  # wrong family refused


def test_factory_runs_the_whole_tree_audits_on_the_loaded_state(tmp_path, roster, monkeypatch):
    """The audits fire from build_backend (the only place they run) and REFUSE
    with SystemExit naming the file. QMBackend.load is stubbed to hand back the
    fixture tree, so no QUAM state is needed -- only the two canonical files the
    factory checks for before loading."""
    from types import SimpleNamespace

    import pytest

    from scqo.labconfig import LabConfig

    from conftest import make_stub_machine
    from scqo_qm import scqo_backend
    from scqo_qm.backend import qm_backend

    folder = tmp_path / "backend_config"
    folder.mkdir()
    for name in ("state.json", "wiring.json"):
        (folder / name).write_text("{}", encoding="utf-8")
    setup = {"backend": "qm", "instrument_config": str(folder)}
    cfg = LabConfig(state_sync="pull")

    machine = make_stub_machine()
    monkeypatch.setattr(qm_backend.QMBackend, "load",
                        lambda **kw: SimpleNamespace(machine=machine))
    assert scqo_backend.build_backend(cfg, setup, roster).machine is machine  # compliant

    machine.qubits["q1"].xy.RF_frequency = 4.9e9  # f_01 stays 4.8e9
    with pytest.raises(SystemExit, match="drive frequencies") as exc:
        scqo_backend.build_backend(cfg, setup, roster)
    assert "qubits.q1.f_01" in str(exc.value) and "state.json" in str(exc.value)


def test_components_inventory_is_a_truthful_witness(backend, roster):
    """``components()`` is the doctor's WITNESS: exactly the entities this backend
    serves a view for, each reported with the ROSTER's kind (a kind disagreement
    is a FAIL in scqo.checks.vendor_checks) and its derived operation. The pair
    composite IS present here — unlike Qblox, QUAM exposes gate macros — and the
    fixed-frequency q3 has no flux channel to miss."""
    from scqo.checks import FAIL, vendor_checks

    inventory = backend.device.components()

    expected = set(roster.channels()) | set(roster.composites())
    assert set(inventory) == expected
    for name, info in inventory.items():
        entity = roster.entities[name]
        assert info.kind == entity.kind
    assert inventory["q1_ro"].operations == ("readout",)
    assert inventory["q1_xy"].operations == ("rx",)
    assert inventory["q1_q2_c_z"].operations == ("flux_bias",)
    # two DECLARED operations on the pair, deliberately of different macro
    # shapes: the vendor CZGate and the lab's ISwapImplementation (conftest)
    assert inventory["q1_q2"].operations == ("cz", "iswap")
    assert inventory["q1_q2"].members["coupler"] == ("q1_q2_c",)

    checks = vendor_checks(roster, inventory)
    assert not [c for c in checks if c.status == FAIL], checks
    assert not [c for c in checks if "does not realize" in c.message], checks
