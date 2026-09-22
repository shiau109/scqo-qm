"""Issue #38: a QM state scqo can load, scqo can also save — with no qualibrate
configuration on the machine.

QUAM looks ``include_defaults`` up in ``~/.qualibrate/config.toml`` on every save
that does not pass it, BEFORE it considers the path, and a missing file raises
FileNotFoundError. On a machine that had never run qualibrate, ``scqo set`` pushed
the new value to the instrument and then failed to write state.json, so the next
session loaded the old value. Machines running the qualibrate GUI have the file,
which is why the live-state tests (they only load) never saw it.

The config is hidden by pointing ``QUAM_CONFIG_FILE`` at a file that does not exist.
QUAM reads that variable on every lookup; ``HOME`` cannot be redirected this way,
because ``qualibrate_config.vars.QUALIBRATE_PATH`` is computed once, at import.
The fixture asserts the lookup really fails, so none of these tests can pass
vacuously on a machine that has the file.

The tree is BUILT here (``scripts/make_opxp_fixture.py``), config hidden, because the
build is a save path too: quam_builder's ``build_quam_wiring`` and ``build_quam``
each call a bare ``machine.save()`` that no scqo code can pass arguments to. The
lab root class pins ``include_defaults`` on its serialiser for exactly those.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

GENERATOR = Path(__file__).resolve().parents[1] / "scripts" / "make_opxp_fixture.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("make_opxp_fixture", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def no_qualibrate_config(tmp_path, monkeypatch):
    """No QUAM/qualibrate config file anywhere QUAM will look."""
    folder = tmp_path / "no_qualibrate"
    folder.mkdir()
    monkeypatch.setenv("QUAM_CONFIG_FILE", str(folder / "config.toml"))
    # QMBackend.load() and the fixture builder both set this process-wide; the
    # monkeypatch restores whatever it was once the test is done.
    monkeypatch.delenv("QUAM_STATE_PATH", raising=False)

    from quam.config import get_quam_config

    with pytest.raises(FileNotFoundError):
        get_quam_config()


def test_the_lab_root_class_pins_include_defaults_and_keeps_the_vendor_split():
    from quam_builder.architecture.superconducting.qpu.flux_tunable_quam import (
        FluxTunableQuam,
    )

    from scqo_qm.quam_builder.architecture.superconducting.qpu.mixed_quam import (
        MixedTransmonQuam,
    )
    from scqo_qm.quam_io import INCLUDE_DEFAULTS

    serialiser = MixedTransmonQuam.get_serialiser()
    assert INCLUDE_DEFAULTS is True  # what the qualibrate stack writes
    assert serialiser.include_defaults is INCLUDE_DEFAULTS
    # EXTENDED, not replaced: the vendor's mapping is what puts wiring/network in
    # wiring.json. A bare JSONSerialiser drops it and folds wiring into state.json.
    vendor = FluxTunableQuam.get_serialiser().content_mapping
    assert vendor.get("wiring") == "wiring.json"
    assert serialiser.content_mapping == vendor


def test_the_device_saves_to_its_load_folder_with_include_defaults_passed():
    from scqo_qm.backend.qm_backend import QMBackend

    saves: list = []
    machine = SimpleNamespace(save=lambda **kw: saves.append(kw))
    QMBackend(machine, roster=None, state_dir="setup/backend_config").device.save()
    assert saves == [{"path": "setup/backend_config", "include_defaults": True}]


def test_a_tree_builds_loads_and_saves_without_a_qualibrate_config(
        no_qualibrate_config, tmp_path, monkeypatch):
    from quam_config import Quam
    from scqo.roster import parse_components

    from scqo_qm.backend.qm_backend import QMBackend
    from scqo_qm.backend.roster_gen import roster_toml_for

    tree = tmp_path / "tree"
    _load_generator().build(tree)
    roster = parse_components(roster_toml_for(Quam.load(str(tree))))
    backend = QMBackend.load(roster=roster, state_path=str(tree))

    resonator = backend.machine.qubits["q1"].resonator
    changed = resonator.time_of_flight + 4
    resonator.time_of_flight = changed

    # The env var no longer points where this session loaded from (another load in
    # the same process, a shell that set it): the save must still land in the tree.
    decoy = tmp_path / "decoy"
    monkeypatch.setenv("QUAM_STATE_PATH", str(decoy))
    backend.device.save()

    assert not decoy.exists()
    assert Quam.load(str(tree)).qubits["q1"].resonator.time_of_flight == changed
    # ...and in the same two files it came from: wiring stays split out
    state = json.loads((tree / "state.json").read_text(encoding="utf-8"))
    wiring = json.loads((tree / "wiring.json").read_text(encoding="utf-8"))
    assert {"wiring", "network"} <= set(wiring)
    assert not {"wiring", "network"} & set(state)
