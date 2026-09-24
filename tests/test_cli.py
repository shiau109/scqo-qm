"""``scqo-qm <command>``: the dispatcher over this driver's operator commands.

The dispatch table IS ``fieldmap.OPERATOR_COMMANDS``, so what is worth pinning
is that every inventory entry is actually reachable (its module exists, takes
``prog``, and answers ``--help`` under its ``scqo-qm`` name) and that the console
script is declared. No cluster and no scqo config: nothing here resolves a setup.
"""

from __future__ import annotations

import inspect
import sys
from importlib import import_module
from pathlib import Path

import pytest

from scqo_qm import cli
from scqo_qm.backend.fieldmap import OPERATOR_COMMANDS

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

SUBCOMMANDS = [cli.subcommand(c) for c in OPERATOR_COMMANDS]


def test_the_console_script_is_declared():
    """Entry points register at INSTALL time: this pins the declaration, and the
    `uv pip install -e` after changing it is what puts scqo-qm on PATH."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]
    assert scripts["scqo-qm"] == "scqo_qm.cli:main"


def test_the_cluster_query_is_in_the_inventory():
    assert "cluster" in SUBCOMMANDS and "close-qm" in SUBCOMMANDS


@pytest.mark.parametrize("entry", OPERATOR_COMMANDS, ids=lambda c: c.name)
def test_every_inventory_entry_dispatches_to_a_main_taking_prog(entry):
    module = import_module(f"scqo_qm.backend.{entry.name}")
    assert "prog" in inspect.signature(module.main).parameters, entry.name


@pytest.mark.parametrize("sub", SUBCOMMANDS)
def test_every_subcommand_answers_help_under_its_scqo_qm_name(sub, capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main([sub, "--help"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: scqo-qm {sub}")


def test_the_dispatcher_hands_over_argv_and_prog(monkeypatch):
    seen = {}

    def fake_main(argv, prog):
        seen.update(argv=argv, prog=prog)
        return 7

    monkeypatch.setattr("scqo_qm.backend.cluster.main", fake_main)
    assert cli.main(["cluster", "--config", "lab.toml"]) == 7
    assert seen == {"argv": ["--config", "lab.toml"], "prog": "scqo-qm cluster"}


@pytest.mark.parametrize("argv", [[], ["-h"], ["--help"]])
def test_usage_lists_every_command_with_its_caution(argv, capsys):
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: scqo-qm <command>")
    for entry in OPERATOR_COMMANDS:
        assert f"\n  {cli.subcommand(entry)}\n" in out
        assert entry.command in out
    assert "CAUTION: DESTRUCTIVE" in out             # close-qm's must not be missed
    assert out.isascii()


def test_an_unknown_command_exits_two_and_shows_the_usage(capsys):
    assert cli.main(["close_qm"]) == 2                # the module name is not the subcommand
    err = capsys.readouterr().err
    assert "unknown command 'close_qm'" in err and "usage: scqo-qm" in err
