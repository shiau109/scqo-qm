"""``scqo-qm <command>`` - this driver's operator commands (in ``.venv-qm``).

The QM-side jobs scqo deliberately does not own: looking at and releasing the
cluster, writing a measured filter into the vendor config, calibrating Octave
mixers, registering a partial swap. They are NOT scqo subcommands - ``scqo run
<name>`` stays the one way to run an experiment, and a QM verb in scqo could only
ever be refused on every other backend - so they live here, beside the vendor
libraries they need.

The command table IS ``fieldmap.OPERATOR_COMMANDS``, the inventory ``scqo state
--fields`` renders: a subcommand is its inventory name with ``_`` -> ``-``, and
its code is ``scqo_qm.backend.<name>.main(argv, prog)``. One table, so the two
discovery doors cannot list different commands.

``scqo-qm -h`` is not instant: any import under ``scqo_qm`` runs the package's
``__init__``, which registers the experiments and so imports qm (~5 s).

Manual dispatch (not argparse subparsers), the shape of scqo's own CLI: each
command module keeps its own parser, so ``scqo-qm <command> --help`` is that
module's full flag list.
"""

from __future__ import annotations

import sys
import textwrap
from importlib import import_module

from scqo.fieldmap import OperatorCommand

from scqo_qm.backend.fieldmap import OPERATOR_COMMANDS


def subcommand(entry: OperatorCommand) -> str:
    """The ``scqo-qm`` subcommand an inventory entry is reached by."""
    return entry.name.replace("_", "-")


def _commands() -> dict[str, OperatorCommand]:
    return {subcommand(c): c for c in OPERATOR_COMMANDS}


def _usage() -> str:
    # ASCII only: this text reaches consoles in whatever codepage the lab runs
    lines = ["usage: scqo-qm <command> [options]   (scqo-qm <command> --help for all its flags)", "",
             "The QM operator commands - vendor tools, NOT scqo subcommands. `scqo state --fields`",
             "lists the same ones, scoped to the commands your setup's tree can use.", ""]
    for name, c in _commands().items():
        lines.append(f"  {name}")
        lines.append(f"      {c.command}")
        lines += textwrap.wrap(c.doc, width=96, initial_indent="      ", subsequent_indent="      ")
        if c.caution:
            lines += textwrap.wrap(f"CAUTION: {c.caution}", width=96,
                                   initial_indent="      ", subsequent_indent="        ")
        lines.append("")
    return "\n".join(lines).rstrip()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0
    if argv[0] == "--version":
        from importlib.metadata import version

        print(version("scqo-qm"))
        return 0
    commands = _commands()
    command = argv[0]
    if command not in commands:
        print(f"unknown command {command!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    module = import_module(f"scqo_qm.backend.{commands[command].name}")
    return module.main(argv[1:], prog=f"scqo-qm {command}") or 0


if __name__ == "__main__":
    sys.exit(main())
