"""How the one-off registration scripts here choose which QUAM tree to edit.

They used to call a bare ``Quam.load()``, which resolves through ``QUAM_STATE_PATH``
and then qualibrate's ``[quam] state_path`` — so a script edited whichever tree that
file happened to name, and on a machine whose repos have moved, that is a folder
nobody loads and nothing says so. The folder is an argument now.

``scqo state --sources`` prints the active setup's ``backend_config/``; the registering
that has an operator command instead of a script is ``scqo-qm register-partial-swap``,
which resolves the setup itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def state_folder(description: str, argv: list[str] | None = None) -> Path:
    """Parse ``<state_path>`` and return it, refusing a folder with no ``state.json``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "state_path",
        help="the QUAM state folder to edit — a setup's backend_config/ "
             "(scqo state --sources names it)")
    folder = Path(parser.parse_args(argv).state_path).expanduser().resolve()
    if not (folder / "state.json").is_file():
        raise SystemExit(f"no state.json in {folder}")
    return folder
