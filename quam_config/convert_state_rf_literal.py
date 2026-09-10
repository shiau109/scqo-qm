"""Make ``RF_frequency`` a literal in a QUAM state folder, so writebacks work.

Per qubit ``xy`` + ``resonator`` channel, swap::

    RF_frequency:           "#./inferred_RF_frequency"  -> literal (= LO + IF)
    intermediate_frequency: <literal>                   -> "#./inferred_intermediate_frequency"

QUAM defaults ``RF_frequency`` to the ``inferred`` reference on BOTH families
(``IQChannel`` and ``MWChannel`` declare it identically). A tree that keeps that
default reads perfectly and refuses every write: QUAM will not overwrite a
reference with a literal, so ``set_drive_freq`` / ``set_readout_freq`` raise
mid-run, at writeback, after the instrument time is spent. This is the conversion
``scqo_qm.quam_fields.rf_frequency_reference_problems`` points at.

Family-agnostic on purpose: it touches ``RF_frequency`` and
``intermediate_frequency`` only, both of which live on the shared channel base,
so an Octave tree converts exactly like an MW-FEM one.

Usage::

    python quam_config/convert_state_rf_literal.py <state folder> [--dry-run]

The folder is the setup's ``backend_config/`` (holding ``state.json`` +
``wiring.json``) — REQUIRED, never guessed. It used to be a hardcoded path into a
repo that has since been renamed, which is a good reason for a one-off script to
take its target as an argument.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "state_path",
        help="the QUAM state folder to convert (a setup's backend_config/)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report what would change and write nothing")
    args = parser.parse_args(argv)

    folder = Path(args.state_path).expanduser().resolve()
    if not (folder / "state.json").is_file():
        print(f"no state.json in {folder}", file=sys.stderr)
        return 2

    from quam_config import Quam

    machine = Quam.load(str(folder))
    converted = 0
    for qname, qubit in machine.qubits.items():
        for line in ("xy", "resonator"):
            channel = getattr(qubit, line, None)
            if channel is None:
                continue
            try:
                raw = channel.get_raw_value("RF_frequency")
            except Exception:
                raw = None
            if not (isinstance(raw, str) and raw.startswith("#")):
                print(f"{qname}.{line}: already a literal, left alone")
                continue
            rf = float(channel.RF_frequency)  # quam resolves inferred = LO + IF
            converted += 1
            if args.dry_run:
                print(f"{qname}.{line}: would set RF_frequency = {rf}")
                continue
            channel.RF_frequency = None  # release the reference first
            channel.RF_frequency = rf
            channel.intermediate_frequency = "#./inferred_intermediate_frequency"
            print(f"{qname}.{line}: RF_frequency = {rf}")

    if args.dry_run:
        print(f"dry run: {converted} channel(s) would change; nothing written")
        return 0
    if not converted:
        print(f"nothing to convert in {folder}")
        return 0
    machine.save()
    print(f"converted {converted} channel(s); saved -> {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
