"""Point a QM device config's qubits at the Thermalizing*Transmon classes.

scqo v0.14 owns the passive-reset wait as the neutral drive-channel knob
``thermalization_time_s``. QUAM's stock ``thermalization_time`` is a READ-ONLY
derived property (``thermalization_time_factor * T1``), so an absolute wait has
nowhere to live until a device's qubits name a class that stores one — see
``scqo_qm/quam_builder/architecture/superconducting/qubit/thermalizing_transmon.py``.
Until a config is migrated, ``scqo set <q>.thermalization_time_s=...`` raises
NotImplementedError naming this script, and the derived factor*T1 keeps working.

Each qubit serializes its own ``__class__``, so this is a per-device-config
edit. It is NOT the ``quam_config/my_quam.py`` ``qubit_type`` toggle, which is
the lab's charge-tunable switch and is governed separately.

Safe by construction: targeted STRING replacement (not a JSON round-trip), so
the file keeps its formatting and the diff is exactly the class lines; the
original is copied to ``<name>.pre-thermalizing.bak`` first; the root
``__class__`` and the top-level key set are asserted unchanged afterwards; and
re-running on a migrated file is a no-op.

Usage:
    python scripts/migrate_thermalizing_transmon.py <backend_config_dir> [...]
    python scripts/migrate_thermalizing_transmon.py --dry-run <dir>

e.g. python scripts/migrate_thermalizing_transmon.py \
         D:/qpu_data_dev/chipA/cd1/qm_OPX1000/backend_config

Verify afterwards by loading the machine and checking one qubit:
    q.thermalization_time            # still factor*T1 until scqo writes one
    q.thermalization_time_ns = 2e5   # now settable
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

_BASE = "quam_builder.architecture.superconducting.qubit"
_NEW = ("scqo_qm.quam_builder.architecture.superconducting.qubit"
        ".thermalizing_transmon")

#: stock QUAM transmon -> the storing subclass that replaces it
REWRITE = {
    f"{_BASE}.fixed_frequency_transmon.FixedFrequencyTransmon":
        f"{_NEW}.ThermalizingFixedFrequencyTransmon",
    f"{_BASE}.flux_tunable_transmon.FluxTunableTransmon":
        f"{_NEW}.ThermalizingFluxTunableTransmon",
}


def migrate(state_path: pathlib.Path, *, dry_run: bool = False) -> bool:
    """Migrate one ``state.json``. Returns True when it changed."""
    text = state_path.read_text(encoding="utf-8")
    before = json.loads(text)
    root = before.get("__class__")
    qubits = before.get("qubits", {})

    print(f"\n=== {state_path}")
    print(f"    root __class__ : {root}")
    print(f"    qubits         : {len(qubits)}")

    if any(new in text for new in REWRITE.values()):
        print("    already migrated - nothing to do")
        return False

    new_text, total = text, 0
    for old, new in REWRITE.items():
        n = new_text.count(f'"{old}"')
        if n:
            new_text = new_text.replace(f'"{old}"', f'"{new}"')
            total += n
            print(f"    {old.rsplit('.', 1)[-1]} -> {new.rsplit('.', 1)[-1]}  x{n}")

    if total == 0:
        found = sorted({q.get("__class__", "?") for q in qubits.values()})
        print(f"    NOTHING REWRITTEN - unrecognised qubit classes: {found}")
        return False

    if dry_run:
        print("    --dry-run: not written")
        return False

    backup = state_path.with_suffix(".json.pre-thermalizing.bak")
    shutil.copy(state_path, backup)
    print(f"    backup         : {backup.name}")
    state_path.write_text(new_text, encoding="utf-8", newline="")

    after = json.loads(state_path.read_text(encoding="utf-8"))
    assert after.get("__class__") == root, "ROOT __class__ changed - restore the backup"
    assert set(after) == set(before), "top-level keys changed - restore the backup"
    assert len(after.get("qubits", {})) == len(qubits), "qubit count changed"
    print("    OK - root class and structure unchanged")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folders", nargs="+", metavar="BACKEND_CONFIG_DIR",
                        help="a setup's backend_config/ folder (holding state.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change and write nothing")
    args = parser.parse_args(argv)

    changed = 0
    for raw in args.folders:
        folder = pathlib.Path(raw)
        state = folder / "state.json" if folder.is_dir() else folder
        if not state.is_file():
            print(f"\n=== {raw}\n    NOT FOUND: {state}", file=sys.stderr)
            return 2
        changed += bool(migrate(state, dry_run=args.dry_run))

    print(f"\n{changed} config(s) migrated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
