"""Migrate QUAM state.json class paths: ``customized.*`` -> ``scqo_qm.*``.

The v1 restructure renamed the Python package ``customized`` to ``scqo_qm``.
QUAM serializes class references as dotted import paths in ``state.json``
(``__class__`` keys), and on an import failure it does NOT crash — it warns and
falls back to the annotated base type (``quam/core/quam_instantiation.py``,
"Falling back to ..."), i.e. a device that loads with the wrong pulse shape or
transmon behavior. That is why this script verifies POSITIVELY: exact per-class
rewrite counts, a load with the fallback warning escalated, a ``to_dict()``
round-trip comparing the serialized class multiset against the file, and a
``generate_config()`` build.

Usage (run from the repo root, in a venv holding this repo + quam):
  python scripts/migrate_state_scqo_qm.py --dry-run <paths...>
  python scripts/migrate_state_scqo_qm.py <paths...>
  python scripts/migrate_state_scqo_qm.py --verify-only <paths...>
  python scripts/migrate_state_scqo_qm.py --reverse <paths...>   # rollback

Each path is a ``state.json`` file or a folder containing one. ``wiring.json``
carries no class paths and is never touched. Backups: ``*.pre-scqo-qm.bak``
(forward) / ``*.pre-rollback.bak`` (reverse); ``.bak`` files are never edited.
A file with zero occurrences reports "already clean" — that IS the census for
states that never carried lab classes (e.g. state_lib/2Q).

Use ``--lenient-generate`` for library states (state_lib/*): their
``generate_config()`` may legitimately fail on partial trees; an import-level
or fallback failure is ALWAYS fatal regardless.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# The exact persisted dotted paths (the cross-agent contract with the code
# restructure). If a later stage relocates any of these MODULES, change this
# dict, not the algorithm.
MAPPING = {
    "customized.quam_builder.architecture.superconducting.qpu.mixed_quam.MixedTransmonQuam":
        "scqo_qm.quam_builder.architecture.superconducting.qpu.mixed_quam.MixedTransmonQuam",
    "customized.quam_builder.architecture.superconducting.qubit.thermalizing_transmon.ThermalizingFluxTunableTransmon":
        "scqo_qm.quam_builder.architecture.superconducting.qubit.thermalizing_transmon.ThermalizingFluxTunableTransmon",
    "customized.quam_builder.architecture.superconducting.qubit.thermalizing_transmon.ThermalizingFixedFrequencyTransmon":
        "scqo_qm.quam_builder.architecture.superconducting.qubit.thermalizing_transmon.ThermalizingFixedFrequencyTransmon",
    "customized.components.pulses.FlatTopCosinePulse":
        "scqo_qm.components.pulses.FlatTopCosinePulse",
    "customized.components.macros.iswap_macro.ISwapImplementation":
        "scqo_qm.components.macros.iswap_macro.ISwapImplementation",
    "customized.components.macros.parametric_reset_macro.ParametricReset":
        "scqo_qm.components.macros.parametric_reset_macro.ParametricReset",
}

FALLBACK_MARKERS = ("Could not load class", "Falling back")


def state_file(path: Path) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "state.json"
    if not p.is_file():
        raise SystemExit(f"no state.json at {path}")
    if p.name != "state.json":
        raise SystemExit(f"refusing non-canonical name: {p} (backups are exempt by design)")
    return p


def class_multiset(obj) -> Counter:
    out: Counter = Counter()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "__class__" and isinstance(v, str):
                out[v] += 1
            else:
                out += class_multiset(v)
    elif isinstance(obj, list):
        for v in obj:
            out += class_multiset(v)
    return out


def rewrite(p: Path, mapping: dict, backup_suffix: str, dry: bool,
            leftover_prefix: str) -> int:
    text = p.read_text(encoding="utf-8")
    before = {old: text.count(f'"{old}"') for old in mapping}
    total = sum(before.values())
    for old, n in sorted(before.items()):
        if n:
            print(f"    {n:3d} x {old.split('.')[-1]}  ({old.split('.')[0]}.-prefixed)")
    if total == 0:
        print("    already clean — nothing to do")
        return 0
    if dry:
        print(f"    DRY RUN: would rewrite {total} occurrence(s)")
        return total

    bak = p.with_name(p.name + backup_suffix)
    shutil.copy2(p, bak)
    new_text = text
    for old, new in mapping.items():
        new_text = new_text.replace(f'"{old}"', f'"{new}"')
    p.write_text(new_text, encoding="utf-8")

    # positive assertions on the written file
    written = p.read_text(encoding="utf-8")
    json.loads(written)  # still valid JSON
    for old, new in mapping.items():
        assert written.count(f'"{new}"') >= before[old], (old, new)
        assert f'"{old}"' not in written, f"needle survived: {old}"
    assert f'"{leftover_prefix}' not in written, (
        f"unmapped {leftover_prefix}* class path remains in {p} — extend MAPPING")
    print(f"    rewrote {total} occurrence(s); backup: {bak.name}")
    return total


def verify(p: Path, expect_prefix: str, lenient_generate: bool) -> None:
    """Load + round-trip + generate_config with the silent fallback escalated."""
    import quam_config

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        machine = quam_config.Quam.load(str(p.parent))
        bad = [w for w in caught
               if any(m in str(w.message) for m in FALLBACK_MARKERS)]
        if bad:
            raise SystemExit(
                f"FATAL: QUAM fell back to base classes loading {p}:\n  "
                + "\n  ".join(str(w.message) for w in bad))

    file_classes = Counter(
        v for v in class_multiset(json.loads(p.read_text(encoding="utf-8"))).elements()
        if v.startswith(expect_prefix))
    dumped_classes = Counter(
        v for v in class_multiset(machine.to_dict()).elements()
        if v.startswith(expect_prefix))
    if file_classes != dumped_classes:
        raise SystemExit(
            f"FATAL: round-trip class multiset mismatch for {p}:\n"
            f"  file:   {dict(file_classes)}\n  to_dict: {dict(dumped_classes)}")

    try:
        machine.generate_config()
        print(f"    verify OK: load + round-trip ({sum(file_classes.values())} "
              f"{expect_prefix}* refs) + generate_config")
    except Exception as e:
        if lenient_generate:
            print(f"    verify: load + round-trip OK; generate_config WARN "
                  f"(lenient): {type(e).__name__}: {e}")
        else:
            raise SystemExit(
                f"FATAL: generate_config failed for {p}: {type(e).__name__}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reverse", action="store_true",
                    help="rollback: scqo_qm.* -> customized.* — NO LONGER USABLE, the "
                         "customized package was deleted after v3.13.0; only a checkout "
                         "at that tag or earlier can load what this writes")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--lenient-generate", action="store_true",
                    help="library states: generate_config failure is a WARN, not fatal")
    args = ap.parse_args()

    if args.reverse:
        mapping = {v: k for k, v in MAPPING.items()}
        backup_suffix, leftover, expect = ".pre-rollback.bak", '"scqo_qm.', "customized."
    else:
        mapping = MAPPING
        backup_suffix, leftover, expect = ".pre-scqo-qm.bak", '"customized.', "scqo_qm."

    grand = 0
    for raw in args.paths:
        p = state_file(Path(raw))
        print(f"== {p}")
        if not args.verify_only:
            grand += rewrite(p, mapping, backup_suffix, args.dry_run,
                             leftover.strip('"'))
        if not args.dry_run and not args.no_verify:
            verify(p, expect, args.lenient_generate)
    print(f"TOTAL rewritten: {grand}")


if __name__ == "__main__":
    main()
