"""One command: add or retune a square partial-swap operation on one qubit pair.

A partial swap is what ``ISwapImplementation`` plays: ONE pulse name carried by the
pair's control-qubit z line and by its coupler, plus the macro that names it. That is
three entries of the ACTIVE setup's QUAM tree, tied together by a naming contract
(``SCQO/procedures/pair-partial-swap``, Step 2):

* ``qubit_control.z.operations["partial_swap_square_<t>"]`` - a ``SquarePulse`` that
  brings the control qubit into resonance with its partner (``--z-amp``)
* ``coupler.operations["partial_swap_square_<t>"]`` - a ``SquarePulse`` riding on the
  coupler's standing bias; its amplitude sets the angle (``--coupler-amp``)
* ``macros["partial_swap_<t>"]`` - ``ISwapImplementation(flux_pulse=<that name>)``

``<t>`` is the target angle x 100 in three digits (0.30 rad -> ``partial_swap_030``).
Experiments play the macro by name (``swap_operation=partial_swap_030``).

Run it (in ``.venv-qm``)::

    python -m scqo_qm.backend.register_partial_swap --pair q1_q2 --name partial_swap_030 \\
        --z-amp -0.1500 --coupler-amp 0.0875
    python -m scqo_qm.backend.register_partial_swap --pair q1_q2 --name partial_swap_030 \\
        --update --z-amp -0.14987         # retune: the amplitudes only
    python -m scqo_qm.backend.register_partial_swap --list    # what the pairs carry

The write is surgical by PROOF, not by intent. The edited tree is saved to a temporary
folder first and compared with the live files, and the live ``state.json`` is replaced
only when the two differ by exactly these three entries. Anything else refuses and
leaves the live folder untouched - including a ``state.json`` that changed on disk after
this command loaded it (an accepted suggestion, another operator's edit), which a plain
save would silently overwrite. An amplitude the port cannot emit is refused too: the
session-start flux-headroom audit would otherwise block every later run.

No separate backup: every ``scqo run`` snapshots the vendor config it ran against
(``<device>/setup_snapshots/``, rebuilt by ``scqo restore``), so the tree before this
edit is the last run's snapshot. Do not run it while a measurement is running: that
run's record would report the edit as setup-snapshot drift. Fully OFFLINE -
``build_session`` loads the QUAM from JSON and never opens a QuantumMachinesManager.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from scqo_qm.quam_io import save_state

#: ``partial_swap_<t>``, t = the target angle x 100 in three digits.
NAME_PATTERN = re.compile(r"partial_swap_(\d{3})")

#: the two files ``machine.save()`` splits a tree into.
FILES = ("state.json", "wiring.json")

#: pulse length of a new operation when ``--length`` is not given (5Q4C: 40 ns).
DEFAULT_LENGTH_NS = 40

_MISSING = object()


def pulse_name(name: str) -> str:
    """``partial_swap_<t>`` -> ``partial_swap_square_<t>``; refuses any other name."""
    match = NAME_PATTERN.fullmatch(name)
    if match is None:
        raise SystemExit(
            f"{name!r} is not a partial-swap operation name: use partial_swap_<t>, <t> "
            f"the target angle x 100 in three digits (0.30 rad -> partial_swap_030)"
        )
    return f"partial_swap_square_{match.group(1)}"


def _pair(machine: Any, pair: str) -> Any:
    try:
        return machine.qubit_pairs[pair]
    except KeyError:
        raise SystemExit(
            f"no qubit pair {pair!r} in the tree; it has {', '.join(machine.qubit_pairs)}"
        ) from None


def _qubit_name(qubit: Any) -> str:
    return getattr(qubit, "name", None) or qubit.id


def list_partial_swaps(machine: Any, pair: str | None = None) -> list[dict[str, Any]]:
    """Every ``partial_swap*`` macro on ``pair`` (all pairs when None) with the two
    pulses it plays. ``managed`` is False for a macro outside the naming contract - a
    legacy name, or one that plays some other pulse - which this command neither
    retunes nor repairs."""
    pairs = {pair: _pair(machine, pair)} if pair is not None else dict(machine.qubit_pairs.items())
    rows = []
    for pair_name, qp in pairs.items():
        control = qp.qubit_control
        for macro_name, macro in (qp.macros or {}).items():
            if not macro_name.startswith("partial_swap"):
                continue
            flux_pulse = getattr(macro, "flux_pulse", None)
            known = isinstance(flux_pulse, str)
            z_op = control.z.operations.get(flux_pulse) if known else None
            c_op = qp.coupler.operations.get(flux_pulse) if known else None
            contract = NAME_PATTERN.fullmatch(macro_name) is not None
            rows.append({
                "pair": pair_name,
                "name": macro_name,
                "control": _qubit_name(control),
                "flux_pulse": flux_pulse,
                "z_amp": getattr(z_op, "amplitude", None),
                "coupler_amp": getattr(c_op, "amplitude", None),
                "length_ns": getattr(z_op, "length", None),
                "managed": bool(contract and flux_pulse == pulse_name(macro_name)
                                and z_op is not None and c_op is not None),
            })
    return rows


def _strip(state: dict, pair: str, control: str, name: str, pulse: str) -> dict:
    """``state`` without the operation's three entries (a copy)."""
    tree = copy.deepcopy(state)
    qp = tree.get("qubit_pairs", {}).get(pair, {})
    qp.get("macros", {}).pop(name, None)
    qp.get("coupler", {}).get("operations", {}).pop(pulse, None)
    tree.get("qubits", {}).get(control, {}).get("z", {}).get("operations", {}).pop(pulse, None)
    return tree


def _differences(a: Any, b: Any, path: str = "", limit: int = 5) -> list[str]:
    """Up to ``limit`` JSON paths at which ``a`` and ``b`` differ."""
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return [path or "/"]
    found: list[str] = []
    for key in sorted(set(a) | set(b), key=str):
        x, y = a.get(key, _MISSING), b.get(key, _MISSING)
        if x != y:
            found += _differences(x, y, f"{path}/{key}", limit - len(found))
            if len(found) >= limit:
                break
    return found[:limit]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def register_partial_swap(
    pair: str,
    name: str,
    *,
    z_amp: float | None = None,
    coupler_amp: float | None = None,
    length_ns: int | None = None,
    update: bool = False,
    config_path: str | None = None,
    session: Any = None,
    cfg: Any = None,
    state_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Add (or, with ``update``, retune) the partial-swap operation ``name`` on ``pair``.

    Resolves the ACTIVE scqo selection unless ``session`` is injected (tests pass one
    with ``state_dir``). Refuses BY NAME before anything is written: a name outside the
    ``partial_swap_<t>`` contract, an unknown pair, adding an operation that exists or
    retuning one that does not, a missing or non-finite amplitude, a length QM cannot
    play, and an amplitude the port would clip. Then saves the edited tree to a
    temporary folder and replaces the live ``state.json`` only when it differs from the
    file on disk by exactly the three entries (``wiring.json`` must not change at all
    and is never rewritten). OFFLINE.

    Returns ``pair``, ``name``, ``pulse``, ``control``, ``action`` (added|retuned),
    ``before`` / ``after`` (``{"z_amp", "coupler_amp"}``), ``length_ns``,
    ``state_dir`` and ``saved``.
    """
    pulse = pulse_name(name)
    for flag, value in (("--z-amp", z_amp), ("--coupler-amp", coupler_amp)):
        if value is not None and not math.isfinite(value):
            raise SystemExit(f"{flag} {value}: not a finite number")

    if session is None:
        from scqo.cli import build_session  # lazy: keep module import scqo-free

        session, cfg = build_session(config_path)
    if state_dir is None and cfg is not None:
        from scqo.datastore import setup_backend_config_dir

        state_dir = setup_backend_config_dir(
            cfg.data_root, cfg.device, session.cooldown_id, session.setup_name
        )
    if state_dir is None:
        raise SystemExit("no setup folder to write: pass state_dir, or run with a scqo config")
    state_dir = Path(state_dir)

    from scqo_qm.quam_fields import flux_headroom_problems

    machine = session.backend.machine
    qp = _pair(machine, pair)
    if getattr(qp, "coupler", None) is None:
        raise SystemExit(f"pair {pair!r} has no coupler; a partial swap plays on it")
    control = _qubit_name(qp.qubit_control)
    z_ops, c_ops = qp.qubit_control.z.operations, qp.coupler.operations
    present = {"macro": name in qp.macros, "z pulse": pulse in z_ops,
               "coupler pulse": pulse in c_ops}
    clipping_before = set(flux_headroom_problems(machine))

    if update:
        missing = [what for what, there in present.items() if not there]
        if missing:
            raise SystemExit(f"{name} on {pair} has no {', '.join(missing)}; drop --update to add it")
        if z_amp is None and coupler_amp is None:
            raise SystemExit("--update needs --z-amp and/or --coupler-amp")
        if length_ns is not None:
            raise SystemExit("--length applies to a new operation; retuning changes the amplitudes only")
        plays = getattr(qp.macros[name], "flux_pulse", None)
        if plays != pulse:
            raise SystemExit(
                f"{name} on {pair} plays {plays!r}, not {pulse!r}: it is outside this "
                f"command's naming contract, so it is not retuned here"
            )
        before = {"z_amp": z_ops[pulse].amplitude, "coupler_amp": c_ops[pulse].amplitude}
        if z_amp is not None:
            z_ops[pulse].amplitude = float(z_amp)
        if coupler_amp is not None:
            c_ops[pulse].amplitude = float(coupler_amp)
        length = z_ops[pulse].length
        action = "retuned"
    else:
        taken = [what for what, there in present.items() if there]
        if taken:
            raise SystemExit(f"{name} already has a {', '.join(taken)} on {pair}; use --update to retune it")
        if z_amp is None or coupler_amp is None:
            raise SystemExit("a new operation needs both --z-amp and --coupler-amp")
        length = DEFAULT_LENGTH_NS if length_ns is None else int(length_ns)
        if length < 16 or length % 4:
            raise SystemExit(f"--length {length} ns: QM plays a multiple of 4 ns, at least 16 ns")
        from quam.components.pulses import SquarePulse

        from scqo_qm.components.macros.iswap_macro import ISwapImplementation

        before = {"z_amp": None, "coupler_amp": None}
        z_ops[pulse] = SquarePulse(amplitude=float(z_amp), length=length)
        c_ops[pulse] = SquarePulse(amplitude=float(coupler_amp), length=length)
        qp.macros[name] = ISwapImplementation(flux_pulse=pulse)
        action = "added"
    after = {"z_amp": z_ops[pulse].amplitude, "coupler_amp": c_ops[pulse].amplitude}

    clipping = [p for p in flux_headroom_problems(machine) if p not in clipping_before]
    if clipping:
        raise SystemExit(
            "refused, the live folder is untouched: every later session would stop at its "
            "flux-headroom audit:\n  - " + "\n  - ".join(clipping)
        )

    with tempfile.TemporaryDirectory(prefix="register_partial_swap_") as tmp:
        staged = Path(tmp)
        save_state(machine, str(staged))
        written = sorted(p.name for p in staged.iterdir())
        if written != sorted(FILES):
            raise SystemExit(f"QUAM wrote {written}, expected {sorted(FILES)}; the live folder is untouched")
        live = {n: _read_json(state_dir / n) for n in FILES}
        new = {n: _read_json(staged / n) for n in FILES}
        if live["wiring.json"] != new["wiring.json"]:
            raise SystemExit(
                "wiring.json would change at " + ", ".join(_differences(live["wiring.json"], new["wiring.json"]))
                + "; the live folder is untouched"
            )
        rest_live = _strip(live["state.json"], pair, control, name, pulse)
        rest_new = _strip(new["state.json"], pair, control, name, pulse)
        if rest_live != rest_new:
            raise SystemExit(
                f"state.json would change beyond {name} at "
                + ", ".join(_differences(rest_live, rest_new))
                + f" - most likely {state_dir / 'state.json'} changed on disk after this command "
                f"loaded it; the live folder is untouched, run the command again"
            )
        saved_state = new["state.json"]
        try:
            macro = saved_state["qubit_pairs"][pair]["macros"][name]
            saved_z = saved_state["qubits"][control]["z"]["operations"][pulse]["amplitude"]
            saved_c = saved_state["qubit_pairs"][pair]["coupler"]["operations"][pulse]["amplitude"]
        except KeyError as err:
            raise SystemExit(f"the saved tree lacks {err}; the live folder is untouched") from None
        if (macro.get("flux_pulse") != pulse or not str(macro.get("__class__", "")).endswith(".ISwapImplementation")
                or (saved_z, saved_c) != (after["z_amp"], after["coupler_amp"])):
            raise SystemExit(f"the saved tree does not carry {name} as written; the live folder is untouched")

        if not dry_run:
            # Replace, never truncate-and-write: a crash mid-copy must not leave a
            # half-written state.json behind. The partial name is not *.json, so a
            # stray one never loads as part of the tree.
            partial = state_dir / ".state.json.register_partial_swap"
            shutil.copyfile(staged / "state.json", partial)
            os.replace(partial, state_dir / "state.json")

    return {
        "pair": pair,
        "name": name,
        "pulse": pulse,
        "control": control,
        "action": action,
        "before": before,
        "after": after,
        "length_ns": length,
        "state_dir": str(state_dir),
        "saved": not dry_run,
    }


def _volts(value: Any) -> str:
    return "-" if value is None else f"{value:.6g} V"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m scqo_qm.backend.register_partial_swap",
        description="Add or retune a square partial-swap operation (control z pulse + coupler "
        "pulse + ISwapImplementation macro) on one qubit pair of the ACTIVE scqo device/setup.",
    )
    p.add_argument("--pair", help="roster pair name, e.g. q1_q2")
    p.add_argument("--name", help="partial_swap_<t>, <t> = target angle x 100 in three digits "
                   "(0.30 rad -> partial_swap_030)")
    p.add_argument("--z-amp", type=float, metavar="V",
                   help="control-qubit z pulse amplitude (V): the swap resonance")
    p.add_argument("--coupler-amp", type=float, metavar="V",
                   help="coupler pulse amplitude (V), riding on the coupler's standing bias: sets the angle")
    p.add_argument("--length", type=int, metavar="NS",
                   help=f"pulse length of a NEW operation (default {DEFAULT_LENGTH_NS} ns)")
    p.add_argument("--update", action="store_true", help="retune the amplitudes of an EXISTING operation")
    p.add_argument("--list", action="store_true",
                   help="show the partial swaps the pairs carry (all pairs, or --pair); writes nothing")
    p.add_argument("--dry-run", action="store_true",
                   help="stage and verify the edit; leave the live folder untouched")
    p.add_argument("--config", default=None, help="scqo config.toml path (default: active selection)")
    args = p.parse_args(argv)

    if args.list:
        extra = [flag for flag, given in (
            ("--name", args.name is not None), ("--z-amp", args.z_amp is not None),
            ("--coupler-amp", args.coupler_amp is not None), ("--length", args.length is not None),
            ("--update", args.update), ("--dry-run", args.dry_run)) if given]
        if extra:
            p.error(f"--list takes only --pair and --config, not {' '.join(extra)}")
        from scqo.cli import build_session

        session, _cfg = build_session(args.config)
        rows = list_partial_swaps(session.backend.machine, args.pair)
        if not rows:
            print(f"no partial_swap operations on {args.pair or 'any pair'}")
        for r in rows:
            if r["managed"]:
                print(f"{r['pair']}  {r['name']}  control {r['control']}  z {_volts(r['z_amp'])}  "
                      f"coupler {_volts(r['coupler_amp'])}  {r['length_ns']} ns")
            else:
                print(f"{r['pair']}  {r['name']}  control {r['control']}  plays {r['flux_pulse']!r} - "
                      f"outside the partial_swap_<t> contract, not managed here")
        return 0

    if not args.pair or not args.name:
        p.error("--pair and --name are required (or --list)")
    out = register_partial_swap(
        args.pair,
        args.name,
        z_amp=args.z_amp,
        coupler_amp=args.coupler_amp,
        length_ns=args.length,
        update=args.update,
        config_path=args.config,
        dry_run=args.dry_run,
    )
    verb = {"added": "would add", "retuned": "would retune"}[out["action"]] if args.dry_run else out["action"]
    print(f"{out['pair']} {out['name']} (control {out['control']}, pulse {out['pulse']}): "
          f"{verb}, {out['length_ns']} ns")
    for label, key in (("z      ", "z_amp"), ("coupler", "coupler_amp")):
        old, new = out["before"][key], out["after"][key]
        change = _volts(new) if old is None else (
            f"{old:.6g} -> {_volts(new)}" if old != new else f"{_volts(new)} (unchanged)")
        print(f"    {label} {change}")
    if args.dry_run:
        print(f"  --dry-run: verified against {out['state_dir']}, nothing written")
    else:
        print(f"  saved to: {out['state_dir']} (only this operation changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
