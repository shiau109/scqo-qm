"""One command: ADOPT a borrowed drive channel - give it an element in the QUAM tree.

SCQO mints a BORROWED drive channel for every drivable mode a drive line reaches
without carrying it by design: ``xy2.q1_q2_c`` is the coupler q1_q2_c driven through
q2's xy wire. Its knobs (``pi_amp``, ``drive_freq_hz``, ...) are real only once the
vendor config holds an element that plays it. This command creates that element in
the ACTIVE setup's tree (SCQO docs/coupler-transmon-plan.md section 2):

* ``machine.borrowed_channels[<address>]`` - a plain ``MWChannel`` whose ``id`` IS the
  address (and so its QM element name), on the line's port - the port of the line's
  designed drive channel - on that port's SECOND upconverter, tuned to the target's
  measured ``f_01_hz`` (or ``--freq-hz``), with ``x180`` / ``x90`` cosine pulses
  (``DragCosinePulse``, DRAG off) of ``--length-ns`` and amplitudes ``--pi-amp`` / half;
* the port, converted to the two-upconverter spelling
  (``upconverters = {1: <its LO>, 2: --lo-hz}``, ``upconverter_frequency`` None - qm-qua
  refuses both), and moved with its MW-FEM port-pair partner to the lowest band that
  holds both LOs when its own band does not. The partner keeps its LO; it is never
  parked (a permanent setting, unlike a coupler tone's run-scoped move).

Run it (in ``.venv-qm``)::

    scqo-qm adopt-channel --list                       # look: ports, LOs, adopted channels
    scqo-qm adopt-channel xy2.q1_q2_c --lo-hz 7.1e9    # 5Q4C: IF -43.05 MHz
    scqo-qm adopt-channel xy2.q2_q3_c --lo-hz 7.1e9    # shares upconverter 2: IF +55.50 MHz

The write is surgical by PROOF, as ``register_partial_swap``'s: the edited tree is
saved to a temporary folder and compiled (``generate_config``), and the live
``state.json`` is replaced only when it differs from the file on disk by exactly the
new element and the two ports' band / LO fields (``wiring.json`` never changes).
Anything else refuses and leaves the live folder untouched - including a
``state.json`` that changed on disk after this command loaded it.

The port's LO pair and band move for EVERY element on the pair (the line's own qubit
and its partner port's): run each qubit's power Rabi after adopting. The port's full
scale is shared too, so the adopted channel's ``pi_amp`` holds only at the current
one (BACKLOG I28). No separate backup - every ``scqo run`` snapshots the tree it ran
against, so ``scqo restore`` of any earlier run rebuilds the tree before this edit.
Do not run it while a measurement is running. MW-FEM only; fully OFFLINE.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from scqo_qm.quam_io import save_state

#: the two files ``machine.save()`` splits a tree into.
FILES = ("state.json", "wiring.json")

#: pulse length of the adopted x180/x90 (5Q4C couplers: 200 ns - plan doc 1.4).
DEFAULT_LENGTH_NS = 200

#: seed amplitude of the adopted x180 (the x90 gets half); a power Rabi calibrates it.
DEFAULT_PI_AMP = 0.25

#: the DragCosinePulse needs an anharmonicity even with DRAG off (alpha 0), where it
#: shapes nothing; the target's measured |anharmonicity_hz| replaces it when stored.
PLACEHOLDER_ANHARMONICITY_HZ = 200e6

_MISSING = object()


def _refuse(message: str) -> SystemExit:
    return SystemExit(f"adopt-channel: {message}; the live folder is untouched")


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


def _line_port(session: Any, line: str) -> tuple[Any, Any]:
    """(the line's designed drive element, its output port): every designed drive
    channel on the line must play through one QUAM element on one port."""
    from scqo_qm._mw_fem import port_info

    roster = session.roster
    designed = [c for c in roster.channels_on(line) if "drive" in c.kinds]
    if not designed:
        raise _refuse(f"line {line!r} carries no designed drive channel, so there is "
                      f"no port to put a borrowed one on")
    elements = []
    for ch in designed:
        try:
            elements.append(session.backend.device.component(ch.name).vendor)
        except KeyError as err:
            raise _refuse(f"{ch.name!r} is not realized by this tree ({err})") from None
    ports = {port_info(getattr(xy, "opx_output", None)) for xy in elements}
    if len(ports) != 1 or None in ports:
        raise _refuse(f"the designed drive channels on {line!r} do not share one MW-FEM "
                      f"port ({sorted(map(str, ports))})")
    return elements[0], elements[0].opx_output


def _partner_port(machine: Any, port: Any) -> Any | None:
    from scqo_qm._mw_fem import partner_port_id, port_info

    ctrl, fem, pid = port_info(port)
    try:
        return machine.ports.get_mw_output(ctrl, fem, partner_port_id(pid))
    except (KeyError, AttributeError):
        return None


def _port_path(port: Any) -> tuple[str, str, str]:
    """The port's keys under ``state.json`` ``ports.mw_outputs`` (JSON keys are strings)."""
    from scqo_qm._mw_fem import port_info

    ctrl, fem, pid = port_info(port)
    return str(ctrl), str(fem), str(pid)


def _strip(state: dict, address: str, ports: list[tuple[str, str, str]]) -> dict:
    """``state`` without the adopted element and the touched ports' band / LO
    fields (a copy) - what must be identical before and after."""
    tree = copy.deepcopy(state)
    borrowed = tree.get("borrowed_channels")
    if isinstance(borrowed, dict):
        borrowed.pop(address, None)
        if not borrowed:
            tree.pop("borrowed_channels")
    outputs = tree.get("ports", {}).get("mw_outputs", {})
    for ctrl, fem, pid in ports:
        entry = outputs.get(ctrl, {}).get(fem, {}).get(pid)
        if isinstance(entry, dict):
            for key in ("band", "upconverter_frequency", "upconverters"):
                entry.pop(key, None)
    return tree


def list_channels(session: Any) -> dict[str, list[dict[str, Any]]]:
    """``{"lines": [...], "adopted": [...]}``: every drive line's port, band, LOs,
    full scale and designed channels, and every adopted borrowed channel. Reads only."""
    from scqo_qm._mw_fem import port_los
    from scqo_qm.backend.qm_backend import port_label

    roster, machine = session.roster, session.backend.machine
    lines = []
    for line in sorted(roster.lines()):
        designed = [c.name for c in roster.channels_on(line) if "drive" in c.kinds]
        if not designed:
            continue
        row: dict[str, Any] = {"line": line, "designed": designed}
        try:
            xy = session.backend.device.component(designed[0]).vendor
            port = xy.opx_output
            row.update(port=port_label(xy), band=getattr(port, "band", None),
                       los=port_los(port),
                       full_scale_power_dbm=getattr(port, "full_scale_power_dbm", None))
        except Exception as err:  # a look must never fail
            row["unavailable"] = f"{type(err).__name__}: {err}"
        lines.append(row)
    adopted = []
    for address, ch in (getattr(machine, "borrowed_channels", None) or {}).items():
        ops = getattr(ch, "operations", {}) or {}
        adopted.append({
            "address": address,
            "port": port_label(ch),
            "upconverter": getattr(ch, "upconverter", None),
            "rf_hz": getattr(ch, "RF_frequency", None),
            "if_hz": getattr(ch, "intermediate_frequency", None),
            "pulses": {name: {"amplitude": getattr(op, "amplitude", None),
                              "length_ns": getattr(op, "length", None)}
                       for name, op in ops.items()},
        })
    return {"lines": lines, "adopted": adopted}


def adopt_channel(
    address: str,
    *,
    lo_hz: float,
    freq_hz: float | None = None,
    length_ns: int = DEFAULT_LENGTH_NS,
    pi_amp: float = DEFAULT_PI_AMP,
    config_path: str | None = None,
    session: Any = None,
    cfg: Any = None,
    state_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Adopt the borrowed drive channel ``address`` (``<line>.<target>``) on the
    ACTIVE setup: the element, the port's two-upconverter form and, when needed,
    the port pair's band. Resolves the active scqo selection unless ``session`` is
    injected (tests pass one with ``state_dir``). Refuses BY NAME before anything is
    written (see the module docstring for what is checked), then stages, compiles
    and verifies the edit and replaces the live ``state.json`` only when it differs
    from the file on disk by exactly what this command writes. OFFLINE.

    Returns ``address``, ``port``, ``band_before`` / ``band``, ``lo_hz``, ``rf_hz``,
    ``if_hz``, ``pi_amp``, ``length_ns``, ``partner`` (its port label or None),
    ``state_dir`` and ``saved``.
    """
    from scqo.entities import Channel

    from scqo_qm._family import RF_MW_FEM, rf_chain
    from scqo_qm._mw_fem import (
        MAX_IF_HZ,
        MW_FEM_BANDS,
        band_holds,
        port_los,
    )
    from scqo_qm.backend.qm_backend import port_label

    for flag, value in (("--lo-hz", lo_hz), ("--freq-hz", freq_hz), ("--pi-amp", pi_amp)):
        if value is not None and not math.isfinite(float(value)):
            raise _refuse(f"{flag} {value}: not a finite number")
    if not 0 < float(pi_amp) < 1:
        raise _refuse(f"--pi-amp {pi_amp}: a pulse amplitude is a fraction of full "
                      f"scale, in (0, 1)")
    length = int(length_ns)
    if length < 16 or length % 4:
        raise _refuse(f"--length-ns {length}: QM plays a multiple of 4 ns, at least 16 ns")

    if session is None:
        from scqo.cli import build_session  # lazy: keep module import scqo-free

        session, cfg = build_session(config_path)
    if state_dir is None and cfg is not None:
        from scqo.datastore import setup_backend_config_dir

        state_dir = setup_backend_config_dir(
            cfg.data_root, cfg.device, session.cooldown_id, session.setup_name)
    if state_dir is None:
        raise _refuse("no setup folder to write: pass state_dir, or run with a scqo config")
    state_dir = Path(state_dir)

    roster, machine = session.roster, session.backend.machine
    e = roster.entities.get(address)
    if not isinstance(e, Channel):
        raise _refuse(str(roster._unknown(address)) if e is None else
                      f"{address!r} is a {type(e).__name__.lower()}, not a channel")
    if not e.borrowed:
        raise _refuse(f"{address!r} is a DESIGNED channel - the tree realizes it through "
                      f"its target's own element, there is nothing to adopt")
    if "drive" not in e.kinds:
        raise _refuse(f"{address!r} is a {e.kind} channel; only drive channels are borrowed")
    adopted = getattr(machine, "borrowed_channels", None)
    if adopted is None:
        raise _refuse("this tree's root class has no borrowed_channels (it is not "
                      "scqo_qm's MixedTransmonQuam)")
    if address in adopted:
        raise _refuse(f"{address!r} is already adopted; retune it with scqo set "
                      f"({address}.pi_amp=..., .drive_freq_hz=..., .pi_duration_s=...)")
    target = e.target[0]

    line_xy, port = _line_port(session, e.line)
    if rf_chain(line_xy) != RF_MW_FEM:
        raise _refuse(f"line {e.line!r} is on {rf_chain(line_xy) or 'an unknown RF chain'}: "
                      f"adopting puts the channel on an MW-FEM port's second upconverter "
                      f"(an Octave output has one LO)")
    if int(getattr(line_xy, "upconverter", 1) or 1) != 1:
        raise _refuse(f"the designed element on {e.line!r} already plays on upconverter "
                      f"{line_xy.upconverter}; upconverter 2 is not free")
    if freq_hz is None:
        stored = session.physical.get(target, "f_01_hz") if session.physical is not None else None
        if stored is None:
            raise _refuse(f"no --freq-hz and no measured {target}.f_01_hz to tune "
                          f"{address} to")
        freq_hz = float(stored)
    freq_hz, lo_hz = float(freq_hz), float(lo_hz)
    if abs(freq_hz - lo_hz) > MAX_IF_HZ:
        raise _refuse(f"{freq_hz / 1e9:.6g} GHz is {(freq_hz - lo_hz) / 1e6:+.1f} MHz from "
                      f"--lo-hz {lo_hz / 1e9:.6g} GHz, past the +-{MAX_IF_HZ / 1e6:.0f} MHz "
                      f"IF window")

    los = port_los(port)
    if 1 not in los:
        raise _refuse(f"port {port_label(line_xy)} declares no LO for upconverter 1")
    if 2 in los and los[2] != lo_hz:
        raise _refuse(f"port {port_label(line_xy)} already runs upconverter 2 at "
                      f"{los[2] / 1e9:.6g} GHz (another adopted channel); a second channel "
                      f"on this port shares it: --lo-hz {los[2]:.6g}")
    band_before = int(port.band)
    if band_holds(band_before, los[1], lo_hz):
        band = band_before
    else:
        holding = [b for b in sorted(MW_FEM_BANDS) if band_holds(b, los[1], lo_hz)]
        if not holding:
            raise _refuse(f"no MW-FEM band holds both the port's LO {los[1] / 1e9:.6g} GHz "
                          f"and --lo-hz {lo_hz / 1e9:.6g} GHz")
        band = holding[0]
    partner = _partner_port(machine, port)
    if partner is not None and band != band_before:
        stranded = {n: lo for n, lo in port_los(partner).items() if not band_holds(band, lo)}
        if stranded:
            raise _refuse(f"band {band} cannot hold the port-pair partner's LO(s) "
                          f"{ {n: f'{lo / 1e9:.6g} GHz' for n, lo in stranded.items()} } - it "
                          f"would have to be parked, which a permanent setting never does")

    physical_alpha = (session.physical.get(target, "anharmonicity_hz")
                      if session.physical is not None else None)
    anharmonicity = (abs(float(physical_alpha)) if physical_alpha is not None
                     else PLACEHOLDER_ANHARMONICITY_HZ)

    from quam.components.channels import MWChannel
    from quam.components.pulses import DragCosinePulse

    if band != band_before:
        port.band = band
        if partner is not None:
            partner.band = band
    if 2 not in los:
        if getattr(port, "upconverters", None):
            port.upconverters[2] = {"frequency": lo_hz}
        else:
            port.upconverter_frequency = None
            port.upconverters = {1: {"frequency": los[1]}, 2: {"frequency": lo_hz}}

    def _pulse(amplitude: float) -> Any:
        return DragCosinePulse(length=length, amplitude=float(amplitude), axis_angle=0.0,
                               alpha=0.0, anharmonicity=anharmonicity, detuning=0.0)

    element = MWChannel(
        id=address,
        opx_output=port.get_reference(),
        upconverter=2,
        RF_frequency=freq_hz,
        intermediate_frequency="#./inferred_intermediate_frequency",
        core=getattr(line_xy, "core", None),
        operations={"x180": _pulse(pi_amp), "x90": _pulse(float(pi_amp) / 2)},
    )
    adopted[address] = element

    try:
        config = machine.generate_config()
    except Exception as err:
        raise _refuse(f"the edited tree does not compile ({type(err).__name__}: {err})") from None
    compiled = config.get("elements", {}).get(address, {})
    if compiled.get("MWInput", {}).get("upconverter") != 2:
        raise _refuse(f"the compiled element {address!r} does not play on upconverter 2 "
                      f"({compiled.get('MWInput')})")
    if_hz = float(element.intermediate_frequency)

    touched = [_port_path(port)] + ([_port_path(partner)] if partner is not None else [])
    with tempfile.TemporaryDirectory(prefix="adopt_channel_") as tmp:
        staged = Path(tmp)
        save_state(machine, str(staged))
        written = sorted(p.name for p in staged.iterdir())
        if written != sorted(FILES):
            raise _refuse(f"QUAM wrote {written}, expected {sorted(FILES)}")
        live = {n: _read_json(state_dir / n) for n in FILES}
        new = {n: _read_json(staged / n) for n in FILES}
        if live["wiring.json"] != new["wiring.json"]:
            raise _refuse("wiring.json would change at "
                          + ", ".join(_differences(live["wiring.json"], new["wiring.json"])))
        rest_live = _strip(live["state.json"], address, touched)
        rest_new = _strip(new["state.json"], address, touched)
        if rest_live != rest_new:
            raise _refuse(f"state.json would change beyond {address} at "
                          + ", ".join(_differences(rest_live, rest_new))
                          + f" - most likely {state_dir / 'state.json'} changed on disk after "
                          f"this command loaded it; run it again")
        saved = new["state.json"]
        try:
            ctrl, fem, pid = _port_path(port)
            saved_port = saved["ports"]["mw_outputs"][ctrl][fem][pid]
            saved_element = saved["borrowed_channels"][address]
        except KeyError as err:
            raise _refuse(f"the saved tree lacks {err}") from None
        ups = {int(k): float(v["frequency"]) for k, v in (saved_port.get("upconverters") or {}).items()}
        if (saved_port.get("upconverter_frequency") is not None or ups.get(2) != lo_hz
                or ups.get(1) != los[1] or saved_port.get("band") != band
                or saved_element.get("upconverter") != 2):
            raise _refuse(f"the saved tree does not carry {address} as written")

        if not dry_run:
            # Replace, never truncate-and-write (register_partial_swap's rule); the
            # partial name is not *.json, so a stray one never loads.
            partial = state_dir / ".state.json.adopt_channel"
            shutil.copyfile(staged / "state.json", partial)
            os.replace(partial, state_dir / "state.json")

    return {
        "address": address,
        "port": port_label(element),
        "band_before": band_before,
        "band": band,
        "lo_hz": lo_hz,
        "rf_hz": freq_hz,
        "if_hz": if_hz,
        "pi_amp": float(pi_amp),
        "length_ns": length,
        "partner": None if partner is None else "/".join(_port_path(partner)),
        "state_dir": str(state_dir),
        "saved": not dry_run,
    }


def _ghz(value: Any) -> str:
    return "-" if value is None else f"{float(value) / 1e9:.6g} GHz"


def main(argv: list[str] | None = None, prog: str = "scqo-qm adopt-channel") -> int:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Adopt a BORROWED drive channel (<line>.<target>, e.g. xy2.q1_q2_c): "
        "an element on the line's MW-FEM port, second upconverter, in the ACTIVE scqo "
        "device/setup's QUAM tree.")
    p.add_argument("address", nargs="?", help="the borrowed channel, <line>.<target>")
    p.add_argument("--lo-hz", type=float, metavar="HZ",
                   help="LO of the port's second upconverter (5Q4C couplers: 7.1e9)")
    p.add_argument("--freq-hz", type=float, metavar="HZ",
                   help="drive frequency (default: the target's measured f_01_hz)")
    p.add_argument("--length-ns", type=int, default=DEFAULT_LENGTH_NS, metavar="NS",
                   help=f"x180/x90 length (default {DEFAULT_LENGTH_NS} ns)")
    p.add_argument("--pi-amp", type=float, default=DEFAULT_PI_AMP, metavar="A",
                   help=f"x180 seed amplitude, x90 = half (default {DEFAULT_PI_AMP})")
    p.add_argument("--list", action="store_true",
                   help="show the drive lines' ports and the adopted channels; writes nothing")
    p.add_argument("--dry-run", action="store_true",
                   help="stage, compile and verify the edit; leave the live folder untouched")
    p.add_argument("--config", default=None, help="scqo config.toml path (default: active selection)")
    args = p.parse_args(argv)

    if args.list:
        extra = [flag for flag, given in (
            ("ADDRESS", args.address is not None), ("--lo-hz", args.lo_hz is not None),
            ("--freq-hz", args.freq_hz is not None), ("--dry-run", args.dry_run)) if given]
        if extra:
            p.error(f"--list takes only --config, not {' '.join(extra)}")
        from scqo.cli import build_session

        session, _cfg = build_session(args.config)
        out = list_channels(session)
        for r in out["lines"]:
            if "unavailable" in r:
                print(f"{r['line']:8s} {', '.join(r['designed'])}  ({r['unavailable']})")
                continue
            los = ", ".join(f"up{n} {_ghz(lo)}" for n, lo in sorted(r["los"].items()))
            print(f"{r['line']:8s} {r['port']:10s} band {r['band']}  {los}  full scale "
                  f"{r['full_scale_power_dbm']} dBm  designed: {', '.join(r['designed'])}")
        if not out["adopted"]:
            print("no borrowed channel is adopted in this tree")
        for a in out["adopted"]:
            pulses = "  ".join(f"{name} {v['amplitude']:.4g} x {v['length_ns']} ns"
                               for name, v in a["pulses"].items())
            print(f"adopted {a['address']}: {a['port']}  RF {_ghz(a['rf_hz'])}  IF "
                  f"{float(a['if_hz']) / 1e6:+.2f} MHz  {pulses}")
        return 0

    if not args.address or args.lo_hz is None:
        p.error("ADDRESS and --lo-hz are required (or --list)")
    out = adopt_channel(
        args.address, lo_hz=args.lo_hz, freq_hz=args.freq_hz, length_ns=args.length_ns,
        pi_amp=args.pi_amp, config_path=args.config, dry_run=args.dry_run)
    verb = "would adopt" if args.dry_run else "adopted"
    print(f"{out['address']}: {verb} on {out['port']}, RF {_ghz(out['rf_hz'])} (LO "
          f"{_ghz(out['lo_hz'])}, IF {out['if_hz'] / 1e6:+.2f} MHz), x180 {out['pi_amp']:.4g} x "
          f"{out['length_ns']} ns, x90 half")
    if out["band"] != out["band_before"]:
        print(f"    band {out['band_before']} -> {out['band']} on this port"
              + (f" and its pair partner {out['partner']}" if out["partner"] else "")
              + ": re-check the qubits on them (qubit_power_rabi)")
    if args.dry_run:
        print(f"  --dry-run: verified against {out['state_dir']}, nothing written")
    else:
        print(f"  saved to: {out['state_dir']} (only this channel and its port changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
