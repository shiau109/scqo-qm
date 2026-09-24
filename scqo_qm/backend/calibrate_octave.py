"""One command: calibrate the Octave mixers for the ACTIVE scqo device/setup.

An Octave up-converts a baseband IQ pair with an analog mixer, so every output
carries LO leakage and an image sideband until the mixer is calibrated against
the exact ``(RF output, LO, gain)`` and ``(that LO, IF)`` it will run at. An
MW-FEM synthesizes microwave directly and has no equivalent step, which is why
nothing in this driver ever called for one -- and why, before this command, an
Octave setup ran every measurement uncalibrated with nothing saying so.

Run it (in ``.venv-qm``)::

    scqo-qm calibrate-octave                 # every active qubit
    scqo-qm calibrate-octave --target q1 q2  # just these
    scqo-qm calibrate-octave --readout-only
    scqo-qm calibrate-octave --dry-run       # say what, calibrate nothing

**When to re-run it.** The results are cached in ``calibration_db.json``, keyed
by ``(RF output, LO frequency, gain)`` for the LO-leakage correction and by
``(that LO mode, IF)`` for the image correction. So re-run after ANY of:

* an LO change -- including one forced by a shared synthesizer (synth2 drives
  RF2 and RF3 from one source, so re-parking either moves both);
* a GAIN change. This is the one with no MW-FEM intuition behind it: changing
  ``full_scale_power_dbm`` on an MW-FEM needs no calibration at all, while gain
  is part of this cache key, so an absolute-power write that had to re-stage the
  gain has just invalidated the correction. ``_power.solve_octave_chain`` holds
  the gain wherever it can precisely to make that rare, and warns by name when it
  cannot;
* a new IF -- the vendor's ``calibrate_octave`` submits only the ONE IF the tree
  currently stores, so a channel parked at a new detuning is uncalibrated there;
* a cold start or a long thermal drift (each cached entry records the mixer
  temperature it was measured at).

**Where the results go.** ``machine.octaves[<name>].calibration_db_path``, which
is NOT part of ``state.json`` -- so a setup snapshot does not capture it, and two
runs with byte-identical QUAM state can have been taken with different mixer
corrections. Unset, quam falls back to ``os.getcwd()``, meaning the calibration
lands in whatever directory the process happened to start in and a later run
launched elsewhere silently finds none. This command always PRINTS the resolved
path, and warns when it is the fallback.

THIS LIVES IN THE DRIVER, NOT IN SCQO, the shape ``close_qm`` and
``apply_distortion`` already use: an operator command that needs the vendor
libraries belongs beside them, and a ``scqo calibrate-octave`` would be a verb
every other backend's operator could only ever be refused by.

Not destructive, but it does take the cluster: it opens a Quantum Machine and
plays tones on the calibrated elements. Do not run it while a measurement is
live - ``scqo-qm cluster`` shows whether one is.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from scqo_qm._family import RF_OCTAVE, rf_chain

#: Timeout (s) handed to ``qm_session`` while the calibration holds the cluster.
#: Generous on purpose: a full sweep is several elements times two calibrations,
#: and a timeout mid-calibration leaves one element done and the rest not.
DEFAULT_TIMEOUT_S = 600


def octave_lines(machine: Any, targets: list[str] | None = None) -> dict[str, dict]:
    """``{qubit: {"readout": bool, "drive": bool}}`` for the ACTIVE qubits.

    Per LINE rather than per qubit, because a tree may legitimately mix chains
    (an Octave readout beside an MW-FEM drive), and asking the vendor to
    calibrate an MW channel raises a RuntimeError that names the element but not
    the reason.
    """
    try:
        names = list(getattr(machine, "active_qubit_names", None)
                     or getattr(machine, "qubits", {}))
    except Exception:
        names = []
    if targets:
        unknown = [t for t in targets if t not in names]
        if unknown:
            raise SystemExit(
                f"not active qubits in this setup: {', '.join(unknown)} "
                f"(active: {', '.join(names) or 'none'})")
        names = [n for n in names if n in targets]

    out: dict[str, dict] = {}
    for name in names:
        qubit = machine.qubits[name]
        lines = {
            "readout": rf_chain(getattr(qubit, "resonator", None)) == RF_OCTAVE,
            "drive": rf_chain(getattr(qubit, "xy", None)) == RF_OCTAVE,
        }
        if lines["readout"] or lines["drive"]:
            out[name] = lines
    return out


def calibration_db(machine: Any) -> dict[str, Any]:
    """Where each Octave's mixer calibration is stored, and whether that was chosen.

    ``fallback`` is the finding: quam resolves an unset ``calibration_db_path``
    to ``os.getcwd()``, so the file follows the process's working directory
    rather than the setup.
    """
    out: dict[str, Any] = {}
    for name, octave in (getattr(machine, "octaves", {}) or {}).items():
        declared = getattr(octave, "calibration_db_path", None)
        out[name] = {
            "path": os.path.abspath(declared if declared else os.getcwd()),
            "fallback": not declared,
        }
    return out


def calibrate(*, targets: list[str] | None = None, drive: bool = True,
              readout: bool = True, config_path: str | None = None,
              dry_run: bool = False, timeout: float = DEFAULT_TIMEOUT_S,
              session: Any = None) -> dict[str, Any]:
    """Calibrate the active setup's Octave mixers; report what was done per line.

    ``session`` is injectable for tests; left None it resolves the active scqo
    selection exactly as ``scqo run`` does. Refuses BY NAME when the setup is not
    served by a QM backend, and when it is but nothing on it is on an Octave --
    a report of zero calibrations looks like success.
    """
    if not (drive or readout):
        raise SystemExit("--readout-only and --drive-only are mutually exclusive")

    if session is None:
        from scqo.cli import build_session  # lazy: keep module import scqo-free

        session, _cfg = build_session(config_path)

    backend = session.backend
    machine = getattr(backend, "machine", None)
    if machine is None:
        label = getattr(session, "backend_label", None) or type(backend).__name__
        raise SystemExit(
            f"the active setup is served by {label!r}, which has no QUAM tree "
            f"and no Octave to calibrate - this command is the QM backend's. "
            f"Check 'scqo user' for the selected device/setup.")

    lines = octave_lines(machine, targets)
    if not lines:
        raise SystemExit(
            "nothing on this setup is on an Octave: every active drive and "
            "readout channel synthesizes its own microwave (MW-FEM), which "
            "carries no mixer and needs no calibration. Nothing to do.")

    planned = {
        name: sorted(line for line, on_octave in per_line.items()
                     if on_octave and (drive if line == "drive" else readout))
        for name, per_line in lines.items()
    }
    report: dict[str, Any] = {
        "success": True,
        "calibration_db": calibration_db(machine),
        "planned": {name: v for name, v in planned.items() if v},
        "calibrated": {},
        "skipped": {},
        "errors": [],
        "dry_run": dry_run,
    }
    if not report["planned"]:
        raise SystemExit(
            "the selected lines are all on an MW-FEM; nothing to calibrate "
            "(drop --readout-only / --drive-only, or pick another target)")
    if dry_run:
        return report

    from qm.octave.octave_mixer_calibration import NoCalibrationElements
    from qualang_tools.multi_user import qm_session

    qmm = machine.connect()
    config = machine.generate_config()
    with qm_session(qmm, config, timeout=timeout) as qm:
        for name, planned in report["planned"].items():
            try:
                machine.qubits[name].calibrate_octave(
                    qm,
                    calibrate_drive="drive" in planned,
                    calibrate_resonator="readout" in planned,
                )
            except NoCalibrationElements:
                # The vendor's own calibrate_octave_ports swallows this with a
                # print. Recorded instead: "this element had nothing to
                # calibrate" is a finding an operator needs, not console noise.
                report["skipped"][name] = "no calibration elements"
            except Exception as exc:  # one bad element must not abort the rest
                report["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
                report["success"] = False
            else:
                report["calibrated"][name] = planned
    return report


def main(argv: list[str] | None = None, prog: str = "scqo-qm calibrate-octave") -> int:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Calibrate the Octave up-conversion mixers (LO leakage and "
                    "image) for the ACTIVE scqo device/setup.",
    )
    p.add_argument("--target", nargs="+", default=None, metavar="QUBIT",
                   help="calibrate only these qubits (default: every active one)")
    p.add_argument("--readout-only", action="store_true",
                   help="calibrate the resonator lines only")
    p.add_argument("--drive-only", action="store_true",
                   help="calibrate the xy lines only")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                   metavar="S", help="cluster session timeout in seconds "
                                     f"(default: {DEFAULT_TIMEOUT_S})")
    p.add_argument("--config", default=None,
                   help="scqo config.toml path (default: active selection)")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be calibrated and where the results "
                        "would be stored; calibrate nothing")
    args = p.parse_args(argv)

    if args.readout_only and args.drive_only:
        raise SystemExit("--readout-only and --drive-only are mutually exclusive")

    report = calibrate(
        targets=args.target,
        drive=not args.readout_only,
        readout=not args.drive_only,
        config_path=args.config,
        dry_run=args.dry_run,
        timeout=args.timeout,
    )

    for octave, where in report["calibration_db"].items():
        note = ("  <- quam's os.getcwd() FALLBACK, not a choice: set "
                "octaves.%s.calibration_db_path to the setup's backend_config/ "
                "folder, or this calibration follows whatever directory you "
                "launch from" % octave) if where["fallback"] else ""
        print(f"octave {octave}: results -> {where['path']}{note}")

    planned = report["planned"]
    if report.get("dry_run"):
        for name, lines in planned.items():
            print(f"  would calibrate {name}: {', '.join(lines)}")
        print(f"--dry-run: {len(planned)} qubit(s) planned, nothing calibrated")
        return 0

    for name, lines in report["calibrated"].items():
        print(f"  calibrated {name}: {', '.join(lines)}")
    for name, why in report["skipped"].items():
        print(f"  skipped {name}: {why}")
    for err in report["errors"]:
        print(f"  error: {err}")

    done, planned_n = len(report["calibrated"]), len(planned)
    print(f"calibrated {done}/{planned_n} qubit(s)")
    return 0 if report.get("success", not report["errors"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
