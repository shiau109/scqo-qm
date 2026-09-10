"""Self-test the scqo stack against a REAL QUAM state — no hardware needed.

    python scripts/check_real_config.py D:\\qpu_data_dev\\5Q4C\\cd1\\qm_5q\\backend_config
    python scripts/check_real_config.py <state_dir> --qubits q1 q2

Loads your ``state.json`` + ``wiring.json``, then runs the full scqo pipeline with
SIMULATED data over the REAL device tree: read neutral fields -> run experiments ->
fit -> write results back -> save (to an explicit scratch path ONLY — the live
quam_state is never targeted) -> reload and compare. Everything happens on a
temporary copy; your original files are never opened for writing.

Needs the QM environment (lab: ``D:\github\.venv-qm``).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `customized`


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("state_dir", help="folder holding state.json + wiring.json")
    parser.add_argument("--qubits", nargs="+", help="qubits to exercise (default: all readable in the state)")
    args = parser.parse_args()

    source = Path(args.state_dir)
    for fname in ("state.json", "wiring.json"):
        if not (source / fname).is_file():
            raise SystemExit(f"{fname} not found in {source}")
    work = Path(tempfile.mkdtemp(prefix="scqo_qm_selftest_"))
    shutil.copy(source / "state.json", work / "state.json")
    shutil.copy(source / "wiring.json", work / "wiring.json")
    print(f"sandbox: {work}")
    print("  (temporary self-test copies + throwaway run data: your originals and your")
    print("   real data_root are NOT touched; real measurements use `scqo run`)")

    try:
        from quam_config import Quam
    except ModuleNotFoundError as err:
        raise SystemExit(
            f"missing package: {err.name}\n"
            "This self-test needs the QM stack (quam/qm + this repo installed). "
            "Run it in the lab's QM environment: D:\github\.venv-qm"
        )

    machine = Quam.load(str(work))
    print(f"[1/5] loaded QUAM | qubits: {list(machine.qubits)}")

    import scqo_qm.experiments  # noqa: F401
    from scqo import Session
    from scqo.roster import parse_components
    from scqo.testing import SimulatedBackend
    from scqo_qm.backend.qm_backend import QMDeviceModel
    from scqo_qm._family import tree_families
    from scqo_qm.backend.roster_gen import roster_toml_for

    # The driver resolves every name through the ROSTER (q1_ro -> the readout
    # knobs of vendor qubit q1), so it is needed before the device model. This
    # throwaway self-test derives one from the QUAM tree itself; the REAL
    # roster lives in <data_root>/<device>/components.toml and is what
    # `scqo run` uses.
    roster = parse_components(roster_toml_for(machine))

    # WHICH hardware families this tree declares, before anything reads a knob.
    # The snapshot below reports a knob this driver cannot realize as None, which
    # is indistinguishable from "uncalibrated" -- so a self-test that printed the
    # knobs without naming the families would reach its PASS line while quietly
    # showing every absolute power as unset.
    families = tree_families(machine)
    print(f"      hardware: RF chain {families['rf_chain']}, "
          f"flux ports {families['flux_port']}"
          + (f", octaves {families['octaves']}" if "octaves" in families else ""))

    dm = QMDeviceModel(machine, roster)
    snap = dm.snapshot()
    for name, fields in snap.items():  # keyed by CHANNEL entity (q1_ro, q1_xy, ...)
        print(f"      {name}: {fields}")

    # Absolute power is the one knob family whose vendor home is chain-specific,
    # so an unreadable one here is a finding about the DRIVER, not the chip. Said
    # out loud rather than left as a None among the other Nones.
    unpriced = sorted(
        name for name, fields in snap.items()
        for field in ("readout_power_dbm", "drive_power_dbm")
        if field in fields and fields[field] is None)
    if unpriced:
        print(f"      note: absolute power unavailable on {unpriced} - expected "
              f"when the RF chain is not mw_fem; see power_context in the run "
              f"record for the reason")

    # The two experiments below anchor on these knobs; a real lab state
    # carries uncalibrated qubits (value None), which are skipped, not run.
    needed = {q: ((f"{q}_ro", "readout_freq_hz"), (f"{q}_xy", "pi_amp"))
              for q in machine.qubits}
    unreadable = [q for q, pairs in needed.items()
                  if any(snap.get(ch, {}).get(field) is None for ch, field in pairs)]
    qubits = args.qubits or [q for q in needed if q not in unreadable]
    if not qubits:
        raise SystemExit(f"no readable qubits in this state (uncalibrated: {unreadable}) "
                         "— nothing to exercise")
    print(f"[2/5] snapshot OK | testing qubits: {qubits}"
          + (f" (skipping uncalibrated: {unreadable})" if unreadable else ""))

    sess = Session(SimulatedBackend(dm), roster, data_root=work / "data", device_name="selftest")
    before = {name: dict(v) for name, v in dm.snapshot().items()}
    failures = []
    for experiment in ("resonator_spectroscopy", "qubit_power_rabi"):
        # update="apply": this self-test exists to exercise writeback (scqo v0.6.0
        # defaults to suggest-only, which would leave the QUAM objects untouched).
        result = sess.run(experiment, {"targets": qubits}, update="apply", tags=["selftest"])
        ok = all(result["outcomes"].get(q) == "successful" for q in qubits) and not result.get("error")
        print(f"[3/5] {experiment}: {result['outcomes']}" + (f" error={result['error']}" if result.get("error") else ""))
        if not ok:
            failures.append(experiment)

    # the knobs those two experiments write live on the CHANNEL entities:
    # readout_freq_hz on <q>_ro, pi_amp on <q>_xy — and dm.snapshot() reads
    # them back THROUGH the views onto QUAM, so "moved" means the vendor tree.
    after = dm.snapshot()
    touched = [f"{q}_ro" for q in qubits] + [f"{q}_xy" for q in qubits]
    moved = [name for name in touched if after[name] != before[name]]
    print(f"[4/5] writeback reached the real QUAM objects for: {moved or 'NONE'}")
    if set(moved) != set(touched):
        failures.append("writeback")

    saved = work / "saved_state"
    machine.save(path=saved)  # explicit scratch path ONLY — never the default quam_state
    reloaded = Quam.load(str(saved))
    dm2 = QMDeviceModel(reloaded, roster)
    round_trip = all(
        abs(dm2.snapshot()[f"{q}_ro"]["readout_freq_hz"]
            - after[f"{q}_ro"]["readout_freq_hz"]) < 1e-3 for q in qubits
    )
    print(f"[5/5] QUAM save/reload round-trip (scratch path): {'OK' if round_trip else 'MISMATCH'}")
    if not round_trip:
        failures.append("save-roundtrip")

    print(f"\nruns saved + indexed under {work / 'data'}: "
          f"{[r['run_id'] for r in sess.find_runs(tag='selftest')]}")
    print("\nPASS - scqo works against this real state" if not failures
          else f"\nFAIL - problems in: {', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
