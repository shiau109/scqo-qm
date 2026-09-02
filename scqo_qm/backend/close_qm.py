"""One command: release the QM cluster's hardware locks.

Halts running jobs and closes the open Quantum Machines on the cluster serving
the ACTIVE scqo device/setup (the same selection ``scqo run`` uses). The
recovery door for the state where a crashed or abandoned session still holds
the locks — the symptom being a job that stalls forever, or an open that never
returns.

Run it (in ``.venv-qm``)::

    python -m scqo_qm.backend.close_qm                # every open QM on the cluster
    python -m scqo_qm.backend.close_qm --qm-id qm-1   # just this one
    python -m scqo_qm.backend.close_qm --dry-run      # list what is open, close nothing

THIS LIVES IN THE DRIVER, NOT IN SCQO, and deliberately so. It is the shape
``scqo_qm.backend.apply_distortion`` already uses: an operator command that
needs the vendor libraries belongs beside them, in the venv where they are
installed. scqo is the vendor-neutral core and has no ``close_qm`` — a
``scqo close-qm`` would be a command every other backend's operator could only
ever be refused by.

DESTRUCTIVE, and there is no confirmation prompt: halting a job discards the
data it had not yet streamed out. That is the point when the cluster is wedged,
but it also means a running measurement someone else started dies too. ``--dry-
run`` first if you are not certain the cluster is idle.

Not a wedged-gateway fix. A cluster in the DEADLINE_EXCEEDED state — where the
gateway itself stops answering — needs the cluster restarted from its web UI by
an operator; no client-side close reaches it. This command is for the case
where the cluster answers fine and the locks are simply still held.
"""

from __future__ import annotations

import argparse
from typing import Any


def close_open_qms(*, qm_id: str | None = None, config_path: str | None = None,
                   dry_run: bool = False, session: Any = None) -> dict[str, Any]:
    """Resolve the active setup's backend and run its ``close_qm`` hook.

    ``session`` is injectable for tests; left None it resolves the active scqo
    selection exactly as ``scqo run`` does. Refuses BY NAME when the selected
    setup is not served by the QM backend — a simulated or Qblox setup has no
    Quantum Machines to close, and saying so beats a benign-looking empty report.
    """
    if session is None:
        from scqo.cli import build_session  # lazy: keep module import scqo-free

        session, _cfg = build_session(config_path)

    backend = session.backend
    hook = getattr(backend, "close_qm", None)
    if hook is None:
        label = getattr(session, "backend_label", None) or type(backend).__name__
        raise SystemExit(
            f"the active setup is served by {label!r}, which has no Quantum "
            f"Machines to close — this command is the QM backend's. Check "
            f"'scqo user' for the selected device/setup."
        )

    if dry_run:
        machine = getattr(backend, "machine", None)
        qmm = machine.connect()
        try:
            listing = (getattr(qmm, "list_open_qms", None)
                       or getattr(qmm, "list_open_quantum_machines", None))
            open_qms = [str(q) for q in listing()] if listing else []
        finally:
            closer = getattr(qmm, "close", None)
            if closer is not None:
                closer()
        return {"success": True, "backend": "qm", "open_qms": open_qms,
                "halted_jobs": [], "closed_qms": [], "errors": [],
                "dry_run": True}

    return hook(qm_id=qm_id)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m scqo_qm.backend.close_qm",
        description="Halt running jobs and close the open Quantum Machines on "
                    "the cluster serving the ACTIVE scqo device/setup.",
    )
    p.add_argument(
        "--qm-id",
        default=None,
        metavar="ID",
        help="close only this Quantum Machine (default: every open one, plus a "
             "whole-cluster sweep for anything the listing missed)",
    )
    p.add_argument(
        "--config", default=None,
        help="scqo config.toml path (default: active selection)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="list the open Quantum Machines and close nothing",
    )
    args = p.parse_args(argv)

    report = close_open_qms(qm_id=args.qm_id, config_path=args.config,
                            dry_run=args.dry_run)

    open_qms = report.get("open_qms") or []
    halted = report.get("halted_jobs") or []
    closed = report.get("closed_qms") or []
    errors = report.get("errors") or []

    if report.get("dry_run"):
        if open_qms:
            print(f"{len(open_qms)} open Quantum Machine(s): {', '.join(open_qms)}")
        else:
            print("no open Quantum Machines on the cluster")
        print("  --dry-run: nothing closed")
        return 0

    if not open_qms and not closed and not halted:
        print("no open Quantum Machines on the cluster — nothing to release")
    else:
        print(f"closed {len(closed)} Quantum Machine(s), halted {len(halted)} job(s)")
        if open_qms:
            print(f"    open:   {', '.join(open_qms)}")
        if halted:
            print(f"    halted: {', '.join(halted)}")
        if closed:
            print(f"    closed: {', '.join(closed)}")

    for err in errors:
        print(f"  error: {err}")
    return 0 if report.get("success", not errors) else 1


if __name__ == "__main__":
    raise SystemExit(main())
