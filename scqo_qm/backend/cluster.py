"""One query: what is holding the QM cluster right now. Read-only.

Lists the Quantum Machines open on the cluster serving the ACTIVE scqo
device/setup (the same selection ``scqo run`` uses) and, under each one, the
jobs that have not finished - running, processing, or waiting in its queue -
with when they were created and started. It opens a manager connection and
nothing else: no Quantum Machine is opened and no job is touched, so it is safe
while someone is measuring. It is the look-first step before
``scqo-qm close-qm``, which is destructive and has no confirmation prompt.

Run it (in ``.venv-qm``)::

    scqo-qm cluster

The listing is CLUSTER-WIDE. Every open QM is shown, whoever opened it and from
whichever setup, because the cluster's locks are shared by all of them. The QOP
job records carry no user, so "whose job is this" is not answerable here; a
job's start time is what tells a neighbour's live measurement from a dead
session's leftover. An open QM with no job still holds its ports - that idle QM
is the classic stuck lock ``close-qm --qm-id`` exists for.

QOP 3.x answers ``get_jobs``, which shows the queue. QOP 2.x does not: there the
only thing the cluster reports per QM is its RUNNING job, and the output says so
rather than printing an empty queue that is really an unknown one.

A cluster whose gateway accepts connections but stalls every request
(DEADLINE_EXCEEDED) needs a restart from its web UI; this query reports it and
stops, the same as every other client.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

#: the QOP 3.x job statuses of a job that has not finished (qm.api.v2.job_api).
ACTIVE_STATUSES = ("In queue", "Running", "Processing")


def cluster_report(*, config_path: str | None = None, session: Any = None) -> dict[str, Any]:
    """Resolve the active setup's backend and list what is open on its cluster.

    ``session`` is injectable for tests; left None it resolves the active scqo
    selection exactly as ``scqo run`` does. Refuses BY NAME when the selected
    setup is not served by the QM backend.

    BEST-EFFORT per step, like ``QMBackend.close_qm``: a QM whose jobs cannot be
    read must not hide the QMs after it. A failure lands in ``errors``; read
    ``success``, never the absence of a raise.
    """
    if session is None:
        from scqo.cli import build_session  # lazy: keep module import scqo-free

        session, _cfg = build_session(config_path)

    from scqo_qm.backend.qm_backend import QMBackend

    backend = session.backend
    if not isinstance(backend, QMBackend):
        label = getattr(session, "backend_label", None) or type(backend).__name__
        raise SystemExit(
            f"the active setup is served by {label!r}, which has no Quantum "
            f"Machines - scqo-qm cluster is the QM backend's. Check 'scqo user' "
            f"for the selected device/setup."
        )

    machine = backend.machine
    net = getattr(machine, "network", None) or {}
    get = net.get if hasattr(net, "get") else lambda k, d=None: getattr(net, k, d)
    report: dict[str, Any] = {
        "success": False,
        "backend": "qm",
        "setup": getattr(session, "setup_name", "") or "",
        "cooldown": getattr(session, "cooldown_id", "") or "",
        "host": get("host"),
        "cluster_name": get("cluster_name"),
        "queue_visible": None,
        "qms": [],
        "errors": [],
    }
    errors: list[str] = report["errors"]

    try:
        qmm = _connect(machine, get("host"), get("port"))
    except Exception as err:
        errors.append(f"connecting to the cluster: {type(err).__name__}: {err}")
        return report

    try:
        try:
            qm_ids = [str(q) for q in qmm.list_open_qms()]
        except Exception as err:
            errors.append(f"list_open_qms: {type(err).__name__}: {err}")
            qm_ids = []
        for qm_id in qm_ids:
            # None = could not be read; [] = read, and nothing unfinished
            jobs: list[dict[str, Any]] | None = None
            if report["queue_visible"] is not False:
                try:
                    jobs = [_job_row(j) for j in qmm.get_jobs(qm_ids=[qm_id], status=list(ACTIVE_STATUSES))]
                    report["queue_visible"] = True
                except NotImplementedError:  # QOP 2.x: qm-qua raises this when there is no v2 api
                    report["queue_visible"] = False
                except Exception as err:
                    errors.append(f"{qm_id}: get_jobs: {type(err).__name__}: {err}")
            if report["queue_visible"] is False:
                jobs = _running_job(qmm, qm_id, errors)
            report["qms"].append({"id": qm_id, "jobs": jobs or [], "jobs_known": jobs is not None})
    finally:
        # qm-qua 1.2.6's QuantumMachinesManager has no close(); newer ones do
        closer = getattr(qmm, "close", None)
        if closer is not None:
            try:
                closer()
            except Exception as err:
                errors.append(f"closing the manager: {type(err).__name__}: {err}")

    report["success"] = not errors
    return report


def _connect(machine: Any, host: Any, port: Any) -> Any:
    """``machine.connect()`` behind a 2 s TCP probe, so a dead host fails in 2 s
    and not after a gRPC timeout (the same guard ``QMBackend.preview`` uses)."""
    if host:
        import socket

        socket.create_connection((str(host), int(port or 80)), timeout=2.0).close()
    return machine.connect()


def _job_row(job: Any) -> dict[str, Any]:
    """One QOP 3.x ``JobData`` as a plain row."""
    meta = getattr(job, "metadata", None)
    return {
        "id": str(job.id),
        "status": str(job.status),
        "created_at": _stamp(getattr(meta, "created_at", None)),
        "started_at": _stamp(getattr(meta, "started_at", None)),
        "simulation": bool(getattr(job, "is_simulation", False)),
    }


def _running_job(qmm: Any, qm_id: str, errors: list[str]) -> list[dict[str, Any]] | None:
    """QOP 2.x: the one thing the cluster reports per QM, its running job.
    None when it could not be read."""
    try:
        job = qmm.get_qm(qm_id).get_running_job()
    except Exception as err:
        errors.append(f"{qm_id}: get_running_job: {type(err).__name__}: {err}")
        return None
    if job is None:
        return []
    try:
        status = str(job.status).capitalize()
    except Exception:  # the status is a server round trip; the id alone still answers
        status = "Running"
    return [{"id": str(getattr(job, "id", job)), "status": status,
             "created_at": None, "started_at": None, "simulation": False}]


def _stamp(value: Any) -> datetime | None:
    """A protobuf timestamp as an aware datetime, None when unset.

    An unset protobuf timestamp decodes to the epoch, not to None - a queued job
    has no start - so anything at or before the epoch counts as unset."""
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)  # protobuf timestamps are UTC
    return value if value.timestamp() > 0 else None


def _age(then: datetime, now: datetime) -> str:
    seconds = max(0, int((now - then).total_seconds()))
    if seconds < 90:
        return f"{seconds} s ago"
    if seconds < 90 * 60:
        return f"{round(seconds / 60)} min ago"
    if seconds < 48 * 3600:
        return f"{round(seconds / 3600)} h ago"
    return f"{round(seconds / 86400)} d ago"


def report_lines(report: dict[str, Any], now: datetime | None = None) -> list[str]:
    """The report as LINES - pure, so tests read these and not a stream."""
    now = now or datetime.now(timezone.utc)
    where = " / ".join(str(v) for v in (report.get("host"), report.get("cluster_name")) if v)
    context = ", ".join(f"{k} {report[k]}" for k in ("setup", "cooldown") if report.get(k))
    lines = [f"# cluster {where or '(no network entry)'}" + (f"   (via {context})" if context else "")]
    if report["queue_visible"] is True:
        lines.append("# QOP 3.x: running and queued jobs shown")
    elif report["queue_visible"] is False:
        lines.append("# QOP 2.x: running job only - the queue is not visible on this QOP")

    qms = report["qms"]
    if not qms:
        if not report["errors"]:
            lines.append("no open Quantum Machines - the cluster is free")
    else:
        lines.append(f"{len(qms)} open Quantum Machine(s):")
    for qm in qms:
        lines.append(f"  {qm['id']}")
        for job in qm["jobs"]:
            if job["status"] == "In queue":
                stamp, verb = job["created_at"], "created"
            else:
                stamp, verb = job["started_at"] or job["created_at"], "started"
            when = ""
            if stamp is not None:
                local = stamp.astimezone()
                when = f"{verb} {local:%Y-%m-%d %H:%M:%S} ({_age(stamp, now)})"
            sim = "  [simulation]" if job["simulation"] else ""
            lines.append(f"    job {job['id']}  {job['status']:<10} {when}{sim}".rstrip())
        if not qm["jobs_known"]:
            lines.append("    jobs unknown - see the error below")
        elif not qm["jobs"]:
            if report["queue_visible"] is False:
                lines.append("    no running job (queue not visible on QOP 2.x)")
            else:
                lines.append(f"    idle - still holds its ports (release: scqo-qm close-qm --qm-id {qm['id']})")

    for err in report["errors"]:
        lines.append(f"  error: {err}")
        if "DEADLINE_EXCEEDED" in err:
            lines.append("    the gateway is wedged - restart the cluster from its web UI; "
                         "no client-side command reaches it")
    return lines


def main(argv: list[str] | None = None, prog: str = "scqo-qm cluster") -> int:
    p = argparse.ArgumentParser(
        prog=prog,
        description="List the Quantum Machines open on the cluster serving the ACTIVE "
        "scqo device/setup and their unfinished jobs. Read-only.",
    )
    p.add_argument("--config", default=None, help="scqo config.toml path (default: active selection)")
    args = p.parse_args(argv)

    report = cluster_report(config_path=args.config)
    for line in report_lines(report):
        print(line)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
