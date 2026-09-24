"""The ``scqo-qm cluster`` query: what is open and running on the QM cluster.

No cluster and no scqo config: the manager, the QMs and the jobs are doubles,
and the report takes an INJECTED session - the live ``build_session`` half is
scqo-owned. Two properties are worth pinning beyond the rows themselves: the
query is READ-ONLY (a double that would record a close or a halt raises
instead), and it is BEST-EFFORT, so a QM whose jobs cannot be read never hides
the QMs after it and never reads as idle.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scqo_qm.backend.cluster import ACTIVE_STATUSES, cluster_report, main, report_lines

NOW = datetime(2026, 9, 24, 6, 0, 0, tzinfo=timezone.utc)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)  # an unset protobuf timestamp


def _job(job_id, status, *, created=None, started=None, simulation=False):
    """A QOP 3.x ``JobData`` double."""
    return SimpleNamespace(id=job_id, status=status, is_simulation=simulation,
                           metadata=SimpleNamespace(created_at=created or EPOCH,
                                                    started_at=started or EPOCH))


class _Untouchable:
    """A QM that must not be closed, and whose job must not be halted."""

    def close(self):
        raise AssertionError("the cluster query closed a QM")

    def get_running_job(self):
        raise AssertionError("QOP 3.x path read the QOP 2.x running job")


class _QMM3:
    """A QOP 3.x manager: ``get_jobs`` answers, per QM."""

    def __init__(self, jobs_by_qm, fail_for=()):
        self.jobs_by_qm = jobs_by_qm
        self.fail_for = set(fail_for)
        self.calls = []
        self.closed = False

    def list_open_qms(self):
        return list(self.jobs_by_qm)

    def get_jobs(self, qm_ids=(), status=()):
        self.calls.append((list(qm_ids), list(status)))
        (qm_id,) = qm_ids
        if qm_id in self.fail_for:
            raise RuntimeError("job service unavailable")
        return self.jobs_by_qm[qm_id]

    def get_qm(self, qm_id):
        return _Untouchable()

    def close_all_qms(self):
        raise AssertionError("the cluster query swept the cluster")

    def close(self):
        self.closed = True


class _QM2:
    def __init__(self, job):
        self._job = job

    def get_running_job(self):
        return self._job

    def close(self):
        raise AssertionError("the cluster query closed a QM")


class _QMM2(_QMM3):
    """A QOP 2.x manager: no ``get_jobs`` (qm-qua raises NotImplementedError)."""

    def __init__(self, running_by_qm):
        super().__init__({k: [] for k in running_by_qm})
        self.running_by_qm = running_by_qm

    def get_jobs(self, qm_ids=(), status=()):
        raise NotImplementedError("This method is not available in the current QOP version")

    def get_qm(self, qm_id):
        return _QM2(self.running_by_qm[qm_id])


def _session(qmm, *, network=None):
    from scqo_qm.backend.qm_backend import QMBackend

    backend = QMBackend.__new__(QMBackend)          # no roster/state needed
    # no host: skips the TCP probe, which would need a listening socket
    backend._machine = SimpleNamespace(connect=lambda: qmm,
                                       network=network or {"cluster_name": "Cluster_1"})
    return SimpleNamespace(backend=backend, backend_label="qm",
                           setup_name="qm_5q", cooldown_id="cd1")


def test_qop3_lists_running_and_queued_jobs_under_each_qm():
    qmm = _QMM3({
        "qm-1": [_job("job-7", "Running", created=NOW - timedelta(minutes=13),
                      started=NOW - timedelta(minutes=12)),
                 _job("job-8", "In queue", created=NOW - timedelta(minutes=5))],
        "qm-2": [],
    })
    report = cluster_report(session=_session(qmm))

    assert report["success"] is True and report["errors"] == []
    assert report["queue_visible"] is True
    assert [q["id"] for q in report["qms"]] == ["qm-1", "qm-2"]
    running, queued = report["qms"][0]["jobs"]
    assert (running["id"], running["status"]) == ("job-7", "Running")
    assert running["started_at"] == NOW - timedelta(minutes=12)
    assert (queued["status"], queued["started_at"]) == ("In queue", None)  # epoch = unset
    assert report["qms"][1] == {"id": "qm-2", "jobs": [], "jobs_known": True}
    # asks for exactly the unfinished statuses, one QM at a time
    assert qmm.calls == [(["qm-1"], list(ACTIVE_STATUSES)), (["qm-2"], list(ACTIVE_STATUSES))]
    assert qmm.closed is True                        # its own manager, nothing else


def test_lines_show_start_times_and_name_the_idle_qm_release():
    qmm = _QMM3({
        "qm-1": [_job("job-7", "Running", started=NOW - timedelta(minutes=12)),
                 _job("job-8", "In queue", created=NOW - timedelta(seconds=40), simulation=True)],
        "qm-2": [],
    })
    lines = report_lines(cluster_report(session=_session(qmm)), now=NOW)
    text = "\n".join(lines)

    assert lines[0] == "# cluster Cluster_1   (via setup qm_5q, cooldown cd1)"
    assert "# QOP 3.x: running and queued jobs shown" in lines
    assert "2 open Quantum Machine(s):" in lines
    job7 = next(line for line in lines if "job-7" in line)
    assert "Running" in job7 and "started" in job7 and "(12 min ago)" in job7
    job8 = next(line for line in lines if "job-8" in line)
    assert "In queue" in job8 and "created" in job8 and "(40 s ago)" in job8
    assert job8.endswith("[simulation]")
    assert "idle - still holds its ports (release: scqo-qm close-qm --qm-id qm-2)" in text
    assert all(line.isascii() for line in lines)     # lab consoles, any codepage


def test_an_empty_cluster_says_it_is_free():
    report = cluster_report(session=_session(_QMM3({})))
    assert report["success"] is True and report["qms"] == []
    assert "no open Quantum Machines - the cluster is free" in report_lines(report, now=NOW)


def test_qop2_falls_back_to_the_running_job_and_says_the_queue_is_unknown():
    job = SimpleNamespace(id="job-3", status="running")
    qmm = _QMM2({"qm-1": job, "qm-2": None})
    report = cluster_report(session=_session(qmm))

    assert report["success"] is True
    assert report["queue_visible"] is False
    assert report["qms"][0]["jobs"] == [{"id": "job-3", "status": "Running", "created_at": None,
                                         "started_at": None, "simulation": False}]
    lines = report_lines(report, now=NOW)
    assert "# QOP 2.x: running job only - the queue is not visible on this QOP" in lines
    # an empty QOP 2.x answer is NOT "idle": the queue behind it is unknown
    assert "    no running job (queue not visible on QOP 2.x)" in lines
    assert not any("idle" in line for line in lines)


def test_one_qm_that_cannot_be_read_does_not_hide_the_next_or_read_as_idle():
    qmm = _QMM3({"qm-1": [], "qm-2": [_job("job-9", "Running", started=NOW)]},
                fail_for={"qm-1"})
    report = cluster_report(session=_session(qmm))

    assert report["success"] is False
    assert report["qms"][0] == {"id": "qm-1", "jobs": [], "jobs_known": False}
    assert report["qms"][1]["jobs"][0]["id"] == "job-9"
    assert any("qm-1: get_jobs" in e for e in report["errors"])
    lines = report_lines(report, now=NOW)
    assert "    jobs unknown - see the error below" in lines
    assert not any("idle" in line for line in lines)


def test_an_unreachable_cluster_is_an_error_not_a_crash():
    def refuse():
        raise ConnectionError("no route to host")

    session = _session(None)
    session.backend._machine.connect = refuse
    report = cluster_report(session=session)

    assert report["success"] is False and report["qms"] == []
    assert report["errors"] == ["connecting to the cluster: ConnectionError: no route to host"]
    lines = report_lines(report, now=NOW)
    assert not any("cluster is free" in line for line in lines)  # unknown is not free


def test_a_wedged_gateway_names_the_restart():
    class _Wedged(_QMM3):
        def list_open_qms(self):
            raise RuntimeError("StatusCode.DEADLINE_EXCEEDED")

    report = cluster_report(session=_session(_Wedged({})))
    lines = report_lines(report, now=NOW)
    assert any("restart the cluster from its web UI" in line for line in lines)


def test_refuses_a_non_qm_setup_by_name():
    session = SimpleNamespace(backend=SimpleNamespace(), backend_label="simulated")
    with pytest.raises(SystemExit) as excinfo:
        cluster_report(session=session)
    assert "simulated" in str(excinfo.value)
    assert "scqo-qm cluster is the QM backend's" in str(excinfo.value)


def test_main_prints_and_exits_one_when_a_step_failed(monkeypatch, capsys):
    qmm = _QMM3({"qm-1": []}, fail_for={"qm-1"})
    monkeypatch.setattr("scqo_qm.backend.cluster.cluster_report",
                        lambda **kw: cluster_report(session=_session(qmm)))
    assert main([]) == 1
    assert "error: qm-1: get_jobs" in capsys.readouterr().out


def test_main_exits_zero_on_a_clean_listing(monkeypatch, capsys):
    monkeypatch.setattr("scqo_qm.backend.cluster.cluster_report",
                        lambda **kw: cluster_report(session=_session(_QMM3({}))))
    assert main([]) == 0
    assert "the cluster is free" in capsys.readouterr().out
