"""``QMBackend.close_qm`` + the ``scqo_qm.backend.close_qm`` operator CLI.

No cluster and no scqo config: the manager, the QMs and the jobs are doubles,
and the CLI takes an INJECTED session — the live ``build_session`` half is
scqo-owned. The point of the hook is that it is BEST-EFFORT, so most of what is
worth pinning is what happens when a step fails: the sweep must continue and
the failure must land in ``errors`` rather than raising.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scqo_qm.backend.close_qm import close_open_qms, main


class _Job:
    def __init__(self, job_id, fail=False):
        self.id = job_id
        self._fail = fail

    def halt(self):
        if self._fail:
            raise RuntimeError("halt refused")


class _QM:
    def __init__(self, qm_id, job=None, close_fails=False):
        self.id = qm_id
        self._job = job
        self._close_fails = close_fails
        self.closed = False

    def get_running_job(self):
        return self._job

    def close(self):
        if self._close_fails:
            raise RuntimeError("close refused")
        self.closed = True


class _QMM:
    def __init__(self, qms):
        self.qms = qms
        self.closed_all = False
        self.closed = False

    def list_open_qms(self):
        return list(self.qms)

    def get_qm(self, qm_id):
        return self.qms.get(qm_id)

    def close_all_qms(self):
        self.closed_all = True

    def close(self):
        self.closed = True


def _backend(qmm):
    """A QMBackend with only what close_qm touches — machine.connect()."""
    from scqo_qm.backend.qm_backend import QMBackend

    backend = QMBackend.__new__(QMBackend)          # no roster/state needed
    backend._machine = SimpleNamespace(connect=lambda: qmm)
    return backend


def test_halts_jobs_closes_machines_and_sweeps_the_cluster():
    qmm = _QMM({"qm-1": _QM("qm-1", job=_Job("job-7")),
                "qm-2": _QM("qm-2", job=None)})
    report = _backend(qmm).close_qm()

    assert report["success"] is True
    assert report["errors"] == []
    assert report["open_qms"] == ["qm-1", "qm-2"]
    assert report["halted_jobs"] == ["job-7"]        # only the one that had a job
    assert report["closed_qms"] == ["qm-1", "qm-2"]
    assert qmm.qms["qm-1"].closed and qmm.qms["qm-2"].closed
    assert qmm.closed_all is True                    # the whole-cluster sweep
    assert qmm.closed is True                        # ... and the manager itself


def test_qm_id_targets_one_and_skips_the_cluster_sweep():
    """--qm-id must not touch its neighbours: no close_all, no listing walk."""
    qmm = _QMM({"qm-1": _QM("qm-1", job=_Job("job-7")),
                "qm-2": _QM("qm-2", job=_Job("job-9"))})
    report = _backend(qmm).close_qm(qm_id="qm-2")

    assert report["closed_qms"] == ["qm-2"]
    assert report["halted_jobs"] == ["job-9"]
    assert qmm.qms["qm-1"].closed is False
    assert qmm.closed_all is False


def test_one_failure_does_not_stop_the_sweep():
    """The whole reason the hook is best-effort: a QM that refuses to close must
    not strand the ones after it. The failure is reported, not raised."""
    qmm = _QMM({"qm-1": _QM("qm-1", job=_Job("job-7", fail=True),
                            close_fails=True),
                "qm-2": _QM("qm-2", job=_Job("job-9"))})
    report = _backend(qmm).close_qm()

    assert report["success"] is False
    assert report["closed_qms"] == ["qm-2"]          # the second one still closed
    assert report["halted_jobs"] == ["job-9"]
    assert any("halting job job-7" in e for e in report["errors"])
    assert any("qm-1: close" in e for e in report["errors"])


def test_a_manager_with_no_listing_api_reports_nothing_open():
    class _Bare:
        def close(self):
            self.closed = True

    report = _backend(_Bare()).close_qm()
    assert report["open_qms"] == [] and report["closed_qms"] == []
    assert report["success"] is True


def test_a_broken_listing_is_an_error_not_a_crash():
    class _Broken(_QMM):
        def list_open_qms(self):
            raise RuntimeError("cluster unreachable")

    report = _backend(_Broken({})).close_qm()
    assert report["success"] is False
    assert any("list_open_qms" in e for e in report["errors"])


# ------------------------------------------------------------------------ CLI

def _session(qmm, *, label="qm"):
    return SimpleNamespace(backend=_backend(qmm), backend_label=label)


def test_cli_refuses_a_non_qm_setup_by_name():
    """A simulated or Qblox setup has nothing to close; say so rather than
    returning a benign-looking empty report."""
    session = SimpleNamespace(backend=SimpleNamespace(), backend_label="simulated")
    with pytest.raises(SystemExit) as excinfo:
        close_open_qms(session=session)
    assert "simulated" in str(excinfo.value)
    assert "no Quantum Machines to close" in str(excinfo.value)


def test_dry_run_lists_without_closing():
    qmm = _QMM({"qm-1": _QM("qm-1", job=_Job("job-7"))})
    report = close_open_qms(session=_session(qmm), dry_run=True)

    assert report["dry_run"] is True
    assert report["open_qms"] == ["qm-1"]
    assert report["closed_qms"] == [] and report["halted_jobs"] == []
    assert qmm.qms["qm-1"].closed is False
    assert qmm.closed_all is False
    assert qmm.closed is True                        # the probe's own manager


def test_cli_main_reports_and_exits_zero(monkeypatch, capsys):
    qmm = _QMM({"qm-1": _QM("qm-1", job=_Job("job-7"))})
    monkeypatch.setattr("scqo_qm.backend.close_qm.close_open_qms",
                        lambda **kw: _backend(qmm).close_qm(**{
                            k: v for k, v in kw.items() if k == "qm_id"}))
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "closed 1 Quantum Machine(s), halted 1 job(s)" in out
    assert "qm-1" in out


def test_cli_main_exits_one_when_a_step_failed(monkeypatch, capsys):
    qmm = _QMM({"qm-1": _QM("qm-1", close_fails=True)})
    monkeypatch.setattr("scqo_qm.backend.close_qm.close_open_qms",
                        lambda **kw: _backend(qmm).close_qm())
    assert main([]) == 1
    assert "error:" in capsys.readouterr().out


def test_cli_help_names_the_flags(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--qm-id" in out and "--dry-run" in out
