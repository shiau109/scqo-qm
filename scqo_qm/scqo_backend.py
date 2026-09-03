"""QM backend factory for the scqo CLI (entry-point group ``scqo.backends``, name ``qm``).

The factory receives the device's SELECTED named setup record from its cooldown
registry (``[<cycle>.setup.<name>]`` — backend + note; since scqo v0.9 the vendor
folder is DERIVED from the keys and injected by ``load_cooldowns``):
``setup["instrument_config"]`` is the folder holding the QUAM files under
canonical names — ``state.json`` + ``wiring.json``. That folder is the single
QUAM-state authority for this device's setup (quam's own resolution via ~/.qualibrate
or QUAM_STATE_PATH is deliberately bypassed). It also receives the device's ROSTER,
the authority on which entities exist: the driver serves views BY ENTITY NAME
(``q1_xy`` -> its drive view over QUAM's ``q1.xy``, ``q1_q2_c_z`` -> the pair's
TunableCoupler), so the roster is threaded into the backend and every name resolves
through it. Vendor imports stay INSIDE the function so loading this module is cheap
and vendor-free. (The virtual-twin ``qm_sim`` mode was retired with v0.5.0.)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from scqo import LabConfig
from scqo.backend import Backend

if TYPE_CHECKING:
    from scqo.roster import Roster


def build_backend(cfg: LabConfig, setup: dict, roster: "Roster") -> Backend:
    if setup.get("backend") != "qm":
        raise SystemExit(f"the qm driver serves backend 'qm', got {setup.get('backend')!r}")
    # State-authority rule checked BEFORE loading QUAM: fail before any state file
    # is touched. Forbidden while qualibrate nodes still write QUAM directly (see
    # scqo-qm CLAUDE.md); the migration finish line is flipping this to "push".
    if cfg.state_sync != "pull":
        raise SystemExit(
            'lab config sets state_sync != "pull" for the QM backend: forbidden while '
            "official qualibrate nodes can still write QUAM (see scqo-qm CLAUDE.md)"
        )
    folder = Path(setup["instrument_config"])
    missing = [n for n in ("state.json", "wiring.json") if not (folder / n).is_file()]
    if missing:
        raise SystemExit(
            f"qm setup: {', '.join(missing)} not found in {folder} — "
            "canonical QUAM filenames required"
        )
    import warnings

    from scqo_qm.quam_fields import (
        drive_frequency_problems,
        flux_headroom_problems,
        flux_headroom_warnings,
        flux_point_problems,
    )
    from scqo_qm.backend.qm_backend import QMBackend

    backend = QMBackend.load(state_path=str(folder), roster=roster)

    # Three whole-tree audits, reported TOGETHER so one `scqo run` surfaces
    # every config problem at once rather than one per attempt.
    #
    # flux_point_problems: the governed knob must BE the applied bias. Checked
    # here rather than inside get/set_idle_flux, because that getter runs during
    # pull-seeding (refusing there would abort session construction with no
    # context) and the accessor is shared with the qualibrate path where a
    # different point is legitimate.
    #
    # flux_headroom_problems: the port must be able to EMIT what the tree
    # declares. Per-sweep refusals live in probes/_flux_limits.py; this catches
    # the config before any probe runs, so "this chip needs three ports in
    # amplified mode" arrives as one message. Only CLIPPING is fatal here —
    # config that merely leaves range on the table warns instead, or the live
    # 5Q4C couplers (const 0.15 V, which have always run) would block every
    # session.
    #
    # drive_frequency_problems: f_01 (what drive_freq_hz READS) must equal
    # xy.RF_frequency (what the drive line PLAYS). set_drive_freq writes both,
    # so only a hand edit or a foreign node can split them - and the remedy is
    # a hand edit of state.json too, because `scqo set` builds its session
    # through this very factory and would be refused the same way.
    for advisory in flux_headroom_warnings(backend.machine):
        warnings.warn(advisory, RuntimeWarning, stacklevel=2)

    complaints = [
        ("flux points", "disagree with the bias every probe applies, so idle_flux "
                        "would be a knob the hardware never sees",
         flux_point_problems(backend.machine)),
        ("flux headroom", "declares voltages its ports cannot emit, which the DAC "
                          "clips silently and the simulator does not show",
         flux_headroom_problems(backend.machine)),
        ("drive frequencies", "disagree between f_01 and xy.RF_frequency, so "
                              "drive_freq_hz would read a number the drive line "
                              "never plays",
         drive_frequency_problems(backend.machine)),
    ]
    report = "\n".join(
        f"{what} in {folder / 'state.json'} {why}:\n  - " + "\n  - ".join(problems)
        for what, why, problems in complaints if problems
    )
    if report:
        raise SystemExit(f"qm setup: {report}")
    return backend
