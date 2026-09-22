"""One command: accepted cryoscope distortion facts -> the QM z-output filter.

After ``scqo run qubit_spectroscopy_cryoscope --target q1`` (or the ramsey one) +
``scqo accept``, the fit's ``distortion_amp``/``distortion_tau_s`` live as FACTS in
scqo's ``physical.json`` — record-only, never auto-pushed. This turns them into the
OPX predistortion filter in one step: resolve the ACTIVE scqo device/setup (the same
selection ``scqo run`` uses), read the accepted taps for ``<target>``'s flux channel,
write them onto ``machine.qubits[<target>].z.opx_output.exponential_filter`` via
:func:`scqo_qm.backend._distortion.apply_exponential_filter`, and save the setup's
``state.json``. Fully OFFLINE — ``build_session`` loads the QUAM from JSON and never
opens a ``QuantumMachinesManager``.

Run it (in ``.venv-qm``)::

    python -m scqo_qm.backend.apply_distortion --target q1
    python -m scqo_qm.backend.apply_distortion --target q1 --dry-run   # preview only
    python -m scqo_qm.backend.apply_distortion --target q1 --extend    # append a residual

It never runs automatically on ``scqo accept`` — applying predistortion is a
deliberate, opt-in step (measure a fresh full correction on a filter-CLEARED line;
``--extend`` refines a residual measured with the current filter active).

Two sources for the taps:

* ``--run <run_id>`` reads the fit straight from that run's ``result.json`` —
  the recommended door for ITERATION (``run -> apply --run <id> --extend ->
  verify``), because it names exactly which measurement feeds the filter. The
  two cryoscopes share ONE fact slot with REPLACE semantics (accepting one
  overwrites the other's taps), so run-addressed apply makes ``scqo accept``
  order irrelevant to the vendor filter.
* No ``--run``: the accepted facts (``physical.json``) — fine when the latest
  accepted cryoscope IS what you want applied.

Facts vs filter — they are DIFFERENT quantities and diverge by design: the facts
hold the latest accepted MEASUREMENT (mid-iteration, a residual as seen through
the then-active filter); the vendor ``exponential_filter`` holds the accumulated
applied CORRECTION. The accumulated total lives in the setup's state.json (itself
a neutral exp-sum, convertible to other backends). We deliberately never write a
composed total back into the facts: scqo accept's staleness guard compares each
pending suggestion's ``before`` to the CURRENT fact value, so a write-back would
strand every pending cryoscope suggestion behind ``--force``.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Any

from scqo_qm.backend._distortion import (
    apply_exponential_filter,
    clear_exponential_filter,
)
from scqo_qm.quam_io import save_state

#: the roster channel kind of a qubit's flux line (catalog CHANNELS).
FLUX_KIND = "flux"

#: experiments whose result.fit carries the distortion taps.
CRYOSCOPE_EXPERIMENTS = ("qubit_spectroscopy_cryoscope", "qubit_ramsey_cryoscope")


def _run_taps(session: Any, run_id: str, target: str) -> tuple[list, list]:
    """The fitted ``(amps, taus_s)`` from one saved run — refuses BY NAME a
    non-cryoscope run, a failed target, or a run without the fit fields."""
    try:
        data = session.load_run(run_id)
    except KeyError as err:
        raise SystemExit(err.args[0] if err.args else str(err)) from None
    record = data["record"]
    experiment = record.get("experiment")
    if experiment not in CRYOSCOPE_EXPERIMENTS:
        raise SystemExit(
            f"run {run_id} is a {experiment!r} run — distortion taps come from "
            f"one of {', '.join(CRYOSCOPE_EXPERIMENTS)}"
        )
    outcome = (record.get("outcomes") or {}).get(target)
    if outcome != "successful":
        raise SystemExit(
            f"run {run_id}: {target!r} outcome is {outcome!r}, not 'successful' "
            f"— refusing to apply a failed fit"
        )
    fit = (data.get("result", {}).get("fit") or {}).get(target) or {}
    amps = fit.get("distortion_amp")
    taus_s = fit.get("distortion_tau_s")
    if amps is None or taus_s is None:
        raise SystemExit(
            f"run {run_id}: no distortion_amp/distortion_tau_s in result.fit "
            f"for {target!r}"
        )
    return amps, taus_s


def clear_distortion(
    target: str,
    *,
    config_path: str | None = None,
    session: Any = None,
    cfg: Any = None,
    save: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Remove ALL exponential_filter taps from ``target``'s z output — the
    fresh-line reset before a clean-slate cryoscope characterization (measure
    the TOTAL response on the bare line, then apply once). Resolves the ACTIVE
    scqo selection like :func:`apply_distortion_from_state`; returns
    ``{"target", "removed", "state_dir", "saved"}``. OFFLINE.
    """
    if session is None:
        from scqo.cli import build_session  # lazy: keep module import scqo-free

        session, cfg = build_session(config_path)
    machine = session.backend.machine
    if dry_run:
        try:
            taps = list(machine.qubits[target].z.opx_output.exponential_filter or [])
        except (KeyError, TypeError, AttributeError):
            taps = []
        removed = [list(p) for p in taps]
    else:
        removed = clear_exponential_filter(machine, target)["removed"]
    state_dir = None
    if cfg is not None:
        from scqo.datastore import setup_backend_config_dir

        state_dir = str(
            setup_backend_config_dir(
                cfg.data_root, cfg.device, session.cooldown_id, session.setup_name
            )
        )
    did_save = bool(save and not dry_run)
    if did_save:
        save_state(machine, state_dir)
    return {"target": target, "removed": removed,
            "state_dir": state_dir, "saved": did_save}


def apply_distortion_from_state(
    target: str,
    *,
    run_id: str | None = None,
    replace: bool = True,
    form: str = "sum",
    config_path: str | None = None,
    session: Any = None,
    cfg: Any = None,
    save: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply distortion taps for ``target`` to the QM z filter.

    Resolves the ACTIVE scqo selection (unless ``session`` is injected — for tests)
    and takes the taps from ``run_id``'s saved fit when given (the run-addressed
    door — accept order becomes irrelevant), else from the accepted facts
    (``distortion_amp``/``distortion_tau_s`` on the target's flux channel). Writes
    them to the QUAM z-output ``exponential_filter`` and (unless ``dry_run`` or
    ``save=False``) saves the setup's ``state.json``. OFFLINE.

    Returns a summary dict: ``target``, ``channel``, ``run_id``, ``amps``,
    ``taus_s``, ``existing_taps``, ``state_dir``, ``saved``, plus
    ``apply_exponential_filter``'s ``exponential_filter`` + ``scale``. Raises
    ``SystemExit`` when no taps are available (no accepted facts / bad run).
    """
    if session is None:
        from scqo.cli import build_session  # lazy: keep module import scqo-free

        session, cfg = build_session(config_path)

    channel = session.backend.roster.default_channel(target, FLUX_KIND)  # q1 -> q1_z
    if run_id is not None:
        amps, taus_s = _run_taps(session, run_id, target)
    else:
        amps = session.physical.get(channel, "distortion_amp")
        taus_s = session.physical.get(channel, "distortion_tau_s")
        if amps is None or taus_s is None:
            raise SystemExit(
                f"no accepted distortion facts for {channel} — run and accept a "
                f"cryoscope for {target!r} first (distortion_amp/distortion_tau_s "
                f"are unset in physical.json), or apply straight from a run with "
                f"--run <run_id>"
            )

    machine = session.backend.machine
    try:
        existing = len(list(machine.qubits[target].z.opx_output.exponential_filter or []))
    except (KeyError, TypeError, AttributeError):
        existing = 0  # apply_exponential_filter raises the friendly target/z error

    if replace and existing:
        warnings.warn(
            f"replacing {existing} existing exponential_filter tap(s) on {channel}; "
            f"a full correction must be MEASURED on a filter-cleared line (use "
            f"--extend to refine a residual instead)",
            stacklevel=2,
        )
    if form == "cascade":
        warnings.warn(
            "form='cascade' (QOP 3.4.1): apply the returned 'scale' to the flux "
            "waveform amplitude yourself — this command does not automate it",
            stacklevel=2,
        )

    result = apply_exponential_filter(
        machine, target, amps, taus_s, replace=replace, form=form
    )

    state_dir = None
    if cfg is not None:
        from scqo.datastore import setup_backend_config_dir

        state_dir = str(
            setup_backend_config_dir(
                cfg.data_root, cfg.device, session.cooldown_id, session.setup_name
            )
        )
    did_save = bool(save and not dry_run)
    if did_save:
        save_state(machine, state_dir)

    return {
        "target": target,
        "channel": channel,
        "run_id": run_id,
        "amps": list(amps),
        "taus_s": list(taus_s),
        "existing_taps": existing,
        "state_dir": state_dir,
        "saved": did_save,
        **result,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m scqo_qm.backend.apply_distortion",
        description="Apply accepted cryoscope distortion taps to the QM z-output "
        "exponential_filter for the ACTIVE scqo device/setup.",
    )
    p.add_argument("--target", required=True, help="qubit/mode name, e.g. q1")
    p.add_argument(
        "--run",
        default=None,
        metavar="RUN_ID",
        help="take the taps from this saved run's fit instead of the accepted "
        "facts (the iteration door; accept order becomes irrelevant)",
    )
    p.add_argument(
        "--extend",
        action="store_true",
        help="append to the existing filter (refine a residual) instead of "
        "overwriting it",
    )
    p.add_argument(
        "--form",
        choices=("sum", "cascade"),
        default="sum",
        help="sum (QOP >= 3.3, default) or cascade (QOP 3.4.1)",
    )
    p.add_argument(
        "--config", default=None, help="scqo config.toml path (default: active selection)"
    )
    p.add_argument(
        "--clear",
        action="store_true",
        help="remove ALL exponential_filter taps from the target's z output "
        "(the fresh-line reset before a clean-slate characterization)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve + preview what would be written; save nothing",
    )
    args = p.parse_args(argv)

    if args.clear:
        if args.run or args.extend or args.form != "sum":
            p.error("--clear takes no --run/--extend/--form")
        out = clear_distortion(args.target, config_path=args.config,
                               dry_run=args.dry_run)
        verb = "would remove" if args.dry_run else "removed"
        print(f"{args.target}: {verb} {len(out['removed'])} exponential_filter tap(s)")
        for pair in out["removed"]:
            a, tau_ns = list(pair)
            print(f"    A={a:+.5g}  tau={tau_ns:.4g} ns")
        if out["state_dir"]:
            print(f"  {'target' if args.dry_run else 'saved to'}: {out['state_dir']}")
        if args.dry_run:
            print("  --dry-run: nothing written")
        return 0

    out = apply_distortion_from_state(
        args.target,
        run_id=args.run,
        replace=not args.extend,
        form=args.form,
        config_path=args.config,
        dry_run=args.dry_run,
    )

    verb = "would write" if args.dry_run else ("appended" if args.extend else "wrote")
    source = f"run {out['run_id']}" if out["run_id"] else "accepted facts"
    print(
        f"{args.target} ({out['channel']}, from {source}): {verb} "
        f"{len(out['exponential_filter'])} exponential_filter tap(s)"
    )
    for pair in out["exponential_filter"]:
        a, tau_ns = list(pair)
        print(f"    A={a:+.5g}  tau={tau_ns:.4g} ns")
    if out["scale"] != 1.0:
        print(f"    scale={out['scale']:.6g}  (apply to the flux waveform amplitude)")
    if out["state_dir"]:
        label = "target" if args.dry_run else "saved to"
        print(f"  {label}: {out['state_dir']}")
    if args.dry_run:
        print("  --dry-run: nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
