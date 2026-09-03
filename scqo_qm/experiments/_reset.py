"""The QM side of the neutral ``reset_method``: one door, opt-in carriers, refusals.

scqo widened ``reset_method`` to ``Literal["thermal", "active"]`` when active
reset landed on the Qblox backend. That widening is repo-wide — the field is on
the shared ``QubitResetParameters`` mixin, so ``reset_method="active"`` passes
validation on EVERY backend, including this one. Every QM shell resolves its
``reset_type`` through :func:`check_reset_method` instead of handing the probes a
literal, so an unrealizable method RAISES rather than thermalizing quietly: the
one failure this field cannot survive is a run that completes, fits fine, and
disagrees only on a wall clock nobody reads. ``reset_method`` crosses the probe
boundary verbatim (the neutral vocabulary was chosen to match QUAM's own
``reset_type``), so this is a guarded pass-through, not a translation.

ACTIVE reset is QUAM's ``BaseTransmon.reset_qubit_active`` — a repeat-until-success
loop: measure -> ``assign(state, I > pulse.threshold)`` -> ``wait(depletion_time
// 2)`` -> conditional pi, looping ``while_((I > pulse.rus_exit_threshold) &
(attempts < max_attempts))`` and ending on a fixed ``wait(500)`` (2 us) on xy. It
costs ~18.8 us against chipA's ~1.86 ms thermal wait — the ~100x that makes long
averaged sweeps affordable. The probes forward ``reset_type`` verbatim to
``qubit.reset(...)``; the four carriers additionally forward ``max_attempts``.

WHICH SHELLS MAY ASK FOR IT. Active reset thresholds every shot against the
readout discriminator ``single_shot_readout`` solved at ONE readout condition,
and it replaces the reset with a measurement. So it is valid exactly where the
readout condition is frozen for the whole run AND the reset is a genuine state
reset — the coherent-drive carriers (relaxation, ramsey, echo, power_rabi, the
two T1 trackers) plus ``qubit_spectroscopy``, whose saturation drive is a finite
pulse that has ended by the time the reset runs and which sweeps only the DRIVE
frequency. ``readout_frequency`` / ``readout_power`` SWEEP the readout condition;
``single_shot_readout`` IS the calibration itself (and a conditional pi driven by
the very threshold being measured biases the |1> blob). Each of those refuses BY
NAME (default DENY via the :data:`ACTIVE_RESET_ATTR` ClassVar). Opting a new
shell in later is one ``supports_active_reset = True`` and one line in the census
test — do NOT add it without a reason and hardware evidence.

FOUR THINGS ARE LOAD-BEARING here, each a SILENT failure if dropped:

1. ``active_reset_rounds`` maps onto QUAM's ``max_attempts`` as an UPPER BOUND, not
   an exact count. QM's loop starts ``attempts`` at 1 and exits early the moment
   ``I <= rus_exit_threshold``, so ``active_reset_rounds=1`` is one measure + one
   conditional pi with the ``while_`` never entered. (Qblox reads the same field
   as EXACTLY-N, because its conditional playback cannot exit a loop early — the
   neutral ``ACTIVE_RESET_ROUNDS_DESC`` documents the asymmetry.)

2. BOTH ``readout_threshold`` AND ``readout_rus_threshold`` are required, even at
   ``active_reset_rounds=1``: the ``while_((I > rus_exit_threshold) & ...)``
   condition is BUILT regardless of how many times the loop runs. On QM an
   uncalibrated value is ``None`` (not 0.0 as on Qblox), and ``I > None`` dies as a
   ``TypeError`` deep inside the QUA DSL naming nothing — so this file reads both
   through the governed device view and refuses by name before the program builds.
   The guard deliberately does NOT ride on ``use_state_discrimination``: active
   reset with averaged-I/Q readout is legal, it thresholds the RESET, not the data.

3. The post-measurement SETTLE lives INSIDE ``reset_qubit_active`` as
   ``wait(depletion_time // 2)`` — note ``wait()`` takes CLOCK CYCLES while
   ``depletion_time`` is ns, so the actual settle is ~2x the calibrated ns value
   (conservative, vendor-owned, not this file's to pick). A missing settle
   Stark-shifts the conditional pi and drifts the fit with nothing attributing it
   to the reset, so active reset REFUSES an ungoverned depletion. QUAM's
   ``ReadoutResonator.depletion_time`` defaults to 16 ns and is never ``None``, so
   the honest QM translation of Qblox's NaN-refusal is to refuse the 16 ns FACTORY
   DEFAULT: a real proposal is ``depletion_factor / (2 pi x kappa_tot_hz)`` from
   ``resonator_spectroscopy`` and is never exactly 16 ns by luck. (A genuinely
   broad resonator whose proposal rounds to 16 ns is the one false positive; the
   remedy message points at the per-run ``readout_depletion_ns`` override.)

4. ``thermalization_time_ns`` + ``active`` is REFUSED, not ignored. The per-run
   thermalization override (``QMBackend._thermalization_override``) brackets the
   QUA build moving ``q.thermalization_time_ns``, which ``reset_qubit_active``
   never reads — honouring both would silently drop the one the caller typed.
   Refusing here means the override context manager never sees the combination.

WHERE THE EVIDENCE CAME FROM. The QUA path itself has lab mileage through the
qualibrate graph scripts (``calibrations/offline_graph/LCH_graph_*.py`` run
``reset_type="active"`` routinely); what is missing is evidence for the scqo
SHELLS specifically, and this driver has been bitten before by a readout-frame
convention that compiled clean and was wrong on the instrument. Offline tests
prove the policy and that the program builds; the feedback loop itself is
hardware, and the chipA walkthrough that closes that gap is the REMAINING step
(update this note to the validation date once it is done). ``QMBackend.acquire``
re-checks through this same function before ``probe()`` — the backstop that fires
even for a shell that forgot to call the helper.
"""

from __future__ import annotations

from typing import Any

from scqo.experiments._depletion import depletion_wait_ns

#: The always-realized method. Kept as a name so the returned value and the
#: refusal messages cannot drift apart.
THERMAL = "thermal"

#: Class attribute a QM shell sets to True to declare active reset VALID for its
#: sequence. Read with ``getattr(..., False)`` so the default is DENY: a shell
#: added later that never thought about it fails closed with a named error
#: instead of silently running a wrong sequence.
ACTIVE_RESET_ATTR = "supports_active_reset"

#: The four that opt in, for the refusal message only (the authority is the class
#: attribute — this is a hint, and the census test keeps it honest).
_CARRIERS = "qubit_relaxation, qubit_ramsey, qubit_echo, qubit_power_rabi"

#: QUAM ``ReadoutResonator.depletion_time`` factory default (ns). An ungoverned
#: depletion sits at exactly this, and active reset refuses it — see rule (3).
_FACTORY_DEPLETION_NS = 16.0


def _experiment_name(experiment: Any) -> str:
    return getattr(type(experiment), "name", type(experiment).__name__)


def _missing_discriminator_knobs(experiment: Any, target: str) -> list[str]:
    """The discriminator knobs ``target`` cannot answer, in the order active reset
    reads them. Empty means both are calibrated.

    Reads through the GOVERNED device view (what ``single_shot_readout`` writes and
    ``scqo accept`` commits), which under ``state_sync="pull"`` agrees with the
    QUAM tree the probe builds from. An uncalibrated knob is ``None`` in QUAM; the
    recording view reads strict, so that surfaces as a ``KeyError`` ("no value
    yet") rather than a returned ``None`` — both mean uncalibrated here.
    """
    view = experiment.device.channel(target, "readout")
    missing: list[str] = []
    for knob in ("readout_threshold", "readout_rus_threshold"):
        try:
            value = getattr(view, knob)
        except KeyError:
            missing.append(knob)
            continue
        if value is None:
            missing.append(knob)
    return missing


def _resolved_depletion_ns(experiment: Any, target: str) -> float | None:
    """``target``'s settle in ns, or ``None`` when nothing has governed it.

    ``depletion_wait_ns`` returns ``None`` for a NaN/unset knob but RAISES
    ``KeyError`` when the vendor never seeded the field at all (the QM stub, and a
    partly-wired node) — both are "ungoverned" to us, so catch it.
    """
    try:
        return depletion_wait_ns(experiment, target)
    except KeyError:
        return None


def check_reset_method(experiment: Any) -> str:
    """This run's QM reset method, after refusing everything QM cannot honour.

    Returns ``"thermal"`` or ``"active"``; raises ``ValueError`` naming the
    experiment and the reason. Side-effect free, because it is called TWICE: once
    by every shell's ``probe()`` and once by ``QMBackend.acquire`` before
    ``probe()`` — the backstop that fires even if a shell forgets the helper.

    Order is load-bearing: the pure checks (which run on a device-less shell)
    come before any device read, and the discriminator before the settle —
    without a discriminator active reset is not a reset at all, whereas a missing
    settle is a real but narrower defect, so the deeper problem is reported first.
    """
    params = experiment.params
    method = getattr(params, "reset_method", THERMAL)
    name = _experiment_name(experiment)

    if method == THERMAL:
        return THERMAL
    if method != "active":
        raise ValueError(
            f"{name}: reset_method={method!r} is not realized on the Quantum "
            f"Machines backend (this driver implements 'thermal' and 'active')."
        )

    if not getattr(type(experiment), ACTIVE_RESET_ATTR, False):
        raise ValueError(
            f"{name}: reset_method='active' is not valid for this experiment. "
            f"Active reset thresholds every shot against the readout "
            f"discriminator, so it needs the readout condition held at the point "
            f"single_shot_readout calibrated — which this experiment does not do "
            f"(it sweeps the readout, or it IS the discriminator's calibration, "
            f"or its reset is a driven dwell rather than a state reset). "
            f"Supported: {_CARRIERS}. Use reset_method='thermal' here."
        )

    if getattr(params, "thermalization_time_ns", None) is not None:
        # Refused, not ignored: the override moves q.thermalization_time_ns, which
        # reset_qubit_active never reads, so honouring both would silently drop
        # the one the caller typed. Same discipline as _thermalization_override's
        # own "refusing to run with the override silently ignored".
        raise ValueError(
            f"{name}: thermalization_time_ns overrides the THERMAL wait "
            f"(q.thermalization_time_ns, baked into the build), which active "
            f"reset never plays — so the two together would silently ignore it. "
            f"Drop one: keep reset_method='active', or set reset_method='thermal' "
            f"to use the override."
        )

    # Device reads last. Discriminator before the settle (see the docstring).
    for target in experiment.params.targets:
        missing = _missing_discriminator_knobs(experiment, target)
        if missing:
            raise ValueError(
                f"{name}: reset_method='active' needs {target}'s readout "
                f"discriminator, and {' and '.join(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} not calibrated (QUAM's "
                f"uncalibrated value is None, which would die as a TypeError deep "
                f"inside the QUA DSL naming nothing). BOTH readout_threshold and "
                f"readout_rus_threshold are required even at active_reset_rounds=1 "
                f"— the repeat-until-success exit condition is built regardless of "
                f"the round count. Run `scqo run single_shot_readout`, review with "
                f"`scqo accept`, then re-run — or set it directly with "
                f"`scqo set {target}.readout_threshold=...`."
            )
    for target in experiment.params.targets:
        settle_ns = _resolved_depletion_ns(experiment, target)
        if settle_ns is None or settle_ns == _FACTORY_DEPLETION_NS:
            raise ValueError(
                f"{name}: reset_method='active' needs {target}'s "
                f"readout_depletion_s, and it has never been governed (it is "
                f"unset or at QUAM's {int(_FACTORY_DEPLETION_NS)} ns factory "
                f"default — a real proposal, depletion_factor / (2 pi x "
                f"kappa_tot_hz), is never exactly that). That is the settle "
                f"reset_qubit_active plays between measuring and the conditional "
                f"pi; without it the readout photons Stark-shift that pi and the "
                f"fit drifts in a way nothing attributes to the reset. Run "
                f"`scqo run resonator_spectroscopy`, review with `scqo accept`, "
                f"then re-run — or set it directly with "
                f"`scqo set {target}.readout_depletion_s=...`."
            )
    return "active"


def reset_max_attempts(experiment: Any) -> int:
    """scqo's ``active_reset_rounds`` as QUAM's ``max_attempts`` — the one place
    the mapping is spelled. QM treats it as an UPPER BOUND (the loop exits early on
    success), where Qblox plays exactly that many rounds; see rule (1)."""
    return int(experiment.params.active_reset_rounds)
