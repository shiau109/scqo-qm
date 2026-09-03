"""Neutral QUAM field accessors — the single place that knows how SCQO's neutral
calibration fields map onto QUAM attribute paths.

The scqo backend's CHANNEL views (``scqo_qm/backend/qm_backend.py``:
``QMDriveChannel`` / ``QMReadoutChannel`` / ``QMFluxChannel``) do every neutral
get/set through these primitives, so the ``f_01`` / ``resonator.RF_frequency`` /
``xy.operations['x180'].amplitude`` mapping is defined exactly once. (The retired
LCH qualibrate shells were the second caller; the vendored official nodes write
QUAM directly and never import this module.) The whole-tree audits
(``flux_point_problems`` / ``flux_headroom_*`` / ``drive_frequency_problems``) are
the same knowledge applied once at session construction by ``scqo_backend.py``.

Pure attribute access on a passed QUAM ``qubit``: no qm/quam import here, so this stays
importable without an instrument (and unit-testable with a stub qubit).

Neutral field -> QUAM path (the neutral names are the greenfield ones; which CHANNEL
entity owns each is in scqo_qm/backend/fieldmap.py, keyed by channel KIND)
    readout_freq_hz       <-> q.resonator.RF_frequency  (and q.resonator.f_01 when present)
    drive_freq_hz         <-> q.f_01 and q.xy.RF_frequency (both written to the same
                              absolute value; drive_frequency_problems refuses a tree
                              where they differ)
    pi_amp                <-> q.xy.operations['x180'].amplitude
    pi_duration_s         <-> q.xy.operations['x180'].length (s <-> ns, 4 ns grid)
    drive_amp             <-> q.xy.operations['saturation'].amplitude
    readout_duration_s    <-> q.resonator.operations['readout'].length (s <-> ns)
    readout_integration_s <-> ...operations['readout'].integration_weights
                              (window w -> [(1, w), (0, length - w)]; default ref when w == length)
    readout_rotation_rad  <-> ...operations['readout'].integration_weights_angle (rad, absolute)
    readout_threshold     <-> ...operations['readout'].threshold
    readout_rus_threshold <-> ...operations['readout'].rus_exit_threshold
    idle_flux             <-> q.z.<flux_point>_offset       (qubit flux channel)
                          <-> qp.coupler.<flux_point>_offset (COUPLER flux channel: the
                              coupler mode's own q*_z entity, decouple/interaction/arbitrary)
"""

from __future__ import annotations

import math
from typing import Any

#: The operation whose amplitude is the calibrated pi pulse (the neutral ``pi_amp``).
PI_OPERATION = "x180"


# ----------------------------------------------------------------- readout / resonator
def get_readout_freq(qubit: Any) -> float:
    return float(qubit.resonator.RF_frequency)


def set_readout_freq(qubit: Any, value: float) -> None:
    """Set the resonator frequency to an absolute Hz: ``RF_frequency`` and, when the
    resonator carries it, ``f_01`` (the two are kept equal)."""
    value = float(value)
    qubit.resonator.RF_frequency = value
    if hasattr(qubit.resonator, "f_01"):
        qubit.resonator.f_01 = value


# ----------------------------------------------------------------------- drive (f_01)
def get_drive_freq(qubit: Any) -> float:
    return float(qubit.f_01)


def set_drive_freq(qubit: Any, value: float) -> None:
    """Set the drive frequency to an absolute Hz on BOTH stores: ``f_01`` (what
    scqo reads back as drive_freq_hz) and ``xy.RF_frequency`` (what the drive
    line plays - QUAM derives intermediate_frequency = RF - LO from it, and every
    probe's update_frequency starts there). The official nodes write the pair the
    same way, and :func:`drive_frequency_problems` refuses a tree where they
    disagree, so a mismatched tree is REPAIRED by the first write rather than
    carried along as a fixed offset (the old delta semantics)."""
    value = float(value)
    qubit.f_01 = value
    qubit.xy.RF_frequency = value


# ---------------------------------------------------------------------- readout amplitude
#: The readout operation whose amplitude is the neutral ``readout_amp``.
READOUT_OPERATION = "readout"


def get_readout_amp(qubit: Any, operation: str = READOUT_OPERATION) -> float:
    return float(qubit.resonator.operations[operation].amplitude)


def set_readout_amp(qubit: Any, value: float, *, operation: str = READOUT_OPERATION) -> None:
    """Write the readout pulse amplitude (within the current output-power config;
    large power reconfiguration — FEM gain — stays with the qualibrate power node's
    ``set_output_power`` path)."""
    qubit.resonator.operations[operation].amplitude = float(value)


# ----------------------------------------------------- readout duration / window
#: The QUAM reference restoring ``ReadoutPulse``'s default weights
#: ([(1, length)] — window == pulse, following the length by reference). Written
#: back whenever the window equals the pulse so state.json keeps the follow form.
DEFAULT_INTEGRATION_WEIGHTS_REF = "#./default_integration_weights"


def _grid_ns(value_s: float, what: str) -> int:
    """A seconds duration as integer ns, REFUSED unless a positive multiple of
    4 ns (the QM pulse/weights grid — also the portable neutral contract, so a
    value accepted here is accepted verbatim on the Qblox backend too)."""
    ns = float(value_s) * 1e9
    grid = round(ns)
    if abs(ns - grid) > 1e-3 or grid <= 0 or grid % 4:
        raise ValueError(
            f"{what}={value_s!r} s: must be a positive multiple of 4 ns "
            f"(the QM pulse/integration-weights grid; no silent rounding)")
    return int(grid)


def _window_ns(pulse: Any) -> int:
    """The effective integration window: total ns of NONZERO integration weight.

    Default weights resolve to [(1, length)] -> the full pulse; a zero-padded
    constant list -> its support; per-sample float weights (1 ns each) -> the
    nonzero-sample count. Shaped (optimized) weights report their total nonzero
    duration — the window length is well-defined even when the shape is not ours."""
    weights = pulse.integration_weights  # QUAM resolves the default reference
    if weights and isinstance(weights[0], (tuple, list)):
        return int(sum(l for w, l in weights if w))
    return int(sum(1 for w in weights if w))


def _set_weights(pulse: Any, value: Any) -> None:
    """Write the integration_weights slot. quam REFUSES overwriting a reference
    attribute with a plain value — the slot must be released (set to None)
    first; setting TO a reference string needs no release but survives one."""
    pulse.integration_weights = None
    pulse.integration_weights = value


def get_readout_duration(qubit: Any, operation: str = READOUT_OPERATION) -> float:
    """Readout pulse length in seconds (QUAM stores ns). Divided by the EXACT
    1e9 (never * 1e-9): division rounds correctly, so 2000 ns reads back as the
    same float the literal 2e-6 parses to — no 1-ulp echo records."""
    return float(qubit.resonator.operations[operation].length) / 1e9


def set_readout_duration(qubit: Any, value: float, *,
                         operation: str = READOUT_OPERATION) -> None:
    """Set the readout pulse length (seconds, multiple of 4 ns).

    The integration weights must span exactly the pulse, so they are rebuilt
    around the new length with the current window PRESERVED numerically —
    clamped down only when the pulse shrinks below it (the scqo layer records
    that as a COUPLED readout_integration_s change). Shaped/optimized weights do
    not survive a length change (they are physically stale anyway): the rebuild
    is constant-window; integration_weights_angle applies on top and is untouched."""
    pulse = qubit.resonator.operations[operation]
    new_ns = _grid_ns(value, "readout_duration_s")
    window = _window_ns(pulse)  # read against the OLD length before changing it
    pulse.length = new_ns
    if window >= new_ns:
        _set_weights(pulse, DEFAULT_INTEGRATION_WEIGHTS_REF)
    else:
        _set_weights(pulse, [(1.0, window), (0.0, new_ns - window)])


def get_readout_integration(qubit: Any, operation: str = READOUT_OPERATION) -> float:
    """Integration window in seconds (nonzero-weight duration; see _window_ns).
    / 1e9, not * 1e-9 — see get_readout_duration."""
    return float(_window_ns(qubit.resonator.operations[operation])) / 1e9


def set_readout_integration(qubit: Any, value: float, *,
                            operation: str = READOUT_OPERATION) -> None:
    """Set the integration window (seconds) as zero-padded constant weights.

    [(1, w), (0, length - w)] spans the pulse exactly (the QUAM contract); the
    calibrated integration_weights_angle rotates whatever weights are set, so it
    survives untouched. A window equal to the pulse restores the default
    reference; a longer one is REFUSED — the QM demodulation cannot outlive the
    pulse, and the neutral field promises only what both backends realize."""
    pulse = qubit.resonator.operations[operation]
    length_ns = int(pulse.length)
    window = _grid_ns(value, "readout_integration_s")
    if window > length_ns:
        raise ValueError(
            f"readout_integration_s={value!r} s exceeds the readout pulse "
            f"({length_ns} ns): the integration weights cannot span past the "
            f"pulse - raise readout_duration_s first (or set both in one "
            f"command; the pulse pushes first)")
    if window == length_ns:
        _set_weights(pulse, DEFAULT_INTEGRATION_WEIGHTS_REF)
    else:
        _set_weights(pulse, [(1.0, window), (0.0, length_ns - window)])


# -------------------------------------------------------------- readout discriminator
# The single-shot discriminator settings — 1:1 with the readout operation's demod
# knobs (no chain solve). integration_weights_angle is the ABSOLUTE demod rotation
# (radians); single_shot_readout proposes a new absolute value (current - measured
# delta) through the governed suggestion flow, never accumulating in place.
def get_readout_rotation(qubit: Any, operation: str = READOUT_OPERATION) -> float:
    return float(qubit.resonator.operations[operation].integration_weights_angle)


def set_readout_rotation(qubit: Any, value: float, *, operation: str = READOUT_OPERATION) -> None:
    qubit.resonator.operations[operation].integration_weights_angle = float(value)


def get_readout_threshold(qubit: Any, operation: str = READOUT_OPERATION) -> float:
    return float(qubit.resonator.operations[operation].threshold)


def set_readout_threshold(qubit: Any, value: float, *, operation: str = READOUT_OPERATION) -> None:
    qubit.resonator.operations[operation].threshold = float(value)


def get_readout_depletion(qubit: Any) -> float:
    """The post-readout photon-depletion wait, ns -> seconds.

    Unlike the thermal wait, this needs no custom transmon class: QUAM's
    ``ReadoutResonator.depletion_time`` is a plain settable int field (ns), and
    QUAM already spends it in four places — after every measurement in the
    single-qubit gates, and as ``depletion_time // 2`` inside
    ``reset_qubit_active``. Until scqo calibrated it, it sat at its 16 ns default
    with nothing governing it."""
    return float(qubit.resonator.depletion_time) / 1e9


def set_readout_depletion(qubit: Any, value: float) -> None:
    """Write the depletion wait (seconds -> ns, rounded to the 4 ns QUA grid).

    Rounded, not refused, for the same reason as :func:`set_thermalization_time`:
    resonator_spectroscopy writes ``depletion_factor / (2 pi x kappa_tot_hz)``,
    which is never on the grid by luck. ZERO IS LEGAL — "measured, needs no
    settle" is a real answer, distinct from never having been calibrated — so
    only a negative value is refused. The field is typed ``int``, so the write
    is an int."""
    ns = float(value) * 1e9
    if ns < 0:
        raise ValueError(f"readout_depletion_s={value!r} s: must not be negative")
    qubit.resonator.depletion_time = int(ns / 4) * 4


def get_readout_rus_threshold(qubit: Any, operation: str = READOUT_OPERATION) -> float:
    return float(qubit.resonator.operations[operation].rus_exit_threshold)


def set_readout_rus_threshold(qubit: Any, value: float, *, operation: str = READOUT_OPERATION) -> None:
    qubit.resonator.operations[operation].rus_exit_threshold = float(value)


# ------------------------------------------------------------------------ idle flux
#: The named flux points the scqo path PINS, because they are the ones every
#: probe's ``initialize_qpu`` actually applies: quam_builder defaults
#: ``flux_point="joint"``, whose qubit leg is ``to_joint_idle()`` ->
#: ``joint_offset`` and whose coupler leg is ``apply_all_couplers_to_min()`` ->
#: ``to_decouple_idle()`` -> ``decouple_offset``.
#:
#: These accessors stay GENERIC over the vendor vocabulary (the qualibrate path
#: legitimately runs "independent"), so nothing here refuses a different point.
#: What refuses is :func:`flux_point_problems`, called once by the scqo backend
#: factory — because a line that DECLARES one point while the probes APPLY
#: another makes ``idle_flux`` a knob that reads and writes a number the
#: hardware never sees. That was live on 5Q4C until 2026-07-29: all five z lines
#: declared "independent" while every run biased them at ``joint_offset``, so
#: both the arch fits' and the resonator map's accepted sweet spots were inert.
GOVERNED_FLUX_POINT = "joint"
GOVERNED_COUPLER_FLUX_POINT = "off"


def flux_point_problems(machine: Any) -> list[str]:
    """Every reason this QUAM tree's flux points disagree with what runs.

    Empty list = the governed knob and the applied bias are the same field.
    Pure (no I/O, no QUA); the caller decides how loudly to fail.
    """
    problems: list[str] = []

    # None (not an empty set) when the tree cannot answer: an absent attribute
    # means "cannot determine", and flagging every qubit would be a false alarm.
    raw_active = getattr(machine, "active_qubits", None)
    active = None if raw_active is None else {getattr(q, "name", None) for q in raw_active}

    for name, qubit in getattr(machine, "qubits", {}).items():
        z = getattr(qubit, "z", None)
        if z is None:
            continue  # fixed-frequency qubit: no flux line, nothing to govern
        point = getattr(z, "flux_point", None)
        if point != GOVERNED_FLUX_POINT:
            problems.append(
                f"qubits.{name}.z.flux_point = {point!r}, but every probe biases "
                f"at {GOVERNED_FLUX_POINT!r} ({GOVERNED_FLUX_POINT}_offset). "
                f"idle_flux would read/write "
                f"{getattr(z, f'{point}_offset', '?')} while the hardware holds "
                f"{getattr(z, f'{GOVERNED_FLUX_POINT}_offset', '?')} V. "
                f"Set it to {GOVERNED_FLUX_POINT!r} in state.json.")
        elif active is not None and name not in active:
            # apply_all_flux_to_joint_idle parks INACTIVE qubits at min_offset,
            # so their idle_flux is a lie in exactly the same way.
            problems.append(
                f"qubits.{name} has a flux line but is not in active_qubit_names: "
                f"the joint path parks inactive qubits at min_offset, so its "
                f"idle_flux ({GOVERNED_FLUX_POINT}_offset) is not what the "
                f"hardware holds. Activate it or remove its flux rider.")

    for name, pair in getattr(machine, "qubit_pairs", {}).items():
        coupler = getattr(pair, "coupler", None)
        if coupler is None:
            continue
        point = getattr(coupler, "flux_point", None)
        if point != GOVERNED_COUPLER_FLUX_POINT:
            problems.append(
                f"qubit_pairs.{name}.coupler.flux_point = {point!r}, but the "
                f"joint path decouples every coupler "
                f"(to_decouple_idle -> decouple_offset), which is "
                f"{GOVERNED_COUPLER_FLUX_POINT!r}. Set it to "
                f"{GOVERNED_COUPLER_FLUX_POINT!r} in state.json.")

    return problems


# ------------------------------------------------------------ drive frequency
#: How far ``f_01`` and ``xy.RF_frequency`` may drift before the drive audit
#: refuses. ulp noise at 5 GHz is ~1e-6 Hz and a hand edit is kHz or more, so
#: 1 Hz separates float echo from a real disagreement with room to spare.
DRIVE_FREQ_TOLERANCE_HZ = 1.0


def _finite_hz(value: Any):
    """``value`` as a finite float, or None when the tree cannot answer: unset
    (None), a ``#``-reference that did not resolve (QUAM hands back the string),
    NaN or inf."""
    try:
        hz = float(value)
    except (TypeError, ValueError):
        return None
    return hz if math.isfinite(hz) else None


def drive_frequency_problems(machine: Any) -> list[str]:
    """Every qubit whose ``f_01`` and ``xy.RF_frequency`` disagree.

    scqo's ``drive_freq_hz`` READS ``f_01`` while the drive line PLAYS
    ``xy.RF_frequency`` (QUAM's ``intermediate_frequency = RF - LO`` feeds every
    probe's ``update_frequency``), so a tree carrying two different numbers makes
    the knob report a frequency the hardware never emits. ``set_drive_freq``
    writes both and the official nodes write both; a hand edit of one, or a node
    that moves only the RF, is the remaining way in - which is why this is a
    startup audit rather than a setter check.

    Skips what it cannot judge: no ``xy`` (nothing plays), ``f_01`` None (an
    uncalibrated qubit - legitimate on a real state file), or an RF that is not
    a finite number. Pure (no I/O, no QUA); the caller decides how loudly to
    fail. Empty list = compliant.
    """
    problems: list[str] = []
    for name, qubit in getattr(machine, "qubits", {}).items():
        xy = getattr(qubit, "xy", None)
        if xy is None:
            continue
        f01 = _finite_hz(getattr(qubit, "f_01", None))
        rf = _finite_hz(getattr(xy, "RF_frequency", None))
        if f01 is None or rf is None:
            continue  # cannot determine - never an accusation
        if abs(f01 - rf) > DRIVE_FREQ_TOLERANCE_HZ:
            problems.append(
                f"qubits.{name}.f_01 = {f01} Hz but qubits.{name}.xy.RF_frequency = "
                f"{rf} Hz (differ by {abs(f01 - rf)} Hz). The drive line PLAYS "
                f"xy.RF_frequency (QUAM's intermediate_frequency = RF - LO feeds every "
                f"probe's update_frequency) while scqo's drive_freq_hz READS f_01, so "
                f"the knob would show a number the hardware never emits. Edit "
                f"state.json so both hold the intended value (the audit cannot know "
                f"which one is right).")
    return problems


#: Every named standing-bias attribute a flux channel can carry — the qubit
#: FluxLine vocabulary plus the TunableCoupler one. Audited together because the
#: DAC does not care which name a bias arrived under.
_IDLE_OFFSET_ATTRS = (
    "joint_offset", "independent_offset", "min_offset", "arbitrary_offset",
    "decouple_offset", "interaction_offset",
)


def _flux_channels(machine: Any):
    """(label, channel) for every flux-emitting channel in the tree."""
    for name, qubit in getattr(machine, "qubits", {}).items():
        z = getattr(qubit, "z", None)
        if z is not None:
            yield f"qubits.{name}.z", z
    for name, pair in getattr(machine, "qubit_pairs", {}).items():
        coupler = getattr(pair, "coupler", None)
        if coupler is not None:
            yield f"qubit_pairs.{name}.coupler", coupler


def _amplitude_of(op: Any):
    """A stored op's amplitude as a float, or None when it is not a number.

    QUAM references (``'#./const'``) and unset fields are legitimately
    non-numeric — the audit skips them rather than inventing a verdict.
    """
    try:
        return float(getattr(op, "amplitude", None))
    except (TypeError, ValueError):
        return None


def _flux_headroom_findings(machine: Any):
    """(clips, message) for every flux-headroom finding in the tree.

    ``clips`` separates the two severities, and the split is the whole design:

    * **clips=True** — the hardware emits something OTHER than what was asked. A
      stored op peak or a standing bias at/past the rail is truncated by the DAC,
      the fit still converges, and the simulator shows nothing. Fatal.
    * **clips=False** — the hardware emits exactly what was asked, there is just
      less of it available than there could be. An undersized ``const`` caps how
      far any volts-based sweep on that line can reach; it corrupts nothing, and
      the ``amplitude_scale`` bound in ``probes/_flux_limits.py`` already refuses
      the moment a probe actually asks for more than the reach. Advisory.

    Making the second one fatal would break every session on configs that work:
    the live 5Q4C couplers sit at 0.15 V and have always run.
    """
    from scqo_qm.experiments._flux_limits import (
        _CONST_AMP_RAIL_FRACTION,
        dac_rail_v,
        rail_remedy,
    )

    for label, channel in _flux_channels(machine):
        rail = dac_rail_v(channel)

        operations = getattr(channel, "operations", None) or {}
        for op_name, op in operations.items():
            amplitude = _amplitude_of(op)
            if amplitude is None:
                continue
            if abs(amplitude) >= rail:
                yield True, (
                    f"{label}.operations['{op_name}'].amplitude = {amplitude} V, "
                    f"at or past the port's {rail} V full scale: the stored "
                    f"waveform peak is clipped on hardware and the simulator "
                    f"hides it. "
                    + rail_remedy(channel, name=f"{label} '{op_name}'",
                                  needed_v=abs(amplitude), rail=rail))

        const = _amplitude_of(operations.get("const")) if "const" in operations else None
        expected = rail * _CONST_AMP_RAIL_FRACTION
        if const is not None and abs(const) < rail and abs(const - expected) > 1e-6 * rail:
            yield False, (
                f"{label}.operations['const'].amplitude = {const} V, but the flux "
                f"convention is half the port's full scale, {expected} V (rail "
                f"{rail} V). Nothing clips - but no volts-based sweep on this line "
                f"can reach the rail, because QUA caps amplitude_scale at +/-2, so "
                f"the reachable excursion is +/-{2 * abs(const)} V instead of "
                f"+/-{rail} V. Set it to {expected}.")

        for attr in _IDLE_OFFSET_ATTRS:
            value = getattr(channel, attr, None)
            if value is None:
                continue
            try:
                offset = float(value)
            except (TypeError, ValueError):
                continue
            if abs(offset) > rail:
                yield True, (
                    f"{label}.{attr} = {offset} V, past the port's {rail} V full "
                    f"scale: the standing bias alone already clips, before any "
                    f"pulse rides on it. "
                    + rail_remedy(channel, name=f"{label}.{attr}",
                                  needed_v=abs(offset), rail=rail))


def flux_headroom_problems(machine: Any) -> list[str]:
    """Every way this tree's flux config would CLIP — the fatal half.

    The per-sweep checks in ``probes/_flux_limits.py`` refuse ONE probe when it
    asks for too much. This is the other half: a whole-tree audit run once at
    session start, so a config that cannot work is reported completely and up
    front — "these three ports need amplified mode" — instead of one probe dying
    on the first one it happens to touch.

    Pure (no I/O, no QUA); the caller decides how loudly to fail.
    """
    return [msg for clips, msg in _flux_headroom_findings(machine) if clips]


def flux_headroom_warnings(machine: Any) -> list[str]:
    """Flux config that works but is leaving range on the table — advisory.

    Today that is exactly the ``const`` = rail/2 convention. This is the ONE
    place it is enforced: the per-sweep helpers deliberately do not, because an
    undersized ``const`` clips nothing and refusing there would reject working
    configs.
    """
    return [msg for clips, msg in _flux_headroom_findings(machine) if not clips]


def get_idle_flux(qubit: Any) -> float:
    """Standing z-line idle bias (V): the offset SELECTED by ``z.flux_point``
    (joint/independent/min/arbitrary; ``zero`` reads as 0.0). Which named point
    is active stays vendor config — the neutral knob is the bias AT that point.

    Under scqo that point must be :data:`GOVERNED_FLUX_POINT`; see
    :func:`flux_point_problems` for why and where it is enforced."""
    z = qubit.z
    if z.flux_point == "zero":
        return 0.0
    return float(getattr(z, f"{z.flux_point}_offset"))


def set_idle_flux(qubit: Any, value: float) -> None:
    """Write the active flux point's offset (V). Applied to hardware by the next
    ``initialize_qpu`` (every probe runs it per batch), like any QUAM knob."""
    z = qubit.z
    if z.flux_point == "zero":
        raise ValueError(
            "z.flux_point='zero': the idle bias is fixed at 0 V - select "
            "joint/independent/min/arbitrary in the vendor config first")
    setattr(z, f"{z.flux_point}_offset", float(value))


# ------------------------------------------------------------- coupler idle flux
#: TunableCoupler flux_point -> the offset attribute it selects. The coupler's
#: named points are its OWN vocabulary (off/on/arbitrary/zero), not the qubit
#: FluxLine's (joint/independent/min/...), so the two idle-flux accessors cannot
#: share one getattr: this map IS the difference. ``zero`` reads 0 V and refuses
#: writes, exactly like the qubit side.
COUPLER_FLUX_POINTS = {
    "off": "decouple_offset",          # the interaction-OFF standing bias
    "on": "interaction_offset",        # the interaction-ON standing bias
    "arbitrary": "arbitrary_offset",
}


def get_coupler_idle_flux(coupler: Any) -> float:
    """Standing bias (V) of a QUAM ``TunableCoupler`` — the offset SELECTED by its
    ``flux_point``.

    Since the greenfield model a coupler is an ordinary roster MODE and its
    standing bias is ``idle_flux`` on its own flux channel (the old pair-level
    coupler_decouple_v / coupler_interaction_v pair is gone): WHICH named point is
    active stays vendor config, and the neutral knob is the bias AT that point.
    """
    point = coupler.flux_point
    if point == "zero":
        return 0.0
    attr = COUPLER_FLUX_POINTS.get(point)
    if attr is None:
        raise ValueError(
            f"coupler flux_point={point!r} is not one of "
            f"{sorted(COUPLER_FLUX_POINTS) + ['zero']} - the standing bias it "
            f"selects is undefined")
    return float(getattr(coupler, attr))


def set_coupler_idle_flux(coupler: Any, value: float) -> None:
    """Write the active coupler point's offset (V); ``zero`` is refused (the
    point IS 0 V - switch flux_point in the vendor config first)."""
    point = coupler.flux_point
    if point == "zero":
        raise ValueError(
            "coupler flux_point='zero': the idle bias is fixed at 0 V - select "
            "off/on/arbitrary in the vendor config first")
    attr = COUPLER_FLUX_POINTS.get(point)
    if attr is None:
        raise ValueError(
            f"coupler flux_point={point!r} is not one of "
            f"{sorted(COUPLER_FLUX_POINTS) + ['zero']} - refusing to guess "
            f"which offset to write")
    setattr(coupler, attr, float(value))


# --------------------------------------------------------------- flux line delay
def get_flux_delay(flux_channel: Any) -> float:
    """Output-path delay of a flux SingleChannel, ns -> seconds.

    ``flux_channel`` is the FLUX-carrying QUAM channel the scqo flux view wraps —
    a qubit's ``z`` FluxLine or a ``TunableCoupler`` (both SingleChannels with an
    ``opx_output`` port). The delay is a PORT-level ``int`` ns field
    (``LFFEMAnalogOutputPort.delay``, 0 by default), shared by everything on that
    physical DAC output; on a per-qubit z wire that is per-qubit in practice.
    """
    return float(flux_channel.opx_output.delay) / 1e9


def set_flux_delay(flux_channel: Any, value: float) -> None:
    """Write the flux line's output-port delay (seconds -> int ns).

    Calibrated by qubit_xyz_delay so a Z pulse and the XY drive it accompanies
    arrive together. The port field is typed ``int`` ns (1 ns resolution, not the
    4 ns wait grid), so the write rounds to the nearest nanosecond. A negative
    delay is refused (a port delay only pushes operations LATER)."""
    ns = float(value) * 1e9
    if ns < 0:
        raise ValueError(f"flux_delay_s={value!r} s: must not be negative")
    flux_channel.opx_output.delay = int(round(ns))


# ------------------------------------------------------------------------- pi amplitude
#: The QUAM storage nodes each drive-amplitude family owns. ``x180``/``x90``/``y90``
#: etc. are usually reference ALIASES that follow their DragCosine node (``_set_op_amp``
#: skips string references), but are written anyway for non-DragCosine setups where
#: they are real pulses.
#:
#: ``-x90_DragCosine`` is a SECOND pi/2 node, not a string alias (the negative sense
#: comes from ``axis_angle = pi`` rather than a negated amplitude). Which of its
#: fields are references varies by config: the current live states reference its
#: ``amplitude`` (and ``alpha``/``detuning``) to ``x90_DragCosine``, but it has also
#: shipped with an independent literal amplitude -- chipA q1 once ran at
#: x90 = 0.07302 with -x90 = 0.09329 against a correct 0.14310. Hence the write
#: covers the whole family, and ``_set_op_amp``/``_set_op_alpha`` skip any field
#: that is a ``#``-reference (it already follows the real node; a literal would
#: sever it) while writing independent literals so the family cannot disagree.
_AMP_NODES: dict[str, tuple[str, ...]] = {
    "x180": ("x180_DragCosine", "x180"),
    "x90": ("x90_DragCosine", "x90", "-x90_DragCosine", "-x90"),
}


def _amp_nodes(operation: str) -> tuple[str, ...]:
    """The storage nodes a read/write of ``operation`` must reach.

    An operation outside the two known families is its own node alone: a bespoke
    pulse has no family to keep consistent."""
    for nodes in _AMP_NODES.values():
        if operation in nodes:
            return nodes
    return (operation,)


def get_pi_amp(qubit: Any, operation: str = PI_OPERATION) -> float:
    """Read the amplitude of the operation's family from its first REAL storage node
    (string-reference aliases carry no number of their own)."""
    if not (hasattr(qubit, "xy") and hasattr(qubit.xy, "operations")):
        return 0.0
    ops = qubit.xy.operations
    for name in _amp_nodes(operation):
        try:
            op = ops[name]
        except (KeyError, TypeError):
            continue
        if not isinstance(op, str) and hasattr(op, "amplitude"):
            return float(op.amplitude)
    return float(ops[operation].amplitude)


def set_pi_amp(qubit: Any, value: float, *, operation: str = PI_OPERATION) -> None:
    """Write a drive amplitude onto every storage node of the operation's family.

    There is deliberately **no** ``lock_x90``. The pi/2 pulse is calibrated in its own
    right (``qubit_deterministic_benchmarking``) and carries its own neutral knob,
    ``pi_amp_x90``, so coupling it to half the pi amplitude would silently overwrite a
    real calibration with a derived guess. One operation, one family, no mode flag.
    """
    val = float(value)
    if not (hasattr(qubit, "xy") and hasattr(qubit.xy, "operations")):
        return
    ops = qubit.xy.operations
    for name in _amp_nodes(operation):
        _set_op_amp(ops, name, val)


def get_thermalization_time(qubit: Any) -> float:
    """Passive-reset wait in SECONDS (QUAM stores ns).

    Reads the ABSOLUTE stored value, never the derived ``thermalization_time``
    property: the property falls back to ``factor * T1`` when nothing has been
    calibrated, and reporting that as scqo state would invent a knob value the
    loop never set. None (nothing stored) is scqo's "not seeded yet"."""
    ns = getattr(qubit, "thermalization_time_ns", None)
    return None if ns is None else float(ns) / 1e9


def set_thermalization_time(qubit: Any, value: float) -> None:
    """Write the passive-reset wait (seconds -> ns, rounded to the 4 ns grid).

    Rounded, NOT refused like :func:`set_pi_duration`: this is a policy wait
    (~10 x T1), not a calibrated pulse, so a sub-cycle difference changes
    nothing measurable and a T1-derived value is never on the grid by luck.
    Requires a thermalizing transmon class (``scqo_qm.quam_builder...
    thermalizing_transmon``) — QUAM's stock ``thermalization_time`` is a
    read-only ``factor * T1`` property with nowhere to store an absolute wait."""
    if not hasattr(qubit, "thermalization_time_ns"):
        raise NotImplementedError(
            f"{getattr(qubit, 'name', qubit)}: this qubit's QUAM class has no "
            f"thermalization_time_ns — its state.json __class__ must name a "
            f"Thermalizing*Transmon (stock QUAM derives the wait as "
            f"thermalization_time_factor * T1 and cannot store an absolute one)")
    ns = float(value) * 1e9
    if ns <= 0:
        raise ValueError(
            f"thermalization_time_s={value!r} s: must be positive")
    qubit.thermalization_time_ns = int(ns / 4) * 4


def get_pi_duration(qubit: Any, operation: str = PI_OPERATION) -> float:
    """Calibrated pi-pulse length in SECONDS (QUAM stores ns).

    / 1e9 (exact), never * 1e-9 - see get_readout_duration for why the division
    keeps the stored float identical to the parsed literal on both backends."""
    return float(qubit.xy.operations[operation].length) / 1e9


def set_pi_duration(qubit: Any, value: float, *, operation: str = PI_OPERATION) -> None:
    """Write the pi-pulse length (seconds, multiple of 4 ns).

    Same storage-node discipline as :func:`set_pi_amp`: the ``x180_DragCosine``
    node holds the real pulse and the plain ``x180`` entry is usually a reference
    alias that follows it. The 4 ns grid is REFUSED rather than rounded - unlike
    the Qblox side, where a pulse length is a plain seconds value, the QM pulse
    grid genuinely cannot realize an off-grid length, and silent rounding would
    de-calibrate the stored pi_amp against a pulse that is not the one measured.
    x90's length is NOT locked to it (a per-gate vendor value; see the
    x90_DragCosine fine print in the fieldmap)."""
    ns = _grid_ns(value, "pi_duration_s")
    if not (hasattr(qubit, "xy") and hasattr(qubit.xy, "operations")):
        return
    ops = qubit.xy.operations
    if operation == "x180":
        _set_op_length(ops, "x180_DragCosine", ns)
    _set_op_length(ops, operation, ns)


def _set_op_length(ops: Any, name: str, value: int) -> None:
    """Set ops[name].length, skipping string-reference aliases (the alias's
    target already carries the value)."""
    try:
        op = ops[name]
    except (KeyError, TypeError):
        return
    if isinstance(op, str):
        return
    if hasattr(op, "length"):
        op.length = int(value)


def _field_is_reference(op: Any, field: str) -> bool:
    """True when ``op.<field>``'s STORED value is a QUAM ``#...`` reference: it
    follows another node, so a literal write would sever the link — and real
    QUAM refuses one outright with a ValueError ("set the attribute to None
    first"). Real QUAM objects expose ``get_raw_value``; the plain test stubs
    carry a ``__quam__`` dict instead."""
    try:
        raw = op.get_raw_value(field)
    except Exception:
        try:
            raw = op.__quam__.get(field) if hasattr(op, "__quam__") else None
        except Exception:
            return False
    return isinstance(raw, str) and raw.startswith("#")


def _set_op_amp(ops: Any, name: str, value: float) -> None:
    """Set ``ops[name].amplitude``, skipping string-reference aliases AND nodes
    whose ``amplitude`` is itself a QUAM reference — the ``_set_op_alpha`` rule.

    Forcing a literal over a ``#``-reference would silently SEVER the alias, so
    later writes to the real node would stop propagating to it (the live states
    carry ``-x90_DragCosine.amplitude`` as exactly such a reference)."""
    try:
        op = ops[name]
    except (KeyError, TypeError):
        return
    if isinstance(op, str):  # a string-reference entry -> skip (target holds the value)
        return
    if _field_is_reference(op, "amplitude"):
        return
    if hasattr(op, "amplitude"):
        op.amplitude = value


# ------------------------------------------------------------------ saturation (spec) drive
#: The operation whose amplitude is the neutral ``drive_amp`` (the saturation /
#: spec drive played by qubit_spectroscopy; drive_power_dbm anchors to it).
SATURATION_OPERATION = "saturation"


def get_saturation_amp(qubit: Any, operation: str = SATURATION_OPERATION) -> float:
    return float(qubit.xy.operations[operation].amplitude)


def set_saturation_amp(qubit: Any, value: float, *, operation: str = SATURATION_OPERATION) -> None:
    """Write the saturation-pulse amplitude (within the current drive-chain
    config; the chain solve for an absolute power lives with drive_power_dbm in
    the scqo backend, next to the readout one)."""
    qubit.xy.operations[operation].amplitude = float(value)


# ------------------------------------------------------------------ DRAG coefficient
def get_drag_beta(qubit: Any, operation: str = PI_OPERATION) -> float:
    """Read the DRAG coefficient (QUAM ``DragCosinePulse.alpha``) from the operation's
    family, falling back to any node that carries one; guards each access so a single
    bad reference does not abort the search."""
    if not (hasattr(qubit, "xy") and hasattr(qubit.xy, "operations")):
        return 0.0
    ops = qubit.xy.operations
    for name in (*_amp_nodes(operation), *list(ops)):
        try:
            op = ops[name]
            if isinstance(op, str):
                continue
            alpha = op.alpha
            if alpha is not None:
                return float(alpha)
        except Exception:
            continue
    return 0.0


def set_drag_beta(qubit: Any, value: float, *, operation: str = PI_OPERATION) -> None:
    """Write the DRAG coefficient (QUAM ``DragCosinePulse.alpha``) on the operation's
    family -- the same storage nodes ``set_pi_amp`` writes.

    No ``lock_x90``: the pi/2 DRAG is calibrated in its own right and carries its own
    neutral knob, ``drag_beta_x90``.
    """
    val = float(value)
    if not (hasattr(qubit, "xy") and hasattr(qubit.xy, "operations")):
        return
    ops = qubit.xy.operations
    for name in _amp_nodes(operation):
        _set_op_alpha(ops, name, val)


def _set_op_alpha(ops: Any, name: str, value: float) -> None:
    """Set ``ops[name].alpha``, skipping string-reference aliases AND nodes whose
    ``alpha`` is itself a QUAM reference.

    Those follow their storage node by design -- ``-x90_DragCosine.alpha`` is exactly
    such a reference. Forcing a literal over one would silently SEVER the alias, so
    later writes to the real node would stop propagating to it. The old
    ``__quam__``-dict check missed references on REAL QUAM objects (which expose
    them through ``get_raw_value``), so QUAM's own ValueError killed the drag
    x90 path on the live state — issue #24.
    """
    try:
        op = ops[name]
    except (KeyError, TypeError):
        return
    if isinstance(op, str):
        return
    if _field_is_reference(op, "alpha"):
        return
    if hasattr(op, "alpha"):
        op.alpha = float(value)


